import os
import re
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import neo4j
from dotenv import load_dotenv
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever, Text2CypherRetriever, ToolsRetriever
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation import RagTemplate, GraphRAG

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 lifespan 이벤트"""
    # Startup
    try:
        initialize_retrievers()
        print("✅ Retriever 초기화 완료")
    except Exception as e:
        print(f"⚠️ 경고: Retriever 초기화 실패: {e}")
    yield
    # Shutdown
    driver.close()
    print("🔌 Neo4j 드라이버 종료")


app = FastAPI(title="GraphRAG Demo API", lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j 연결
URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
AUTH = ("neo4j", os.getenv("NEO4J_PASSWORD", "password"))
driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

# LLM 및 Embedder 설정
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={"temperature": 0}
)
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# 전역 변수로 retriever 저장
INDEX_NAME = "content_vector_index"
vector_retriever = None
vector_cypher_retriever = None
text2cypher_retriever = None
tools_retriever = None
graphrag = None


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    used_nodes: List[str]
    used_edges: List[str]
    retriever_used: str
    context: str = ""


def extract_all_field_values(text: str, field_name: str) -> List[str]:
    """텍스트에서 특정 필드의 모든 값을 추출 (다양한 형식 지원)"""
    values = []

    # 패턴 1: field_name='value' 또는 field_name="value" (Python repr 스타일)
    pattern1 = rf"['\"]?{field_name}['\"]?\s*=\s*['\"]([^'\"]+)['\"]"
    values.extend(re.findall(pattern1, text))

    # 패턴 2: 'field_name': 'value' (딕셔너리 스타일)
    pattern2 = rf"['\"]?{field_name}['\"]?\s*:\s*['\"]([^'\"]+)['\"]"
    values.extend(re.findall(pattern2, text))

    # 패턴 3: ART_로 시작하는 article_id 직접 추출
    if field_name == "article_id":
        pattern3 = r"(ART_\d{3}_\d{10})"
        values.extend(re.findall(pattern3, text))

    # 패턴 4: content_id 직접 추출 (ART_XXX_XXXXXXXXXX_chunk_N 형식)
    if field_name == "content_id":
        pattern4 = r"(ART_\d{3}_\d{10}_chunk_\d+)"
        values.extend(re.findall(pattern4, text))

    # 중복 제거 후 반환
    return list(set(values))


def is_valid_article_id(article_id: str) -> bool:
    """유효한 article_id 형식인지 확인"""
    # ART_로 시작하고 숫자 패턴을 따르는지 확인
    return bool(re.match(r"^ART_\d{3}_\d{10}$", article_id))


def is_valid_category(category: str) -> bool:
    """유효한 카테고리명인지 확인"""
    # 한글 카테고리만 허용 (정치, 경제, 사회 등)
    invalid_values = ["Unknown", "No title", "text2cypher_retriever",
                      "vector_retriever", "vectorcypher_retriever"]
    if category in invalid_values:
        return False
    # retriever 관련 문자열 제외
    if "retriever" in category.lower():
        return False
    return len(category) > 0


def extract_nodes_edges_from_cypher(cypher_query: str) -> tuple[List[str], List[str]]:
    """Cypher 쿼리에서 사용된 노드와 엣지 추출"""
    nodes = []
    edges = []

    # 노드 레이블 추출 (예: (a:Article), (c:Category))
    node_pattern = r'\([a-z]+:(\w+)(?:\s+\{[^}]*\})?\)'
    node_matches = re.findall(node_pattern, cypher_query)
    nodes.extend(node_matches)

    # 관계 타입 추출 (예: [:BELONGS_TO], [:PUBLISHED])
    edge_pattern = r'\[:(\w+)\]'
    edge_matches = re.findall(edge_pattern, cypher_query)
    edges.extend(edge_matches)

    return list(set(nodes)), list(set(edges))


def extract_nodes_from_content(content: str) -> tuple[List[str], List[str]]:
    """검색 결과 content에서 노드/엣지 추출 (모든 값 추출)"""
    nodes = []
    edges = []

    # article_id - 모든 article_id 값 추출 (유효성 검사 포함)
    article_ids = extract_all_field_values(content, "article_id")
    for article_id in article_ids:
        if article_id and is_valid_article_id(article_id):
            nodes.append(f"Article_{article_id}")

    # category_name - 모든 category 값 추출 (유효성 검사 포함)
    categories = extract_all_field_values(content, "category_name")
    for category in categories:
        if is_valid_category(category):
            nodes.append(f"Category_{category}")

    # content_id - 모든 content_id 값 추출
    content_ids = extract_all_field_values(content, "content_id")
    for content_id in content_ids:
        if content_id:
            nodes.append(f"Content_{content_id}")

    # 관계 추정
    if article_ids and categories:
        edges.append("BELONGS_TO")
    if article_ids and content_ids:
        edges.append("HAS_CHUNK")

    return list(set(nodes)), list(set(edges))


def extract_vectorcypher_nodes(content: str) -> tuple[List[str], List[str]]:
    """VectorCypherRetriever 결과에서 핵심 노드만 추출 (related_articles 제외)"""
    nodes = []
    edges = []

    # related_articles 이전 부분만 파싱 (관련 기사 제외)
    related_idx = content.find("related_articles")
    if related_idx > 0:
        main_content = content[:related_idx]
    else:
        main_content = content

    # content_id 추출 (이스케이프된 따옴표 포함)
    # 형식: content_id=\'ART_138_0002210299_chunk_0\'
    content_id_match = re.search(r"content_id=\\?['\"]?(ART_\d{3}_\d{10}_chunk_\d+)", main_content)
    if content_id_match:
        content_id = content_id_match.group(1)
        nodes.append(f"Content_{content_id}")

        # content_id에서 article_id 추출 (chunk 부분 제거)
        article_id = re.sub(r"_chunk_\d+$", "", content_id)
        if is_valid_article_id(article_id):
            nodes.append(f"Article_{article_id}")
            edges.append("HAS_CHUNK")

    # article_id 추출 (메인 기사) - content_id에서 못 찾은 경우
    if not any("Article_" in n for n in nodes):
        article_match = re.search(r"article_id=\\?['\"]?(ART_\d{3}_\d{10})", main_content)
        if article_match:
            article_id = article_match.group(1)
            if is_valid_article_id(article_id):
                nodes.append(f"Article_{article_id}")

    # category_name 추출
    category_match = re.search(r"category_name=\\?['\"]?([가-힣/]+)", main_content)
    if category_match:
        category = category_match.group(1).strip()
        if is_valid_category(category):
            nodes.append(f"Category_{category}")
            edges.append("BELONGS_TO")

    return list(set(nodes)), list(set(edges))


def initialize_retrievers():
    """Retrievers 초기화"""
    global vector_retriever, vector_cypher_retriever, text2cypher_retriever, tools_retriever, graphrag

    # Vector Retriever
    vector_retriever = VectorRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        embedder=embedder
    )

    # VectorCypher Retriever
    retrieval_query = """
    WITH node AS content, score
    MATCH (content)<-[:HAS_CHUNK]-(article:Article)
    OPTIONAL MATCH (article)-[:BELONGS_TO]->(category:Category)
    OPTIONAL MATCH (category)<-[:BELONGS_TO]-(related_article:Article)
    WHERE related_article <> article

    RETURN
        content.content_id AS content_id,
        content.chunk AS chunk,
        content.title AS content_title,
        article.article_id AS article_id,
        article.title AS article_title,
        article.url AS article_url,
        article.published_date AS article_date,
        category.name AS category_name,
        score AS similarity_score,
        collect(DISTINCT {
            article_id: related_article.article_id,
            title: related_article.title,
            url: related_article.url,
            published_date: related_article.published_date
        })[0..5] AS related_articles
    """

    vector_cypher_retriever = VectorCypherRetriever(
        driver=driver,
        index_name=INDEX_NAME,
        retrieval_query=retrieval_query,
        embedder=embedder
    )

    # Text2Cypher Retriever
    neo4j_schema = get_neo4j_schema()
    examples = [
        """
        USER INPUT: 경제 분야의 최신 뉴스 알려주세요
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: "경제"})
        RETURN a.article_id, a.title, a.url, a.published_date
        ORDER BY a.published_date DESC
        LIMIT 10
        """,
        """
        USER INPUT: 정치 카테고리의 최신 기사 5개를 보여주세요
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: "정치"})
        RETURN a.article_id, a.title, a.url, a.published_date
        ORDER BY a.published_date DESC
        LIMIT 5
        """,
        """
        USER INPUT: 카테고리별 기사 개수를 알려주세요
        CYPHER QUERY:
        MATCH (a:Article)-[:BELONGS_TO]->(c:Category)
        RETURN c.name as category, count(a) as article_count
        ORDER BY article_count DESC
        """,
    ]

    text2cypher_retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    vector_tool = vector_retriever.convert_to_tool(
        name="vector_retriever",
        description="키워드나 개념으로 유사한 기사를 빠르게 찾을 때 사용. 단순 검색용."
    )
    vector_cypher_tool = vector_cypher_retriever.convert_to_tool(
        name="vectorcypher_retriever",
        description="기사의 '상세 정보', '전체 정보', '내용', '본문'을 요청할 때 사용. 특정 주제/키워드로 기사를 찾고 제목, URL, 날짜, 카테고리, 관련 기사까지 상세 정보를 반환."
    )
    text2cypher_tool = text2cypher_retriever.convert_to_tool(
        name="text2cypher_retriever",
        description="카테고리별 기사 수, 특정 카테고리의 기사 목록, 통계, 집계 등 그래프 구조 기반 쿼리에 사용. 예: '경제 카테고리 기사', '카테고리별 기사 개수'"
    )

    tools_retriever = ToolsRetriever(
        driver=driver,
        llm=llm,
        tools=[vector_tool, vector_cypher_tool, text2cypher_tool],
    )

    prompt_template = RagTemplate(
        template="""당신은 뉴스 기사 정보를 제공하는 전문 어시스턴트입니다.

질문: {query_text}

검색된 문서 정보:
{context}

지침:
1. 사용자의 질문에 직접적으로 답변하세요.
2. **검색된 모든 기사**를 빠짐없이 답변에 포함하세요. 일부만 선택하지 마세요.
3. 먼저 검색된 기사들을 종합 분석한 답변을 제공하세요.
4. 답변 마지막에 출처(기사 목록)를 정리하세요.
5. 각 기사의 제목(title), URL(url), 발행일(published_date)을 모두 포함하세요.
6. 검색 결과에 없는 내용은 추측하지 마세요.

기사 목록 답변 형식:

**검색된 기사 목록 (총 N건):**
1. **[기사 제목]** (발행일)
   - URL: [기사 URL]
   - 핵심: [한 줄 요약]

2. **[기사 제목]** (발행일)
   - URL: [기사 URL]
   - 핵심: [한 줄 요약]

답변:""",
        expected_inputs=["context", "query_text"]
    )

    graphrag = GraphRAG(
        llm=llm,
        retriever=tools_retriever,
        prompt_template=prompt_template
    )


def get_neo4j_schema() -> str:
    """Neo4j 스키마 정보 가져오기"""
    with driver.session() as session:
        node_info = session.run("""
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType, propertyName, propertyTypes
            RETURN nodeType, collect(propertyName) as properties
        """).data()

        patterns = session.run("""
            MATCH (n)-[r]->(m)
            RETURN DISTINCT labels(n)[0] as source, type(r) as relationship, labels(m)[0] as target
            LIMIT 20
        """).data()

        schema_text = "=== Neo4j Schema ===\n\n노드 타입:\n"
        for node in node_info:
            schema_text += f"- {node['nodeType']}: {node['properties']}\n"

        schema_text += "\n관계 패턴:\n"
        for pattern in patterns:
            schema_text += f"- ({pattern['source']})-[:{pattern['relationship']}]->({pattern['target']})\n"

        return schema_text


@app.get("/")
async def root():
    """루트 페이지 - 시각화 HTML 반환"""
    return FileResponse("index.html")


@app.get("/graph")
async def get_graph():
    """전체 그래프 구조 반환"""
    try:
        with driver.session() as session:
            # 노드 가져오기 (모든 노드)
            nodes_result = session.run("""
                MATCH (n)
                WHERE n:Article OR n:Category OR n:Content OR n:Media
                RETURN
                    elementId(n) as id,
                    labels(n)[0] as label,
                    CASE
                        WHEN n:Article THEN n.title
                        WHEN n:Category THEN n.name
                        WHEN n:Content THEN substring(n.chunk, 0, 50) + '...'
                        WHEN n:Media THEN n.name
                        ELSE 'Unknown'
                    END as title,
                    properties(n) as properties
            """)

            nodes = []
            for record in nodes_result:
                nodes.append({
                    "id": record["id"],
                    "label": record["label"],
                    "title": record["title"] or "No title",
                    "properties": dict(record["properties"])
                })

            # 엣지 가져오기 (모든 엣지)
            edges_result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE (n:Article OR n:Category OR n:Content OR n:Media)
                  AND (m:Article OR m:Category OR m:Content OR m:Media)
                RETURN
                    elementId(r) as id,
                    elementId(n) as source,
                    elementId(m) as target,
                    type(r) as relationship
            """)

            edges = []
            for record in edges_result:
                edges.append({
                    "id": record["id"],
                    "source": record["source"],
                    "target": record["target"],
                    "relationship": record["relationship"]
                })

            return {
                "nodes": nodes,
                "edges": edges
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph retrieval failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_graphrag(req: QueryRequest):
    """GraphRAG 쿼리 실행 및 사용된 노드/엣지 반환"""
    try:
        if graphrag is None:
            raise HTTPException(status_code=500, detail="GraphRAG not initialized")

        result = graphrag.search(query_text=req.question, return_context=True)

        # 사용된 노드와 엣지 추출
        used_nodes = []
        used_edges = []
        retriever_used = "unknown"
        context_str = ""

        print(f"\n=== 쿼리: {req.question} ===")

        # result 객체 처리
        try:
            if hasattr(result, 'retriever_result') and result.retriever_result:
                retriever_result = result.retriever_result

                # 선택된 retriever 확인
                if hasattr(retriever_result, 'metadata') and retriever_result.metadata:
                    tools_selected = retriever_result.metadata.get('tools_selected', [])
                    print(f"📌 선택된 Retriever: {tools_selected}")

                # items 처리
                if hasattr(retriever_result, 'items') and retriever_result.items:
                    filtered_count = 0

                    for idx, item in enumerate(retriever_result.items):
                        # retriever 이름 추출
                        if hasattr(item, 'metadata') and item.metadata:
                            if 'tool' in item.metadata:
                                retriever_used = item.metadata['tool']
                            elif 'retriever_name' in item.metadata:
                                retriever_used = item.metadata['retriever_name']

                            # VectorRetriever: score 기반 필터링
                            if 'id' in item.metadata and 'nodeLabels' in item.metadata:
                                node_id = item.metadata['id']
                                node_labels = item.metadata['nodeLabels']
                                score = item.metadata.get('score', 1.0)

                                if 'Content' in node_labels and score >= 0.7:
                                    used_nodes.append(f"ElementId_{node_id}")
                                    used_edges.append("HAS_CHUNK")
                                    filtered_count += 1
                            else:
                                filtered_count += 1

                        # content 처리
                        if hasattr(item, 'content'):
                            content = str(item.content)
                            context_str += content + "\n\n"

                            # VectorRetriever가 아닌 경우 노드/엣지 추출
                            if retriever_used != "vector_retriever":
                                if retriever_used == "vectorcypher_retriever":
                                    nodes, edges = extract_vectorcypher_nodes(content)
                                else:
                                    nodes, edges = extract_nodes_from_content(content)

                                used_nodes.extend(nodes)
                                used_edges.extend(edges)

                            # Text2Cypher: 카테고리 추론
                            if retriever_used == "text2cypher_retriever":
                                if not any("Category" in n for n in used_nodes):
                                    for cat in ["정치", "경제", "사회", "생활/문화", "스포츠", "IT/과학"]:
                                        if cat in req.question:
                                            used_nodes.append(f"Category_{cat}")
                                            used_edges.append("BELONGS_TO")
                                            break

        except Exception as parse_error:
            print(f"⚠️ 파싱 오류: {parse_error}")

        # 중복 제거
        used_nodes = list(set(used_nodes))
        used_edges = list(set(used_edges))

        print(f"🎯 최종 반환 노드: {used_nodes}")
        print(f"🔗 최종 반환 엣지: {used_edges}")

        return QueryResponse(
            answer=result.answer if hasattr(result, 'answer') else str(result),
            used_nodes=used_nodes,
            used_edges=used_edges,
            retriever_used=retriever_used,
            context=context_str[:1000] if context_str else "No context available"
        )

    except Exception as e:
        import traceback
        error_detail = f"Query failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/health")
async def health_check():
    """헬스 체크"""
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return {"status": "healthy", "neo4j": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
