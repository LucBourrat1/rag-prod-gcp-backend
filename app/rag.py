import asyncio
import os
import tempfile
import uuid
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "rag-tutorial"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "rag-tutorial-gcp"

_qdrant_client: QdrantClient | None = None

# ── State du graph ──────────────────────────────────────────────────────────


class RAGState(TypedDict):
    question: str
    document_ids: list[str]
    context: list[Document]  # rempli par le nœud retrieve
    answer: str  # rempli par le nœud generate


# ── Clients partagés ────────────────────────────────────────────────────────


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        if not _qdrant_client.collection_exists(COLLECTION_NAME):
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(size=1536, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams())
                },
            )
    return _qdrant_client


def get_vectorstore() -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


# ── Nœuds du graph ──────────────────────────────────────────────────────────


def retrieve(state: RAGState) -> RAGState:
    """Nœud 1 : récupère les chunks pertinents depuis Qdrant (hybrid search)."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {
                "must": [{"key": "doc_id", "match": {"any": state["document_ids"]}}]
            },
        }
    )
    docs = retriever.invoke(state["question"])
    return {**state, "context": docs}


def generate(state: RAGState) -> RAGState:
    """Nœud 2 : génère la réponse à partir du contexte récupéré."""
    context_text = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = ChatPromptTemplate.from_template("""
Réponds à la question en te basant uniquement sur le contexte suivant.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte:
{context}

Question: {question}
""")
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": state["question"]})
    return {**state, "answer": answer}


# ── Construction du graph ────────────────────────────────────────────────────


def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# Graph compilé une seule fois au démarrage du process
_rag_graph = build_rag_graph()

# ── Fonctions publiques appelées par main.py ─────────────────────────────────


async def process_pdf(
    content: bytes, filename: str, user_email: str, gcs_path: str
) -> str:
    doc_id = str(uuid.uuid4())
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = filename
            chunk.metadata["user_email"] = user_email
            chunk.metadata["gcs_path"] = gcs_path
        vectorstore = get_vectorstore()
        await asyncio.to_thread(vectorstore.add_documents, chunks)
    finally:
        if tmp_path:
            os.unlink(tmp_path)
    return doc_id


async def get_user_documents(user_email: str) -> list:
    """Récupère les documents d'un user directement depuis les payloads Qdrant.
    On déduplique par doc_id — chaque PDF a plusieurs chunks mais on n'affiche qu'une ligne.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_qdrant_client()
    results, _ = await asyncio.to_thread(
        client.scroll,
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="user_email", match=MatchValue(value=user_email))]
        ),
        with_payload=True,
        limit=1000,
    )
    seen = {}
    for point in results:
        doc_id = point.payload.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": point.payload.get("filename"),
                "gcs_path": point.payload.get("gcs_path"),
                "status": "indexed",
            }
    return list(seen.values())


async def ask_question(question: str, document_ids: list[str], user_email: str) -> dict:
    """Lance le graph LangGraph et formate la réponse avec les sources."""
    initial_state: RAGState = {
        "question": question,
        "document_ids": document_ids,
        "context": [],
        "answer": "",
    }
    # graph.invoke est synchrone — on wrappe pour ne pas bloquer FastAPI
    final_state = await asyncio.to_thread(_rag_graph.invoke, initial_state)

    sources = [
        {
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page", 0) + 1,
            "excerpt": doc.page_content[:200] + "...",
        }
        for doc in final_state["context"]
    ]
    return {"answer": final_state["answer"], "sources": sources}
