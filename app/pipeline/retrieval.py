"""
Hybrid retrieval: combines BM25 (keyword) and FAISS (dense vector) search
using LangChain's EnsembleRetriever, which merges results with Reciprocal
Rank Fusion (RRF).
"""
import pickle

from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from app.config import settings

# NOTE: swap this import to match whatever embedding class your
# ingestion script actually used (OpenAIEmbeddings or a local
# sentence-transformers wrapper like HuggingFaceEmbeddings) —
# it must be the SAME embedding model used to build the index,
# or similarity scores will be meaningless.
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_bm25_retriever(path: str = "data/bm25_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_vector_retriever(index_dir: str = None, k: int = 8):
    index_dir = index_dir or settings.VECTOR_INDEX_DIR
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_hybrid_retriever(bm25_weight: float = 0.4, vector_weight: float = 0.6):
    bm25_retriever = load_bm25_retriever()
    bm25_retriever.k = settings.HYBRID_TOP_K

    vector_retriever = load_vector_retriever(k=settings.HYBRID_TOP_K)

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )
    return ensemble


if __name__ == "__main__":
    # quick manual test
    retriever = get_hybrid_retriever()
    test_query = "unemployment rate decline"
    results = retriever.invoke(test_query)

    print(f"Query: {test_query}\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. [{doc.metadata.get('source')}] {doc.metadata.get('title')}")
        print(f"   {doc.page_content[:100]}...\n")