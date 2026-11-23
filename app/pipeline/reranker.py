"""
Reranks hybrid retrieval results using a cross-encoder, which scores the
query and each candidate document together (more accurate than the
separate-embeddings approach used in retrieval.py, but slower — which is
why it only runs on the top K candidates, not the whole corpus).
"""
from sentence_transformers import CrossEncoder

from app.config import settings
from .retrieval import get_hybrid_retriever


_reranker_model = None


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker_model


def rerank(query: str, documents: list, top_k: int = None):
    """
    documents: list of LangChain Document objects (from the hybrid retriever)
    Returns the same documents, reordered by cross-encoder relevance score.
    """
    top_k = top_k or settings.RERANK_TOP_K
    if not documents:
        return []

    model = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_k]]


def retrieve_and_rerank(query: str):
    retriever = get_hybrid_retriever()
    candidates = retriever.invoke(query)
    reranked = rerank(query, candidates)
    return reranked


if __name__ == "__main__":
    test_query = "unemployment rate decline"

    retriever = get_hybrid_retriever()
    before = retriever.invoke(test_query)

    print("BEFORE reranking:")
    for i, doc in enumerate(before, 1):
        print(f"{i}. [{doc.metadata.get('source')}] {doc.metadata.get('title')}")

    after = rerank(test_query, before)

    print("\nAFTER reranking:")
    for i, doc in enumerate(after, 1):
        print(f"{i}. [{doc.metadata.get('source')}] {doc.metadata.get('title')}")
