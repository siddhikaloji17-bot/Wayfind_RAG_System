"""
source_independence.py

Groups retrieved sources that likely share a common origin (e.g. wire
service reprints), so the pipeline doesn't overcount "5 sources agree"
when it's really 1 story republished 5 times.
"""
from difflib import SequenceMatcher


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def group_by_independence(documents: list, similarity_threshold: float = 0.9) -> list[list]:
    """
    Returns a list of groups (each a list of Documents). Documents whose
    page_content is near-identical (likely reprints of the same story)
    are grouped together; each group counts as ONE independent source.
    """
    groups = []

    for doc in documents:
        placed = False
        for group in groups:
            representative = group[0]
            if _similarity(doc.page_content, representative.page_content) >= similarity_threshold:
                group.append(doc)
                placed = True
                break
        if not placed:
            groups.append([doc])

    return groups


if __name__ == "__main__":
    from .reranker import retrieve_and_rerank

    test_query = "unemployment rate decline"
    docs = retrieve_and_rerank(test_query)

    groups = group_by_independence(docs)

    print(f"Query: {test_query}")
    print(f"Retrieved {len(docs)} documents, grouped into {len(groups)} independent source(s):\n")
    for i, group in enumerate(groups, 1):
        sources = [d.metadata.get("source") for d in group]
        print(f"Group {i} ({len(group)} doc(s), likely same origin): {sources}")
