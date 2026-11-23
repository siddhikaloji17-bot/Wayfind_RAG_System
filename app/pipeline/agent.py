"""
agent.py

LangGraph orchestration of the full Veritrace/Wayfind pipeline.

Flow per claim:
  extract claims
    -> for each claim:
         check fact-check DB
           -> if matched: use it directly, done
           -> if not matched:
                hybrid retrieve -> rerank -> group by independence
                  -> if evidence too thin (< MIN_SOURCES independent
                     groups): broaden retrieval once, then continue
                     regardless (avoid infinite loops)
                -> stance detection per independent source
                -> assemble evidence report
"""
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from .claim_extraction import extract_claims
from .factcheck_lookup import lookup_factcheck
from .reranker import retrieve_and_rerank
from .retrieval import get_hybrid_retriever
from .source_independence import group_by_independence
from .stance_detection import stance_report


MIN_INDEPENDENT_SOURCES = 2


class ClaimState(TypedDict):
    claim: str
    factcheck_matches: List[Dict[str, Any]]
    retrieved_docs: List[Any]
    independent_groups: List[List[Any]]
    retry_count: int
    evidence_report: List[Dict[str, Any]]
    final_result: Dict[str, Any]


def node_factcheck_lookup(state: ClaimState) -> ClaimState:
    matches = lookup_factcheck(state["claim"])
    state["factcheck_matches"] = matches
    return state


def route_after_factcheck(state: ClaimState) -> str:
    return "have_factcheck" if state["factcheck_matches"] else "need_retrieval"


def node_retrieve(state: ClaimState) -> ClaimState:
    docs = retrieve_and_rerank(state["claim"])
    state["retrieved_docs"] = docs
    state["independent_groups"] = group_by_independence(docs)
    return state


def route_after_retrieve(state: ClaimState) -> str:
    enough_sources = len(state["independent_groups"]) >= MIN_INDEPENDENT_SOURCES
    already_retried = state.get("retry_count", 0) >= 1
    if enough_sources or already_retried:
        return "proceed_to_stance"
    return "broaden_retrieval"


def node_broaden_retrieval(state: ClaimState) -> ClaimState:
    # Simple broadening strategy: pull more candidates from the base
    # hybrid retriever directly (bypassing the tighter rerank top-k),
    # then re-group. A more advanced version could rewrite the query
    # itself (e.g. drop specific numbers, generalize the phrasing).
    retriever = get_hybrid_retriever()
    docs = retriever.invoke(state["claim"])
    state["retrieved_docs"] = docs
    state["independent_groups"] = group_by_independence(docs)
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def node_stance_detection(state: ClaimState) -> ClaimState:
    report = stance_report(state["claim"], state["independent_groups"])
    state["evidence_report"] = report
    return state


def node_assemble_factcheck_result(state: ClaimState) -> ClaimState:
    state["final_result"] = {
        "claim": state["claim"],
        "resolution": "existing_factcheck",
        "matches": state["factcheck_matches"],
    }
    return state


def node_assemble_evidence_result(state: ClaimState) -> ClaimState:
    state["final_result"] = {
        "claim": state["claim"],
        "resolution": "evidence_report",
        "sources_evaluated": len(state["independent_groups"]),
        "evidence": state["evidence_report"],
    }
    return state


def build_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("factcheck_lookup", node_factcheck_lookup)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("broaden_retrieval", node_broaden_retrieval)
    graph.add_node("stance_detection", node_stance_detection)
    graph.add_node("assemble_factcheck_result", node_assemble_factcheck_result)
    graph.add_node("assemble_evidence_result", node_assemble_evidence_result)

    graph.set_entry_point("factcheck_lookup")

    graph.add_conditional_edges(
        "factcheck_lookup",
        route_after_factcheck,
        {
            "have_factcheck": "assemble_factcheck_result",
            "need_retrieval": "retrieve",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "proceed_to_stance": "stance_detection",
            "broaden_retrieval": "broaden_retrieval",
        },
    )

    # after broadening, always proceed (retry_count now >= 1 prevents looping)
    graph.add_edge("broaden_retrieval", "stance_detection")

    graph.add_edge("stance_detection", "assemble_evidence_result")
    graph.add_edge("assemble_factcheck_result", END)
    graph.add_edge("assemble_evidence_result", END)

    return graph.compile()


def verify_text(text: str) -> Dict[str, Any]:
    """
    Full pipeline entry point: raw text in, list of per-claim
    verification results out.
    """
    claims = extract_claims(text)
    graph = build_graph()

    results = []
    for c in claims:
        initial_state: ClaimState = {
            "claim": c["claim"],
            "factcheck_matches": [],
            "retrieved_docs": [],
            "independent_groups": [],
            "retry_count": 0,
            "evidence_report": [],
            "final_result": {},
        }
        final_state = graph.invoke(initial_state)
        results.append(final_state["final_result"])

    return {"input_text": text, "claims_checked": len(claims), "results": results}


if __name__ == "__main__":
    test_input = (
        "National unemployment declined significantly this year. "
        "TechCorp also laid off thousands of workers last quarter."
    )

    output = verify_text(test_input)

    print(f"Input: {output['input_text']}")
    print(f"Claims checked: {output['claims_checked']}\n")

    for r in output["results"]:
        print(f"Claim: {r['claim']}")
        print(f"Resolution: {r['resolution']}")
        if r["resolution"] == "evidence_report":
            for e in r["evidence"]:
                print(f"  [{e['stance'].upper()}] {e['source']}: {e['rationale']}")
        else:
            for m in r["matches"]:
                print(f"  [{m['publisher']}] {m['textual_rating']}: {m['title']}")
        print()
