"""
stance_detection.py

For a given claim and a retrieved source, classifies the source's stance
toward the claim: supports / disputes / unrelated — with a short
rationale grounded in the source text (not the model's own knowledge).
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.config import settings


STANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You determine how a source relates to a claim.

Given a CLAIM and a SOURCE TEXT, classify the source's stance as exactly
one of:
  - "supports": the source text corroborates the claim
  - "disputes": the source text contradicts or complicates the claim
  - "unrelated": the source text doesn't actually address the claim

Base your judgment ONLY on the source text provided, not on outside
knowledge. Return ONLY a JSON object with:
  - "stance": one of "supports" / "disputes" / "unrelated"
  - "rationale": one sentence explaining why, referencing the source text
    in your own words (do not quote it directly)

Do not include any text outside the JSON object."""),
    ("user", "CLAIM: {claim}\n\nSOURCE TEXT: {source_text}")
])


def get_llm():
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )


def detect_stance(claim: str, source_text: str) -> dict:
    llm = get_llm()
    parser = JsonOutputParser()
    chain = STANCE_PROMPT | llm | parser

    try:
        result = chain.invoke({"claim": claim, "source_text": source_text})
        return result
    except Exception as e:
        print(f"Stance detection failed: {e}")
        return {"stance": "unrelated", "rationale": "Could not process source."}


def stance_report(claim: str, independent_groups: list[list]) -> list[dict]:
    """
    Takes claim + groups from source_independence.py (each group =
    one independent source, possibly multiple reprints). Returns one
    stance result per group, using the first document in each group
    as the representative text.
    """
    report = []
    for group in independent_groups:
        representative = group[0]
        result = detect_stance(claim, representative.page_content)
        report.append({
            "claim": claim,
            "stance": result.get("stance"),
            "rationale": result.get("rationale"),
            "source": representative.metadata.get("source"),
            "title": representative.metadata.get("title"),
            "url": representative.metadata.get("url"),
            "corroborating_reprints": len(group) - 1,
        })
    return report


if __name__ == "__main__":
    from .reranker import retrieve_and_rerank
    from .source_independence import group_by_independence

    test_claim = "National unemployment declined significantly over the past year."

    docs = retrieve_and_rerank(test_claim)
    groups = group_by_independence(docs)
    report = stance_report(test_claim, groups)

    print(f"Claim: {test_claim}\n")
    for i, r in enumerate(report, 1):
        print(f"{i}. [{r['stance'].upper()}] {r['source']} — {r['title']}")
        print(f"   {r['rationale']}")
        if r["corroborating_reprints"] > 0:
            print(f"   (+ {r['corroborating_reprints']} reprint(s) of this same source)")
        print()
