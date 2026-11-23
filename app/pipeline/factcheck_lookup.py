"""
Queries the Google Fact Check Tools API for existing professional
fact-checks matching a claim. If found, this is faster and more
authoritative than running our own retrieval pipeline from scratch.
"""
import requests

from app.config import settings


def lookup_factcheck(claim: str, max_results: int = 5) -> list[dict]:
    """
    Returns a list of matching fact-checks, each with:
      - publisher, title, url, textual_rating, review_date
    Returns an empty list if no API key is set, the request fails,
    or no matches are found.
    """
    if not settings.GOOGLE_FACTCHECK_API_KEY:
        print("No GOOGLE_FACTCHECK_API_KEY set — skipping fact-check DB lookup.")
        return []

    params = {
        "query": claim,
        "key": settings.GOOGLE_FACTCHECK_API_KEY,
        "languageCode": "en",
        "pageSize": max_results,
    }

    try:
        url = settings.GOOGLE_FACTCHECK_ENDPOINT
        print(f"Fact-check request URL: {url}")
        response = requests.get(url, params=params, timeout=10)
        print(f"Request final URL: {response.url}")
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        message = str(e)
        if hasattr(e, 'response') and e.response is not None:
            try:
                message += f" | response={e.response.text}"
            except Exception:
                pass
        print(f"Fact-check API request failed: {message}")
        return []

    claims = data.get("claims", [])
    results = []
    for c in claims:
        reviews = c.get("claimReview", [])
        for review in reviews:
            results.append({
                "claim_text": c.get("text"),
                "publisher": review.get("publisher", {}).get("name"),
                "title": review.get("title"),
                "url": review.get("url"),
                "textual_rating": review.get("textualRating"),
                "review_date": review.get("reviewDate"),
            })

    return results


if __name__ == "__main__":
    test_claim = "unemployment rate fell significantly this year"

    matches = lookup_factcheck(test_claim)

    print(f"Claim: {test_claim}\n")
    if not matches:
        print("No existing fact-checks found — would fall through to retrieval pipeline.")
    else:
        print(f"Found {len(matches)} existing fact-check(s):\n")
        for i, m in enumerate(matches, 1):
            print(f"{i}. [{m['publisher']}] {m['title']}")
            print(f"   Rating: {m['textual_rating']}")
            print(f"   {m['url']}\n") 