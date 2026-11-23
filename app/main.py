"""Wayfind API: evidence-grounded retrieval, ingestion, and claim checking."""
import json
import math
import os
import re
import base64
import binascii
import hmac
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Wayfind — Evidence Workspace",
    description="Evidence-grounded answers and claim analysis over a document collection.",
    version="1.0.0",
)

# Keep browser access local by default; set WAYFIND_ALLOWED_ORIGINS to a
# comma-separated list when deploying behind an authenticated frontend.
_allowed_origins = [origin.strip() for origin in os.getenv(
    "WAYFIND_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=12000, description="Text to check.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: list[str] | None = None


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    document_ids: list[str] = Field(..., min_length=2, max_length=10)


class VerifyResponse(BaseModel):
    input_text: str
    claims_checked: int
    results: list


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SAMPLE_CORPUS_PATH = PROJECT_DIR / "data" / "sample_corpus.jsonl"
DATA_DIR = Path(os.getenv("WAYFIND_DATA_DIR", str(PROJECT_DIR / "data")))
CORPUS_PATH = DATA_DIR / "user_corpus.jsonl"
MAX_UPLOAD_BYTES = 2_000_000
_corpus_lock = RLock()
_app_environment = os.getenv("WAYFIND_ENV", "development").casefold()
_access_username = os.getenv("WAYFIND_ACCESS_USERNAME", "")
_access_password = os.getenv("WAYFIND_ACCESS_PASSWORD", "")

if _app_environment == "production" and (not _access_username or not _access_password):
    raise RuntimeError("Set WAYFIND_ACCESS_USERNAME and WAYFIND_ACCESS_PASSWORD in production.")


@app.middleware("http")
async def require_access_credentials(request, call_next):
    """Protect a public deployment while leaving its platform health check open."""
    if request.url.path != "/health" and _app_environment == "production":
        auth = request.headers.get("authorization", "")
        try:
            scheme, encoded = auth.split(" ", 1)
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            supplied_user, separator, supplied_password = decoded.partition(":")
            valid = (
                scheme.casefold() == "basic" and separator
                and hmac.compare_digest(supplied_user, _access_username)
                and hmac.compare_digest(supplied_password, _access_password)
            )
        except (ValueError, UnicodeDecodeError, binascii.Error):
            valid = False
        if not valid:
            return JSONResponse(
                status_code=401,
                content={"detail": "Sign in to use Wayfind."},
                headers={"WWW-Authenticate": 'Basic realm="Wayfind", charset="UTF-8"'},
            )
    return await call_next(request)


def _load_documents() -> list[dict]:
    """Load the shipped sample corpus plus user-added documents."""
    docs = []
    for path in (SAMPLE_CORPUS_PATH, CORPUS_PATH):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    if item.get("text"):
                        docs.append(item)
                except (json.JSONDecodeError, TypeError):
                    continue
    return docs


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w'-]{2,}", text.casefold())


def _retrieve(question: str, top_k: int = 5, document_ids: list[str] | None = None) -> list[dict]:
    """Small, transparent BM25-style scorer; no model call or network required."""
    docs = _load_documents()
    if document_ids:
        allowed = set(document_ids)
        docs = [doc for doc in docs if str(doc.get("document_id", doc.get("id"))) in allowed]
    query_terms = set(_tokens(question))
    if not query_terms or not docs:
        return []
    tokenized = [_tokens(f"{d.get('title', '')} {d.get('text', '')}") for d in docs]
    df = Counter(term for terms in tokenized for term in set(terms))
    avg_len = sum(map(len, tokenized)) / max(len(tokenized), 1)
    ranked = []
    for doc, terms in zip(docs, tokenized):
        freq = Counter(terms)
        score = 0.0
        for term in query_terms:
            count = freq[term]
            if count:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * len(terms) / max(avg_len, 1)))
        if score > 0:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [{**doc, "relevance": round(1 - math.exp(-score / 5), 3)} for score, doc in ranked[:top_k]]


def _chunks(text: str, limit: int = 1100) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs or [text.strip()]:
        while len(para) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:limit])
            para = para[limit:]
        if current and len(current) + len(para) + 1 > limit:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def _verify_text(text: str) -> dict:
    """Return source matches without claiming that lexical similarity is fact verification."""
    claims = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    results = []
    for claim in claims[:20]:
        matches = _retrieve(claim, 5)
        evidence = [{"stance": "context", "source": doc.get("source", "Unknown source"),
                     "title": doc.get("title", "Untitled"), "url": doc.get("url"),
                     "rationale": doc.get("text", "")[:700], "relevance": doc["relevance"]}
                    for doc in matches if doc["relevance"] >= 0.12]
        results.append({"claim": claim, "resolution": "evidence_report",
                        "sources_evaluated": len(evidence), "evidence": evidence})
    return {"input_text": text, "claims_checked": len(results), "results": results}


@app.get("/health")
def health():
    docs = _load_documents()
    return {"status": "ok", "service": "wayfind", "documents": len(docs), "retrieval": "local BM25-style search"}


@app.get("/documents")
def list_documents():
    docs = _load_documents()
    unique = {}
    for doc in docs:
        doc_id = str(doc.get("document_id", doc.get("id")))
        if doc_id not in unique:
            unique[doc_id] = {"id": doc_id, "title": doc.get("title", doc.get("source", "Untitled")),
                              "source": doc.get("source", "Unknown source"), "url": doc.get("url"),
                              "chunks": 0, "added_at": doc.get("added_at")}
        unique[doc_id]["chunks"] += 1
    return {"documents": list(unique.values()), "count": len(unique)}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    name = Path(file.filename or "document").name
    suffix = Path(name).suffix.casefold()
    if suffix not in {".txt", ".md", ".csv"}:
        raise HTTPException(status_code=415, detail="Upload a UTF-8 .txt, .md, or .csv file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than the 2 MB limit.")
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded.")
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Add at least 20 characters of readable text.")
    document_id = str(uuid4())
    records = [{"id": f"{document_id}:{i + 1}", "document_id": document_id, "source": name, "title": f"{name} · part {i + 1}",
                "text": chunk, "url": None, "added_at": datetime.now(timezone.utc).isoformat()}
               for i, chunk in enumerate(_chunks(text))]
    with _corpus_lock:
        CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CORPUS_PATH.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"filename": name, "chunks_added": len(records), "document_id": document_id}


@app.post("/ask")
def ask(request: AskRequest):
    matches = _retrieve(request.question, request.top_k, request.document_ids)
    if not matches or matches[0]["relevance"] < 0.12:
        return {"question": request.question, "answer": "I couldn't find enough relevant evidence in this collection to answer reliably. Try adding a source document or rephrasing the question.", "confidence": "low", "abstained": True, "sources": []}
    evidence = []
    for doc in matches:
        sentences = re.split(r"(?<=[.!?])\s+", doc.get("text", ""))
        query_terms = set(_tokens(request.question))
        sentence = max(sentences, key=lambda s: len(query_terms.intersection(_tokens(s))), default=doc.get("text", ""))
        evidence.append({"id": doc.get("id"), "source": doc.get("source", "Unknown source"),
                         "title": doc.get("title", "Untitled"), "url": doc.get("url"),
                         "excerpt": sentence[:700], "relevance": doc["relevance"]})
    strong = len([e for e in evidence if e["relevance"] >= 0.45])
    return {"question": request.question,
            "answer": "The collection contains these relevant findings:\n\n" + "\n\n".join(f"• {e['excerpt']} [{i + 1}]" for i, e in enumerate(evidence[:3])),
            "confidence": "moderate" if strong > 1 else "limited", "abstained": False, "sources": evidence}


@app.post("/compare")
def compare(request: CompareRequest):
    docs = _load_documents()
    available = {str(d.get("document_id", d.get("id"))) for d in docs}
    missing = [doc_id for doc_id in request.document_ids if doc_id not in available]
    if missing:
        raise HTTPException(status_code=404, detail="One or more selected documents were not found.")
    rows = []
    for doc_id in request.document_ids:
        hits = _retrieve(request.question, 1, [doc_id])
        doc = hits[0] if hits else next(d for d in docs if str(d.get("document_id", d.get("id"))) == doc_id)
        rows.append({"id": doc_id, "title": doc.get("title", doc.get("source")), "source": doc.get("source"),
                     "relevance": hits[0]["relevance"] if hits else 0,
                     "excerpt": hits[0].get("text", "")[:500] if hits else "No matching passage found."})
    rows.sort(key=lambda row: row["relevance"], reverse=True)
    return {"question": request.question, "documents": rows,
            "note": "Relevance indicates topical match, not whether a source is correct or authoritative."}


@app.post("/verify", response_model=VerifyResponse)
def verify(request: VerifyRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    return _verify_text(request.text)


FRONTEND_DIR = BASE_DIR / "static"


# Serve the frontend at the root URL: http://localhost:8000/
@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
