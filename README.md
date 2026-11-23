# Wayfind — Evidence Workspace

Wayfind is a local-first document research and claim-evidence workspace. It can search a small sample corpus, accept UTF-8 text/Markdown/CSV uploads, show cited passages, compare selected sources, and review claims against the local collection.

## Features

- **Ask:** retrieve relevant passages and see where each finding came from.
- **Verify:** break text into claims and show related source passages for each claim.
- **Library:** upload `.txt`, `.md`, and `.csv` files (up to 2 MB each), then select sources to focus Ask or Compare.
- **Compare:** inspect the most relevant passage from each selected source side by side.
- **Evidence-first UI:** low-match questions abstain; relevance is shown as topical similarity, not a truth score.
- **API:** FastAPI endpoints with interactive docs at `/docs`.

The current Ask and Compare implementation uses a lightweight local BM25-style keyword scorer. It does not require an API key or make a model call. The optional `app/pipeline` modules contain the earlier LangChain/FAISS hybrid retrieval and claim-analysis pipeline.

## Project layout

```text
wayfind-rag-system/
├── app/
│   ├── main.py                 # FastAPI routes and local retrieval
│   ├── config.py               # Environment-backed settings
│   ├── pipeline/               # Optional LangChain/FAISS pipeline
│   │   ├── agent.py
│   │   ├── claim_extraction.py
│   │   ├── factcheck_lookup.py
│   │   ├── ingestion.py
│   │   ├── reranker.py
│   │   ├── retrieval.py
│   │   ├── source_independence.py
│   │   └── stance_detection.py
│   └── static/
│       └── index.html           # Wayfind web interface
├── data/
│   └── sample_corpus.jsonl      # Safe example corpus
├── .env.example                # Configuration template (no credentials)
├── .gitignore
├── README.md
└── requirements.txt
```

Uploaded documents and generated indexes stay in local `data/` files and are excluded from Git by default.

## Run locally

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000). API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

## Deploy on Render

The repository includes a Render Blueprint at `render.yaml`. It deploys the FastAPI app in Singapore, installs only the web runtime dependencies from `requirements-deploy.txt`, mounts a 1 GB persistent disk at `/var/data` for uploaded documents, and enables HTTP Basic Authentication for the site and API. The `/health` endpoint remains available to Render's health checker. Install `requirements.txt` only if you also need the optional LangChain/FAISS pipeline locally.

1. In Render, choose **New → Blueprint** and connect `siddhikaloji17-bot/Wayfind_RAG_System` on the `main` branch.
2. Review the Blueprint resources and enter a username and a strong, unique password when Render asks for `WAYFIND_ACCESS_USERNAME` and `WAYFIND_ACCESS_PASSWORD`. Keep these values in Render's environment settings, not in GitHub.
3. Create the resources and wait for the first deploy to finish. Render will provide the live `onrender.com` URL; opening it prompts for the Basic Auth credentials.

This Blueprint uses paid compute and a persistent disk. Render's documented persistent disk rate is $0.25 per GB per month; compute is billed separately according to the selected service plan. Render's free web services do not support persistent disks and their files are discarded on restarts or spin-down, so the free plan is unsuitable for keeping uploaded sources. Check the current [Render pricing](https://render.com/pricing) before creating the service. See the official [FastAPI deploy guide](https://render.com/docs/deploy-fastapi), [persistent disk guide](https://render.com/docs/disks), and [free plan limitations](https://render.com/docs/free).

## API routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service and corpus status |
| `GET` | `/documents` | List the local sample and uploaded sources |
| `POST` | `/documents` | Upload a UTF-8 text, Markdown, or CSV file |
| `POST` | `/ask` | Search the corpus, optionally scoped to selected document IDs |
| `POST` | `/compare` | Compare a question across selected document IDs |
| `POST` | `/verify` | Show corpus evidence for statements in submitted text |

## Optional hybrid index

The legacy ingestion command builds local BM25 and FAISS indexes from a JSONL corpus:

```bash
python -m app.pipeline.ingestion --input data/sample_corpus.jsonl
```

Generated index files are local artifacts and should not be committed.

## GitHub safety

- Keep API keys in `.env`; never commit that file.
- `.env.example` contains blank placeholders for setup.
- Uploaded documents, local indexes, virtual environments, and Python cache files are ignored.
- Review `git status` before pushing to make sure no private data was added.

## Current limits

The local ranker is keyword-based, so paraphrases can be missed. The included corpus is a small demonstration dataset. This deployment protects the shared workspace with one Basic Auth login; it does not isolate documents by user. Do not use it for multiple users' private documents without adding per-user access controls and storage isolation.
