# Wayfind — Evidence Workspace

Wayfind is a local-first document research and claim-evidence workspace. It can search a small sample corpus, accept UTF-8 text/Markdown/CSV uploads, show cited passages, compare selected sources, and review claims against the local collection.

## Features

- **Ask:** retrieve relevant passages and see where each finding came from.
- **Verify:** break text into claims and show related source passages for each claim.
- **Library:** upload `.txt`, `.md`, and `.csv` files (up to 2 MB each), then select sources to focus Ask or Compare.
- **Compare:** inspect the most relevant passage from each selected source side by side.
- **Evidence-first UI:** low-match questions abstain; relevance is shown as topical similarity, not a truth score.
- **API:** FastAPI endpoints with interactive docs at `/docs`.

The current Ask, Verify, and Compare implementation uses a lightweight local BM25-style keyword scorer. It does not require an API key or make a model call. The optional `app/pipeline` modules contain the earlier LangChain/FAISS hybrid retrieval and claim-analysis pipeline; those modules are not currently wired into the web routes.

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
├── index.py                     # Vercel FastAPI entry point
├── .python-version              # Vercel Python runtime
├── vercel.json                  # Function duration limit
├── .env.example                # Configuration template (no credentials)
├── .gitignore
├── README.md
├── requirements.txt            # Web application dependencies
├── requirements-pipeline.txt   # Optional ML pipeline dependencies
└── requirements-deploy.txt     # Render web runtime dependencies
```

Uploaded documents and generated indexes stay in local `data/` files and are excluded from Git by default. On Vercel, uploads use `/tmp/wayfind`; Vercel's function filesystem is temporary, so uploaded sources are not durable across cold starts or separate function instances.

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

Open [http://localhost:8000](http://localhost:8000). API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs). To install the optional hybrid retrieval pipeline locally, also run `pip install -r requirements-pipeline.txt`.

## Deploy on Vercel

The repository has a root FastAPI entry point at `index.py`; Vercel deploys the complete current web app (the Ask, Verify, Library, and Compare interface and its API routes) as a Python Function. `requirements.txt` intentionally contains only the web app's dependencies so the optional ML pipeline does not exceed the function bundle limit.

1. Import `siddhikaloji17-bot/Wayfind_RAG_System` from GitHub at [vercel.com/new](https://vercel.com/new). Keep the project root at the repository root. Vercel should detect FastAPI and Python 3.12 from the repository files.
2. In the project's environment variables, set `WAYFIND_ENV=production`, `WAYFIND_ACCESS_USERNAME`, and a strong unique `WAYFIND_ACCESS_PASSWORD` for Production. Keep the credentials out of GitHub.
3. Deploy. Open the resulting `vercel.app` URL to use the web interface; `/health` is only the status endpoint.

The Hobby plan is free within its included usage limits. The Vercel Function filesystem is read-only except for temporary `/tmp` storage, so uploaded documents are not durable on this setup. Use durable object storage (for example Vercel Blob) before relying on uploads across function instances or cold starts. See Vercel's [FastAPI guide](https://vercel.com/docs/frameworks/backend/fastapi), [Python runtime limits](https://vercel.com/docs/functions/runtimes), [Hobby plan](https://vercel.com/docs/plans/hobby), and [Blob pricing](https://vercel.com/docs/vercel-blob/usage-and-pricing).

## Deploy on Render

The repository includes a Render Blueprint at `render.yaml`. It deploys the FastAPI app on Render's free tier in Virginia, installs only the web runtime dependencies from `requirements-deploy.txt`, and enables HTTP Basic Authentication for the site and API. The `/health` endpoint remains available to Render's health checker. Install `requirements-pipeline.txt` if you need the optional LangChain/FAISS pipeline locally.

1. In Render, choose **New → Blueprint** and connect `siddhikaloji17-bot/Wayfind_RAG_System` on the `main` branch.
2. Review the Blueprint resources and enter a username and a strong, unique password when Render asks for `WAYFIND_ACCESS_USERNAME` and `WAYFIND_ACCESS_PASSWORD`. Keep these values in Render's environment settings, not in GitHub.
3. Create the resources and wait for the first deploy to finish. Render will provide the live `onrender.com` URL; opening it prompts for the Basic Auth credentials.

This free demo does not include a persistent disk. The app's uploaded corpus is stored on Render's ephemeral filesystem, so uploads can disappear when the service restarts or spins down. Choose a paid plan and attach a persistent disk if uploaded sources must survive restarts. Check the current [Render pricing](https://render.com/pricing) before changing plans. See the official [FastAPI deploy guide](https://render.com/docs/deploy-fastapi), [persistent disk guide](https://render.com/docs/disks), and [free plan limitations](https://render.com/docs/free).

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
