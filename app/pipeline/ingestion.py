"""
Reads the JSONL corpus and builds two indexes:
  - a BM25 keyword index
  - a FAISS dense vector index
Both are saved to disk so retrieval.py can load them without re-embedding
every time.
"""
import argparse
import json
import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from app.config import settings


def load_corpus(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            docs.append(
                Document(
                    page_content=record["text"],
                    metadata={
                        "id": record["id"],
                        "source": record["source"],
                        "title": record["title"],
                        "url": record["url"],
                    },
                )
            )
    return docs


def build_bm25_index(docs: list[Document], out_path: str):
    retriever = BM25Retriever.from_documents(docs)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(retriever, f)
    print(f"BM25 index saved to {out_path}")


class LocalSentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()


def build_vector_index(docs: list[Document], out_dir: str):
    if settings.USE_LOCAL_EMBEDDINGS:
        print(f"Using local embeddings model: {settings.LOCAL_EMBEDDING_MODEL}")
        embeddings = LocalSentenceTransformerEmbeddings(settings.LOCAL_EMBEDDING_MODEL)
    else:
        print(f"Using OpenAI embeddings model: {settings.EMBEDDING_MODEL}")
        embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(docs, embeddings)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(out_dir)
    print(f"FAISS vector index saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL corpus")
    args = parser.parse_args()

    input_path = Path(args.input)
    documents = load_corpus(input_path)
    print(f"Loaded {len(documents)} documents from {input_path}")

    build_bm25_index(documents, out_path="data/bm25_index.pkl")
    build_vector_index(documents, out_dir=settings.VECTOR_INDEX_DIR)

    print("Ingestion complete.")