import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.config import (
    KNOWLEDGE_DIR,
    RAG_DIR,
    RAG_EMBEDDING_MODEL,
    RAG_INDEX_PATH,
    RAG_MANIFEST_PATH,
    RAG_TOP_K,
    admin_log,
)


def _list_markdown_files(source_dir: str) -> list[Path]:
    root = Path(source_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.md") if p.is_file())


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned = "\n".join(lines).strip()
    return cleaned


def _chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        step = max(1, chunk_size - chunk_overlap)
        while start < len(paragraph):
            piece = paragraph[start : start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            start += step
        current = ""

    if current:
        chunks.append(current)

    return chunks


def load_knowledge_chunks(
    source_dir: str = KNOWLEDGE_DIR,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for path in _list_markdown_files(source_dir):
        text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(source_dir).as_posix()
        title = next(
            (
                line.lstrip("#").strip()
                for line in text.splitlines()
                if line.strip().startswith("#")
            ),
            path.stem.replace("_", " ").title(),
        )
        for chunk_id, content in enumerate(
            _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": relative_path,
                    "title": title,
                    "content": content,
                }
            )

    return chunks


def build_index_from_directory(
    source_dir: str = KNOWLEDGE_DIR,
    output_dir: str = RAG_DIR,
    embedding_model: str = RAG_EMBEDDING_MODEL,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> dict[str, Any]:
    if embedding_model.lower() != "tfidf":
        raise ValueError(
            f"Unsupported embedding model '{embedding_model}'. v1 currently supports 'tfidf'."
        )

    chunks = load_knowledge_chunks(
        source_dir=source_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise FileNotFoundError(
            f"No markdown knowledge files found in {os.path.abspath(source_dir)}"
        )

    texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, os.path.basename(RAG_INDEX_PATH))
    manifest_path = os.path.join(output_dir, os.path.basename(RAG_MANIFEST_PATH))
    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "embedding_model": embedding_model,
                "vectorizer": vectorizer,
                "matrix": matrix,
                "chunks": chunks,
            },
            f,
        )

    manifest = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "embedding_model": embedding_model,
        "source_dir": os.path.abspath(source_dir),
        "index_path": os.path.abspath(index_path),
        "documents": len({chunk["source"] for chunk in chunks}),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    admin_log(
        "RAG",
        f"Built knowledge index ({manifest['documents']} docs, {manifest['chunks']} chunks)",
    )
    return manifest


def load_index(index_path: str = RAG_INDEX_PATH) -> dict[str, Any] | None:
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def get_knowledge_status() -> dict[str, Any]:
    status = {
        "knowledge_dir": os.path.abspath(KNOWLEDGE_DIR),
        "rag_dir": os.path.abspath(RAG_DIR),
        "embedding_model": RAG_EMBEDDING_MODEL,
        "index_ready": os.path.exists(RAG_INDEX_PATH),
        "knowledge_files": len(_list_markdown_files(KNOWLEDGE_DIR)),
    }
    if os.path.exists(RAG_MANIFEST_PATH):
        with open(RAG_MANIFEST_PATH, "r", encoding="utf-8") as f:
            status["manifest"] = json.load(f)
    return status


def retrieve_knowledge(query: str, top_k: int = RAG_TOP_K) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    index = load_index()
    if not index:
        return []

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunks = index["chunks"]
    query_vector = vectorizer.transform([query])
    scores = (query_vector @ matrix.T).toarray().ravel()

    if scores.size == 0:
        return []

    ranked = np.argsort(scores)[::-1]
    results: list[dict[str, Any]] = []
    for idx in ranked:
        score = float(scores[idx])
        if score <= 0:
            continue
        chunk = chunks[int(idx)]
        results.append(
            {
                "score": round(score, 4),
                "source": chunk["source"],
                "title": chunk["title"],
                "content": chunk["content"],
            }
        )
        if len(results) >= max(1, int(top_k)):
            break

    return results


def format_knowledge_context(query: str, top_k: int = RAG_TOP_K) -> str:
    results = retrieve_knowledge(query, top_k=top_k)
    if not results:
        return ""

    sections = []
    for item in results:
        sections.append(
            "\n".join(
                [
                    f"Source: {item['source']}",
                    f"Title: {item['title']}",
                    f"Score: {item['score']}",
                    item["content"],
                ]
            )
        )
    return "\n\n---\n\n".join(sections)
