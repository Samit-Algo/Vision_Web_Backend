"""ChromaDB vector store for video analysis chunks. Uses built-in embeddings."""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from ..core.config import get_settings


COLLECTION_NAME = "video_analysis"


def _get_db_path() -> Path:
    return Path(get_settings().static_video_vector_db_dir)


def _get_client():
    db_path = _get_db_path()
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    if not text or not text.strip():
        return []
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size // 2:
                chunk = chunk[: last_period + 1]
                end = start + last_period + 1
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap if end < len(text) else len(text)
    return chunks


def store_analysis(video_id: str, full_analysis: str) -> None:
    """Store video analysis in vector DB. Chunks and embeds using ChromaDB default."""
    client = _get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    existing = collection.get(where={"video_id": video_id}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunks = _chunk_text(full_analysis)
    if not chunks:
        return

    ids = [f"{video_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"video_id": video_id, "chunk_idx": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)


def search(video_id: str, query: str, top_k: int = 5) -> list[str]:
    """RAG search: get top-k relevant chunks for video_id."""
    client = _get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={"video_id": video_id},
    )

    if results and results.get("documents") and results["documents"][0]:
        return results["documents"][0]
    return []


def has_video(video_id: str) -> bool:
    """Check if we have any chunks for this video."""
    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
        existing = collection.get(where={"video_id": video_id}, include=[])
        return len(existing["ids"]) > 0
    except Exception:
        return False
