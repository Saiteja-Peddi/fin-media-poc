"""Persistent Chroma vector store: index chunks and run similarity search."""

from pathlib import Path

import chromadb

from . import config, models

_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
# Cosine distance (0=identical, 2=opposite); set only at collection creation.
_collection = _client.get_or_create_collection(
    name="segments", metadata={"hnsw:space": "cosine"}
)


def add_chunks(chunks, media_file, media_kind, original_file_path):
    """Embed and store a file's chunks, replacing any prior rows for that file.

    original_file_path is the real source (e.g. data/input/example.mp4), not
    the WAV, so clips can later be cut from it. Returns the count stored.
    """
    basename = Path(media_file).stem

    _collection.delete(where={"media_file": media_file})  # replace, don't dup

    if not chunks:
        return 0

    ids = [f"{basename}_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    embeddings = models.embed(documents)
    metadatas = [
        {
            "text": c["text"],
            "start_ms": c["start_ms"],
            "end_ms": c["end_ms"],
            "media_file": media_file,
            "chunk_index": c["chunk_index"],
            "media_kind": media_kind,
            "original_file_path": original_file_path,
        }
        for c in chunks
    ]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query_text, n_results=3):
    """Return the nearest chunks as dicts of stored metadata plus 'distance'."""
    query_embedding = models.embed([query_text], is_query=True)[0]
    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    # Chroma nests results per query; we only send one.
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {**metadata, "distance": distance}
        for metadata, distance in zip(metadatas, distances)
    ]
