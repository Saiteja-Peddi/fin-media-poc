"""Vector store access.

Wraps the persistent Chroma collection: upserting embedded transcript chunks
(with their timestamps and source metadata) and running similarity queries.
"""

from pathlib import Path

import chromadb

from . import config, models

_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
# Cosine distance (0=identical, 2=opposite) — bounded, magnitude-independent
# scores that are easy to threshold. Only applied at collection creation.
_collection = _client.get_or_create_collection(
    name="segments", metadata={"hnsw:space": "cosine"}
)


def add_chunks(chunks, media_file):
    """Embed and store a file's chunks, replacing any previous run's rows.

    Idempotent per media_file: existing rows for this file are deleted first so
    re-ingesting the same file replaces rather than duplicates. Returns the
    number of chunks stored.
    """
    basename = Path(media_file).stem

    # Drop prior rows for this file so a re-run replaces cleanly.
    _collection.delete(where={"media_file": media_file})

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
    """Embed a query and return matching chunks with metadata and distance.

    Returns a list of dicts: the stored metadata plus 'distance'.
    """
    query_embedding = models.embed([query_text])[0]
    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    hits = []
    # Chroma nests results one level deep (per query); we only send one query.
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    for metadata, distance in zip(metadatas, distances):
        hits.append({**metadata, "distance": distance})
    return hits
