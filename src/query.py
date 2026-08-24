"""Query pipeline: embed-search, LLM re-rank, then cut a clip per ranked hit."""

import json

from . import clip, config, models, store

# Force valid JSON from small local models (see models.llm format=).
_RERANK_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
}


def _build_rerank_prompt(question, candidates, n_results):
    """Numbered candidate passages -> LLM re-ranking prompt."""
    lines = []
    for i, c in enumerate(candidates):
        title = c.get("title", "")
        text = c["text"][:400]
        lines.append(f"[{i}] {title}: {text}")
    passages = "\n".join(lines)

    return (
        "You are ranking transcript passages by how well each ANSWERS a "
        "question. A passage that directly addresses the question is more "
        "relevant than one that only mentions the topic in passing.\n\n"
        f"QUESTION: {question}\n\n"
        f"PASSAGES:\n{passages}\n\n"
        f"Return the indices of the {n_results} most relevant passages, best "
        'first, as a JSON object: {"ranking": [i, j, ...]}. Use only indices '
        "shown above, no duplicates."
    )


def _rerank(question, candidates, n_results):
    """Reorder candidates by LLM relevance; fall back to embedding order.

    The LLM may omit or repeat indices, so its output is treated as a partial
    preference: valid unique indices first, then any remaining candidates in
    their original (embedding) order. Any failure falls back to that order.
    """
    if len(candidates) <= 1:
        return candidates[:n_results]

    try:
        raw = models.llm(
            _build_rerank_prompt(question, candidates, n_results),
            format=_RERANK_SCHEMA,
            temperature=0,
        )
        order = json.loads(raw).get("ranking", [])
    except Exception:
        return candidates[:n_results]

    seen = set()
    ranked = []
    for i in order:
        if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
            seen.add(i)
            ranked.append(candidates[i])
    # Backfill anything the LLM dropped, preserving embedding order.
    for i, c in enumerate(candidates):
        if i not in seen:
            ranked.append(c)

    return ranked[:n_results]


def ask(question, n_results=3):
    """Search, LLM re-rank, and cut a fresh clip per ranked hit (ranked from 1).

    Pulls config.RERANK_CANDIDATES by embedding, re-ranks to n_results, then
    adds "clip_path" to each hit ("media_kind" is already in the metadata).
    """
    clip.clear_clips_folder()

    candidates = store.search(question, n_results=config.RERANK_CANDIDATES)
    results = _rerank(question, candidates, n_results)

    for rank, hit in enumerate(results, start=1):
        hit["clip_path"] = str(clip.cut_clip(
            hit["original_file_path"],
            hit["media_kind"],
            hit["start_ms"],
            hit["end_ms"],
            rank,
        ))

    return results
