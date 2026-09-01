"""Query pipeline: embed-search, LLM re-rank, then cut a clip per ranked hit.

A hit may optionally be *merged* with other hits from the SAME parent video
(never across sources) into one clip — see ask()'s `merge` argument.
"""

import json

from . import clip, config, models, segment, store

# Merge modes for ask(): keep hits separate, merge only time-adjacent same-parent
# hits into a continuous clip, or stitch all same-parent hits (jump-cuts allowed).
MERGE_NONE = "none"
MERGE_ADJACENT = "adjacent"
MERGE_PARENT = "parent"

# Two spans join into one continuous cut when their gap is within this margin;
# a larger gap (a real topic jump in "parent" mode) stays a separate span that
# concat stitches with a hard cut. Sized to the clip padding on each side.
_MERGE_GAP_MS = 2 * clip.PAD_MS

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


def _group_hits(ranked, merge):
    """Group ranked hits per the merge mode; groups ordered by best member rank.

    Returns a list of groups (each a list of hit dicts). Grouping NEVER crosses
    parent videos. MERGE_NONE -> one hit per group. MERGE_PARENT -> all hits of a
    parent in one group. MERGE_ADJACENT -> only runs of consecutive segments
    (chunk_index n, n+1, ...) within a parent; a gap starts a new group.
    """
    if merge == MERGE_NONE:
        return [[h] for h in ranked]

    # Best (lowest) rank per hit drives group ordering; ranked is already sorted.
    rank_of = {id(h): i for i, h in enumerate(ranked)}

    by_parent = {}  # original_file_path -> hits, in rank order
    for h in ranked:
        by_parent.setdefault(h["original_file_path"], []).append(h)

    groups = []
    for hits in by_parent.values():
        if merge == MERGE_PARENT:
            groups.append(list(hits))
            continue
        # MERGE_ADJACENT: split into consecutive-chunk_index runs.
        for run in _adjacent_runs(hits):
            groups.append(run)

    groups.sort(key=lambda g: min(rank_of[id(h)] for h in g))
    return groups


def _adjacent_runs(hits):
    """Split same-parent hits into runs whose chunk_index is consecutive."""
    ordered = sorted(hits, key=lambda h: h["chunk_index"])
    runs, current = [], [ordered[0]]
    for prev, h in zip(ordered, ordered[1:]):
        if h["chunk_index"] == prev["chunk_index"] + 1:
            current.append(h)
        else:
            runs.append(current)
            current = [h]
    runs.append(current)
    return runs


def _merge_spans(members):
    """Time-sorted members -> minimal [start_ms, end_ms] spans (gaps within
    _MERGE_GAP_MS collapse into one continuous span)."""
    spans = []
    for m in sorted(members, key=lambda h: h["start_ms"]):
        if spans and m["start_ms"] - spans[-1][1] <= _MERGE_GAP_MS:
            spans[-1][1] = max(spans[-1][1], m["end_ms"])
        else:
            spans.append([m["start_ms"], m["end_ms"]])
    return [tuple(s) for s in spans]


def _group_result(members, spans):
    """Build the result dict for a group: the best-ranked member, with span
    bounds; a multi-segment group gets one regenerated label over its text."""
    result = dict(members[0])  # best-ranked (groups keep rank order within)
    result["start_ms"] = spans[0][0]
    result["end_ms"] = spans[-1][1]
    if len(members) > 1:
        combined = "\n".join(m["text"] for m in sorted(members, key=lambda h: h["start_ms"]))
        result["text"] = combined
        label = segment.label_span(combined)
        if label:
            result["title"], result["summary"] = label
        # A merged clip may span several speakers; show all distinct ones.
        speakers = sorted({m.get("speaker_id", "") for m in members if m.get("speaker_id")})
        result["speaker_id"] = ", ".join(speakers)
    return result


def ask(question, n_results=3, merge=MERGE_NONE):
    """Search, LLM re-rank, and cut a fresh clip per result (ranked from 1).

    Pulls config.RERANK_CANDIDATES by embedding, re-ranks them, then groups per
    `merge` (MERGE_NONE | MERGE_ADJACENT | MERGE_PARENT) — only same-parent hits
    ever merge — and cuts one clip per group, backfilling to n_results groups.
    Adds "clip_path" to each returned hit ("media_kind" is already metadata).
    """
    clip.clear_clips_folder()

    candidates = store.search(question, n_results=config.RERANK_CANDIDATES)
    ranked = _rerank(question, candidates, len(candidates))
    groups = _group_hits(ranked, merge)[:n_results]

    results = []
    for rank, members in enumerate(groups, start=1):
        spans = _merge_spans(members)
        result = _group_result(members, spans)
        result["clip_path"] = str(clip.concat_clips(
            result["original_file_path"],
            result["media_kind"],
            spans,
            rank,
        ))
        results.append(result)

    return results
