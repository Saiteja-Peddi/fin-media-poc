"""Topic segmentation: raw ASR words -> coherent, titled topic segments.

Tier-1 (implemented): while the transcript fits the local model's context
budget, the LLM segments the whole thing in one pass — best coherence for
short clips. Tier-2 (planned): for transcripts that outgrow the budget, an
embedding pass first cuts large topic-complete blocks, each then refined by
the LLM. The embedding helpers below are kept for that tier-2 coarse pass.
"""

import json
import math
import re

from . import config, models

_SENTENCE_ENDINGS = (".", "!", "?")


def build_sentences(words):
    """Group ASR words into sentences.

    Heuristic: split after any word ending in ".", "!", or "?". No abbreviation
    handling ("Inc." / "U.S." over-split) — good enough as a base unit.

    words: [{"word", "start_ms", "end_ms"}, ...] from models.asr().
    Returns [{"text", "start_ms", "end_ms", "word_count"}, ...] in order.
    """
    sentences = []
    current = []

    for w in words:
        current.append(w)
        if w["word"].rstrip().endswith(_SENTENCE_ENDINGS):
            sentences.append(_make_sentence(current))
            current = []

    if current:  # trailing words with no closing punctuation
        sentences.append(_make_sentence(current))

    return sentences


def _make_sentence(words):
    """Assemble one sentence dict from its words."""
    return {
        "text": " ".join(w["word"] for w in words),
        "start_ms": words[0]["start_ms"],
        "end_ms": words[-1]["end_ms"],
        "word_count": len(words),
    }


# --- Embedding helpers (reserved for the tier-2 coarse pre-segmentation) -----

def _cosine(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _consecutive_similarities(sentences):
    """Cosine similarity for each consecutive sentence pair (one batch embed).

    Returns a list of length len(sentences) - 1, where item i is the similarity
    between sentence i and i+1. Empty if fewer than two sentences.
    """
    if len(sentences) < 2:
        return []
    embeddings = models.embed([s["text"] for s in sentences])
    return [
        _cosine(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]


def detect_candidate_boundaries(sentences):
    """Find likely topic shifts between consecutive sentences.

    Embeds every sentence in one batch call, then flags a boundary wherever the
    cosine similarity between neighbours drops below config.SIMILARITY_THRESHOLD.

    Returns sentence indices AFTER which a boundary falls (a boundary at i means
    the shift happens between sentence i and i+1).

    The threshold is a starting point and will likely need tuning against real
    output — inspect actual similarity scores before trusting it.
    """
    similarities = _consecutive_similarities(sentences)
    return [
        i for i, sim in enumerate(similarities)
        if sim < config.SIMILARITY_THRESHOLD
    ]


# Structured-output schema: forces small local models to emit valid JSON.
_SEGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sentence_idx": {"type": "integer"},
                    "end_sentence_idx": {"type": "integer"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": [
                    "start_sentence_idx", "end_sentence_idx", "title", "summary",
                ],
            },
        },
    },
    "required": ["segments"],
}


def _build_segment_prompt(sentences):
    """Numbered transcript -> LLM topic-segmentation prompt."""
    last_idx = len(sentences) - 1
    transcript = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(sentences))

    return (
        "You are an expert editor who divides financial audio and video "
        "transcripts into coherent, self-contained topic segments used for "
        "search and chaptering.\n\n"
        f"The transcript below has {len(sentences)} sentences, numbered 0 to "
        f"{last_idx}, one per line.\n\n"
        "TASK: group consecutive sentences into topic segments. Each segment is "
        "a single, self-contained topic or subtopic that reads sensibly on its "
        "own.\n\n"
        "RULES:\n"
        "1. Put a boundary ONLY at a genuine topic shift — a new asset class, "
        "theme, question, or section. Never break in the middle of a thought, "
        "explanation, example, or list.\n"
        f"2. Segments must be contiguous, ordered, and cover every sentence "
        f"exactly once: the first starts at index 0; each later segment's "
        f"start_sentence_idx equals the previous segment's end_sentence_idx + 1; "
        f"the last ends at index {last_idx}. No gaps, no overlaps.\n"
        "3. Consolidate aggressively. Keep a topic together with ALL its "
        "supporting sentences — elaborations, examples, figures, and caveats — "
        "in ONE segment. Do not open a new segment for a follow-on sentence that "
        "continues the same topic (e.g. keep all discussion of gold in a single "
        "segment). Aim for the fewest segments that still keep each to one "
        "topic; a typical segment spans several sentences. If the whole "
        "transcript is a single topic, return exactly one segment.\n"
        "4. Base each title and summary ONLY on that segment's own sentences. Do "
        "not invent facts, figures, names, or opinions that are not stated.\n"
        "5. title: 2-6 words, specific to the topic (name the asset class or "
        "theme; avoid generic words like 'Overview'), no trailing punctuation. "
        "summary: one factual sentence in the transcript's language.\n\n"
        "Return ONLY a JSON object of this exact shape (indices inclusive):\n"
        '{"segments": [{"start_sentence_idx": 0, "end_sentence_idx": 2, '
        '"title": "Equities Outlook", "summary": "..."}, ...]}\n\n'
        f"TRANSCRIPT:\n{transcript}"
    )


def _parse_segments_json(raw):
    """Parse the LLM's reply into a list of segment dicts.

    Accepts the schema's {"segments": [...]} object, or a bare array as a
    fallback. Tolerates a ```json fence or surrounding prose.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("segments", [])
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON array of segments")
    return parsed


def _repair_coverage(raw_segments, n_sentences):
    """Force LLM segment ranges into a clean, contiguous cover of 0..n-1.

    Despite the prompt, an LLM can emit overlaps, gaps, or out-of-order ranges,
    and clips are cut from these ranges — so we never trust it for this
    invariant. This deterministically rewrites boundaries: segments are ordered,
    each starts right after the previous ends (no gaps, no overlaps), and the
    last reaches the final sentence. Collapsed/subsumed entries are dropped;
    titles/summaries are preserved. Returns index dicts, never empty.
    """
    last = n_sentences - 1

    cleaned = []
    for seg in raw_segments:
        try:
            start = int(seg["start_sentence_idx"])
            end = int(seg["end_sentence_idx"])
        except (KeyError, TypeError, ValueError):
            continue  # malformed entry
        start = max(0, min(start, last))
        end = max(0, min(end, last))
        if end < start:
            continue  # reversed range
        cleaned.append({
            "start": start, "end": end,
            "title": seg.get("title", ""), "summary": seg.get("summary", ""),
        })

    cleaned.sort(key=lambda s: (s["start"], s["end"]))

    repaired = []
    cursor = 0
    for seg in cleaned:
        if seg["end"] < cursor:
            continue  # entirely behind the cursor -> already covered
        repaired.append({
            "start_sentence_idx": cursor,          # chain: no gap, no overlap
            "end_sentence_idx": max(seg["end"], cursor),
            "title": seg["title"], "summary": seg["summary"],
        })
        cursor = repaired[-1]["end_sentence_idx"] + 1
        if cursor > last:
            break

    if not repaired:  # nothing usable -> one segment covering everything
        return [{"start_sentence_idx": 0, "end_sentence_idx": last,
                 "title": "", "summary": ""}]

    if repaired[-1]["end_sentence_idx"] < last:  # cover any uncovered tail
        repaired[-1]["end_sentence_idx"] = last

    return repaired


def llm_segment(sentences):
    """Segment a numbered sentence list into titled topics with one LLM call.

    The whole transcript goes in one pass (caller ensures it fits the context
    budget). A JSON schema constrains the output; one retry then a raise guards
    against a stray unparseable reply. A deterministic repair pass then
    guarantees contiguous, non-overlapping coverage regardless of the model.

    Returns [{"text", "start_ms", "end_ms", "title", "summary"}, ...] in order.
    """
    if not sentences:
        return []

    prompt = _build_segment_prompt(sentences)
    raw = models.llm(prompt, format=_SEGMENT_SCHEMA, temperature=0)
    try:
        segments = _parse_segments_json(raw)
    except (json.JSONDecodeError, ValueError):
        raw = models.llm(prompt, format=_SEGMENT_SCHEMA, temperature=0)
        try:
            segments = _parse_segments_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"LLM did not return parseable segment JSON:\n{raw}"
            ) from e

    segments = _repair_coverage(segments, len(sentences))
    result = []
    for seg in segments:
        start = seg["start_sentence_idx"]
        end = seg["end_sentence_idx"]
        result.append(
            {
                "text": " ".join(s["text"] for s in sentences[start:end + 1]),
                "start_ms": sentences[start]["start_ms"],
                "end_ms": sentences[end]["end_ms"],
                "title": seg["title"],
                "summary": seg["summary"],
            }
        )
    return result


def _combine(first, second, keep):
    """Merge two adjacent segments; `keep` supplies the title/summary."""
    return {
        "text": f"{first['text']} {second['text']}",
        "start_ms": first["start_ms"],
        "end_ms": second["end_ms"],
        "title": keep["title"],
        "summary": keep["summary"],
    }


# Title/summary regeneration for merged segments (structured output).
_RELABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "summary"],
}


def _relabel(text):
    """Generate a (title, summary) grounded in the given segment text.

    Returns None on any failure so the caller keeps the existing label rather
    than crashing the pipeline.
    """
    prompt = (
        "Write a short title and one-sentence summary for this financial "
        "transcript segment. They must describe the segment's MAIN topic as a "
        "whole.\n\n"
        "Rules: title 2-6 words, specific to the topic (name the asset class or "
        "theme), no trailing punctuation; summary one factual sentence grounded "
        "only in the text.\n\n"
        'Respond as JSON: {"title": "...", "summary": "..."}\n\n'
        f"SEGMENT:\n{text}"
    )
    try:
        obj = json.loads(models.llm(prompt, format=_RELABEL_SCHEMA, temperature=0))
        title, summary = obj.get("title", ""), obj.get("summary", "")
    except Exception:
        return None
    return (title, summary) if title else None


def merge_short_segments(segments):
    """Fold sub-minimum segments into a neighbour (a light guardrail only).

    A short segment merges into the following one; a trailing short segment
    merges into the previous one. No maximum bound — length follows topic
    coherence, not the clock. Prints the merge count. Titles are refreshed
    afterwards by _label_segments, so surviving-title choice here doesn't stick.
    """
    if not segments:
        return []

    min_ms = config.MIN_SEGMENT_SECONDS * 1000

    merges = 0
    merged = []
    pending = None  # a short segment waiting to attach to the next one
    for seg in segments:
        cur = dict(seg)
        if pending is not None:
            cur = _combine(pending, cur, keep=cur)  # following segment survives
            merges += 1
            pending = None
        if (cur["end_ms"] - cur["start_ms"]) < min_ms:
            pending = cur
        else:
            merged.append(cur)
    if pending is not None:
        if merged:  # trailing short segment: fold into the previous one
            merged[-1] = _combine(merged[-1], pending, keep=merged[-1])
            merges += 1
        else:  # a single short segment with nothing to merge into
            merged.append(pending)

    print(f"merge_short_segments: {merges} merge(s)")
    return merged


def _label_segments(segments):
    """Regenerate every segment's title/summary from its final text.

    Boundaries (LLM) and merging are settled first, so labels are grounded in
    exactly the content each final segment holds — fixing draft titles that no
    longer fit after consolidation. Failures keep the existing label.
    """
    relabelled = 0
    for seg in segments:
        new_label = _relabel(seg["text"])
        if new_label:
            seg["title"], seg["summary"] = new_label
            relabelled += 1
    print(f"_label_segments: {relabelled}/{len(segments)} labelled")
    return segments


def _estimate_tokens(sentences):
    """Rough token estimate for the transcript (~4 chars per token)."""
    chars = sum(len(s["text"]) for s in sentences)
    return chars // 4


def build_segments(words):
    """Raw ASR words -> final titled topic segments.

    Tier router: while the transcript fits config.SAFE_CONTEXT_TOKENS, the LLM
    segments it in one pass (tier-1). Larger transcripts need the tier-2
    hierarchical split, which is not implemented yet.
    """
    sentences = build_sentences(words)
    if not sentences:
        return []

    est_tokens = _estimate_tokens(sentences)
    if est_tokens > config.SAFE_CONTEXT_TOKENS:
        raise NotImplementedError(
            f"Transcript ~{est_tokens} tokens exceeds SAFE_CONTEXT_TOKENS "
            f"({config.SAFE_CONTEXT_TOKENS}); tier-2 hierarchical segmentation "
            f"is not implemented yet."
        )

    segments = llm_segment(sentences)          # boundaries + draft labels
    segments = merge_short_segments(segments)  # length guardrail
    return _label_segments(segments)           # final labels from final text


if __name__ == "__main__":
    # Eyeball the full segmentation on the most recently transcribed file.
    def _fmt_ms(ms):
        s = ms // 1000
        return f"{s // 60:02d}:{s % 60:02d}"

    caches = sorted(
        config.DATA_MEDIA.glob("*_words.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not caches:
        print("No cached transcripts in data/media — ingest a file first.")
    else:
        latest = caches[-1]
        with open(latest) as f:
            words = json.load(f)

        segments = build_segments(words)
        print(f"\n{latest.name}: {len(words)} words -> {len(segments)} segments\n")
        for i, seg in enumerate(segments, start=1):
            time_range = f"{_fmt_ms(seg['start_ms'])}-{_fmt_ms(seg['end_ms'])}"
            print(f"[{i}] {time_range}  {seg['title']}")
            print(f"    {seg['summary']}\n")
