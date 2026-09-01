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
from collections import Counter

from . import config, models

_SENTENCE_ENDINGS = (".", "!", "?")


def build_sentences(words):
    """Group ASR words into sentences.

    Heuristic: split after any word ending in ".", "!", or "?". No abbreviation
    handling ("Inc." / "U.S." over-split) — good enough as a base unit.

    words: [{"word", "start_ms", "end_ms", "speaker"?}, ...] from models.asr()
    (the "speaker" field is present once diarization has tagged the words).
    Returns [{"text", "start_ms", "end_ms", "word_count", "speaker"}, ...] in
    order.
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
        "speaker": _sentence_speaker(words),
    }


def _sentence_speaker(words):
    """Speaker for a sentence: its first word's speaker, else the most common
    non-None one. None only when every word is unassigned (or untagged, e.g. a
    pre-diarization transcript)."""
    first = words[0].get("speaker")
    if first is not None:
        return first

    speakers = [s for w in words if (s := w.get("speaker")) is not None]
    if not speakers:
        return None
    return Counter(speakers).most_common(1)[0][0]


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


def get_speaker_boundaries(sentences):
    """Sentence indices where a speaker change forces a HARD segment boundary.

    Index i means sentence i starts a new segment (its known speaker differs
    from the last known one) — non-negotiable and enforced after the LLM runs,
    unlike detect_candidate_boundaries' adjustable suggestions. speaker=None is
    uncertainty, not a new person: it never creates a boundary and is compared
    against the last KNOWN speaker (so S0, None, S1 splits before S1).

    Returns a set of sentence indices (each in 1..len(sentences)-1).
    """
    boundaries = set()
    prev_known = None
    for i, s in enumerate(sentences):
        speaker = s.get("speaker")
        if speaker is None:
            continue  # uncertain tag: neither starts nor breaks a boundary
        if prev_known is not None and speaker != prev_known:
            boundaries.add(i)  # sentence i starts a new segment
        prev_known = speaker
    return boundaries


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


def _boundary_hints(candidate_indices, hard_boundary_indices):
    """Render the CANDIDATE/REQUIRED boundary hint block for the prompt.

    Both index sets use the "new segment starts AT this sentence number"
    convention. Returns an empty string when there are no hints at all.
    """
    candidates = sorted(candidate_indices or [])
    required = sorted(hard_boundary_indices or [])
    if not candidates and not required:
        return ""

    required_str = ", ".join(map(str, required)) if required else "none"
    candidate_str = ", ".join(map(str, candidates)) if candidates else "none"
    return (
        "BOUNDARY HINTS (sentence numbers where a new segment may or must "
        "start):\n"
        f"- REQUIRED (speaker changes): {required_str}. A new segment MUST start "
        "at each of these sentence numbers. The sentence before it and the "
        "sentence at it are spoken by different people and must NEVER share a "
        "segment. Respect these exactly.\n"
        f"- CANDIDATE (possible topic shifts): {candidate_str}. These are only "
        "hints. Use your judgment: confirm one, shift it by a sentence or two, "
        "or ignore it entirely based on the actual topic flow.\n\n"
    )


def _build_segment_prompt(sentences, candidate_indices=None, hard_boundary_indices=None):
    """Numbered transcript -> LLM topic-segmentation prompt.

    candidate_indices / hard_boundary_indices are sentence numbers where a new
    segment may (candidate) or must (required/speaker-change) start; both are
    surfaced to the model, clearly distinguished.
    """
    last_idx = len(sentences) - 1
    transcript = "\n".join(f"{i}: {s['text']}" for i, s in enumerate(sentences))

    return (
        "You are an expert editor who divides financial audio and video "
        "transcripts into coherent, self-contained topic segments used for "
        "search and chaptering.\n\n"
        f"The transcript below has {len(sentences)} sentences, numbered 0 to "
        f"{last_idx}, one per line.\n\n"
        + _boundary_hints(candidate_indices, hard_boundary_indices) +
        "TASK: group consecutive sentences into topic segments. Each segment is "
        "a single, self-contained topic or subtopic that reads sensibly on its "
        "own.\n\n"
        "RULES:\n"
        "1. Put a boundary ONLY at a genuine topic shift — a new asset class, "
        "theme, question, or section — OR wherever a REQUIRED boundary above "
        "demands one. Never break in the middle of a thought, explanation, "
        "example, or list (except where a REQUIRED boundary falls).\n"
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


def _enforce_hard_boundaries(segments, hard_boundary_indices):
    """Split any segment that straddles a REQUIRED (speaker-change) boundary.

    Overrides the LLM's grouping so no output segment holds a speaker change in
    its interior. Contiguity/coverage preserved; titles/summaries are duplicated
    across a split (final labels are regenerated downstream anyway).
    """
    if not hard_boundary_indices:
        return segments

    result = []
    for seg in segments:
        start = seg["start_sentence_idx"]
        end = seg["end_sentence_idx"]
        # Cut points strictly inside the segment: a boundary at b means b begins
        # a new segment, so it's interior when start < b <= end.
        cuts = sorted(b for b in hard_boundary_indices if start < b <= end)
        prev = start
        for b in cuts:
            result.append({**seg, "start_sentence_idx": prev, "end_sentence_idx": b - 1})
            prev = b
        result.append({**seg, "start_sentence_idx": prev, "end_sentence_idx": end})
    return result


def llm_segment(sentences, candidate_indices=None, hard_boundary_indices=None):
    """Segment a numbered sentence list into titled topics with one LLM call.

    The whole transcript goes in one pass (caller ensures it fits the context
    budget). A JSON schema constrains the output; one retry then a raise guards
    against a stray unparseable reply, and a deterministic repair pass then
    guarantees contiguous, non-overlapping coverage.

    candidate_indices (topic-shift hints) are adjustable suggestions;
    hard_boundary_indices (speaker changes) are both passed as requirements AND
    enforced after parsing, so the LLM can never merge two speakers.

    Returns [{"text", "start_ms", "end_ms", "title", "summary", "speaker"}, ...]
    in order.
    """
    if not sentences:
        return []

    prompt = _build_segment_prompt(sentences, candidate_indices, hard_boundary_indices)
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
    # Speaker changes are non-negotiable: split anything the LLM left straddling.
    segments = _enforce_hard_boundaries(segments, hard_boundary_indices)
    result = []
    for seg in segments:
        start = seg["start_sentence_idx"]
        end = seg["end_sentence_idx"]
        span = sentences[start:end + 1]
        result.append(
            {
                "text": " ".join(s["text"] for s in span),
                "start_ms": sentences[start]["start_ms"],
                "end_ms": sentences[end]["end_ms"],
                "title": seg["title"],
                "summary": seg["summary"],
                # Single-speaker after enforcement; carried so merging can't
                # later fold two speakers together.
                "speaker": _sentence_speaker(span),
            }
        )
    return result


def _combine(first, second, keep):
    """Merge two adjacent segments; `keep` supplies the title/summary.

    Only called for speaker-compatible pairs (see _mergeable), so the combined
    speaker is simply whichever side is known (or None if both are).
    """
    first_speaker = first.get("speaker")
    return {
        "text": f"{first['text']} {second['text']}",
        "start_ms": first["start_ms"],
        "end_ms": second["end_ms"],
        "title": keep["title"],
        "summary": keep["summary"],
        "speaker": first_speaker if first_speaker is not None else second.get("speaker"),
    }


def _mergeable(a, b):
    """True unless merging a and b would mix two different known speakers.

    A speaker change is a hard boundary, so segments on opposite sides of one are
    never merged. An unknown (None) speaker is compatible with anything — an
    uncertain tag shouldn't block an otherwise valid length merge.
    """
    sa, sb = a.get("speaker"), b.get("speaker")
    return sa is None or sb is None or sa == sb


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


def label_span(text):
    """Public: (title, summary) for a merged span, or None on failure.

    Used by the query layer to label a clip stitched from several segments.
    """
    return _relabel(text)


def merge_short_segments(segments):
    """Fold sub-minimum segments into a neighbour (a light guardrail only).

    A short segment merges into the following one; a trailing short one merges
    into the previous. No maximum bound — length follows topic coherence, not
    the clock. Never merges across a speaker change: a short segment that can't
    merge without mixing two known speakers stays standalone. Titles are
    refreshed afterwards by _label_segments, so the surviving title doesn't stick.
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
            if _mergeable(pending, cur):
                cur = _combine(pending, cur, keep=cur)  # following segment survives
                merges += 1
            else:  # speaker boundary: keep the short segment on its own
                merged.append(pending)
            pending = None
        if (cur["end_ms"] - cur["start_ms"]) < min_ms:
            pending = cur
        else:
            merged.append(cur)
    if pending is not None:
        if merged and _mergeable(merged[-1], pending):  # fold into previous one
            merged[-1] = _combine(merged[-1], pending, keep=merged[-1])
            merges += 1
        else:  # nothing to merge into, or doing so would cross a speaker boundary
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


def _assign_speaker_ids(segments, sentences):
    """Stamp each segment with a final speaker_id, verifying the invariant.

    Re-derives the speaker from the sentences a segment covers (matched by time)
    and ASSERTS they agree — a mismatch means enforcement is broken and we want a
    loud failure over a silently mislabelled clip. speaker_id is None when the
    whole segment was diarization-uncertain (no guessing).
    """
    for seg in segments:
        covered = [
            s for s in sentences
            if s["start_ms"] >= seg["start_ms"] and s["end_ms"] <= seg["end_ms"]
        ]
        speakers = {s["speaker"] for s in covered if s.get("speaker") is not None}
        if len(speakers) > 1:
            raise AssertionError(
                f"Segment [{seg['start_ms']}..{seg['end_ms']}] '{seg.get('title')}' "
                f"mixes speakers {sorted(speakers)} — hard-boundary enforcement "
                f"(get_speaker_boundaries/_enforce_hard_boundaries) failed to "
                f"split on a speaker change."
            )
        seg["speaker_id"] = speakers.pop() if speakers else None
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

    # Two boundary sources: adjustable topic-shift candidates (embeddings) and
    # non-negotiable speaker changes. detect_candidate_boundaries returns the
    # index BEFORE which the shift sits, so +1 converts to a start-of-segment
    # index, matching get_speaker_boundaries' convention.
    candidate_indices = {i + 1 for i in detect_candidate_boundaries(sentences)}
    hard_boundaries = get_speaker_boundaries(sentences)

    segments = llm_segment(sentences, candidate_indices, hard_boundaries)  # + labels
    segments = merge_short_segments(segments)  # length guardrail (speaker-aware)
    segments = _label_segments(segments)       # final labels from final text
    return _assign_speaker_ids(segments, sentences)  # speaker_id + invariant check


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

        sentences = build_sentences(words)
        print(f"\n{latest.name}: {len(words)} words -> {len(sentences)} sentences\n")
        for i, s in enumerate(sentences, start=1):
            time_range = f"{_fmt_ms(s['start_ms'])}-{_fmt_ms(s['end_ms'])}"
            speaker = s["speaker"] or "—"
            print(f"[{i}] {time_range}  ({speaker})  {s['text']}")

        segments = build_segments(words)
        print(f"\n{latest.name}: {len(words)} words -> {len(segments)} segments\n")
        for i, seg in enumerate(segments, start=1):
            time_range = f"{_fmt_ms(seg['start_ms'])}-{_fmt_ms(seg['end_ms'])}"
            print(f"[{i}] {time_range}  {seg['title']}")
            print(f"    {seg['summary']}\n")
