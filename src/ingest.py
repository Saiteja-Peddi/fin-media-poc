"""Ingestion pipeline.

Takes raw files from data/input, normalizes them to audio in data/media,
runs ASR to get timestamped transcripts, chunks the segments, and hands them
to the store for embedding + indexing.
"""

from pathlib import Path

import ffmpeg

from . import config


def extract_audio(video_path):
    """Extract a video's audio track to 16kHz mono WAV for ASR.

    Writes to config.DATA_MEDIA using the input's basename with a .wav
    extension. Skips extraction if the output already exists. Returns the
    output path.
    """
    video_path = Path(video_path)
    out_path = config.DATA_MEDIA / f"{video_path.stem}.wav"

    if out_path.exists():
        return out_path

    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(out_path), ar=16000, ac=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        # ffmpeg-python swallows ffmpeg's own error text; surface it.
        print(e.stderr.decode())
        raise

    return out_path


def chunk_words(words, window_seconds=30):
    """Group a flat word list into consecutive fixed-length time windows.

    Deliberately crude: each chunk spans roughly `window_seconds`, anchored at
    its first word. A chunk ends at the last word that *starts* before the
    cutoff, so words are never split. To be replaced by topic segmentation
    later.

    Returns a list of:
        {"text": str, "start_ms": int, "end_ms": int, "chunk_index": int}
    """
    window_ms = window_seconds * 1000
    chunks = []
    current = []
    cutoff = None

    for w in words:
        if cutoff is None:
            cutoff = w["start_ms"] + window_ms

        # Word starts past the window edge -> flush and open a new window.
        if w["start_ms"] >= cutoff:
            chunks.append(_make_chunk(current, len(chunks)))
            current = []
            cutoff = w["start_ms"] + window_ms

        current.append(w)

    if current:
        chunks.append(_make_chunk(current, len(chunks)))

    return chunks


def _make_chunk(words, chunk_index):
    """Assemble one chunk dict from a non-empty list of word dicts."""
    return {
        "text": " ".join(w["word"] for w in words),
        "start_ms": words[0]["start_ms"],
        "end_ms": words[-1]["end_ms"],
        "chunk_index": chunk_index,
    }
