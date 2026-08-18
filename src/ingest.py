"""Ingestion: detect kind, extract audio, and chunk the transcript."""

from pathlib import Path

import ffmpeg

from . import config


def detect_media_kind(file_path):
    """Return "video" or "audio" from the file's actual streams, not its name.

    Extensions lie, so we probe the container. Embedded cover art is ignored so
    an MP3 isn't misread as video. Raises if neither stream type is found.
    """
    file_path = Path(file_path)

    try:
        info = ffmpeg.probe(str(file_path))
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffprobe failed for {file_path}:\n{e.stderr.decode()}")

    has_video = False
    has_audio = False
    for stream in info.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            if stream.get("disposition", {}).get("attached_pic") == 1:
                continue  # cover art / thumbnail, not real video
            has_video = True
        elif codec_type == "audio":
            has_audio = True

    if has_video:
        return config.MEDIA_KIND_VIDEO
    if has_audio:
        return config.MEDIA_KIND_AUDIO
    raise ValueError(
        f"No audio or video streams found in {file_path} — "
        "the file may be corrupt or an unsupported format."
    )


def original_reference(file_path):
    """Absolute path to the user's source file — what clips are cut from.

    Kept separate from the 16kHz WAV extract_audio writes (which has no video
    and lower quality) and carried into the stored metadata.
    """
    return str(Path(file_path).resolve())


def extract_audio(video_path):
    """Extract a 16kHz mono WAV into data/media for ASR (skips if it exists)."""
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
        print(e.stderr.decode())  # ffmpeg-python otherwise swallows the message
        raise

    return out_path


def chunk_words(words, window_seconds=30):
    """Group words into fixed ~window_seconds chunks, anchored at each chunk's
    first word (words are never split). Placeholder for topic segmentation.

    Returns [{"text", "start_ms", "end_ms", "chunk_index"}, ...].
    """
    window_ms = window_seconds * 1000
    chunks = []
    current = []
    cutoff = None

    for w in words:
        if cutoff is None:
            cutoff = w["start_ms"] + window_ms

        if w["start_ms"] >= cutoff:  # past the window edge -> new chunk
            chunks.append(_make_chunk(current, len(chunks)))
            current = []
            cutoff = w["start_ms"] + window_ms

        current.append(w)

    if current:
        chunks.append(_make_chunk(current, len(chunks)))

    return chunks


def _make_chunk(words, chunk_index):
    return {
        "text": " ".join(w["word"] for w in words),
        "start_ms": words[0]["start_ms"],
        "end_ms": words[-1]["end_ms"],
        "chunk_index": chunk_index,
    }
