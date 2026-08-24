"""Cut playable clips from a hit's ORIGINAL source file (never the ASR WAV).

Uses the ffmpeg-python library and pathlib.Path (cross-platform).
"""

from pathlib import Path

import ffmpeg

from . import config

PAD_MS = 400  # padding each side so clips don't cut off mid-word

# Used only when the source file has no extension of its own.
_FALLBACK_SUFFIX = {
    config.MEDIA_KIND_VIDEO: ".mp4",
    config.MEDIA_KIND_AUDIO: ".wav",
}


def _source_duration_ms(media_path):
    """Source duration in ms via ffprobe."""
    try:
        info = ffmpeg.probe(str(media_path))
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffprobe failed for {media_path}:\n{e.stderr.decode()}")
    return int(float(info["format"]["duration"]) * 1000)


def _pad_clamp(start_ms, end_ms, duration_ms):
    """Pad a span by PAD_MS each side, clamped to [0, duration]. Returns secs."""
    padded_start_ms = max(0, start_ms - PAD_MS)
    padded_end_ms = min(duration_ms, end_ms + PAD_MS)
    return padded_start_ms / 1000, padded_end_ms / 1000


def _video_output_opts(media_kind):
    """Encoder options: re-encode video (fresh keyframe at the cut, no black
    lead-in); stream-copy audio-only sources (no keyframe dependency)."""
    if media_kind == config.MEDIA_KIND_VIDEO:
        return {
            "vcodec": "libx264",
            "acodec": "aac",
            "preset": "veryfast",
            "movflags": "+faststart",
        }
    return {"c": "copy"}


def cut_clip(original_file_path, media_kind, start_ms, end_ms, rank, out_dir=None):
    """Cut a padded clip [start_ms, end_ms] from the source, written fresh.

    Padding is clamped to [0, duration]. The clip keeps the source's extension
    (falling back to a media_kind default only if it has none), so it's video
    or audio depending on the source. Returns the output Path.
    """
    if out_dir is None:
        out_dir = config.DATA_CLIPS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Path(original_file_path)
    duration_ms = _source_duration_ms(source)

    start_s, end_s = _pad_clamp(start_ms, end_ms, duration_ms)

    suffix = source.suffix or _FALLBACK_SUFFIX.get(media_kind, ".mp4")
    out_path = out_dir / f"clip_{rank}{suffix}"

    try:
        (
            ffmpeg
            .input(str(source))
            .output(
                str(out_path),
                ss=f"{start_s:.3f}",
                to=f"{end_s:.3f}",
                **_video_output_opts(media_kind),
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg failed cutting {source}:\n{e.stderr.decode()}")

    return out_path


def concat_clips(original_file_path, media_kind, spans_ms, rank, out_dir=None):
    """Stitch several [start_ms, end_ms] spans of ONE source into a single clip.

    All spans MUST come from the same parent file (the query layer never mixes
    sources). Spans are cut and joined in the order given via the ffmpeg concat
    filter in a single re-encode — no temp files. With one span this is just a
    continuous cut; with several the joins are hard (jump-)cuts. Returns the Path.
    """
    if not spans_ms:
        raise ValueError("concat_clips needs at least one span")
    if len(spans_ms) == 1:
        start_ms, end_ms = spans_ms[0]
        return cut_clip(original_file_path, media_kind, start_ms, end_ms, rank, out_dir)

    if out_dir is None:
        out_dir = config.DATA_CLIPS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Path(original_file_path)
    duration_ms = _source_duration_ms(source)

    suffix = source.suffix or _FALLBACK_SUFFIX.get(media_kind, ".mp4")
    out_path = out_dir / f"clip_{rank}{suffix}"

    is_video = media_kind == config.MEDIA_KIND_VIDEO
    # Trim each span (input-side seek) and feed the streams into concat. The
    # concat filter resets timestamps at every join, so the re-encode below
    # starts each piece cleanly — same reason cut_clip re-encodes video.
    streams = []
    for start_ms, end_ms in spans_ms:
        start_s, end_s = _pad_clamp(start_ms, end_ms, duration_ms)
        piece = ffmpeg.input(str(source), ss=f"{start_s:.3f}", to=f"{end_s:.3f}")
        if is_video:
            streams.extend([piece.video, piece.audio])
        else:
            streams.append(piece.audio)

    node = ffmpeg.concat(
        *streams, n=len(spans_ms), v=1 if is_video else 0, a=1
    ).node
    if is_video:
        out = ffmpeg.output(
            node[0], node[1], str(out_path),
            vcodec="libx264", acodec="aac", preset="veryfast",
            movflags="+faststart",
        )
    else:
        out = ffmpeg.output(node[0], str(out_path), acodec="aac")

    try:
        out.overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg failed concatenating {source}:\n{e.stderr.decode()}")

    return out_path


def clear_clips_folder():
    """Empty data/clips (keeping .gitkeep) so queries' clips never mix."""
    clips_dir = Path(config.DATA_CLIPS)
    clips_dir.mkdir(parents=True, exist_ok=True)
    for entry in clips_dir.iterdir():
        if entry.is_file() and entry.name != ".gitkeep":
            entry.unlink()


if __name__ == "__main__":
    # Cut one clip from the first chunk in Chroma (requires an ingested file).
    from . import store

    clear_clips_folder()

    metadatas = store._collection.get(include=["metadatas"])["metadatas"]
    if not metadatas:
        print("No chunks in the store — ingest a file first.")
    else:
        first = metadatas[0]
        path = cut_clip(
            first["original_file_path"],
            first["media_kind"],
            first["start_ms"],
            first["end_ms"],
            rank=1,
        )
        print(f"Cut clip: {path}  (media_kind: {first['media_kind']})")
