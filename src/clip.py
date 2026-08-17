"""Clip extraction.

Given a source media file and a start/end timestamp from a search hit, uses
ffmpeg (via the ffmpeg-python library, like ingest.extract_audio) to cut a
short clip into data/clips for playback.

All path handling goes through pathlib.Path so this works identically on
Windows and macOS.
"""

from pathlib import Path

import ffmpeg

from . import config

# Milliseconds of padding added before/after each hit so clips don't start or
# end mid-word. Clamped to the source's bounds.
PAD_MS = 400


def _source_duration_ms(media_path):
    """Return the source file's duration in milliseconds via ffprobe."""
    try:
        info = ffmpeg.probe(str(media_path))
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"ffprobe failed for {media_path}:\n{e.stderr.decode()}"
        )
    return int(float(info["format"]["duration"]) * 1000)


def _resolve_media(media_file):
    """Resolve a media reference to an actual file path.

    Accepts a full path or a bare basename; a basename is looked up in
    config.DATA_MEDIA (where extract_audio writes the WAVs).
    """
    path = Path(media_file)
    if path.is_file():
        return path
    return config.DATA_MEDIA / path.name


def cut_clip(media_file, start_ms, end_ms, rank, out_dir=None):
    """Cut a padded clip from a source file and write it to out_dir.

    Cuts directly from the whole source with ffmpeg seek flags (no re-encode).
    The clip is padded by PAD_MS on each side, clamped to [0, duration]. Output
    is named clip_{rank} with the source's extension so it also works for video.
    Always writes fresh. Returns the output path.
    """
    if out_dir is None:
        out_dir = config.DATA_CLIPS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = _resolve_media(media_file)
    duration_ms = _source_duration_ms(source)

    # Pad, then clamp so we never seek before 0 or past the end of the file.
    padded_start_ms = max(0, start_ms - PAD_MS)
    padded_end_ms = min(duration_ms, end_ms + PAD_MS)

    out_path = out_dir / f"clip_{rank}{source.suffix}"

    try:
        (
            ffmpeg
            .input(str(source))
            .output(
                str(out_path),
                ss=f"{padded_start_ms / 1000:.3f}",  # seek relative to file start
                to=f"{padded_end_ms / 1000:.3f}",
                c="copy",                            # stream copy, no re-encode
            )
            .overwrite_output()                      # always write fresh
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        # ffmpeg-python swallows ffmpeg's own error text; surface it.
        raise RuntimeError(
            f"ffmpeg failed cutting {source}:\n{e.stderr.decode()}"
        )

    return out_path


def clear_clips_folder():
    """Delete every file in config.DATA_CLIPS, keeping the folder itself.

    Creates the folder first if missing, so it's safe to call anytime.
    """
    clips_dir = Path(config.DATA_CLIPS)
    clips_dir.mkdir(parents=True, exist_ok=True)
    for entry in clips_dir.iterdir():
        # Keep .gitkeep so the (otherwise-ignored) folder stays tracked in git.
        if entry.is_file() and entry.name != ".gitkeep":
            entry.unlink()


if __name__ == "__main__":
    # Manual test: clear the clips folder, then cut one clip from the first
    # chunk currently stored in Chroma.
    from . import store

    clear_clips_folder()

    stored = store._collection.get(include=["metadatas"])
    metadatas = stored["metadatas"]
    if not metadatas:
        print("No chunks in the store — ingest a file first.")
    else:
        first = metadatas[0]
        path = cut_clip(
            first["media_file"],
            first["start_ms"],
            first["end_ms"],
            rank=1,
        )
        print(f"Cut clip: {path}")
