"""Interactive menu for the POC: ingest files or ask questions.

    python app.py
"""

from pathlib import Path

from src import config, ingest, models, query, segment, store

# Extensions the folder scanner picks up from data/input.
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv",   # video
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",  # audio
}


def _fmt_ms(ms):
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _clean_path(raw):
    """Normalize a pasted/dragged path (quotes, escaped spaces, ~) to an
    absolute Path."""
    raw = raw.strip()

    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1]
    raw = raw.replace("\\ ", " ").replace("\\", "")

    return Path(raw).expanduser().resolve()


def _run_pipeline(path):
    """Detect -> extract -> transcribe -> segment -> store. Returns the WAV name."""
    media_kind = ingest.detect_media_kind(path)
    print(f"  Detected media kind: {media_kind}")

    # Original source (what clips are cut from), not the transcription WAV.
    original_file_path = ingest.original_reference(path)

    print("  Extracting audio...")
    audio_path = ingest.extract_audio(path)

    print("  Transcribing (first run downloads the model and is slow)...")
    words = models.asr(audio_path)
    print(f"    {len(words)} words transcribed.")

    print("  Segmenting into topics...")
    segments = segment.build_segments(words)
    print(f"    {len(segments)} segments produced:")
    for i, seg in enumerate(segments, start=1):
        time_range = f"{_fmt_ms(seg['start_ms'])}-{_fmt_ms(seg['end_ms'])}"
        print(f"      [{i}] {time_range}  {seg['title']}")

    stored = store.add_chunks(segments, audio_path.name, media_kind, original_file_path)
    print(f"  Stored {stored} segments from '{audio_path.name}'.")
    return audio_path.name


def _processed_media_files():
    """Return the set of media_file names already present in the index."""
    stored = store._collection.get(include=["metadatas"])
    return {m["media_file"] for m in stored["metadatas"]}


def do_ingest():
    """Prompt for a media file and run the full ingest pipeline."""
    raw = input("Path to the video/audio file (any location on your system): ").strip()
    if not raw:
        print("No path entered.\n")
        return

    path = _clean_path(raw)
    if not path.is_file():
        print(f"File not found: {path}\n")
        return

    print()
    try:
        _run_pipeline(path)
    except Exception as e:
        print(f"  Could not process this file: {e}")
    print()


def do_ingest_folder():
    """Scan data/input and ingest any files not yet in the index."""
    input_dir = Path(config.DATA_INPUT)
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS
    )

    if not files:
        print(f"No media files found in {input_dir}.\n")
        return

    processed = _processed_media_files()
    # Already ingested if its .wav name is in the index.
    pending = [p for p in files if f"{p.stem}.wav" not in processed]

    print(f"Found {len(files)} media file(s) in {input_dir}; "
          f"{len(pending)} unprocessed.\n")

    if not pending:
        print("Everything is already processed.\n")
        return

    processed_count = 0
    for i, path in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] {path.name}")
        try:
            _run_pipeline(path)
            processed_count += 1
        except Exception as e:
            print(f"  Skipped (could not process): {e}")
        print()

    print(f"Done. Processed {processed_count} of {len(pending)} new file(s).\n")


def do_ask():
    """Prompt for a question and print ranked search results."""
    question = input("Your question: ").strip()
    if not question:
        print("No question entered.\n")
        return

    k_raw = input("How many results? [3]: ").strip()
    n_results = int(k_raw) if k_raw.isdigit() and int(k_raw) > 0 else 3

    # Merge lets one result cover several topics from the SAME source video.
    merge_map = {"1": query.MERGE_NONE, "2": query.MERGE_ADJACENT, "3": query.MERGE_PARENT}
    m_raw = input(
        "Merge same-video clips? [1] none  [2] adjacent topics  [3] all matches: "
    ).strip()
    merge = merge_map.get(m_raw, query.MERGE_NONE)

    results = query.ask(question, n_results=n_results, merge=merge)

    if not results:
        print("\nNo matches found.\n")
        return

    print(f'\nResults for: "{question}"\n')
    for rank, hit in enumerate(results, start=1):
        time_range = f"{_fmt_ms(hit['start_ms'])}-{_fmt_ms(hit['end_ms'])}"
        print(f"[{rank}] {time_range}  ({hit['media_file']})")
        print(f"    {hit['text']}")
        print(f"    Clip [{hit['media_kind']}]: {hit['clip_path']}\n")


def main():
    print("fin-media-poc — searchable spoken financial content\n")
    actions = {"1": do_ingest, "2": do_ingest_folder, "3": do_ask}

    while True:
        print("What would you like to do?")
        print("  1) Transcribe / ingest a file")
        print("  2) Process new files in data/input")
        print("  3) Ask a question")
        print("  4) Quit")
        choice = input("Select an option [1-4]: ").strip()
        print()

        if choice == "4" or choice.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            break

        action = actions.get(choice)
        if action is None:
            print("Please enter 1, 2, 3, or 4.\n")
            continue

        action()


if __name__ == "__main__":
    main()
