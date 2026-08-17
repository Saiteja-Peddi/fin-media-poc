"""Interactive entry point for the fin-media POC.

Presents a menu — transcribe/ingest a file, or ask a question — and prompts for
the file path or question based on the choice. Wraps the same pipeline the
scripted CLIs (run_ingest.py, ask.py) use.

    python app.py
"""

from pathlib import Path

from src import config, ingest, models, query, store

# Media extensions the batch scanner will pick up from data/input.
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv",   # video
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",  # audio
}


def _fmt_ms(ms):
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def _clean_path(raw):
    """Normalize a pasted/dragged path into an absolute Path.

    Accepts any absolute or relative path anywhere on the system and tidies up
    the forms terminals produce: surrounding quotes, backslash-escaped spaces,
    a leading `~`, and stray whitespace. Relative paths resolve against the
    current working directory.
    """
    raw = raw.strip()

    # Strip a matching pair of surrounding quotes (common when pasting).
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1]

    # Un-escape backslash-escaped characters (e.g. dragged-in "my\ file.mp4").
    raw = raw.replace("\\ ", " ").replace("\\", "")

    return Path(raw).expanduser().resolve()


def _run_pipeline(path):
    """Run extract -> transcribe -> chunk -> store for one file.

    Returns the name of the stored media file (the .wav basename).
    """
    print("  Extracting audio...")
    audio_path = ingest.extract_audio(path)

    print("  Transcribing (first run downloads the model and is slow)...")
    words = models.asr(audio_path)
    print(f"    {len(words)} words transcribed.")

    chunks = ingest.chunk_words(words)
    print(f"    {len(chunks)} chunks produced.")

    stored = store.add_chunks(chunks, audio_path.name)
    print(f"  Stored {stored} chunks from '{audio_path.name}'.")
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
    _run_pipeline(path)
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
    # A file is already ingested if its extracted .wav name is in the index.
    pending = [p for p in files if f"{p.stem}.wav" not in processed]

    print(f"Found {len(files)} media file(s) in {input_dir}; "
          f"{len(pending)} unprocessed.\n")

    if not pending:
        print("Everything is already processed.\n")
        return

    for i, path in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] {path.name}")
        _run_pipeline(path)
        print()

    print(f"Done. Processed {len(pending)} new file(s).\n")


def do_ask():
    """Prompt for a question and print ranked search results."""
    question = input("Your question: ").strip()
    if not question:
        print("No question entered.\n")
        return

    k_raw = input("How many results? [3]: ").strip()
    n_results = int(k_raw) if k_raw.isdigit() and int(k_raw) > 0 else 3

    results = query.ask(question, n_results=n_results)

    if not results:
        print("\nNo matches found.\n")
        return

    print(f'\nResults for: "{question}"\n')
    for rank, hit in enumerate(results, start=1):
        time_range = f"{_fmt_ms(hit['start_ms'])}-{_fmt_ms(hit['end_ms'])}"
        print(f"[{rank}] {time_range}  ({hit['media_file']})")
        print(f"    {hit['text']}\n")


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
