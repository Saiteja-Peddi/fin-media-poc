"""Interactive entry point for the fin-media POC.

Presents a menu — transcribe/ingest a file, or ask a question — and prompts for
the file path or question based on the choice. Wraps the same pipeline the
scripted CLIs (run_ingest.py, ask.py) use.

    python app.py
"""

from pathlib import Path

from src import ingest, models, query, store


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

    print("\nExtracting audio...")
    audio_path = ingest.extract_audio(path)

    print("Transcribing (first run downloads the model and is slow)...")
    words = models.asr(audio_path)
    print(f"  {len(words)} words transcribed.")

    chunks = ingest.chunk_words(words)
    print(f"  {len(chunks)} chunks produced.")

    stored = store.add_chunks(chunks, audio_path.name)
    print(f"\nStored {stored} chunks from '{audio_path.name}' in the index.\n")


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
    actions = {"1": do_ingest, "2": do_ask}

    while True:
        print("What would you like to do?")
        print("  1) Transcribe / ingest a file")
        print("  2) Ask a question")
        print("  3) Quit")
        choice = input("Select an option [1-3]: ").strip()
        print()

        if choice == "3" or choice.lower() in ("q", "quit", "exit"):
            print("Goodbye.")
            break

        action = actions.get(choice)
        if action is None:
            print("Please enter 1, 2, or 3.\n")
            continue

        action()


if __name__ == "__main__":
    main()
