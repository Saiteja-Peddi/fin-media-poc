"""CLI entry point for ingesting media into the searchable index.

Usage:
    python run_ingest.py [--input PATH]
"""

import argparse

from src import ingest, models, store


def main():
    parser = argparse.ArgumentParser(description="Ingest media into the search index.")
    parser.add_argument("path", help="Path to a video/audio file to ingest.")
    args = parser.parse_args()

    audio_path = ingest.extract_audio(args.path)
    print(audio_path)

    words = models.asr(audio_path)
    print(f"{len(words)} words transcribed. First 20:")
    for w in words[:20]:
        print(f"  [{w['start_ms']:>7} - {w['end_ms']:>7} ms]  {w['word']}")

    chunks = ingest.chunk_words(words)
    print(f"\n{len(chunks)} chunks produced. First chunk:")
    first = chunks[0]
    print(f"  [{first['start_ms']} - {first['end_ms']} ms]")
    print(f"  {first['text']}")

    stored = store.add_chunks(chunks, audio_path.name)
    print(f"\nStored {stored} chunks in the vector index.")


if __name__ == "__main__":
    main()
