"""CLI entry point for asking questions against the indexed media.

Usage:
    python ask.py "your question here" [--k N]
"""

import argparse

from src import query


def _fmt_ms(ms):
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description="Search indexed media by question.")
    parser.add_argument("question", help="Natural-language question to search for.")
    parser.add_argument(
        "--k", type=int, default=3, help="Number of results to return."
    )
    args = parser.parse_args()

    results = query.ask(args.question, n_results=args.k)

    if not results:
        print("No matches found.")
        return

    print(f'Results for: "{args.question}"\n')
    for rank, hit in enumerate(results, start=1):
        time_range = f"{_fmt_ms(hit['start_ms'])}-{_fmt_ms(hit['end_ms'])}"
        print(f"[{rank}] {time_range}  ({hit['media_file']})")
        print(f"    {hit['text']}")
        print(f"    Clip: {hit['clip_path']}\n")


if __name__ == "__main__":
    main()
