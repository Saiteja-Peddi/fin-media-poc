"""Query pipeline.

Embeds a natural-language question, searches the vector store for matching
transcript chunks, and returns ranked hits with their source and timestamps.
"""

from . import clip, store


def ask(question, n_results=3):
    """Search the index and cut a fresh clip for each ranked hit.

    Clears clips left from the previous question, runs the search, then cuts one
    clip per result (in ranked order). Each returned hit gains a "clip_path".
    """
    clip.clear_clips_folder()

    results = store.search(question, n_results=n_results)

    for rank, hit in enumerate(results, start=1):
        hit["clip_path"] = clip.cut_clip(
            hit["media_file"],
            hit["start_ms"],
            hit["end_ms"],
            rank,
        )

    return results
