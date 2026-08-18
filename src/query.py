"""Query pipeline: search the index and cut a clip per ranked hit."""

from . import clip, store


def ask(question, n_results=3):
    """Clear old clips, search, and cut a fresh clip per hit (ranked from 1).

    Each returned hit gains "clip_path"; "media_kind" is already present from
    the stored metadata.
    """
    clip.clear_clips_folder()

    results = store.search(question, n_results=n_results)

    for rank, hit in enumerate(results, start=1):
        hit["clip_path"] = str(clip.cut_clip(
            hit["original_file_path"],
            hit["media_kind"],
            hit["start_ms"],
            hit["end_ms"],
            rank,
        ))

    return results
