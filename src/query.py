"""Query pipeline.

Embeds a natural-language question, searches the vector store for matching
transcript chunks, and returns ranked hits with their source and timestamps.
"""

from . import store


def ask(question, n_results=3):
    """Search the index for a question. Returns ranked hit dicts (see store.search)."""
    return store.search(question, n_results=n_results)
