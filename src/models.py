"""Thin wrappers around the ML models used by the POC.

Each function is a single, self-contained call to one model so the rest of the
codebase depends on these signatures rather than on WhisperX / sentence-
transformers / Ollama directly. This is the one place to change when swapping a
local model for a hosted API.
"""

import json
from functools import lru_cache
from pathlib import Path

from . import config


@lru_cache(maxsize=1)
def _asr_model():
    """Load and cache the WhisperX transcription model (lazy, once)."""
    import whisperx

    return whisperx.load_model(
        config.WHISPER_MODEL,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
    )


def asr(audio_path):
    """Transcribe audio and return a flat list of word-level timestamps.

    Runs WhisperX transcription followed by forced alignment so every word
    gets its own start/end. Returns a list of dicts:
        [{"word": str, "start_ms": int, "end_ms": int}, ...]

    Results are cached to {basename}_words.json next to the audio, since
    reloading WhisperX is slow; a cached file is loaded verbatim.
    """
    audio_path = Path(audio_path)
    cache_path = config.DATA_MEDIA / f"{audio_path.stem}_words.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import whisperx

    model = _asr_model()
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16)

    # Forced alignment: turns segment-level output into word-level timestamps.
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=config.DEVICE
    )
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, config.DEVICE
    )

    words = []
    for segment in aligned["segments"]:
        for w in segment.get("words", []):
            # Some tokens (e.g. digits/punctuation) can't be aligned and lack
            # timestamps; skip them so every entry has valid integer ms.
            if "start" not in w or "end" not in w:
                continue
            words.append(
                {
                    "word": w["word"],
                    "start_ms": int(w["start"] * 1000),
                    "end_ms": int(w["end"] * 1000),
                }
            )

    with open(cache_path, "w") as f:
        json.dump(words, f)

    return words


def embed(texts):
    """Embed a list of strings into a list of vectors via Ollama.

    Ollama's embeddings endpoint takes one text at a time, so we loop. Assumes
    `ollama serve` is running locally. Raises a clear error if a call fails.
    """
    import ollama

    vectors = []
    for text in texts:
        try:
            response = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
        except Exception as e:
            raise RuntimeError(
                f"Ollama embedding call failed for model '{config.EMBED_MODEL}' "
                f"(is `ollama serve` running on {config.OLLAMA_HOST}?): {e}"
            ) from e
        vectors.append(response["embedding"])
    return vectors


def llm(prompt):
    """Generate text from the local LLM. Wired up later."""
    raise NotImplementedError


def vision(image_path, prompt):
    """Describe/answer about an image with the local vision model. Wired up later."""
    raise NotImplementedError
