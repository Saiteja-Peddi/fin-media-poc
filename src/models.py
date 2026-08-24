"""Model wrappers (WhisperX ASR, Ollama embeddings). The one place to swap a
local model for a hosted API."""

import json
from functools import lru_cache
from pathlib import Path

from . import config


@lru_cache(maxsize=1)
def _asr_model():
    """Load the WhisperX model once per process."""
    import whisperx

    return whisperx.load_model(
        config.WHISPER_MODEL,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
    )


def asr(audio_path):
    """Transcribe audio to word-level timestamps.

    Returns [{"word": str, "start_ms": int, "end_ms": int}, ...], cached to
    {basename}_words.json and loaded verbatim if present.
    """
    audio_path = Path(audio_path)
    cache_path = config.DATA_MEDIA / f"{audio_path.stem}_words.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    import whisperx

    model = _asr_model()
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=8)

    # Forced alignment turns segment-level output into per-word timestamps.
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=config.DEVICE
    )
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, config.DEVICE
    )

    words = []
    for segment in aligned["segments"]:
        for w in segment.get("words", []):
            # Unalignable tokens (some digits/punctuation) lack timestamps.
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


def embed(texts, is_query=False):
    """Embed strings via Ollama (one call each). Requires `ollama serve`.

    mxbai-embed-large is asymmetric: pass is_query=True for search queries so
    the instruction prefix is applied; documents get no prefix.
    """
    import ollama

    prefix = config.EMBED_QUERY_PREFIX if is_query else ""

    vectors = []
    for text in texts:
        try:
            response = ollama.embeddings(
                model=config.EMBED_MODEL, prompt=f"{prefix}{text}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Ollama embedding call failed for model '{config.EMBED_MODEL}' "
                f"(is `ollama serve` running on {config.OLLAMA_HOST}?): {e}"
            ) from e
        vectors.append(response["embedding"])
    return vectors


def llm(prompt, model=None, format=None, temperature=None):
    """Generate a text reply from the local LLM via Ollama. Requires `ollama serve`.

    format: optional JSON schema (dict) or "json" to constrain output to valid
    JSON — needed for reliable structured output from small local models.
    temperature: optional decoding temperature (0 = deterministic).
    """
    import ollama

    model = model or config.LLM_MODEL
    # Ollama defaults num_ctx to 2048 and silently truncates; request the
    # configured budget so full transcripts actually fit.
    options = {"num_ctx": config.MODEL_CONTEXT_TOKENS}
    if temperature is not None:
        options["temperature"] = temperature
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=format,
            options=options,
        )
    except Exception as e:
        raise RuntimeError(
            f"Ollama chat call failed for model '{model}' "
            f"(is `ollama serve` running on {config.OLLAMA_HOST}?): {e}"
        ) from e
    return response["message"]["content"]


def vision(image_path, prompt):
    """Answer about an image with the local vision model. Wired up later."""
    raise NotImplementedError
