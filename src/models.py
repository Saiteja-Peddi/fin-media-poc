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


@lru_cache(maxsize=1)
def _diarize_pipeline(device):
    """Load WhisperX's diarization pipeline once per process (kept separate from
    diarize() so the slow model load is cached and the token check runs first)."""
    from whisperx.diarize import DiarizationPipeline

    # WhisperX renamed this kwarg from use_auth_token -> token, and its default
    # model is now the separately-gated community-1; pin the model we document.
    return DiarizationPipeline(
        model_name=config.DIARIZATION_MODEL, token=config.HF_TOKEN, device=device
    )


def diarize(audio_path, device="cpu"):
    """Run speaker diarization ("who spoke when") over an audio file.

    Returns WhisperX's DataFrame (start/end/speaker) unreshaped, for
    assign_speakers_to_words(). Needs an HF token for the gated pyannote models;
    raises a setup-pointing error rather than a raw stack trace when it's missing.
    """
    if not config.HF_TOKEN:
        raise RuntimeError(
            "Speaker diarization needs a HuggingFace token, but HF_TOKEN is empty.\n"
            "  1. Copy .env.example to .env and paste a Read-role token.\n"
            "  2. Accept the licences for pyannote/speaker-diarization-3.1 and\n"
            "     pyannote/segmentation-3.0 (see .env.example for the links).\n"
            "To run without diarization, set ENABLE_DIARIZATION = False in "
            "src/config.py."
        )

    import whisperx

    pipeline = _diarize_pipeline(device)
    audio = whisperx.load_audio(str(audio_path))
    return pipeline(audio)


def assign_speakers_to_words(words, diarize_result):
    """Tag each word with the speaker who said it, using diarize() output.

    Returns a new copy of the flat word list where every word also carries a
    "speaker" field (e.g. "SPEAKER_00"); words WhisperX can't confidently assign
    get speaker=None rather than a guess.
    """
    import whisperx

    # assign_word_speakers works on whisperx's own shape (segments of words with
    # second-based start/end), so adapt our flat ms list, tag in place, read back.
    wx_words = [
        {
            "word": w["word"],
            "start": w["start_ms"] / 1000,
            "end": w["end_ms"] / 1000,
        }
        for w in words
    ]
    transcript = {"segments": [{"words": wx_words}], "word_segments": wx_words}
    whisperx.assign_word_speakers(diarize_result, transcript)

    return [
        {**w, "speaker": wx.get("speaker")}
        for w, wx in zip(words, wx_words)
    ]


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
