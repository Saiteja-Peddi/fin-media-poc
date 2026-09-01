"""Central config: paths, model names, and toggles used across the POC."""

import os
from pathlib import Path

import certifi
from dotenv import load_dotenv

# macOS python.org builds lack a usable CA bundle, so model downloads fail with
# CERTIFICATE_VERIFY_FAILED; point TLS at certifi's bundle unless already set.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("SSL_CERT_DIR", str(Path(certifi.where()).parent))

# --- Paths (created on import so the rest of the code can assume they exist) --
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load secrets from the repo-root .env (copied from .env.example) into the
# environment; never commit .env. Nothing overrides already-set env vars.
load_dotenv(PROJECT_ROOT / ".env")

DATA_INPUT = PROJECT_ROOT / "data" / "input"   # user's source files
DATA_MEDIA = PROJECT_ROOT / "data" / "media"   # extracted WAVs + transcripts
DATA_CLIPS = PROJECT_ROOT / "data" / "clips"   # clips cut around search hits
CHROMA_DIR = PROJECT_ROOT / "db" / "chroma"    # persistent vector store

for _p in (DATA_INPUT, DATA_MEDIA, DATA_CLIPS, CHROMA_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Models ------------------------------------------------------------------
# distil-large-v3: near large-v3 accuracy, ~1.5GB int8, good on 8GB. Its
# CTranslate2 backend has no Metal support, so DEVICE must stay "cpu".
WHISPER_MODEL = "distil-large-v3"

# mxbai-embed-large needs a query-only instruction prefix (see models.embed).
EMBED_MODEL = "mxbai-embed-large"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Vision not wired up yet (models.vision raises). Pull before use:
#   ollama pull llama3.2:3b        ollama pull moondream
LLM_MODEL = "llama3.2:3b"
VISION_MODEL = "moondream"

# HuggingFace READ token for speaker diarization (gated pyannote models). Read
# from .env; empty means diarization is disabled and the rest runs local-only.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Master switch: turn off to skip diarization and fall back to plain
# transcription (e.g. when no HF token is set up), without removing any code.
ENABLE_DIARIZATION = True

# Pinned because newer WhisperX defaults to the separately-gated community-1.
# We target 3.1 — the model .env.example tells users to accept.
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# Shared by ingest.detect_media_kind() and the stored metadata.
MEDIA_KIND_VIDEO = "video"
MEDIA_KIND_AUDIO = "audio"

# Below this cosine similarity, consecutive sentences are treated as a topic
# shift. Reserved for tier-2 coarse pre-segmentation; needs tuning.
SIMILARITY_THRESHOLD = 0.5

# Segments shorter than this are merged into a neighbour (segment.
# merge_short_segments) — also curbs the small model's tendency to over-split a
# topic into tiny fragments. No maximum: length follows coherence, not time.
MIN_SEGMENT_SECONDS = 15

# Two-stage retrieval: pull this many candidates by embedding, then the LLM
# re-ranks them down to the requested number (query._rerank).
RERANK_CANDIDATES = 10

# Context budget for one-pass (tier-1) topic segmentation, sized to the LOCAL
# model. num_ctx is what we ask Ollama to allocate; llama3.2:3b handles 8192
# comfortably on 8GB. Bump this up with bigger models (e.g. a hosted Sonnet's
# ~200K) so tier-1 covers longer transcripts before the tier-2 split kicks in.
MODEL_CONTEXT_TOKENS = 8192
# Transcript budget only — leave headroom for the prompt scaffold and response.
SAFE_CONTEXT_TOKENS = int(MODEL_CONTEXT_TOKENS * 0.6)

# --- Toggles / endpoints / compute -------------------------------------------
USE_LOCAL_ASR = True
USE_LOCAL_EMBED = True
USE_LOCAL_LLM = True
USE_LOCAL_VISION = True

OLLAMA_HOST = "http://localhost:11434"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
