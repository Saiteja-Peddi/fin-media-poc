"""Central config: paths, model names, and toggles used across the POC."""

import os
from pathlib import Path

import certifi

# macOS python.org builds lack a usable CA bundle, so model downloads fail with
# CERTIFICATE_VERIFY_FAILED; point TLS at certifi's bundle unless already set.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("SSL_CERT_DIR", str(Path(certifi.where()).parent))

# --- Paths (created on import so the rest of the code can assume they exist) --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

# Not wired up yet (models.llm/vision raise). Sized for 8GB; pull before use:
#   ollama pull llama3.2:3b        ollama pull moondream
LLM_MODEL = "llama3.2:3b"
VISION_MODEL = "moondream"

# Shared by ingest.detect_media_kind() and the stored metadata.
MEDIA_KIND_VIDEO = "video"
MEDIA_KIND_AUDIO = "audio"

# --- Toggles / endpoints / compute -------------------------------------------
USE_LOCAL_ASR = True
USE_LOCAL_EMBED = True
USE_LOCAL_LLM = True
USE_LOCAL_VISION = True

OLLAMA_HOST = "http://localhost:11434"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
