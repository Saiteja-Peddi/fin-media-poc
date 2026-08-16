"""Central configuration for the fin-media POC.

Holds all filesystem paths, model names, and feature toggles in one place so
the rest of the code never hardcodes locations or model identifiers. Paths are
resolved relative to the project root and created on import if missing.
"""

import os
from pathlib import Path

# --- TLS certificates ------------------------------------------------------
# macOS python.org builds ship without a usable CA bundle, so model downloads
# (WhisperX/torch alignment weights) fail with CERTIFICATE_VERIFY_FAILED.
# Point urllib/requests at certifi's bundle unless the user already set it.
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("SSL_CERT_DIR", str(Path(certifi.where()).parent))

# --- Paths -----------------------------------------------------------------
# Project root is the parent of the src/ directory this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_INPUT = PROJECT_ROOT / "data" / "input"   # raw source files dropped by the user
DATA_MEDIA = PROJECT_ROOT / "data" / "media"   # normalized audio/video extracted from input
DATA_CLIPS = PROJECT_ROOT / "data" / "clips"   # short clips cut around search hits
CHROMA_DIR = PROJECT_ROOT / "db" / "chroma"    # persistent Chroma vector store

# Create any missing directories so downstream code can assume they exist.
for _p in (DATA_INPUT, DATA_MEDIA, DATA_CLIPS, CHROMA_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Model names -----------------------------------------------------------
WHISPER_MODEL = "small"                     # WhisperX ASR model size
EMBED_MODEL = "nomic-embed-text"            # Ollama embedding model
LLM_MODEL = "llama3.1"                       # Ollama model for local generation
VISION_MODEL = "llava"                       # Ollama vision model for frames

# --- Toggle flags ----------------------------------------------------------
# Flip these to swap local models for hosted APIs later on.
USE_LOCAL_ASR = True
USE_LOCAL_EMBED = True
USE_LOCAL_LLM = True
USE_LOCAL_VISION = True

# --- Local service endpoints ----------------------------------------------
OLLAMA_HOST = "http://localhost:11434"

# --- Compute settings ------------------------------------------------------
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
