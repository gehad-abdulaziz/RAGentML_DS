"""Shared configuration and constants.

Owner: Member 6 (feature/agent-interface).

Loads environment variables from `.env` and exposes shared constants used by every
part of the project. Other modules should import from here instead of reading
os.environ directly.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)  # load .env variables into os.environ

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCUMENTS_DIR = DATA_DIR / "documents"
GRAPH_SEED_DIR = DATA_DIR / "graph_seed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure logs dir exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Agent constants ---------------------------------------------------------
MAX_ITERATIONS = 5  # hard limit on agent loop iterations (course requirement)

# --- LLM ---------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# --- Neo4j -------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "change-me-please")

# --- ChromaDB & Embeddings ---------------------------------------------------
_chroma_path = Path(os.getenv("CHROMA_PATH", "chroma_db"))
CHROMA_PATH = _chroma_path if _chroma_path.is_absolute() else PROJECT_ROOT / _chroma_path
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "study_guides")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Tool Execution Flags ----------------------------------------------------
USE_MOCK_TOOLS = os.getenv("USE_MOCK_TOOLS", "0").strip().lower() in ("1", "true", "yes")
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "0").strip().lower() in ("1", "true", "yes")
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "agent.log")
