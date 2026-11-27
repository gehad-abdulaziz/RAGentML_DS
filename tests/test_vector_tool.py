"""Tests for the vector search tool. Owner: Member 3 (feature/rag-retrieval).

Two groups:

1. Unit tests (always run, no model download needed). They build a tiny Chroma index in
   a temporary folder with hand-made 3-dimensional vectors, and replace the query
   embedder with a fake one, so the expected ranking is known exactly.

2. Retrieval quality tests (need the real index and the embedding model). They run
   English and Arabic questions and check that the right study guide is in the top 3.
   They are skipped with a reason if the index or the model is not available.
   Setup: python -m src.vectordb.build_index   (once, with internet access)

Run:  pytest tests/test_vector_tool.py -v -rs   (-rs shows why tests were skipped)
Save the output to logs/ (e.g., pytest tests/test_vector_tool.py -v | tee logs/vector_tool_tests.log).
"""
import json
import re
from pathlib import Path

import chromadb
import pytest

from src import config
from src.tools import vector_tool
from src.tools.vector_tool import EMBEDDING_MODEL_NAME, MAX_K, search_documents

BUILD_INDEX_FILE = config.PROJECT_ROOT / "src" / "vectordb" / "build_index.py"

# ---------------------------------------------------------------------------
# Fake index: 3-dimensional unit vectors, so cosine similarity is easy to predict.
# ---------------------------------------------------------------------------
FAKE_DOCS = [
    # (id, text, source, page, embedding)
    ("spaced_p2_c0", "Spaced repetition reviews material at increasing intervals.",
     "06_Active_Recall_Spaced_Repetition_Guide.pdf", 2, [1.0, 0.0, 0.0]),
    ("exam_p1_c0", "Practise with past exam papers under timed conditions.",
     "05_Exam_Preparation_Guide.pdf", 1, [0.0, 1.0, 0.0]),
    ("notes_p1_c0", "Use the Cornell method to structure lecture notes.",
     "07_Note_Taking_Lecture_Learning_Guide.pdf", 1, [0.0, 0.0, 1.0]),
    ("notes_p3_c1", "Review your notes within 24 hours of the lecture.",
     "07_Note_Taking_Lecture_Learning_Guide.pdf", 3, [0.6, 0.0, 0.8]),
]

FAKE_QUERY_VECTORS = {
    "spaced repetition": [1.0, 0.0, 0.0],
    "exam prep": [0.0, 1.0, 0.0],
    "note taking": [0.0, 0.0, 1.0],
}


def fake_embed(query: str) -> list[float]:
    return FAKE_QUERY_VECTORS.get(query, [0.5773502691896258] * 3)


def build_collection(db_path: Path, docs) -> None:
    """Create the study_guides collection exactly like build_index.py does (cosine space)."""
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    if docs:
        collection.upsert(
            ids=[d[0] for d in docs],
            documents=[d[1] for d in docs],
            metadatas=[d[5] if len(d) > 5 else {
                "source": d[2], "title": d[2][3:-4].replace("_", " "),
                "category": "test", "page": d[3], "chunk_id": d[0],
            } for d in docs],
            embeddings=[d[4] for d in docs],
        )


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    """A small index with known contents, and a fake query embedder."""
    db = tmp_path / "chroma"
    build_collection(db, FAKE_DOCS)
    monkeypatch.setattr(config, "CHROMA_PATH", db)
    monkeypatch.setattr(vector_tool, "_embed_query", fake_embed)
    return db


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_valid_query_returns_results(fake_index):
    result = search_documents("spaced repetition", k=3)
    assert result["status"] == "ok"
    results = result["data"]["results"]
    assert isinstance(results, list)
    assert 0 < len(results) <= 3
    for item in results:
        assert set(["text", "source", "score"]) <= set(item)
        assert isinstance(item["text"], str) and item["text"]
        assert isinstance(item["score"], float)


def test_results_have_source_metadata(fake_index):
    results = search_documents("note taking", k=4)["data"]["results"]
    known_sources = {d[2] for d in FAKE_DOCS}
    for item in results:
        assert item["source"], "every result needs a source for citations"
        assert item["source"] in known_sources  # taken from metadata, never invented
        assert item["title"] and item["page"] is not None and item["chunk_id"]


def test_best_match_first_and_score_is_similarity(fake_index):
    results = search_documents("note taking", k=4)["data"]["results"]
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)      # best first
    assert results[0]["chunk_id"] == "notes_p1_c0"
    assert results[0]["score"] == pytest.approx(1.0)   # same direction as the query
    assert results[1]["chunk_id"] == "notes_p3_c1"
    assert results[1]["score"] == pytest.approx(0.8)   # cosine of [0.6,0,0.8] with [0,0,1]
    assert results[-1]["score"] == pytest.approx(0.0)  # orthogonal chunks


def test_query_is_stripped_before_embedding(fake_index, monkeypatch):
    seen = []
    monkeypatch.setattr(vector_tool, "_embed_query", lambda q: seen.append(q) or fake_embed(q))
    search_documents("   note taking  ", k=1)
    assert seen == ["note taking"]


def test_k_limits_number_of_results(fake_index):
    assert len(search_documents("note taking", k=1)["data"]["results"]) == 1
    assert len(search_documents("note taking", k=2)["data"]["results"]) == 2


def test_k_larger_than_index_returns_everything_without_error(fake_index):
    result = search_documents("note taking", k=MAX_K)
    assert result["status"] == "ok"
    assert len(result["data"]["results"]) == len(FAKE_DOCS)


def test_default_k_is_three(fake_index):
    assert len(search_documents("note taking")["data"]["results"]) == 3


def test_result_is_json_serializable(fake_index):
    json.dumps(search_documents("exam prep", k=3))  # must not raise


@pytest.mark.parametrize("bad_query", ["", "   ", None, 123, ["note taking"]])
def test_empty_or_non_string_query_returns_error(fake_index, bad_query):
    result = search_documents(bad_query, k=3)
    assert result["status"] == "error"
    assert result["error"]


def test_too_long_query_returns_error(fake_index):
    result = search_documents("x" * (vector_tool.MAX_QUERY_CHARS + 1), k=3)
    assert result["status"] == "error"
    assert "too long" in result["error"]


@pytest.mark.parametrize("bad_k", [0, -1, MAX_K + 1, 2.5, "3", True, None])
def test_invalid_k_returns_error(fake_index, bad_k):
    result = search_documents("note taking", k=bad_k)
    assert result["status"] == "error"
    assert result["error"].startswith("k must")


def test_k_at_upper_bound_is_allowed(fake_index):
    assert search_documents("note taking", k=MAX_K)["status"] == "ok"


def test_missing_index_folder_returns_error(tmp_path, monkeypatch):
    missing = tmp_path / "no_such_folder"
    monkeypatch.setattr(config, "CHROMA_PATH", missing)
    result = search_documents("note taking", k=3)  # must NOT raise
    assert result["status"] == "error"
    assert "build_index" in result["error"]
    assert not missing.exists()  # the tool must not create an empty index as a side effect


def test_empty_folder_returns_error_and_creates_nothing(tmp_path, monkeypatch):
    empty = tmp_path / "empty_chroma"
    empty.mkdir()
    monkeypatch.setattr(config, "CHROMA_PATH", empty)
    result = search_documents("note taking", k=3)
    assert result["status"] == "error"
    assert "build_index" in result["error"]
    assert not (empty / "chroma.sqlite3").exists()


def test_empty_collection_returns_error(tmp_path, monkeypatch):
    db = tmp_path / "chroma"
    build_collection(db, docs=[])
    monkeypatch.setattr(config, "CHROMA_PATH", db)
    monkeypatch.setattr(vector_tool, "_embed_query", fake_embed)
    result = search_documents("note taking", k=3)
    assert result["status"] == "error"
    assert "empty" in result["error"]


def test_embedding_failure_returns_error(fake_index, monkeypatch):
    def broken(query):
        raise RuntimeError("download of the model was refused")

    monkeypatch.setattr(vector_tool, "_embed_query", broken)
    result = search_documents("note taking", k=3)  # must NOT raise
    assert result["status"] == "error"
    assert EMBEDDING_MODEL_NAME in result["error"]


def test_unexpected_error_never_raises(fake_index, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("something unexpected")

    monkeypatch.setattr(vector_tool, "_open_collection", boom)
    result = search_documents("note taking", k=3)
    assert result == {"status": "error", "error": result["error"]}
    assert "ValueError" in result["error"]


def test_chunk_without_source_metadata_is_skipped(tmp_path, monkeypatch):
    docs = FAKE_DOCS[:2] + [("orphan", "Chunk with no source.", "x", 1, [0.0, 0.0, 1.0],
                             {"title": "unknown", "page": 1, "chunk_id": "orphan"})]
    db = tmp_path / "chroma"
    build_collection(db, docs)
    monkeypatch.setattr(config, "CHROMA_PATH", db)
    monkeypatch.setattr(vector_tool, "_embed_query", fake_embed)
    results = search_documents("note taking", k=3)["data"]["results"]
    assert all(r["source"] for r in results)
    assert "orphan" not in [r["chunk_id"] for r in results]


def test_embedding_model_matches_ingestion():
    """The query must be embedded with the same model as the chunks (build_index.py)."""
    match = re.search(r'^EMBEDDING_MODEL_NAME\s*=\s*"([^"]+)"',
                      BUILD_INDEX_FILE.read_text(encoding="utf-8"), re.M)
    assert match, "EMBEDDING_MODEL_NAME not found in build_index.py"
    assert match.group(1) == EMBEDDING_MODEL_NAME


def test_ingestion_uses_cosine_space():
    """Scores are only 'similarity' if the index uses cosine distance."""
    assert '"hnsw:space": "cosine"' in BUILD_INDEX_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Retrieval quality tests (real index + real embedding model)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_index():
    """Skip with a clear reason unless the real index and the embedding model are present."""
    db = Path(config.CHROMA_PATH)
    if not (db / "chroma.sqlite3").exists():
        pytest.skip(f"No real index at {db}. Run: python -m src.vectordb.build_index")
    try:
        vector_tool._get_embedding_model()
    except Exception as exc:
        pytest.skip(f"Embedding model not available ({type(exc).__name__}). "
                    "Run once with internet access so it can be downloaded.")
    return db


# Each query is paired with the study guide that should answer it.
# English queries.
QUALITY_EN = [
    ("What is spaced repetition?", "06_Active_Recall_Spaced_Repetition_Guide.pdf"),
    ("How should I prepare for exams?", "05_Exam_Preparation_Guide.pdf"),
    ("How can I manage my time better as a student?", "04_Time_Management_Strategies_Guide.pdf"),
    ("How do I take effective notes in lectures?", "07_Note_Taking_Lecture_Learning_Guide.pdf"),
    ("I keep procrastinating and lack motivation to study", "08_Study_Motivation_Procrastination_Guide.pdf"),
    ("How do I set academic goals and learn from feedback?", "09_Feedback_Academic_Goal_Setting_Guide.pdf"),
    ("How should I learn programming and debug my code?", "03_Programming_Learning_Guide.pdf"),
    ("How do I solve differentiation and integration problems?", "02_Mathematics_Learning_Guide.pdf"),
]

# Arabic queries against the English guides (the model is multilingual, but this is
# cross-language retrieval, which is the weaker case; see README, Member 2 limitation).
QUALITY_AR = [
    ("ما هو التكرار المتباعد؟", "06_Active_Recall_Spaced_Repetition_Guide.pdf"),
    ("كيف أستعد للامتحانات؟", "05_Exam_Preparation_Guide.pdf"),
    ("كيف أدير وقتي بشكل أفضل؟", "04_Time_Management_Strategies_Guide.pdf"),
    ("كيف أكتب ملاحظات المحاضرة بشكل فعال؟", "07_Note_Taking_Lecture_Learning_Guide.pdf"),
    ("كيف أتغلب على التسويف وقلة الحافز للدراسة؟", "08_Study_Motivation_Procrastination_Guide.pdf"),
    ("كيف أضع أهدافا أكاديمية وأستفيد من التغذية الراجعة؟", "09_Feedback_Academic_Goal_Setting_Guide.pdf"),
    ("كيف أتعلم البرمجة وأصحح الأخطاء في الكود؟", "03_Programming_Learning_Guide.pdf"),
    ("كيف أحل مسائل التفاضل والتكامل؟", "02_Mathematics_Learning_Guide.pdf"),
]


@pytest.mark.parametrize("query, expected", QUALITY_EN, ids=[f"en-{exp[:2]}" for _, exp in QUALITY_EN])
def test_quality_english_expected_guide_in_top3(real_index, query, expected):
    result = search_documents(query, k=3)
    assert result["status"] == "ok", result
    sources = [r["source"] for r in result["data"]["results"]]
    assert expected in sources, f"top-3 sources were {sources}"


@pytest.mark.parametrize("query, expected", QUALITY_AR, ids=[f"ar-{exp[:2]}" for _, exp in QUALITY_AR])
def test_quality_arabic_expected_guide_in_top3(real_index, query, expected):
    result = search_documents(query, k=3)
    assert result["status"] == "ok", result
    sources = [r["source"] for r in result["data"]["results"]]
    assert expected in sources, f"top-3 sources were {sources}"