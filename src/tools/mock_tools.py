"""Mock tools with the SAME signatures and return shapes as the real tools.

Owner: Member 6 (feature/agent-interface). Used by Members 5 and 6 until the real
tools are merged into develop, and for deterministic testing.

Follow docs/contracts.md exactly. Mocks return deterministic, realistic data
and provide a mechanism to simulate errors for testing.
"""
from __future__ import annotations

import re
from typing import Any

# Global switch for error-injection tests
MOCK_FORCE_ERROR = False
MOCK_ERROR_MESSAGE = "Forced mock error for testing error handling."


def set_mock_force_error(force: bool, message: str | None = None) -> None:
    """Enable or disable forced error responses across mock tools."""
    global MOCK_FORCE_ERROR, MOCK_ERROR_MESSAGE
    MOCK_FORCE_ERROR = force
    if message:
        MOCK_ERROR_MESSAGE = message


def mock_predict_performance(features: dict) -> dict:
    """Fake version of ml_tool.predict_performance.

    Validates features and computes deterministic score based on study hours,
    attendance, sleep hours, etc.
    """
    if MOCK_FORCE_ERROR:
        return {"status": "error", "error": MOCK_ERROR_MESSAGE}

    if not isinstance(features, dict):
        return {"status": "error", "error": "features must be a dictionary."}

    # Range and type checks
    num_ranges = {
        "Hours_Studied": (0, 100),
        "Attendance": (0, 100),
        "Sleep_Hours": (0, 24),
        "Previous_Scores": (0, 100),
        "Tutoring_Sessions": (0, 20),
        "Physical_Activity": (0, 30),
    }

    for key, (min_val, max_val) in num_ranges.items():
        if key in features:
            val = features[key]
            if not isinstance(val, (int, float)):
                return {"status": "error", "error": f"{key} must be a number."}
            if val < min_val or val > max_val:
                return {
                    "status": "error",
                    "error": f"{key} value {val} is out of allowed range [{min_val}, {max_val}].",
                }

    # Required key features for prediction
    # If the student didn't provide any key metrics, return error specifying missing features
    key_features = ["Hours_Studied", "Attendance"]
    missing_key = [k for k in key_features if k not in features]
    if len(missing_key) == len(key_features):
        return {
            "status": "error",
            "error": "Missing required features: Hours_Studied and Attendance are required for score prediction.",
        }

    # Deterministic prediction model
    hours = float(features.get("Hours_Studied", 15))
    attendance = float(features.get("Attendance", 80))
    sleep = float(features.get("Sleep_Hours", 7))
    prev_scores = float(features.get("Previous_Scores", 75))
    tutoring = float(features.get("Tutoring_Sessions", 1))

    # Formula providing intuitive responses to changes in Hours_Studied
    score = 25.0 + (hours * 1.1) + (attendance * 0.3) + (sleep * 0.8) + (prev_scores * 0.2) + (tutoring * 1.5)
    score = max(0.0, min(100.0, round(score, 1)))

    return {
        "status": "ok",
        "data": {
            "prediction": score,
            "unit": "exam score (0-100)",
            "inputs_used": features,
        },
    }


def mock_search_documents(query: str, k: int = 3) -> dict:
    """Fake version of vector_tool.search_documents."""
    if MOCK_FORCE_ERROR:
        return {"status": "error", "error": MOCK_ERROR_MESSAGE}

    if not isinstance(query, str) or not query.strip():
        return {"status": "error", "error": "Query cannot be empty."}

    if len(query) > 1000:
        return {"status": "error", "error": "Query exceeds maximum limit of 1000 characters."}

    q_lower = query.lower()

    # Canned documents matching repository PDFs
    canned_docs = [
        {
            "keywords": ["spaced repetition", "active recall", "flashcard", "memory", "retention"],
            "source": "06_Active_Recall_Spaced_Repetition_Guide.pdf",
            "title": "Active Recall Spaced Repetition Guide",
            "page": 2,
            "chunk_id": "06_Active_Recall_Spaced_Repetition_Guide_page_2_chunk_0",
            "text": "Active recall involves actively stimulating memory during the learning process by testing yourself. Spaced repetition spaces review sessions over increasing intervals to optimize long-term retention.",
            "score": 0.88,
        },
        {
            "keywords": ["time management", "pomodoro", "schedule", "procrastination"],
            "source": "04_Time_Management_Strategies_Guide.pdf",
            "title": "Time Management Strategies Guide",
            "page": 1,
            "chunk_id": "04_Time_Management_Strategies_Guide_page_1_chunk_0",
            "text": "Effective time management uses techniques like the Pomodoro technique (25 minutes study, 5 minutes break) and structured weekly calendars to prevent procrastination.",
            "score": 0.84,
        },
        {
            "keywords": ["calculus", "math", "derivatives", "limits", "integrals", "mathematics"],
            "source": "02_Mathematics_Learning_Guide.pdf",
            "title": "Mathematics Learning Guide",
            "page": 1,
            "chunk_id": "02_Mathematics_Learning_Guide_page_1_chunk_0",
            "text": "Mastering calculus requires solid foundational knowledge of algebra and functions. Solve step-by-step practice problems and understand intuitive geometric interpretations.",
            "score": 0.82,
        },
        {
            "keywords": ["programming", "code", "debugging", "python", "data structures", "algorithm"],
            "source": "03_Programming_Learning_Guide.pdf",
            "title": "Programming Learning Guide",
            "page": 1,
            "chunk_id": "03_Programming_Learning_Guide_page_1_chunk_0",
            "text": "Learn programming through deliberate practice, tracing code execution on paper, and working on small, incremental projects to solidify core syntax and data structures.",
            "score": 0.81,
        },
        {
            "keywords": ["exam", "preparation", "test", "anxiety", "revision"],
            "source": "05_Exam_Preparation_Guide.pdf",
            "title": "Exam Preparation Guide",
            "page": 1,
            "chunk_id": "05_Exam_Preparation_Guide_page_1_chunk_0",
            "text": "Prepare for exams by simulating timed test conditions, reviewing past papers, and targeting weak topics identified in early self-assessments.",
            "score": 0.79,
        },
    ]

    matched: list[dict[str, Any]] = []
    for doc in canned_docs:
        if any(kw in q_lower for kw in doc["keywords"]):
            matched.append({
                "text": doc["text"],
                "source": doc["source"],
                "title": doc["title"],
                "page": doc["page"],
                "chunk_id": doc["chunk_id"],
                "score": doc["score"],
            })

    # Default fallback guide if no specific keyword matched
    if not matched:
        matched.append({
            "text": "Effective academic performance relies on consistent study habits, active engagement in lectures, and regular self-assessment.",
            "source": "01_Academic_Performance_Improvement_Guide.pdf",
            "title": "Academic Performance Improvement Guide",
            "page": 1,
            "chunk_id": "01_Academic_Performance_Improvement_Guide_page_1_chunk_0",
            "score": 0.75,
        })

    return {"status": "ok", "data": {"results": matched[: max(1, min(k, 10))]}}


def mock_query_graph(query_type: str, params: dict) -> dict:
    """Fake version of graph_tool.query_graph."""
    if MOCK_FORCE_ERROR:
        return {"status": "error", "error": MOCK_ERROR_MESSAGE}

    valid_types = {
        "prerequisites": ["course"],
        "topics_in_course": ["course"],
        "resources_for_topic": ["topic"],
        "courses_in_subject": ["subject"],
        "strategies_for_topic": ["topic"],
        "topic_prerequisites": ["topic"],
    }

    if query_type not in valid_types:
        return {
            "status": "error",
            "error": f"Unknown query_type '{query_type}'. Valid types: {list(valid_types.keys())}",
        }

    if not isinstance(params, dict):
        return {"status": "error", "error": "params must be a dictionary."}

    req_param = valid_types[query_type][0]
    if req_param not in params or not str(params[req_param]).strip():
        return {
            "status": "error",
            "error": f"Missing required parameter '{req_param}' for query_type '{query_type}'.",
        }

    val = str(params[req_param]).strip().lower()

    if query_type == "prerequisites":
        if "calculus ii" in val:
            return {
                "status": "ok",
                "data": {"rows": [{"course": "Calculus II", "prerequisite": "Calculus I", "distance": 1}]},
            }
        elif "calculus i" in val:
            return {"status": "ok", "data": {"rows": []}}
        elif "data structures" in val:
            return {
                "status": "ok",
                "data": {"rows": [{"course": "Data Structures and Algorithms", "prerequisite": "Programming I", "distance": 1}]},
            }
        return {"status": "ok", "data": {"rows": []}}

    elif query_type == "topics_in_course":
        if "calculus i" in val:
            return {
                "status": "ok",
                "data": {
                    "rows": [
                        {"course": "Calculus I", "topic": "Limits", "description": "The idea of a value a function approaches"},
                        {"course": "Calculus I", "topic": "Differentiation", "description": "Rates of change and derivative rules"},
                        {"course": "Calculus I", "topic": "Integration", "description": "Accumulation and integration techniques"},
                    ]
                },
            }
        elif "calculus ii" in val:
            return {
                "status": "ok",
                "data": {
                    "rows": [
                        {"course": "Calculus II", "topic": "Integration", "description": "Accumulation and integration techniques"},
                        {"course": "Calculus II", "topic": "Differential Equations", "description": "Equations involving derivatives"},
                    ]
                },
            }
        return {"status": "ok", "data": {"rows": []}}

    elif query_type == "resources_for_topic":
        if "limit" in val:
            return {
                "status": "ok",
                "data": {
                    "rows": [
                        {
                            "topic": "Limits",
                            "title": "Khan Academy - Limits and Continuity",
                            "url": "https://www.khanacademy.org/math/calculus-1/cs1-limits-and-continuity",
                            "type": "Video Series",
                        }
                    ]
                },
            }
        return {"status": "ok", "data": {"rows": []}}

    elif query_type == "courses_in_subject":
        if "math" in val:
            return {
                "status": "ok",
                "data": {
                    "rows": [
                        {"subject": "Mathematics", "course": "Calculus I", "code": "MATH101"},
                        {"subject": "Mathematics", "course": "Calculus II", "code": "MATH102"},
                        {"subject": "Mathematics", "course": "Linear Algebra", "code": "MATH201"},
                    ]
                },
            }
        elif "program" in val:
            return {
                "status": "ok",
                "data": {
                    "rows": [
                        {"subject": "Programming", "course": "Programming I", "code": "CS101"},
                        {"subject": "Programming", "course": "Data Structures and Algorithms", "code": "CS201"},
                    ]
                },
            }
        return {"status": "ok", "data": {"rows": []}}

    elif query_type == "strategies_for_topic":
        return {
            "status": "ok",
            "data": {
                "rows": [
                    {
                        "topic": params[req_param],
                        "strategy": "Active Recall",
                        "description": "Test yourself on formulas and key definitions without looking at notes.",
                    },
                    {
                        "topic": params[req_param],
                        "strategy": "Practice-Based Learning",
                        "description": "Solve diverse end-of-chapter problems under timed conditions.",
                    },
                ]
            },
        }

    elif query_type == "topic_prerequisites":
        if "differentiation" in val or "integration" in val:
            return {
                "status": "ok",
                "data": {"rows": [{"topic": params[req_param], "requires": "Limits", "distance": 1}]},
            }
        return {"status": "ok", "data": {"rows": []}}

    return {"status": "ok", "data": {"rows": []}}
