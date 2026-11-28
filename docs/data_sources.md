# Data Sources

Shared document. Each member edits only the section for their own part.

## 1. ML dataset (Member 1)

- **Chosen dataset:** Kaggle "Student Performance Factors" (`StudentPerformanceFactors.csv`)
- **URL:** https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
- **License:** Open Database License (ODbL) / Public Domain
- **Target:** `Exam_Score` (continuous numeric score 0-100)
- **Features used:** 19 student factors (6 numerical, 8 ordinal categorical, 5 nominal categorical).
- **Preprocessing notes:** Missing numericals imputed with median and scaled via `StandardScaler`; ordinal features encoded via `OrdinalEncoder` with domain ordering; nominal features one-hot encoded with `OneHotEncoder(handle_unknown='ignore')`. Pipeline wrapped in scikit-learn `ColumnTransformer`.
- Raw file location: `data/raw/StudentPerformanceFactors.csv` (git-ignored, downloadable via Kaggle).

## 2. Study-guide documents for the vector DB (Member 2)

- Target: 15-30 documents in `data/documents/` (active recall, spaced repetition, sleep, time management, exam anxiety, ...).
- Written by the team or taken from open sources (Wikipedia, OpenStax) with the source recorded in **each file's metadata** (suggested front matter below).

```markdown
---
title: Active Recall
source: Written by the team  # or the URL
license: CC BY-SA 4.0        # or "Team-authored"
topics: [Active Recall]      # must match topic names in the graph
---
Document text...
```

| File | Title | Source / URL | License |
|------|-------|--------------|---------|
| TODO | TODO | TODO | TODO |

## 3. Graph seed data (Member 4)

- Target: hand-designed 30-60 nodes (courses, subjects, topics, resources with real links, strategies) stored as CSV/JSON in `data/graph_seed/`.
- Resource links must be real and checked.

| File | Contents | Source |
|------|----------|--------|
| subjects.csv | 2 subjects (Mathematics, Programming) | Team-designed |
| courses.csv | 8 courses with code and subject | Team-designed (course codes are fictional) |
| topics.csv | 21 topics with short descriptions | Names taken from the headings of the PDFs in `data/documents/` |
| strategies.csv | 9 study strategies | Names taken from the PDFs in `data/documents/` |
| resources.csv | 15 learning resources with real URLs (checked with `python -m src.graphdb.check_links` and manually in a browser) | Khan Academy, MIT OCW, CS50, Python docs, etc. |
| course_topics.csv, course_prerequisites.csv, topic_requires.csv, topic_strategies.csv, resource_topics.csv | Relationship files | Team-designed |

## Shared topic list (Members 2 & 4)
Every name below appears literally in the PDFs in `data/documents/`.

Topics: Algebra, Functions and Graphs, Limits, Differentiation, Integration, Matrix Operations, Systems of Linear Equations, Eigenvalues and Eigenvectors, Probability, Probability Distributions, Statistical Inference, Variables, Data Types, and Operators, Conditional Statements and Loops, Functions, Arrays, Lists, and Strings, Debugging, Classes and Objects, Encapsulation and Inheritance, Data Structures, Searching and Sorting, Complexity Analysis.

Strategies: Active Recall, Spaced Repetition, Practice-Based Learning, Step-by-Step Problem Solving, Deliberate Practice, Code Tracing, Project-Based Learning, Peer Explanation, Charting Method.
