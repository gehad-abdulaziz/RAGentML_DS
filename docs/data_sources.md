# Data Sources

Shared document. Each member edits only the section for their own part.

## 1. ML dataset (Member 1)

- Candidates: Kaggle "Student Performance Factors" (target `Exam_Score`; features such as `Hours_Studied`, `Attendance`, `Sleep_Hours`), or UCI "Student Performance" (student-mat).
- **Chosen dataset:** TODO
- **URL:** TODO
- **License:** TODO
- **Target:** TODO
- **Features used:** TODO
- **Preprocessing notes:** TODO
- Raw file location: `data/raw/` (large raw files are git-ignored; document how to download them here).

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
| TODO | TODO | TODO |

## Shared topic list (Members 2 & 4)

TODO: the agreed topic names used both in `data/documents/` metadata and in graph `Topic` nodes.