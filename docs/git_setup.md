# Git Setup (run once, by one team member)

## 1. Create `main` and `develop`, push them

```bash
cd RAGentML_DS
git init -b main
git add .
git commit -m "chore: initial project scaffold"
git remote add origin https://github.com/gehad-abdulaziz/RAGentML_DS.git
git push -u origin main

git checkout -b develop
git push -u origin develop
```

## 2. Create the six feature branches from `develop` and push them

```bash
git checkout develop
for b in feature/ml-tool feature/rag-ingestion feature/rag-retrieval \
         feature/graph-tool feature/agent-core feature/agent-interface; do
  git checkout -b "$b" develop
  git push -u origin "$b"
  git checkout develop
done
```

Explicit version (same result):

```bash
git checkout develop
git checkout -b feature/ml-tool develop            && git push -u origin feature/ml-tool            && git checkout develop
git checkout -b feature/rag-ingestion develop      && git push -u origin feature/rag-ingestion      && git checkout develop
git checkout -b feature/rag-retrieval develop      && git push -u origin feature/rag-retrieval      && git checkout develop
git checkout -b feature/graph-tool develop         && git push -u origin feature/graph-tool         && git checkout develop
git checkout -b feature/agent-core develop         && git push -u origin feature/agent-core         && git checkout develop
git checkout -b feature/agent-interface develop    && git push -u origin feature/agent-interface    && git checkout develop
```

## 3. Each member starts working

```bash
git clone https://github.com/gehad-abdulaziz/RAGentML_DS.git
cd RAGentML_DS
git checkout feature/<your-branch>
```

## 4. Recommended branch protection for `main`

GitHub -> Settings -> Branches -> Add branch protection rule (or Rulesets), branch name pattern `main`:

- [x] Require a pull request before merging
  - [x] Require approvals: **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
- [x] Require conversation resolution before merging
- [x] Do not allow bypassing the above settings (include administrators)
- [x] Block force pushes
- [x] Restrict deletions

Only PRs from `develop` into `main`, at milestones.

Recommended (lighter) protection for `develop`:

- [x] Require a pull request before merging, **1** approval
- [x] Block force pushes

Note: on a private repo with a free GitHub plan, branch protection may be unavailable; in that case follow the rules by convention.

## 5. Safety check before the first push

```bash
git status
git ls-files | grep -E "(^|/)\.env$" && echo "STOP: .env is tracked" || echo "OK: no .env tracked"
```