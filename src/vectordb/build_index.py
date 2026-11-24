"""Build the ChromaDB vector index from study-guide documents.

Owner: Member 2 (feature/rag-ingestion).

Run:  python -m src.vectordb.build_index

Pipeline: data/documents/*.md|*.txt -> chunk -> embed -> store in Chroma at CHROMA_PATH.
Every chunk must carry metadata: source (file name/URL) and title.
"""
from __future__ import annotations

# TODO (Member 2): agree on topic names with Member 4 (graph) so documents and graph match.
# TODO (Members 2 & 3): agree on the collection name (see src/config.py CHROMA_COLLECTION)
#                       and on the embedding model; Member 3 must use the SAME embedding model.


def load_documents(documents_dir: str) -> list[dict]:
    """Read all .md/.txt files.

    TODO: return [{"text": str, "source": str, "title": str}, ...]
    Parse source/title from each file's metadata header (front matter).
    """
    raise NotImplementedError


def chunk_document(doc: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split one document into overlapping chunks.

    TODO: return chunks that keep the parent's source/title metadata.
    """
    raise NotImplementedError


def build_index() -> None:
    """Embed chunks (sentence-transformers) and persist them in Chroma.

    TODO: create/reset the collection, add chunks with ids, documents and metadatas.
    """
    raise NotImplementedError


def main() -> None:
    """TODO: call build_index() and print how many documents/chunks were indexed."""
    raise NotImplementedError


if __name__ == "__main__":
    main()