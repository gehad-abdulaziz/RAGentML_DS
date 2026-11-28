from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DOCUMENTS_DIR, CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL_NAME


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

categories = {
    "01": "academic_performance",
    "02": "mathematics",
    "03": "programming",
    "04": "time_management",
    "05": "exam_preparation",
    "06": "memory_learning",
    "07": "note_taking",
    "08": "motivation",
    "09": "feedback_goal_setting"
}

document_topics = {
    "01": "Academic Performance",
    "02": "Mathematics",
    "03": "Programming",
    "04": "Time Management",
    "05": "Exam Preparation",
    "06": "Active Recall",
    "07": "Note Taking",
    "08": "Study Motivation",
    "09": "Goal Setting",
}

chunk_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# --------------------------------------------------
# 1. Load PDF Documents
# --------------------------------------------------

def load_documents(documents_dir: str = None) -> list[dict]:

    folder = Path(documents_dir) if documents_dir else DOCUMENTS_DIR

    if not folder.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {folder}"
        )

    pdf_files = sorted(folder.glob("*.pdf"))

    print("PDF files found:", len(pdf_files))

    documents = []

    for file_path in pdf_files:

        print(f"Processing: {file_path.name}")

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        prefix = file_path.name[:2]
        category = categories.get(prefix, "general")
        topic = document_topics.get(prefix, "General")

        title = file_path.stem[3:].replace("_", " ")

        for page in pages:

            documents.append({
                "text": page.page_content,
                "source": file_path.name,
                "title": title,
                "category": category,
                "topic": topic,
                "page": page.metadata.get("page", 0) + 1
            })

    print("Total pages loaded:", len(documents))

    return documents


# --------------------------------------------------
# 2. Split Documents into Chunks
# --------------------------------------------------

def chunk_document(
    doc: dict,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:

    splitter = chunk_splitter

    if chunk_size != CHUNK_SIZE or overlap != CHUNK_OVERLAP:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    text_chunks = splitter.split_text(doc["text"])

    chunks = []

    for index, text in enumerate(text_chunks):

        chunks.append({
            "text": text,
            "source": doc["source"],
            "title": doc["title"],
            "category": doc["category"],
            "topic": doc.get("topic", "General"),
            "page": doc["page"],
            "chunk_id": (
                f"{Path(doc['source']).stem}_"
                f"page_{doc['page']}_chunk_{index}"
            )
        })

    return chunks


# --------------------------------------------------
# 3. Build ChromaDB Index
# --------------------------------------------------

def build_index() -> None:

    documents = load_documents()

    if not documents:
        print("No PDF documents found.")
        return

    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    print("Total chunks:", len(all_chunks))

    if not all_chunks:
        print("No chunks generated.")
        return

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    texts = [chunk["text"] for chunk in all_chunks]

    metadatas = [
        {
            "source": chunk["source"],
            "title": chunk["title"],
            "category": chunk["category"],
            "topic": chunk.get("topic", "General"),
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in all_chunks
    ]

    ids = [chunk["chunk_id"] for chunk in all_chunks]

    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    try:
        client.delete_collection(name=CHROMA_COLLECTION)
        print(f"Deleted existing collection '{CHROMA_COLLECTION}' for clean rebuild.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("\nIndex built successfully!")
    print("Pages loaded:", len(documents))
    print("Chunks created:", len(all_chunks))
    print("Total chunks in ChromaDB:", collection.count())
    print("Collection:", CHROMA_COLLECTION)
    print("Database path:", CHROMA_PATH)


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
