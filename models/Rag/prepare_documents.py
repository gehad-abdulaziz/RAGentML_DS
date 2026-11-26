
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_FOLDER = Path("documents")

chunk_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""]
)

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

all_chunks = []

pdf_files = list(DOCUMENTS_FOLDER.glob("*.pdf"))

print("PDF files found:", len(pdf_files))

for file_path in pdf_files:

    print(f"\nProcessing: {file_path.name}")

    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    print("Pages loaded:", len(pages))

    prefix = file_path.name[:2]
    category = categories.get(prefix, "general")

    for page in pages:
        page.metadata.update({
            "source": file_path.name,
            "category": category
        })

    chunks = chunk_splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):

        chunk.metadata.update({
            "chunk_id": f"{file_path.stem}_{index}",
            "category": category,
            "page": chunk.metadata.get("page", 0) + 1
        })

        all_chunks.append(chunk)

print("\nTotal PDF documents:", len(pdf_files))
print("Total chunks:", len(all_chunks))

for chunk in all_chunks[:5]:
    print("\n-------------------------")
    print("Metadata:", chunk.metadata)
    print("Content:", chunk.page_content[:500])


import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="student_advisor_knowledge",
    metadata={"hnsw:space": "cosine"}
)

if all_chunks:

    texts = [chunk.page_content for chunk in all_chunks]
    metadatas = [chunk.metadata for chunk in all_chunks]
    ids = [chunk.metadata["chunk_id"] for chunk in all_chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("\nChunks stored successfully!")
    print("Total chunks in ChromaDB:", collection.count())

else:
    print("\nNo chunks found. Check your PDF files.")

