import os
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi


# =========================
# CONFIG
# =========================

DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore/faiss_index"
BM25_PATH = "vectorstore/bm25.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =========================
# LOAD DOCUMENTS
# =========================

def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)

    return documents


# =========================
# SPLIT DOCUMENTS
# =========================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)


# =========================
# CREATE FAISS INDEX (FIXED)
# =========================

def create_vectorstore(chunks):

    print("🧠 Creating embeddings + FAISS index...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    texts = [chunk.page_content for chunk in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings
    )

    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print("✅ FAISS Vector Store Saved")


# =========================
# CREATE BM25 INDEX
# =========================

def create_bm25_index(chunks):

    texts = [chunk.page_content for chunk in chunks]
    tokenized_texts = [text.lower().split() for text in texts]

    bm25 = BM25Okapi(tokenized_texts)

    os.makedirs("vectorstore", exist_ok=True)

    with open(BM25_PATH, "wb") as f:
        pickle.dump((bm25, texts), f)

    print("✅ BM25 Index Saved")


# =========================
# MAIN PIPELINE
# =========================

def main():

    print("📄 Loading documents...")

    documents = load_documents()
    print(f"✅ Loaded {len(documents)} pages")

    print("✂ Splitting documents...")

    chunks = split_documents(documents)

    # metadata fix
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "source": chunk.metadata.get("source", "unknown"),
            "chunk_id": i
        }

    print(f"✅ Created {len(chunks)} chunks")

    print("🧠 Creating FAISS vectorstore...")
    create_vectorstore(chunks)

    print("🔍 Creating BM25 index...")
    create_bm25_index(chunks)

    print("🚀 Ingestion completed successfully!")


if __name__ == "__main__":
    main()