import os
import sys
import argparse
from dotenv import load_dotenv

# Silence warnings
import warnings
warnings.filterwarnings("ignore")

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    import torch
except ImportError as e:
    print(f"Error: Missing required packages for RAG. {e}")
    sys.exit(1)

def build_rag_index(rag_dir):
    print(f"Starting isolated RAG initialization for directory: {rag_dir}")
    if not os.path.exists(rag_dir):
        print(f"Error: RAG directory {rag_dir} not found.")
        sys.exit(1)

    try:
        import glob, hashlib
        pdf_files = sorted(glob.glob(os.path.join(rag_dir, "*.pdf")))
        dir_hasher = hashlib.md5()
        for f in pdf_files:
            dir_hasher.update(f.encode('utf-8'))
            dir_hasher.update(str(os.path.getmtime(f)).encode('utf-8'))
        current_hash = dir_hasher.hexdigest()

        index_path = os.path.join(rag_dir, "faiss_index")
        hash_path = os.path.join(index_path, "dir_hash.txt")

        # Check if up to date before loading the heavy embedding model
        if os.path.exists(index_path) and os.path.exists(hash_path):
            with open(hash_path, 'r') as hf:
                saved_hash = hf.read().strip()
            if saved_hash == current_hash:
                print(f"Index is up to date. Skipping re-indexing.")
                sys.exit(0)
            else:
                print("Documents changed. Re-indexing...")

        print("Loading BAAI/bge-m3 embedding model onto GPU...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3", 
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("Loading PDFs...")
        loader = PyPDFDirectoryLoader(rag_dir)
        raw_docs = loader.load()
        print(f"Loaded {len(raw_docs)} raw documents.")

        print("Chunking text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(raw_docs)
        print(f"Split documents into {len(docs)} semantic chunks.")

        print("Building FAISS vector index...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Save index to disk
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        with open(hash_path, 'w') as hf:
            hf.write(current_hash)
        
        print("RAG initialization complete and saved to disk.")
        sys.exit(0)
        
    except Exception as e:
        print(f"Fatal error initializing RAG: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate RAG Index Building")
    parser.add_argument("--dir", type=str, required=True, help="Path to the RAG documents directory")
    args = parser.parse_args()
    build_rag_index(args.dir)
