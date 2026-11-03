"""Build and load a content-addressed scientific PDF index without pickle files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_VERSION = 2
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250


class RAGIndexError(RuntimeError):
    pass


def resolve_rag_directory(directory):
    if not str(directory).strip():
        raise RAGIndexError("Choose a nonempty scientific PDF directory.")
    path = Path(directory).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_snapshot(directory):
    """The loader and cache fingerprint use exactly the same recursive PDF set."""
    root = resolve_rag_directory(directory)
    if not root.is_dir():
        raise RAGIndexError(f"Scientific PDF directory does not exist: {root}")
    paths = sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf"),
                   key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise RAGIndexError(f"No PDF files found in {root} (including subdirectories).")
    records = [{"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)} for path in paths]
    settings = {"schema_version": INDEX_VERSION, "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "pdfs": records}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return paths, fingerprint


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def index_generation(directory, fingerprint=None):
    """Validate the current generation and checksums before loading any index."""
    root = resolve_rag_directory(directory) / "faiss_index"
    try:
        manifest = json.loads((root / "current.json").read_text(encoding="utf-8"))
        if (manifest["schema_version"] != INDEX_VERSION or manifest["embedding_model"] != EMBEDDING_MODEL
                or (fingerprint is not None and manifest["corpus_fingerprint"] != fingerprint)):
            raise ValueError("PDF corpus or index format changed")
        generation_name = manifest["generation"]
        if (not isinstance(generation_name, str) or not generation_name.startswith("generation-")
                or Path(generation_name).name != generation_name or "/" in generation_name or "\\" in generation_name):
            raise ValueError("invalid index generation path")
        generation = root / generation_name
        if generation.is_symlink() or generation.resolve().parent != root.resolve():
            raise ValueError("index generation points outside its directory")
        for name in ("index.faiss", "documents.json"):
            artifact = generation / name
            if artifact.is_symlink() or not artifact.is_file() or file_sha256(artifact) != manifest["artifacts"][name]:
                raise ValueError(f"missing or changed {name}")
        return generation
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RAGIndexError(f"No valid current scientific index: {exc}") from exc


def create_embeddings():
    # Querying and indexing must not reserve GPU memory needed by SAM3 and Ollama.
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"},
                                encode_kwargs={"normalize_embeddings": True})


def _save_safe_store(store, directory):
    import faiss
    mapping = store.index_to_docstore_id
    if set(mapping) != set(range(store.index.ntotal)) or len(set(mapping.values())) != len(mapping):
        raise RAGIndexError("FAISS returned an invalid document mapping.")
    records = []
    for index in range(store.index.ntotal):
        doc_id = mapping[index]
        document = store.docstore.search(doc_id)
        if not isinstance(doc_id, str) or not isinstance(document.page_content, str) or not isinstance(document.metadata, dict):
            raise RAGIndexError("FAISS returned invalid document data.")
        records.append({"id": doc_id, "page_content": document.page_content, "metadata": document.metadata})
    _write_json(Path(directory) / "documents.json", records)
    faiss.write_index(store.index, str(Path(directory) / "index.faiss"))


def load_rag_index(directory, embeddings):
    """Load native FAISS vectors plus JSON documents; never deserialize index.pkl."""
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    _, fingerprint = corpus_snapshot(directory)
    generation = index_generation(directory, fingerprint)
    records = json.loads((generation / "documents.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise RAGIndexError("The scientific index contains no document records.")
    documents, mapping = {}, {}
    for position, record in enumerate(records):
        if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                or record["id"] in documents or not isinstance(record.get("page_content"), str)
                or not isinstance(record.get("metadata"), dict)):
            raise RAGIndexError("The scientific index has invalid document records.")
        documents[record["id"]] = Document(page_content=record["page_content"], metadata=record["metadata"])
        mapping[position] = record["id"]
    index = faiss.read_index(str(generation / "index.faiss"))
    if index.ntotal != len(records):
        raise RAGIndexError("Scientific vector and document counts differ.")
    return FAISS(embeddings, index, InMemoryDocstore(documents), mapping)


def build_rag_index(rag_dir):
    root = resolve_rag_directory(rag_dir)
    paths, fingerprint = corpus_snapshot(root)
    try:
        generation = index_generation(root, fingerprint)
    except RAGIndexError:
        generation = None
    if generation is not None:
        print("Scientific index is up to date; no embedding model was loaded.")
        return generation

    index_root = root / "faiss_index"
    index_root.mkdir(exist_ok=True)
    lock = index_root / ".build.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RAGIndexError(f"Another RAG build is active: {lock}. A stopped build may have left this lock.") from exc
    os.close(descriptor)
    staging = None
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        staging = Path(tempfile.mkdtemp(prefix=".building-", dir=index_root))
        raw_docs = []
        for path in paths:
            raw_docs.extend(PyPDFLoader(str(path)).load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                                  separators=["\n\n", "\n", " ", ""])
        docs = [document for document in splitter.split_documents(raw_docs) if document.page_content.strip()]
        if not docs:
            raise RAGIndexError("PDFs contain no extractable text. OCR is required for image-only documents.")
        print(f"Embedding {len(docs)} scientific text chunks on CPU...")
        store = FAISS.from_documents(docs, create_embeddings())
        _save_safe_store(store, staging)
        if corpus_snapshot(root)[1] != fingerprint:
            raise RAGIndexError("PDF files changed during indexing; retry with a stable document directory.")
        generation = index_root / f"generation-{uuid.uuid4().hex}"
        staging.rename(generation)
        staging = None
        manifest = {"schema_version": INDEX_VERSION, "embedding_model": EMBEDDING_MODEL,
                    "corpus_fingerprint": fingerprint, "generation": generation.name,
                    "artifacts": {name: file_sha256(generation / name) for name in ("index.faiss", "documents.json")}}
        pending = index_root / f".current-{uuid.uuid4().hex}.json"
        try:
            _write_json(pending, manifest)
            os.replace(pending, index_root / "current.json")
        finally:
            pending.unlink(missing_ok=True)
        print(f"Scientific RAG index published: {generation}")
        return generation
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Directory containing scientific PDFs (searched recursively).")
    args = parser.parse_args(argv)
    try:
        build_rag_index(args.dir)
    except Exception as exc:
        print(f"Error initializing scientific RAG: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
