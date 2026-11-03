"""RAG publication, content invalidation and failures with small inference adapters."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import build_rag_index as indexer


def indexing_adapters():
    """Replace PDF/embedding/native vector engines; keep all publication code real."""
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content, self.metadata = page_content, metadata

    class Loader:
        def __init__(self, path):
            self.path = path
        def load(self):
            return [Document(Path(self.path).read_text(encoding="utf-8"), {"source": self.path, "page": 0})]

    class Splitter:
        def __init__(self, **kwargs):
            pass
        def split_documents(self, docs):
            return docs

    class Store:
        def __init__(self, documents):
            self.documents = documents
        def search(self, doc_id):
            return self.documents[doc_id]

    class VectorStore:
        def __init__(self, embeddings, index, docstore, mapping):
            self.index, self.docstore, self.index_to_docstore_id = index, docstore, mapping
        @classmethod
        def from_documents(cls, docs, embeddings):
            mapping = {i: f"document-{i}" for i in range(len(docs))}
            return cls(embeddings, types.SimpleNamespace(ntotal=len(docs)), Store(dict(zip(mapping.values(), docs))), mapping)

    modules = {}
    for name in ("langchain_community", "langchain_community.document_loaders", "langchain_community.vectorstores",
                 "langchain_community.docstore", "langchain_community.docstore.in_memory", "langchain_core",
                 "langchain_core.documents", "langchain_text_splitters", "faiss"):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    modules["langchain_community.document_loaders"].PyPDFLoader = Loader
    modules["langchain_community.vectorstores"].FAISS = VectorStore
    modules["langchain_community.docstore.in_memory"].InMemoryDocstore = Store
    modules["langchain_core.documents"].Document = Document
    modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = Splitter
    modules["faiss"].write_index = lambda index, path: Path(path).write_text(str(index.ntotal), encoding="ascii")
    modules["faiss"].read_index = lambda path: types.SimpleNamespace(ntotal=int(Path(path).read_text(encoding="ascii")))
    return modules


class IndexPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paper = self.root / "paper.pdf"
        self.paper.write_text("Scientific literature about penguins", encoding="utf-8")
        self.adapters = indexing_adapters()
        patch = mock.patch.dict(sys.modules, self.adapters)
        patch.start()
        self.addCleanup(patch.stop)
        embeddings = mock.patch.object(indexer, "create_embeddings", return_value=object())
        self.embeddings = embeddings.start()
        self.addCleanup(embeddings.stop)

    def build(self):
        return indexer.build_rag_index(self.root)

    def test_safe_roundtrip_does_not_read_legacy_pickle(self):
        generation = self.build()
        (self.root / "faiss_index" / "index.pkl").write_bytes(b"untrusted legacy pickle")
        with mock.patch("pickle.loads", side_effect=AssertionError("pickle must not run")), \
                mock.patch("pickle.load", side_effect=AssertionError("pickle must not run")):
            store = indexer.load_rag_index(self.root, object())
        self.assertEqual(store.index.ntotal, 1)
        self.assertEqual(store.docstore.search(store.index_to_docstore_id[0]).page_content, self.paper.read_text())
        self.assertEqual(sorted(path.name for path in generation.iterdir()), ["documents.json", "index.faiss"])

    def test_unchanged_cache_does_not_load_embeddings(self):
        first = self.build()
        self.embeddings.reset_mock()
        self.assertEqual(self.build(), first)
        self.embeddings.assert_not_called()

    def test_nested_uppercase_pdf_is_loaded_and_invalidates_cache(self):
        first = self.build()
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "new.PDF").write_text("New nested research", encoding="utf-8")
        second = self.build()
        self.assertNotEqual(first, second)
        store = indexer.load_rag_index(self.root, object())
        self.assertEqual(store.index.ntotal, 2)
        self.assertTrue(any(doc.page_content == "New nested research" for doc in store.docstore.documents.values()))

    def test_same_size_edit_with_same_mtime_rebuilds(self):
        first = self.build()
        previous = self.paper.stat()
        text = self.paper.read_text()
        self.paper.write_text("X" + text[1:], encoding="utf-8")
        os.utime(self.paper, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertNotEqual(self.build(), first)

    def test_deleted_pdf_is_removed_from_new_index(self):
        other = self.root / "other.pdf"
        other.write_text("Other paper", encoding="utf-8")
        first = self.build()
        other.unlink()
        self.assertNotEqual(self.build(), first)
        self.assertEqual(indexer.load_rag_index(self.root, object()).index.ntotal, 1)

    def test_empty_directory_rejects_old_cache(self):
        self.build()
        self.paper.unlink()
        with self.assertRaisesRegex(indexer.RAGIndexError, "No PDF"):
            self.build()
        with self.assertRaisesRegex(indexer.RAGIndexError, "No PDF"):
            indexer.load_rag_index(self.root, object())

    def test_corrupt_cached_artifact_is_not_reused(self):
        first = self.build()
        (first / "documents.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(indexer.RAGIndexError):
            indexer.load_rag_index(self.root, object())
        self.assertNotEqual(self.build(), first)

    def test_failed_publication_keeps_previous_generation_and_removes_lock(self):
        first = self.build()
        pointer = self.root / "faiss_index" / "current.json"
        prior_manifest = pointer.read_bytes()
        self.paper.write_text("Changed paper", encoding="utf-8")
        with mock.patch.object(indexer.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                self.build()
        self.assertEqual(pointer.read_bytes(), prior_manifest)
        self.assertTrue(first.exists())
        self.assertFalse((pointer.parent / ".build.lock").exists())
        self.assertFalse(list(pointer.parent.glob(".building-*")))
        self.assertFalse(list(pointer.parent.glob(".current-*")))
        with self.assertRaises(indexer.RAGIndexError):
            indexer.load_rag_index(self.root, object())

    def test_documents_changed_mid_build_cannot_be_published(self):
        real_save = indexer._save_safe_store
        def save_then_change(*args):
            real_save(*args)
            self.paper.write_text("Changed while embedding", encoding="utf-8")
        with mock.patch.object(indexer, "_save_safe_store", side_effect=save_then_change):
            with self.assertRaisesRegex(indexer.RAGIndexError, "changed during"):
                self.build()
        self.assertFalse((self.root / "faiss_index" / "current.json").exists())

    def test_scanned_pdf_with_no_text_requires_ocr(self):
        self.paper.write_text("  \n", encoding="utf-8")
        with self.assertRaisesRegex(indexer.RAGIndexError, "OCR"):
            self.build()
        self.embeddings.assert_not_called()

    def test_concurrent_build_lock_is_respected(self):
        folder = self.root / "faiss_index"
        folder.mkdir()
        (folder / ".build.lock").write_text("active", encoding="utf-8")
        with self.assertRaisesRegex(indexer.RAGIndexError, "Another RAG"):
            self.build()
        self.assertEqual((folder / ".build.lock").read_text(), "active")


class RAGInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ollama = types.ModuleType("ollama")
        path = Path(__file__).resolve().parents[1] / "behave_rag.py"
        with mock.patch.dict(sys.modules, {"ollama": self.ollama}):
            spec = importlib.util.spec_from_file_location("_rag_initialization_test", path)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.agent = self.module.RAGMixin()
        self.agent.rag_dir = self.root
        self.agent.vector_store, self.agent.embeddings = object(), object()

    def test_missing_directory_clears_old_store_and_raises(self):
        self.agent.rag_dir = self.root / "missing"
        self.agent.vector_store = object()
        with mock.patch.object(self.module.subprocess, "run") as run:
            with self.assertRaisesRegex(indexer.RAGIndexError, "does not exist"):
                self.agent.initialize_rag()
        run.assert_not_called()
        self.assertIsNone(self.agent.vector_store)

    def test_failed_builder_never_loads_or_retains_previous_index(self):
        failed = types.SimpleNamespace(returncode=1, stdout="", stderr="missing required package")
        with mock.patch.object(self.module.subprocess, "run", return_value=failed), \
                mock.patch.object(self.module, "create_embeddings") as embeddings:
            with self.assertRaisesRegex(indexer.RAGIndexError, "missing required package"):
                self.agent.initialize_rag()
        embeddings.assert_not_called()
        self.assertIsNone(self.agent.vector_store)
        self.assertIsNone(self.agent.embeddings)

    def test_false_success_without_published_files_does_not_load_embedding_model(self):
        success = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(self.module.subprocess, "run", return_value=success), \
                mock.patch.object(self.module, "create_embeddings") as embeddings:
            with self.assertRaises(indexer.RAGIndexError):
                self.agent.initialize_rag()
        embeddings.assert_not_called()
        self.assertIsNone(self.agent.vector_store)

    def test_loader_failure_does_not_publish_partial_embeddings(self):
        with mock.patch.object(self.module.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="", stderr="")), \
                mock.patch.object(self.module, "corpus_snapshot", return_value=([], "fingerprint")), \
                mock.patch.object(self.module, "index_generation"), \
                mock.patch.object(self.module, "create_embeddings", return_value=object()), \
                mock.patch.object(self.module, "load_rag_index", side_effect=ValueError("bad vector data")):
            with self.assertRaisesRegex(indexer.RAGIndexError, "bad vector data"):
                self.agent.initialize_rag()
        self.assertIsNone(self.agent.vector_store)
        self.assertIsNone(self.agent.embeddings)

    def test_success_uses_absolute_builder_path_and_explicit_project_cwd(self):
        store = object()
        with mock.patch.object(self.module.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stdout="", stderr="")) as run, \
                mock.patch.object(self.module, "corpus_snapshot", return_value=([], "fingerprint")), \
                mock.patch.object(self.module, "index_generation"), \
                mock.patch.object(self.module, "create_embeddings", return_value=object()), \
                mock.patch.object(self.module, "load_rag_index", return_value=store):
            self.assertIs(self.agent.initialize_rag(), True)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[2]), indexer.PROJECT_ROOT / "build_rag_index.py")
        self.assertEqual(run.call_args.kwargs["cwd"], str(indexer.PROJECT_ROOT))
        self.assertEqual(command[-1], str(self.root.resolve()))
        self.assertIs(self.agent.vector_store, store)

    def test_switching_directory_immediately_invalidates_old_knowledge(self):
        self.agent.current_rag_refs = {1: "old"}
        self.agent.rag_dir = self.root / "different"
        self.assertIsNone(self.agent.vector_store)
        self.assertIsNone(self.agent.embeddings)
        self.assertEqual(self.agent.current_rag_refs, {})

    def test_same_directory_preserves_loaded_index(self):
        old = self.agent.vector_store
        self.agent.rag_dir = self.root / "."
        self.assertIs(self.agent.vector_store, old)


class AgentConfigurationTests(unittest.TestCase):
    def test_configuration_is_project_relative_and_does_not_pollute_other_instances(self):
        project = Path(__file__).resolve().parents[1]
        dotenv = types.ModuleType("dotenv")
        settings = ({"OLLAMA_MODEL": "first", "RAG_DATA_DIR": "papers-a", "VIDEOS_DIR": "videos-a"},
                    {"OLLAMA_MODEL": "second", "RAG_DATA_DIR": "papers-b", "VIDEOS_DIR": "videos-b"})
        dotenv.dotenv_values = mock.Mock(side_effect=settings)
        ollama = types.ModuleType("ollama")
        modules = {"dotenv": dotenv, "ollama": ollama, "transformers": None}
        for name, cls_name in (("behave_tracking", "TrackingMixin"), ("behave_report", "ReportMixin"),
                               ("behave_huddle", "HuddleMixin"), ("behave_analysis", "AnalysisMixin")):
            module = types.ModuleType(name)
            setattr(module, cls_name, type(cls_name, (), {}))
            modules[name] = module
        rag_path = project / "behave_rag.py"
        with mock.patch.dict(sys.modules, modules):
            rag_spec = importlib.util.spec_from_file_location("behave_rag", rag_path)
            rag_module = importlib.util.module_from_spec(rag_spec)
            rag_spec.loader.exec_module(rag_module)
            modules["behave_rag"] = rag_module
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(Path, "is_file", return_value=True):
            os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"
            spec = importlib.util.spec_from_file_location("_config_test", project / "behave_ai.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(os.environ["PYTORCH_ALLOC_CONF"], "max_split_size_mb:64")
            first = module.BehaviorAgent("first.env")
            second = module.BehaviorAgent("second.env")
            self.assertEqual((first.model, second.model), ("first", "second"))
            self.assertEqual(first.config_path, project / "first.env")
            self.assertEqual(first.videos_dir, str(project / "videos-a"))
            self.assertEqual(second.rag_dir, str(project / "papers-b"))
            self.assertNotIn("OLLAMA_MODEL", os.environ)
            self.assertFalse(dotenv.dotenv_values.call_args.kwargs["interpolate"])
            dotenv.dotenv_values.side_effect = None
            dotenv.dotenv_values.return_value = settings[0]
            os.environ["OLLAMA_MODEL"] = "environment-override"
            self.assertEqual(module.BehaviorAgent().model, "environment-override")
            with mock.patch.object(Path, "is_file", return_value=False):
                with self.assertRaisesRegex(FileNotFoundError, "configuration file"):
                    module.BehaviorAgent("missing.env")

    def test_embeddings_are_explicitly_cpu_resident(self):
        package = types.ModuleType("langchain_huggingface")
        package.HuggingFaceEmbeddings = mock.Mock()
        with mock.patch.dict(sys.modules, {"langchain_huggingface": package}):
            indexer.create_embeddings()
        self.assertEqual(package.HuggingFaceEmbeddings.call_args.kwargs["model_kwargs"], {"device": "cpu"})


if __name__ == "__main__":
    unittest.main()
