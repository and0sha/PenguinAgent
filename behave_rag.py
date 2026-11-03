import ollama
import os
from pathlib import Path
import subprocess
import sys
from build_rag_index import (
    PROJECT_ROOT, RAGIndexError, corpus_snapshot, create_embeddings, index_generation,
    load_rag_index, resolve_rag_directory,
)

class RAGMixin:
    @property
    def rag_dir(self):
        return self._rag_dir

    @rag_dir.setter
    def rag_dir(self, value):
        try:
            directory = str(resolve_rag_directory(value))
        except (OSError, ValueError, RAGIndexError):
            self.invalidate_rag()
            raise
        if directory != getattr(self, "_rag_dir", None):
            self.invalidate_rag()
        self._rag_dir = directory

    def invalidate_rag(self):
        self.vector_store = None
        self.embeddings = None
        self.current_rag_refs = {}
        self.rag_ref_counter = 1

    def initialize_rag(self):
        """Publish a queryable index only after the isolated builder and safe loader succeed."""
        self.invalidate_rag()
        try:
            directory = resolve_rag_directory(self.rag_dir)
            if not directory.is_dir():
                raise RAGIndexError(f"Scientific PDF directory does not exist: {directory}")
            result = subprocess.run(
                [sys.executable, "-B", str(PROJECT_ROOT / "build_rag_index.py"), "--dir", str(directory)],
                cwd=str(PROJECT_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "No diagnostic output.").strip()[-4000:]
                raise RAGIndexError(f"Scientific RAG builder failed (exit {result.returncode}): {detail}")
            index_generation(directory, corpus_snapshot(directory)[1])
            embeddings = create_embeddings()
            store = load_rag_index(directory, embeddings)
            self.embeddings, self.vector_store = embeddings, store
            return True
        except Exception as exc:
            self.invalidate_rag()
            raise RAGIndexError(f"Cannot initialize scientific RAG: {exc}") from exc

    def ask(self, query, output_dir=None):
        # Video reports are scoped explicitly by the caller to a validated run.
        # A literature-only question must never sweep reports from other videos.
        local_context = ""
        report_files = []
        if output_dir and Path(output_dir).is_dir():
            report_files = [Path(output_dir) / name for name in (
                "final_scientific_report.md", "final_huddle_report.md",
            )]
            
        for filepath in report_files:
            if filepath.is_file():
                try:
                    if filepath.is_symlink() or filepath.resolve().parent != Path(output_dir).resolve():
                        raise RAGIndexError(f"Current video report points outside its run directory: {filepath.name}")
                    with open(filepath, "r", encoding="utf-8") as f:
                        contents = f.read(4501)
                    excerpt = contents[:4500]
                    if len(contents) > 4500:
                        excerpt += "\n[Report excerpt truncated]"
                    local_context += f"\n--- Report Context from {filepath.name} ---\n{excerpt}\n"
                except (OSError, UnicodeError) as exc:
                    raise RAGIndexError(f"Cannot read current video report {filepath.name}: {exc}") from exc
                

        rag_context = ""
        if self.vector_store is not None:
            try:
                # Use MMR for more diverse interactive context
                docs = self.vector_store.max_marginal_relevance_search(query, k=3, fetch_k=10, lambda_mult=0.7)
                rag_context = "\n".join([d.page_content for d in docs])
            except Exception as exc:
                raise RAGIndexError(f"Cannot search the scientific index: {exc}") from exc
                
        sys_prompt = ("You are PenguinAgent, an expert ornithologist. Answer using the provided contexts. "
                      "Distinguish observations, scientific literature and hypotheses. Video reports are prior interpretations, "
                      "not independent measurements. State when the supplied evidence is insufficient.")
        
        combined_context = ""
        if local_context:
            combined_context += f"\n\n[PRIORITY] Context from Local Video Reports:\n{local_context[:10000]}"
        if rag_context:
            combined_context += f"\n\n[SECONDARY] Context from Scientific RAG Database:\n{rag_context}"
            
        if combined_context:
            sys_prompt += f"\n\nContext:\n{combined_context}"
            
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': query}
            ], options={"num_ctx": 16384}, keep_alive=0)
            answer = response['message']['content']
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Ollama returned an empty answer")
            return answer
        except Exception as exc:
            raise RuntimeError(f"Cannot communicate with Ollama: {exc}") from exc
