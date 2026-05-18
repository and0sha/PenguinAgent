import os
import torch
import ollama
import gc
try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    pass

class RAGMixin:
    def initialize_rag(self):
        print("Initializing Local RAG Database...")
        if not os.path.exists(self.rag_dir):
            print(f"RAG directory {self.rag_dir} not found.")
            return

        try:
           
            import subprocess
            import sys
            
            
            result = subprocess.run(
                [sys.executable, "build_rag_index.py", "--dir", self.rag_dir],
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
                
            if result.returncode != 0:
                print("Error: Isolated RAG indexing failed. Check logs above.")
                return

            index_path = os.path.join(self.rag_dir, "faiss_index")
            if os.path.exists(index_path):
               
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3", 
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
                print("RAG successfully loaded into main process for querying.")
                
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print("Error: Index path not found after successful subprocess run.")
                
        except Exception as e:
            print(f"Error initializing RAG: {e}")

    def ask(self, query, output_dir=None):
        import glob
        # Gather Context from Local Reports (Priority)
        local_context = ""
        report_files = []
        if output_dir and os.path.exists(output_dir):
            report_files.extend(glob.glob(os.path.join(output_dir, "*.md")))
        else:
            report_files.extend(glob.glob(os.path.join(self.videos_dir, "**/*.md"), recursive=True))
            
        for filepath in set(report_files):
            if "final_scientific_report" in filepath:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        local_context += f"\n--- Report Context from {filepath} ---\n{f.read()}\n"
                except: pass
                

        rag_context = ""
        if self.vector_store:
            try:
                # Use MMR for more diverse interactive context
                docs = self.vector_store.max_marginal_relevance_search(query, k=3, fetch_k=10, lambda_mult=0.7)
                rag_context = "\n".join([d.page_content for d in docs])
            except Exception as e:
                print(f"Error searching RAG: {e}")
                
        sys_prompt = "<|think|>You are PenguinAgent, an expert ornithologist. Answer the user's question using the provided contexts."
        
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
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with Ollama: {e}"