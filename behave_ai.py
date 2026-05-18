import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import logging
import warnings

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)
import json
import ollama
import torch
import gc
from dotenv import load_dotenv

from tools.feature_tracking import extract_tracks

try:
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("Warning: Missing some Langchain/FAISS modules. RAG might not work.")

from behave_rag import RAGMixin
from behave_tracking import TrackingMixin
from behave_report import ReportMixin
from behave_huddle import HuddleMixin
from behave_analysis import AnalysisMixin

class BehaviorAgent(RAGMixin, TrackingMixin, ReportMixin, HuddleMixin, AnalysisMixin):
    def __init__(self, config_path="config.env"):
        load_dotenv(config_path)
        self.model = os.getenv("OLLAMA_MODEL", "gemma4:26b")
        self.rag_dir = os.getenv("RAG_DATA_DIR", "./data")
        self.videos_dir = os.getenv("VIDEOS_DIR", "./videos")
        self.tracks = None
        self.vector_store = None
        self.embeddings = None
        self.current_rag_refs = {}
        self.rag_ref_counter = 1
