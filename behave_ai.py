import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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
from pathlib import Path
from dotenv import dotenv_values

from behave_rag import RAGMixin
from behave_tracking import TrackingMixin
from behave_report import ReportMixin
from behave_huddle import HuddleMixin
from behave_analysis import AnalysisMixin

class BehaviorAgent(RAGMixin, TrackingMixin, ReportMixin, HuddleMixin, AnalysisMixin):
    def __init__(self, config_path="config.env"):
        project_root = Path(__file__).resolve().parent
        config = Path(config_path).expanduser()
        self.config_path = (config if config.is_absolute() else project_root / config).resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Application configuration file does not exist: {self.config_path}")
        settings = dotenv_values(self.config_path, interpolate=False, encoding="utf-8-sig")
        def setting(name, default):
            return os.environ.get(name, settings.get(name) or default)
        def directory(value):
            if not str(value).strip():
                raise ValueError("Application directory paths must be nonempty.")
            path = Path(value).expanduser()
            return str((path if path.is_absolute() else project_root / path).resolve())
        self.model = setting("OLLAMA_MODEL", "gemma4:26b")
        if not self.model.strip():
            raise ValueError("OLLAMA_MODEL must be nonempty.")
        self.rag_dir = setting("RAG_DATA_DIR", "./data")
        self.videos_dir = directory(setting("VIDEOS_DIR", "./videos"))
        self.tracks = None
        self.tracking_result = None
        self._tracking_context = None
        self._tracking_failed_context = None
        self._tracking_failed_run_id = None
        self.vector_store = None
        self.embeddings = None
        self.current_rag_refs = {}
        self.rag_ref_counter = 1
