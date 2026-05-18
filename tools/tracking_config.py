import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)

try:
    import transformers
    transformers.utils.logging.set_verbosity_error()
except ImportError:
    pass
logging.getLogger("transformers").setLevel(logging.ERROR)

TARGET_FPS = 1               
TARGET_HEIGHT = 720              
CHUNK_DURATION_SEC = 10         
MAX_DIST_REL_WIDTH = 0.10        
SIGLIP_CONFIDENCE_THRESH = 0.1  
