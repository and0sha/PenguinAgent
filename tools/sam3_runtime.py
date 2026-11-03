"""Resolve SAM3 files without importing PyTorch or downloading model weights."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_REVISION = "847e1a3b15115a04c87c0760297f044f0555d970"
_SAM3_KEYS = {"SAM3_ROOT", "SAM3_CHECKPOINT"}
_ESCAPES = {"\\": "\\", "'": "'", '"': '"', "a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v"}


class Sam3ConfigurationError(RuntimeError):
    """The configured SAM3 source or checkpoint cannot be used."""


@dataclass(frozen=True)
class Sam3Runtime:
    source_root: Path
    bpe_path: Path
    checkpoint_path: Optional[Path]

    def builder_kwargs(self):
        kwargs = {"bpe_path": str(self.bpe_path), "async_loading_frames": False}
        if self.checkpoint_path is not None:
            kwargs["checkpoint_path"] = str(self.checkpoint_path)
        return kwargs


def _read_sam3_config(config_path):
    """Read the two literal, single-line SAM3 settings from an env file.

    Other application settings are left to python-dotenv. Quotes and their escape
    sequences follow dotenv syntax. Optional ``export`` prefixes and comments are
    supported; paths must be literal (no variable interpolation or shell expansion).
    """
    try:
        lines = config_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Sam3ConfigurationError(
            f"Cannot read SAM3 configuration at {config_path}: {exc}. "
            "Provide a readable config.env or choose it with --config."
        ) from exc

    settings = {}
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if key not in _SAM3_KEYS:
            continue
        value = value.strip()
        error = None
        if not separator:
            error = "expected KEY=value"
        elif value.startswith(("'", '"')):
            quote = value[0]
            pattern = r'^"((?:\\"|[^"])*)"\s*(?:#.*)?$' if quote == '"' else r"^'((?:\\'|[^'])*)'\s*(?:#.*)?$"
            match = re.fullmatch(pattern, value)
            if match is None:
                error = "unterminated quoted path or unexpected text after it"
            else:
                value = match.group(1)
                escape_pattern = r"\\[\\'\"abfnrtv]" if quote == '"' else r"\\[\\']"
                value = re.sub(escape_pattern, lambda match: _ESCAPES[match.group(0)[1]], value)
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if "${" in value:
            error = "variable interpolation is not supported for SAM3 paths"
        if error:
            raise Sam3ConfigurationError(
                f"Invalid {key} in {config_path}, line {line_number}: {error}. "
                "Use a single-line literal path, quoting it if needed."
            )
        settings[key] = value
    return settings


def _project_path(value, project_root, setting):
    if any(ord(character) < 32 for character in value):
        raise Sam3ConfigurationError(
            f"Invalid {setting} path: it contains a control character. "
            "Use forward slashes (C:/models/sam3) or single quotes for Windows paths in config.env."
        )
    try:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise Sam3ConfigurationError(f"Invalid {setting} path {value!r}: {exc}") from exc


def resolve_sam3_runtime(config_path=None, *, project_root=None, environ: Optional[Mapping[str, str]] = None):
    """Validate SAM3 paths, resolving relative paths from the project root.

    Environment values override config.env. An empty/unset SAM3_CHECKPOINT keeps
    SAM3's default Hugging Face download/cache behavior. No ML packages are loaded.
    """
    project_root = Path(project_root or PROJECT_ROOT).resolve()
    config_path = _project_path(str(config_path or "config.env"), project_root, "config")
    settings = _read_sam3_config(config_path)
    environment = os.environ if environ is None else environ

    root_value = environment.get("SAM3_ROOT", settings.get("SAM3_ROOT", "./sam3")).strip()
    if not root_value:
        raise Sam3ConfigurationError("SAM3_ROOT is empty. Set it to ./sam3 or a SAM3 checkout directory.")
    source_root = _project_path(root_value, project_root, "SAM3_ROOT")
    bpe_path = source_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    required_files = [source_root / "sam3" / "model_builder.py", bpe_path]
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_files = ", ".join(str(path) for path in missing)
        raise Sam3ConfigurationError(
            f"SAM3_ROOT does not contain a complete SAM3 checkout: {source_root}. "
            f"Missing files: {missing_files}. From the PenguinAgent directory run "
            "`git submodule update --init -- sam3` to restore the pinned SAM3 source, "
            "or set SAM3_ROOT to an existing checkout containing sam3/model_builder.py and sam3/assets/."
        )

    checkpoint_value = environment.get("SAM3_CHECKPOINT", settings.get("SAM3_CHECKPOINT", "")).strip()
    checkpoint_path = None
    if checkpoint_value:
        checkpoint_path = _project_path(checkpoint_value, project_root, "SAM3_CHECKPOINT")
        if not checkpoint_path.is_file():
            raise Sam3ConfigurationError(
                f"SAM3_CHECKPOINT file does not exist: {checkpoint_path}. "
                "Set it to a local SAM3 sam3.pt file, or leave it empty to use Hugging Face loading."
            )

    return Sam3Runtime(source_root=source_root, bpe_path=bpe_path, checkpoint_path=checkpoint_path)
