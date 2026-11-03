"""Check SAM3 source/checkpoint paths without importing ML packages or downloading weights."""

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sam3_runtime import SAM3_REVISION, Sam3ConfigurationError, resolve_sam3_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.env", help="Config path; relative paths start at the project directory.")
    args = parser.parse_args()
    try:
        runtime = resolve_sam3_runtime(args.config)
    except Sam3ConfigurationError as exc:
        print(f"SAM3 configuration error: {exc}", file=sys.stderr)
        return 1

    print(f"SAM3 source: {runtime.source_root}")
    print(f"Tokenizer: {runtime.bpe_path}")
    print(f"Expected bundled revision: {SAM3_REVISION}")
    if runtime.checkpoint_path is None:
        print("Checkpoint: automatic Hugging Face loading (access/authentication and cache or network required).")
    else:
        print(f"Checkpoint: {runtime.checkpoint_path}")
    print("Path checks passed. GPU support, installed dependencies and model inference have not been checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
