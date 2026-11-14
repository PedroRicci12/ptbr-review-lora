"""Upload the human-label QLoRA adapter and its model card to the Hugging Face Hub (Stage 8).

The token is read from `.env` (HF_TOKEN, write access) or from the environment; it is never
printed or written anywhere else. The repository is created **private** by default, so the page
can be reviewed before it is made public (`--public`).

    uv run --group publish python scripts/publish_hf.py --adapter-dir <unzipped qlora_human_adapter>

Only the adapter itself is uploaded (`adapter_config.json`, `adapter_model.safetensors`) plus the
two cards. The tokenizer files in the zip are the base model's and are loaded from the base
repository; the zip's auto-generated README is replaced by the model card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "PedroRicci12/smollm3-3b-ptbr-review-sentiment-qlora"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
CARD_FILES = {"README.md": "README.md", "README.pt-BR.md": "README.pt-BR.md"}
EXPECTED_CONFIG = {"base_model_name_or_path": "HuggingFaceTB/SmolLM3-3B", "r": 16, "lora_alpha": 32}


def read_token(env_file: Path) -> str:
    """HF_TOKEN from the environment, else from a KEY=VALUE .env file."""
    token = os.environ.get("HF_TOKEN", "")
    if not token and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "HF_TOKEN":
                token = value.strip().strip('"').strip("'")
    if not token:
        raise SystemExit("HF_TOKEN not set: copy .env.example to .env and paste a write token")
    return token


def check_adapter(adapter_dir: Path) -> str:
    """Fail early if files are missing or the config is not the expected adapter; return its sha256."""
    for name in ADAPTER_FILES:
        if not (adapter_dir / name).is_file():
            raise SystemExit(f"missing {name} in {adapter_dir}")
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    for key, value in EXPECTED_CONFIG.items():
        if config.get(key) != value:
            raise SystemExit(f"adapter_config.json: {key}={config.get(key)!r}, expected {value!r}")
    return hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--public", action="store_true", help="create the repository as public")
    args = parser.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi

    sha256 = check_adapter(args.adapter_dir)
    print("adapter_model.safetensors sha256:", sha256)
    api = HfApi(token=read_token(ROOT / ".env"))
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(args.adapter_dir / name))
        for name in ADAPTER_FILES
    ] + [
        CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(ROOT / "model_card" / src))
        for src, dest in CARD_FILES.items()
    ]
    commit = api.create_commit(
        args.repo_id, operations=operations, commit_message="Upload QLoRA adapter and model card"
    )
    print("uploaded:", commit.commit_url)


if __name__ == "__main__":
    main()
