"""Small config and run metadata helpers; JSON is valid YAML syntax."""
from __future__ import annotations
import json, platform, subprocess, sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        config = json.loads(text)
        validate_config(config)
        return config
    except json.JSONDecodeError as exc: raise ValueError("Configs use JSON-compatible YAML; install PyYAML for general YAML") from exc

def validate_config(config):
    if config.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if int(config.get("max_steps", 1)) <= 0:
        raise ValueError("max_steps must be positive")
    if int(config.get("num_workers", 0)) < 0:
        raise ValueError("num_workers must be non-negative")
    if not isinstance(config.get("mixed_precision", False), bool):
        raise ValueError("mixed_precision must be boolean")

def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"

def runtime_metadata(config, resolved_device=None):
    framework = {"name": None, "version": None, "cuda": None, "gpu": None}
    try:
        framework["name"] = "torch"
        framework["version"] = version("torch")
    except PackageNotFoundError:
        pass
    # Avoid loading native CUDA/CPU DLLs during a CPU-only smoke run. The
    # server CUDA path still records runtime details from the active backend.
    if resolved_device == "cuda":
        try:
            import torch
            framework["cuda"] = getattr(torch.version, "cuda", None)
            framework["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            pass
    return {"python": sys.version, "platform": platform.platform(), "requested_device": config.get("device"), "resolved_device": resolved_device or config.get("device"), "seed": config.get("seed"), "git_commit": git_commit(), "framework": framework}
