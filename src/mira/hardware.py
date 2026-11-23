from __future__ import annotations

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _cpu_model() -> str:
    info = Path("/proc/cpuinfo")
    if not info.exists():
        return platform.processor() or "unknown"
    for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gib() -> float | None:
    info = Path("/proc/meminfo")
    if not info.exists():
        return None
    for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal"):
            kib = int(line.split()[1])
            return round(kib / 1024 / 1024, 2)
    return None


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def collect_hardware() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "device": "cpu",
        "packages": {
            name: _pkg(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "faiss-cpu",
                "pymupdf",
                "llama-cpp-python",
                "streamlit",
                "numpy",
            )
        },
    }
