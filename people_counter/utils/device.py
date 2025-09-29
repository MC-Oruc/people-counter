from __future__ import annotations

from typing import Optional, List


def resolve_device(device: str) -> str:
    """
    Resolve a user-provided device string to a concrete compute target.

    Rules (copied from YoloDetector._resolve_device / cuda-test):
      - "cpu" -> "cpu"
      - "gpu" or "cuda" -> "cuda" if CUDA is available else "cpu"
      - "cuda:<index>" -> that index if available, else fallback to "cuda" or "cpu"
    """
    dev = (device or "cpu").lower()
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    if dev.startswith("cuda") or dev == "gpu":
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            if ":" in dev:
                try:
                    idx = int(dev.split(":", 1)[1])
                    if idx < torch.cuda.device_count():  # type: ignore[attr-defined]
                        return f"cuda:{idx}"
                except Exception:
                    pass
            return "cuda"
        return "cpu"
    return "cpu"


def is_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return hasattr(torch, 'cuda') and torch.cuda.is_available()
    except Exception:
        return False


def list_cuda_devices() -> List[str]:
    if not is_cuda_available():
        return []
    try:
        import torch  # type: ignore
        cnt = int(torch.cuda.device_count())  # type: ignore[attr-defined]
        return [f"cuda:{i}" for i in range(cnt)]
    except Exception:
        return []


def cuda_diagnostics() -> str:
    """Return a single-line diagnostic string about CUDA/Torch availability."""
    try:
        import torch  # type: ignore
        lines = []
        lines.append(f"torch: {getattr(torch, '__version__', 'unknown')}")
        try:
            lines.append(f"torch.version.cuda: {getattr(getattr(torch, 'version', None), 'cuda', None)}")
        except Exception:
            lines.append("torch.version.cuda: (error)")
        avail = hasattr(torch, 'cuda') and torch.cuda.is_available()
        lines.append(f"cuda.is_available: {avail}")
        try:
            cnt = torch.cuda.device_count() if avail else 0
            lines.append(f"cuda.device_count: {cnt}")
            for i in range(int(cnt)):
                try:
                    name = torch.cuda.get_device_name(i)
                    lines.append(f"cuda:{i} -> {name}")
                except Exception:
                    pass
        except Exception:
            lines.append("cuda.device_count: (error)")
        return " | ".join(lines)
    except Exception as e:
        return f"torch import failed: {e}"


def default_half_for(device: str) -> bool:
    """Whether half precision should be enabled by default for the resolved device."""
    return str(device).startswith('cuda')
