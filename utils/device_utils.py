import os

import torch


def resolve_torch_device(requested: str | None) -> torch.device:
    """Resolve a requested accelerator with explicit CUDA/MPS/CPU fallbacks."""
    requested = (requested or "auto").lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch.device("mps")
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        print(f"[Init] Requested device '{requested}' is unavailable; falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch.device("mps")
        print("[Init] Requested MPS but it is unavailable; falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")
