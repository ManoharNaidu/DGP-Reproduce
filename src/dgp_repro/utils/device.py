"""Central device selection. Nothing else in the code base should hard-code 'cuda'."""

from __future__ import annotations


def resolve_device(spec: str = "auto") -> str:
    """'auto' -> 'cuda:0' when available, else 'cpu'. Explicit CUDA requests fail loudly without a GPU."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if spec == "auto":
        return "cuda:0" if has_cuda else "cpu"
    if spec.startswith("cuda") and not has_cuda:
        raise RuntimeError(f"device {spec!r} requested but no CUDA GPU is available on this machine")
    return spec


def describe_device(device: str) -> dict:
    info = {"device": device}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        if device.startswith("cuda"):
            index = int(device.split(":")[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(index)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
    except ImportError:
        pass
    return info
