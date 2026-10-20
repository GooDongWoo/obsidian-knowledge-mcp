"""Shared PyTorch inference policy; importing this module does not import torch."""


def resolve_torch_precision(policy: str, device: str) -> str:
    if policy not in ("auto", "float32"):
        raise ValueError("inference_precision must be auto or float32")
    return "bfloat16" if policy == "auto" and device.split(":", 1)[0] == "cuda" else "float32"


def model_dtype_kwargs(model_class: type, dtype: str) -> dict[str, dict[str, str]]:
    import inspect

    parameters = inspect.signature(model_class).parameters
    for name in ("model_kwargs", "automodel_args"):
        if name in parameters:
            return {name: {"torch_dtype": dtype}}
    raise ValueError("model constructor has no supported dtype argument")


def require_cuda_bf16(device: str, dtype: str) -> None:
    """Fail closed unless the selected CUDA device proves native bf16 support."""
    if device.split(":", 1)[0] != "cuda" or dtype != "bfloat16":
        return

    import inspect

    message = f"CUDA native bfloat16 support could not be verified for {device}"
    try:
        import torch

        with torch.cuda.device(device):
            probe = torch.cuda.is_bf16_supported
            if "including_emulation" in inspect.signature(probe).parameters:
                supported = probe(including_emulation=False)
            else:
                # Older probes can accept emulated bf16 on pre-Ampere devices.
                supported = torch.cuda.get_device_capability()[0] >= 8 and probe()
            if not supported:
                raise RuntimeError(message)
    except Exception as exc:
        raise RuntimeError(message) from exc
