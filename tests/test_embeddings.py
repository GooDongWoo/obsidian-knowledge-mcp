from fastembed.common.types import Device
import onnxruntime as ort


def test_prepare_cuda_runtime_preloads_optional_nvidia_dlls(monkeypatch):
    from knowledge_mcp.embeddings import prepare_cuda_runtime

    calls = []
    monkeypatch.setattr(ort, "preload_dlls", lambda: calls.append(True), raising=False)
    prepare_cuda_runtime()
    assert calls == [True]


def test_select_device_uses_cuda_provider_when_available():
    from knowledge_mcp.embeddings import select_device

    assert select_device(["CUDAExecutionProvider", "CPUExecutionProvider"]) is Device.CUDA


def test_select_device_falls_back_to_cpu_without_cuda_provider():
    from knowledge_mcp.embeddings import select_device

    assert select_device(["CPUExecutionProvider"]) is Device.CPU


def test_runtime_provider_selection_matches_onnxruntime():
    from knowledge_mcp.embeddings import select_device

    expected = Device.CUDA if "CUDAExecutionProvider" in ort.get_available_providers() else Device.CPU
    assert select_device() is expected
