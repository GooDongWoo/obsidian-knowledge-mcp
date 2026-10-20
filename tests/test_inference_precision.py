"""Precision contracts without importing torch or loading real models."""

from contextlib import contextmanager
from dataclasses import fields, replace
import sys
from types import SimpleNamespace

import pytest

from knowledge_mcp.config import Settings


@pytest.mark.parametrize("policy,device,expected", [
    ("auto", "cuda", "bfloat16"), ("auto", "cuda:1", "bfloat16"),
    ("auto", "cpu", "float32"), ("auto", "mps", "float32"),
    ("float32", "cuda:1", "float32"), ("float32", "cpu", "float32"),
])
def test_resolve_precision(policy, device, expected):
    from knowledge_mcp.inference_precision import resolve_torch_precision
    assert resolve_torch_precision(policy, device) == expected


@pytest.mark.parametrize("policy", ["bf16", "float16", "", None])
def test_invalid_precision_rejected(policy, tmp_path):
    from knowledge_mcp.inference_precision import resolve_torch_precision
    with pytest.raises(ValueError, match="inference_precision must be auto or float32"):
        resolve_torch_precision(policy, "cpu")
    base = Settings.from_paths(vault_root=tmp_path / "vault", project_root=tmp_path / "project")
    with pytest.raises(ValueError, match="inference_precision must be auto or float32"):
        replace(base, inference_precision=policy)


def test_settings_append_precision_preserving_positional_and_replace(tmp_path):
    settings = Settings(tmp_path / "vault", tmp_path / "runtime", Settings.DEFAULT_QDRANT_URL,
                        "collection", Settings.DEFAULT_DENSE_MODEL, "codex")
    assert settings.inference_precision == "auto"
    assert fields(Settings)[-1].name == "inference_precision"
    assert replace(settings, inference_precision="float32").inference_precision == "float32"
    assert Settings.from_paths(vault_root=tmp_path / "vault",
                               project_root=tmp_path / "project").inference_precision == "auto"


@pytest.mark.parametrize("policy", [None, "auto", "float32", "float16"])
def test_settings_precision_from_env(policy, tmp_path, monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_VAULT_ROOT", str(tmp_path / "vault"))
    monkeypatch.setenv("KNOWLEDGE_PROJECT_ROOT", str(tmp_path / "project"))
    if policy is not None:
        monkeypatch.setenv("KNOWLEDGE_INFERENCE_PRECISION", policy)
    if policy == "float16":
        with pytest.raises(ValueError, match="inference_precision"):
            Settings.from_env("codex")
    else:
        assert Settings.from_env("codex").inference_precision == (policy or "auto")


def test_constructor_kwargs_select_supported_parameter_without_instantiation():
    from knowledge_mcp.inference_precision import model_dtype_kwargs

    class Current:
        def __init__(self, model_kwargs=None, automodel_args=None):
            raise AssertionError("must only inspect constructor")

    class Legacy:
        def __init__(self, automodel_args=None):
            raise AssertionError("must only inspect constructor")

    assert model_dtype_kwargs(Current, "bfloat16") == {"model_kwargs": {"torch_dtype": "bfloat16"}}
    assert model_dtype_kwargs(Legacy, "float32") == {"automodel_args": {"torch_dtype": "float32"}}
    with pytest.raises(ValueError, match="model constructor has no supported dtype argument"):
        model_dtype_kwargs(object, "bfloat16")


def install_cuda_stub(monkeypatch, *, legacy=False, supported=True, capability=(8, 0), broken=False):
    state = {"device": None, "calls": []}

    @contextmanager
    def device(name):
        previous = state["device"]
        state["device"] = name
        try:
            yield
        finally:
            state["device"] = previous

    def result():
        assert state["device"] == "cuda:1"
        if broken:
            raise RuntimeError("driver unavailable")
        return supported

    def modern_probe(including_emulation=True):
        state["calls"].append(("modern", including_emulation))
        return result()

    def legacy_probe():
        state["calls"].append(("legacy",))
        return result()

    def get_device_capability():
        assert state["device"] == "cuda:1"
        state["calls"].append(("capability",))
        return capability

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device=device, is_bf16_supported=legacy_probe if legacy else modern_probe,
        get_device_capability=get_device_capability)))
    return state


def test_native_bf16_modern_probe_disables_emulation_on_selected_device(monkeypatch):
    from knowledge_mcp.inference_precision import require_cuda_bf16
    state = install_cuda_stub(monkeypatch)
    require_cuda_bf16("cuda:1", "bfloat16")
    assert state == {"device": None, "calls": [("modern", False)]}


def test_native_bf16_legacy_probe_checks_native_capability(monkeypatch):
    from knowledge_mcp.inference_precision import require_cuda_bf16
    state = install_cuda_stub(monkeypatch, legacy=True)
    require_cuda_bf16("cuda:1", "bfloat16")
    assert state["device"] is None
    assert ("legacy",) in state["calls"]
    assert ("capability",) in state["calls"]


@pytest.mark.parametrize("options", [
    {"supported": False}, {"legacy": True, "supported": False},
    {"legacy": True, "capability": (7, 5)}, {"broken": True},
])
def test_native_bf16_fails_closed_without_changing_device(monkeypatch, options):
    from knowledge_mcp.inference_precision import require_cuda_bf16
    state = install_cuda_stub(monkeypatch, **options)
    with pytest.raises(RuntimeError, match="CUDA native bfloat16 support could not be verified for cuda:1"):
        require_cuda_bf16("cuda:1", "bfloat16")
    assert state["device"] is None


@pytest.mark.parametrize("device,dtype", [("cpu", "float32"), ("cpu", "bfloat16"), ("cuda:1", "float32")])
def test_other_precisions_do_not_import_or_probe_torch(monkeypatch, device, dtype):
    from knowledge_mcp.inference_precision import require_cuda_bf16
    monkeypatch.setitem(sys.modules, "torch", None)
    require_cuda_bf16(device, dtype)


def test_imports_do_not_pull_ml_dependencies(tmp_path):
    import os
    import subprocess

    subprocess.run([sys.executable, "-c", "import sys; import knowledge_mcp.cli, knowledge_mcp.proxy; "
                    "import knowledge_mcp.inference_precision; "
                    "assert not {'torch', 'transformers', 'sentence_transformers'} & sys.modules.keys()"],
                   cwd=tmp_path, env=os.environ.copy(), check=True, timeout=30)
