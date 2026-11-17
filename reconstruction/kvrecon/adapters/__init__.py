






from importlib import import_module
from typing import Protocol
from ..backend import Capabilities, RecoveryBackend
from .litert_gemma import LiteRtTensorAdapter
from .qwen_verify import QwenVerifyAdapter


class ModelAdapter(Protocol):
    def read_manifest(self, path): ...
    def dimensions(self, sections, profile) -> dict: ...
    def capabilities(self, sections, profile, layer) -> Capabilities: ...
    def open(self, sections, profile, layer, output, model_hash, **options) -> RecoveryBackend: ...
    def import_observation(self, snapshot, output, sections, model_hash, prefix, start, count, layer): ...
    def save_fixture(self, output, vectors, positions, prefix, layer, sections, model_hash, source): ...


ADAPTERS = {
    "litert_tensor_gemma": LiteRtTensorAdapter,
    "litert_qwen_verify": QwenVerifyAdapter,
}


def get_adapter(name) -> ModelAdapter:
    if name in ADAPTERS:
        factory = ADAPTERS[name]
    elif ":" in name:
        module, symbol = name.split(":", 1)
        factory = getattr(import_module(module), symbol)
    else:
        raise ValueError(f"Unknown adapter: {name}")
    adapter = factory()
    for method in ("read_manifest", "dimensions", "capabilities", "open", "import_observation", "save_fixture"):
        if not callable(getattr(adapter, method, None)):
            raise TypeError(f"Adapter missing method: {method}")
    return adapter


__all__ = ["ADAPTERS", "ModelAdapter", "LiteRtTensorAdapter", "QwenVerifyAdapter", "get_adapter"]
