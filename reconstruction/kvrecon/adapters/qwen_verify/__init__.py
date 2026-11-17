











import json
from pathlib import Path
from ...backend import Capabilities
from ...config import model_dimensions
from ...common import quantization, kv_dtype
from ...observation import import_snapshot, save_observation

DEFAULT_PACKAGE = "com.example.annoymous"
QUERY_MODES = ("verify", "decode", "exact_prefill")


def kv_layers(sections, signature):
    outputs = {x["name"] for x in sections["tf_lite_prefill_decode"]["signatures"][signature]["outputs"]}
    return tuple(sorted(int(n.removeprefix("kv_cache_k_")) for n in outputs
                        if n.startswith("kv_cache_k_") and n.replace("kv_cache_k_", "kv_cache_v_", 1) in outputs))


class QwenVerifyAdapter:
    

    def read_manifest(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))["model_sections"]

    def dimensions(self, sections, profile):
        return model_dimensions(sections, profile["batch_signature"])

    @staticmethod
    def options(profile):
        opts = dict(profile.get("adapter_options") or {})
        mode = opts.get("query_mode", "verify")
        if mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}")
        opts["query_mode"] = mode
        opts.setdefault("package", DEFAULT_PACKAGE)
        return opts

    def capabilities(self, sections, profile, layer):
        signatures = sections["tf_lite_prefill_decode"]["signatures"]
        mode = self.options(profile)["query_mode"]
        layers = set(kv_layers(sections, profile["batch_signature"]))
        if mode == "verify":
            if "verify_64" not in signatures:
                raise ValueError("Qwen verify mode requires a model compiled with the verify_64 signature")
            layers &= set(kv_layers(sections, "verify_64"))
        layers = tuple(sorted(layers))
        if layer not in layers:
            raise ValueError(f"Layer {layer} has no per-token K/V in this Qwen bundle; full-attention layers: {list(layers)}")
        dims = self.dimensions(sections, profile)
        quant = quantization(sections, layer, profile["batch_signature"])
        return Capabilities(**dims, layers=layers, kv_dtype=str(kv_dtype(quant)),
                            kv_width=sum(x["width"] for x in quant))

    def open(self, sections, profile, layer, output, model_hash, **options):
        from ...npu_standalone import StandaloneNpuBackend
        self.capabilities(sections, profile, layer)
        opts = self.options(profile)
        worker = options.pop("native_executable", None) or opts.get("worker")
        if not worker:
            raise ValueError("Set adapter_options.worker to the built Qwen verify worker (see build.py)")
        options.pop("package", None)
        return StandaloneNpuBackend(sections, layer, output, model_hash, native_executable=worker,
                                    device_model=profile["device_model"],
                                    batch_signature=profile["batch_signature"],
                                    mask_token_id=profile["mask_token_id"], package=opts["package"],
                                    qwen_recovery=True, qwen_query_mode=opts["query_mode"], **options)

    def import_observation(self, *args):
        return import_snapshot(*args)

    def save_fixture(self, *args):
        return save_observation(*args)
