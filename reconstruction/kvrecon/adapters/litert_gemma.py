
import json
from ..backend import Capabilities
from ..config import model_dimensions
from ..common import quantization, kv_dtype
from ..observation import import_snapshot, save_observation


class LiteRtTensorAdapter:
    
    def read_manifest(self, path):
        return json.loads(path.read_text())["model_sections"]

    def dimensions(self, sections, profile):
        return model_dimensions(sections, profile["batch_signature"])

    def capabilities(self, sections, profile, layer):
        dims = self.dimensions(sections, profile)
        outputs = sections["tf_lite_prefill_decode"]["signatures"][profile["batch_signature"]]["outputs"]
        names = {x["name"] for x in outputs}
        layers = tuple(sorted(int(n.removeprefix("kv_slice_k_")) for n in names
                              if n.startswith("kv_slice_k_") and n.replace("kv_slice_k_", "kv_slice_v_", 1) in names))
        if layer not in layers:
            raise ValueError("Selected layer is unavailable in this adapter")
        quant = quantization(sections, layer, profile["batch_signature"])
        return Capabilities(**dims, layers=layers, kv_dtype=str(kv_dtype(quant)),
                            kv_width=sum(x["width"] for x in quant))

    def open(self, sections, profile, layer, output, model_hash, **options):
        from ..npu import NpuBackend
        self.capabilities(sections, profile, layer)
        return NpuBackend(sections, layer, output, model_hash,
                          device_model=profile["device_model"], batch_signature=profile["batch_signature"],
                          mask_token_id=profile["mask_token_id"], **options)

    def import_observation(self, *args):
        return import_snapshot(*args)

    def save_fixture(self, *args):
        return save_observation(*args)

    def first_v_spec(self, sections, profile):
                                                                                      
                                                                                      
        return dict(layer=0,quantization=quantization(sections,0,profile["batch_signature"]),
                    prefix=[2],context_independent_by_architecture=True,
                    scope="First text-token V of supported Gemma split graphs; not multimodal embeddings")
