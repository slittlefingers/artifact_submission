
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.adapters import ADAPTERS, get_adapter, LiteRtTensorAdapter, QwenVerifyAdapter
from kvrecon.config import MODEL_CONFIGS, load_model_config

VOCAB, EMB, CTX, KV = 248320, 2048, 2048, 256


def tensor(name, shape, kind=0):
    return dict(name=name, shape=shape, type=kind, scale=[], zero_point=[])


def qwen_sections(verify=True, layers=(3, 7)):
    def kv(prefix):
        rows = []
        for layer in layers:
            rows += [tensor(f"{prefix}_k_{layer}", [1, 2, CTX, KV]), tensor(f"{prefix}_v_{layer}", [1, 2, KV, CTX])]
        return rows + [tensor("kv_cache_c_0", [1, 3, 6144]), tensor("kv_cache_r_0", [1, 16, 128, 128])]
    sig = {
        "decode": dict(inputs=[tensor("embeddings", [1, 1, EMB])] + kv("kv_cache"),
                       outputs=[tensor("logits", [1, 1, VOCAB])] + kv("kv_cache")),
        "prefill_128": dict(inputs=[tensor("embeddings", [1, 128, EMB])] + kv("kv_cache"), outputs=kv("kv_cache")),
    }
    if verify:
        sig["verify_64"] = dict(inputs=[tensor("embeddings", [1, 64, EMB]), tensor("tree", [64, 64])] + kv("kv_cache"),
                                outputs=[tensor("logits", [1, 64, VOCAB])] + kv("kv_cache"))
    return {"tf_lite_prefill_decode": {"signatures": sig}}


def profile(**options):
    return dict(batch_signature="prefill_128", mask_token_id=0, default_layer=3,
                device_model="/sdcard/Android/data/com.example.annoymous/files/qwen.litertlm",
                adapter="litert_qwen_verify", adapter_options=options)


def test_registry_lists_both_adapters():
    assert set(ADAPTERS) >= {"litert_tensor_gemma", "litert_qwen_verify"}
    assert isinstance(get_adapter("litert_tensor_gemma"), LiteRtTensorAdapter)
    assert isinstance(get_adapter("litert_qwen_verify"), QwenVerifyAdapter)


def test_qwen_capabilities_use_full_attention_layers():
    caps = get_adapter("litert_qwen_verify").capabilities(qwen_sections(), profile(), 3)
    assert caps.layers == (3, 7)
    assert caps.vocab_size == VOCAB and caps.embedding_dim == EMB
    assert caps.batch_capacity == 128 and caps.context_capacity == CTX
    assert caps.kv_dtype == "float32" and caps.kv_width == 2 * 2 * KV


def test_qwen_rejects_linear_layer_and_missing_verify():
    adapter = get_adapter("litert_qwen_verify")
    with pytest.raises(ValueError, match="full-attention"):
        adapter.capabilities(qwen_sections(), profile(), 0)
    with pytest.raises(ValueError, match="verify_64"):
        adapter.capabilities(qwen_sections(verify=False), profile(), 3)
                                                 
    assert adapter.capabilities(qwen_sections(verify=False), profile(query_mode="decode"), 3).layers == (3, 7)
    with pytest.raises(ValueError, match="query_mode"):
        adapter.capabilities(qwen_sections(), profile(query_mode="bogus"), 3)


def test_qwen_open_requires_worker_and_sets_qwen_backend(tmp_path):
    adapter = get_adapter("litert_qwen_verify")
    with pytest.raises(ValueError, match="worker"):
        adapter.open(qwen_sections(), profile(), 3, tmp_path / "a", "0" * 64)
    worker = tmp_path / "qwen_verify_worker"
    worker.write_bytes(b"\x7fELF")
    backend = adapter.open(qwen_sections(), profile(worker=str(worker), package="com.example.test"), 3,
                           tmp_path / "b", "0" * 64, adb="adb", serial=None, install=False)
    assert backend.qwen_recovery and backend.qwen_query_mode == "verify"
    assert backend.package == "com.example.test" and backend.native_executable == worker.resolve()
    assert backend.width == 2 * 2 * KV and backend.mask_token_id == 0
    assert backend.proc is None                                  


def test_bundled_model_configs_load_and_name_registered_adapters():
    for name, path in MODEL_CONFIGS.items():
        cfg = load_model_config(path)
        assert cfg["adapter"] in ADAPTERS, name
    qwen = load_model_config(MODEL_CONFIGS["qwen3.5-2b"])
    assert qwen["adapter"] == "litert_qwen_verify" and qwen["default_layer"] == 3 and qwen["mask_token_id"] == 0
    QwenVerifyAdapter.options(qwen)


def test_bundled_qwen_inspection_matches_shipped_verify_bundle():
    cfg = load_model_config(MODEL_CONFIGS["qwen3.5-2b"])
    adapter = get_adapter(cfg["adapter"])
    manifest = adapter.read_manifest(cfg["manifest"])
    assert cfg["model"].name == "qwen3_5_2b_vl_spec_Google_Tensor_G5.litertlm"
    assert {"verify_64", "prefill_128"} <= set(manifest["tf_lite_prefill_decode"]["signatures"])
    assert "tf_lite_vision_encoder" in manifest
    caps = adapter.capabilities(manifest, cfg, cfg["default_layer"])
    assert caps.layers == (3, 7, 11, 15, 19, 23) and caps.kv_width == 1024 and caps.batch_capacity == 128


def test_qwen_read_manifest(tmp_path):
    path = tmp_path / "inspection.json"
    path.write_text(json.dumps({"model_sha256": "0" * 64, "model_sections": qwen_sections()}), encoding="utf-8")
    assert "verify_64" in QwenVerifyAdapter().read_manifest(path)["tf_lite_prefill_decode"]["signatures"]
