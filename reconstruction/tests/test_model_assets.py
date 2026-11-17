
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.adapters import get_adapter
from kvrecon.config import MODEL_CONFIGS, load_model_config
from kvrecon.litertlm import Bundle, FlatReader


@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_inspection_matches_bundle(name):
    cfg = load_model_config(MODEL_CONFIGS[name])
    if not cfg["model"].is_file():
        pytest.skip(f"{cfg['model'].name} not present")
    shipped = get_adapter(cfg["adapter"]).read_manifest(cfg["manifest"])
    with Bundle(cfg["model"]) as bundle:
        assert bundle.model_sections() == shipped
        assert bundle.tokenizers(), "bundle carries no tokenizer"


@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_prepared_assets_have_model_dimensions(name):
    cfg = load_model_config(MODEL_CONFIGS[name])
    if not cfg["embeddings"].is_file():
        pytest.skip("run reconstruction/run.py prepare-assets first")
    adapter = get_adapter(cfg["adapter"])
    dims = adapter.dimensions(adapter.read_manifest(cfg["manifest"]), cfg)
    table = np.load(cfg["embeddings"], mmap_mode="r")
    assert table.shape == (dims["vocab_size"], dims["embedding_dim"]) and table.dtype == np.float32


def test_flat_reader_rejects_implausible_vector(tmp_path):
    data = bytearray(64)                                                         
    data[0:2], data[2:4] = (6).to_bytes(2, "little"), (8).to_bytes(2, "little")
    data[4:6] = (4).to_bytes(2, "little")                        
    data[8:12] = (8).to_bytes(4, "little", signed=True)
    data[12:16] = (4).to_bytes(4, "little")                
    data[16:20] = (2_000_000).to_bytes(4, "little")
    with pytest.raises(ValueError):
        FlatReader(bytes(data)).vector(8, 0)


def test_bundle_rejects_non_litertlm(tmp_path):
    path = tmp_path / "x.litertlm"
    path.write_bytes(b"NOTLITERTLM" + bytes(64))
    with pytest.raises(ValueError):
        with Bundle(path):
            pass
