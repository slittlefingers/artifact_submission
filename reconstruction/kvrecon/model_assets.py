




import json
import os
from pathlib import Path

import numpy as np

from .common import digest
from .litertlm import Bundle


def _replace(tmp, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, path)


def check_tokenizer_ids(tokenizer_path, spm_bytes):
    
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import Tokenizer
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(spm_bytes)
    vocab = Tokenizer.from_file(str(tokenizer_path)).get_vocab(with_added_tokens=True)
    wrong = [i for i, piece in enumerate(proto.pieces) if vocab.get(piece.piece) != i]
    if wrong:
        raise ValueError(f"{tokenizer_path} disagrees with the bundled SentencePiece model on {len(wrong)} IDs, e.g. {wrong[:5]}")
    return len(proto.pieces)


def export_embeddings(bundle, sections, dims, output, threads=4):
    from ai_edge_litert.interpreter import Interpreter
    section = sections.get("tf_lite_embedder")
    if not section:
        raise ValueError("Bundle has no tf_lite_embedder section")
    blob = bundle.read(section["offset"], section["length"])
    interpreter = Interpreter(model_content=blob, num_threads=threads)
    signatures = interpreter.get_signature_list()
    batched = [s for s in signatures if s.startswith("prefill_embedder_") and s.rsplit("_", 1)[1].isdigit()]
    if not batched:
        raise ValueError(f"No prefill_embedder_N signature among {sorted(signatures)}")
    name = max(batched, key=lambda s: int(s.rsplit("_", 1)[1]))
    width = int(name.rsplit("_", 1)[1])
    (inp,), (out,) = signatures[name]["inputs"], signatures[name]["outputs"]
    runner = interpreter.get_signature_runner(name)
    vocab, dim = dims["vocab_size"], dims["embedding_dim"]
    tmp = output.with_name(output.name + ".partial.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = np.lib.format.open_memmap(tmp, mode="w+", dtype="<f4", shape=(vocab, dim))
    for start in range(0, vocab, width):
        ids = np.arange(start, start + width, dtype=np.int32)
        ids[ids >= vocab] = 0                                                                 
        result = runner(**{inp: ids[None, :]})[out]
        if result.shape != (1, width, dim) or not np.isfinite(result).all():
            raise ValueError(f"Embedder returned {result.shape}; expected (1, {width}, {dim}) finite values")
        stop = min(start + width, vocab)
        table[start:stop] = result[0, :stop - start]
    table.flush()
    del table
    _replace(tmp, output)
    return dict(signature=name, section_offset=section["offset"], section_length=section["length"],
                table_shape=[vocab, dim], dtype="float32")


def prepare_assets(profile, adapter, threads=4, log=print):
    
    model, manifest = Path(profile["model"]), Path(profile["manifest"])
    if not model.is_file():
        raise FileNotFoundError(f"Model bundle missing: {model}")
    meta = json.loads(manifest.read_text(encoding="utf-8-sig"))
    sections = adapter.read_manifest(manifest)
    expected = meta.get("model_sha256") if isinstance(meta, dict) else None
    model_hash = digest(model)
    if expected and expected != model_hash:
        raise ValueError(f"{model.name} sha256 {model_hash} differs from the inspection ({expected})")
    dims = adapter.dimensions(sections, profile)
    report = dict(model=model.name, model_sha256=model_hash, vocab_size=dims["vocab_size"], embedding_dim=dims["embedding_dim"])
    with Bundle(model) as bundle:
        if bundle.model_sections() != sections:
            raise ValueError(f"{manifest.name} does not describe {model.name}")
        bundled = bundle.tokenizers()
        tokenizer = Path(profile["tokenizer"])
        if not tokenizer.exists():
            if "hf_json" not in bundled:
                raise FileNotFoundError(f"{tokenizer} is missing and {model.name} bundles no HF tokenizer; supply tokenizer.json")
            tmp = tokenizer.with_name(tokenizer.name + ".partial")
            tokenizer.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(bundled["hf_json"])
            _replace(tmp, tokenizer)
            report["tokenizer"] = "extracted from bundle"
            log(f"Tokenizer extracted: {tokenizer}")
        elif "hf_json" in bundled:
            report["tokenizer"] = "matches bundle" if tokenizer.read_bytes() == bundled["hf_json"] else "differs from bundle"
        else:
            report["tokenizer"] = "existing file"
        if tokenizer.suffix == ".json" and "sentencepiece" in bundled:
            report["sentencepiece_ids_checked"] = check_tokenizer_ids(tokenizer, bundled["sentencepiece"])
            log(f"Tokenizer IDs match the bundled SentencePiece model ({report['sentencepiece_ids_checked']} pieces)")
        report["tokenizer_sha256"] = digest(tokenizer)
        embeddings = Path(profile["embeddings"])
        if embeddings.exists():
            table = np.load(embeddings, mmap_mode="r")
            if table.shape != (dims["vocab_size"], dims["embedding_dim"]):
                raise ValueError(f"{embeddings} has shape {table.shape}; expected {(dims['vocab_size'], dims['embedding_dim'])}")
            report["embeddings"] = "existing file"
        else:
            log(f"Exporting {dims['vocab_size']} x {dims['embedding_dim']} embeddings from the bundled embedder ...")
            report["embeddings"] = export_embeddings(bundle, sections, dims, embeddings, threads)
            log(f"Embeddings saved: {embeddings}")
        report["embeddings_sha256"] = digest(embeddings)
    provenance = embeddings.with_name("assets_provenance.json")
    provenance.write_text(json.dumps(report, indent=2) + "\n")
    return report
