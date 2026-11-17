
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
from ...common import dump
from ...media.boundaries import VBoundaryLocator

CODE = Path(__file__).resolve().parents[3] / 'components'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from image_reconstruction import qwen_vision
from image_reconstruction.checkpoint import Reconstructor
from image_reconstruction.grid_search import QwenFamily, render
from image_reconstruction.storage import Store

TOKEN_LIMITS = [qwen_vision.CAPS[0] // qwen_vision.MERGE ** 2, qwen_vision.CAPS[-1] // qwen_vision.MERGE ** 2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def open_replay(cfg, inspection, output, prefix_ids):
    
    from ...npu_standalone import StandaloneNpuBackend
    backend = StandaloneNpuBackend(inspection["model_sections"], cfg.get("layer", 3), output, inspection["model_sha256"],
                                   native_executable=cfg.get("image_worker", str(QwenFamily.worker)), adb=cfg["adb"],
                                   serial=cfg.get("serial"), package=cfg["package"], device_model=cfg["device_model"],
                                   batch_signature="prefill_128", mask_token_id=0, install=False)
    return backend, QwenFamily(backend, prefix_ids, cfg["tokenizer"])


def end_references(replay, store, names, end_id, folder):
    
    folder.mkdir(parents=True, exist_ok=True)
    values, records = [], []
    for name in names:
        path = folder / Path(name).name
        path.write_bytes(store.read(name))
        with Image.open(path) as image:
            _, patch_grid = qwen_vision.preprocess(image.convert("RGB"))
        tokens = patch_grid[0] * patch_grid[1] // qwen_vision.MERGE ** 2
        rows, grid = replay(path, tokens, suffix=[end_id])
        values.append(rows[-1, rows.shape[1] // 2:])
        records.append(dict(image=name, sha256=store.sha256(name), grid=grid, visual_tokens=tokens))
    return np.stack(values), records


def locate_end(observed, visual_start, references):
    
    locator = VBoundaryLocator(dict(image=references), observed.shape[1] // 2, dict(image=TOKEN_LIMITS), "original")
    span, proposal = locator.locate("image", visual_start - 1, observed, np.arange(len(observed)),
                                    SimpleNamespace(distance_space="original"))
    return span["end"] - span["start"], proposal


def select_grid(replay, reconstructor, observed, visual_start, tokens, folder):
    
    folder.mkdir(parents=True, exist_ok=True)
    target = observed[visual_start:visual_start + tokens].astype(np.float64)
    half = observed.shape[1] // 2
    features = reconstructor.features(target[:, :half].astype(np.float32), target[:, half:].astype(np.float32))
    trials = []
    for candidate in QwenFamily.candidates(tokens):
        h, w = candidate["grid"]
        path = folder / f"{h}x{w}.png"
        render(reconstructor, features, candidate, path)
        rows, grid = replay(path, tokens)
        if grid != candidate["grid"]:
            raise ValueError(f"Candidate {candidate['grid']} replayed as {grid}")
        trials.append(dict(grid=grid, image=path.name, rmse=float(np.sqrt(np.mean((rows - target) ** 2)))))
        dump(folder / "grid_trials.json", dict(tokens=tokens, metric="joint K+V RMSE", candidates=trials))
    if not trials:
        raise ValueError(f"No preprocessing-compatible grid for {tokens} visual tokens")
    return min(trials, key=lambda t: t["rmse"]), trials


def recover_span(replay, reconstructor, observed, visual_start, references, folder):
    
    tokens, proposal = locate_end(observed, visual_start, references)
    best, trials = select_grid(replay, reconstructor, observed, visual_start, tokens, folder / "grids")
    selection = dict(visual_tokens=tokens, grid=best["grid"], rmse=best["rmse"], candidates=len(trials),
                     runner_up=min((t["rmse"] for t in trials if t is not best), default=None), boundary=proposal,
                     image=f"grids/{best['image']}", recorded_span_used=False)
    dump(folder / "selection.json", selection)
    return tokens, best["grid"], folder / "grids" / best["image"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build Qwen end-marker V references from public images on the phone")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Directory or zip of public images outside the recovery cohort")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True, help="New .npz file")
    args = parser.parse_args(argv)
    cfg = read(args.config)
    if args.output.exists():
        raise FileExistsError(f"Refusing overwrite: {args.output}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(cfg["tokenizer"])
    prefix = tokenizer.encode(cfg.get("image_prefix", "<|im_start|>user\n<|vision_start|>"), add_special_tokens=False).ids
    end_id = tokenizer.token_to_id(cfg.get("image_end", "<|vision_end|>"))
    store = Store(args.images)
    names = sorted(n for n in store.names() if Path(n).suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.count]
    work = args.output.with_suffix("")
    backend, replay = open_replay(cfg, read(cfg["inspection"]), work / "worker", prefix)
    with backend:
        values, records = end_references(replay, store, names, end_id, work / "images")
    np.savez_compressed(args.output, image=values)
    dump(args.output.with_suffix(".json"), dict(prefix_ids=prefix, end_id=end_id, layer=cfg.get("layer", 3),
                                                 references=records, use="Boundary proposal only; acceptance remains replay"))
    print(f"Saved {len(values)} end-marker references to {args.output}")


if __name__ == "__main__":
    main()
