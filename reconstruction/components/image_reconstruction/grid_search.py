"""Recover the unknown visual grid: render every legal grid from the K/V, replay it on the phone, keep the closest."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile

import numpy as np
from PIL import Image
from torch.nn import functional as F

from . import qwen_vision
from .capture import GEMMA_PIXEL_BUDGET, GEMMA_PIXELS_PER_TOKEN, gemma_visual_grid, recorded_grid, visual_kv
from .checkpoint import Reconstructor
from .cohort import Cohort
from .metrics import to_image

HERE = Path(__file__).resolve().parent
RECONSTRUCTION = HERE.parent.parent
METRICS = ('k', 'v', 'kv')


class GemmaFamily:
    

    upload_limit, upload_chunk = 900_000, 512_000

    def __init__(self, backend, prefix_ids):
        self.backend, self.prefix_ids = backend, list(prefix_ids)

    @staticmethod
    def open(args, sections, layer, model_sha256, output):
        sys.path.insert(0, str(RECONSTRUCTION))
        from kvrecon.npu import NpuBackend
        return NpuBackend(sections, layer, output, model_sha256, adb=args.adb, serial=args.serial, package=args.package,
                          device_model=args.device_model, install=not args.no_install)

    @staticmethod
    def replay_size(h, w):
        
        if gemma_visual_grid(h * GEMMA_PIXELS_PER_TOKEN, w * GEMMA_PIXELS_PER_TOKEN) == (h, w):
            return h * GEMMA_PIXELS_PER_TOKEN, w * GEMMA_PIXELS_PER_TOKEN
        low, high = max(w * w / 280, 280 / (h + 1) ** 2), min((w + 1) ** 2 / 280, 280 / h ** 2)
        if low >= high:
            return None
        ratio = math.sqrt(low * high)
        for scale in (1, 2, 4, 8):
            height = round(math.sqrt(GEMMA_PIXEL_BUDGET / ratio) * scale)
            width = round(height * ratio)
            if gemma_visual_grid(height, width) == (h, w):
                return height, width
        return None

    @classmethod
    def candidates(cls, tokens):
        
        return [dict(grid=[h, tokens // h], input_hw=list(size)) for h in range(1, tokens + 1)
                if tokens % h == 0 and (size := cls.replay_size(h, tokens // h)) is not None]

    def __call__(self, path, tokens):
        b = self.backend
        blob = Path(path).read_bytes()
        b.reset(self.prefix_ids)
        if len(blob) > self.upload_limit:
            for offset in range(0, len(blob), self.upload_chunk):
                b.rpc('upload_collection_image', offset=offset, data_hex=blob[offset:offset + self.upload_chunk].hex())
            args = {}
        else:
            args = dict(image_hex=blob.hex())
        raw = b.rpc('collect_image', include_pixels=True, collect_kv=True, training_acquisition=True, position=b.position, **args)
        call = b.calls[-1]
        if call['image_rows'] != tokens:
            raise ValueError(f"Device produced {call['image_rows']} visual tokens, expected {tokens}")
                                                                                                             
        offset = tokens * (768 + 1536) * 4 + call['resized_height'] * call['resized_width'] * 3
        kv = np.frombuffer(raw[offset:], dtype=np.int8).reshape(tokens, -1).copy()
        return kv, [call['resized_height'] // GEMMA_PIXELS_PER_TOKEN, call['resized_width'] // GEMMA_PIXELS_PER_TOKEN]


class QwenFamily:
    

    worker = HERE / 'native/qwen_image_worker'
    image_pad = '<|image_pad|>'

    def __init__(self, backend, prefix_ids, tokenizer):
        from tokenizers import Tokenizer
        self.backend, self.prefix_ids = backend, list(prefix_ids)
        self.image_pad_id = Tokenizer.from_file(str(tokenizer)).token_to_id(self.image_pad)
        if self.image_pad_id is None:
            raise ValueError(f'{self.image_pad} is not in {tokenizer}')
        self.packet = Path(tempfile.mkdtemp()) / 'vision_inputs.bin'

    @classmethod
    def open(cls, args, sections, layer, model_sha256, output):
        sys.path.insert(0, str(RECONSTRUCTION))
        from kvrecon.npu_standalone import StandaloneNpuBackend
        backend = StandaloneNpuBackend(sections, layer, output, model_sha256, native_executable=cls.worker, adb=args.adb,
                                       serial=args.serial, install=False, package=args.package, device_model=args.device_model,
                                       batch_signature='prefill_128', mask_token_id=0)
        return backend

    @staticmethod
    def candidates(tokens):
        
        side = qwen_vision.FACTOR
        grids = []
        for h in range(1, tokens + 1):
            w = tokens // h
            if tokens % h or qwen_vision.target_size(w * side, h * side) != (w * side, h * side):
                continue
            qwen_vision.pick_cap(tokens * qwen_vision.MERGE ** 2)
            grids.append(dict(grid=[h, w], input_hw=[h * side, w * side]))
        return grids

    def __call__(self, path, tokens, suffix=()):
        b = self.backend
        with Image.open(path) as opened:
            patches, patch_grid = qwen_vision.preprocess(opened.convert('RGB'))
        signature, inputs, count = qwen_vision.encoder_inputs(patches, patch_grid)
        if count != tokens:
            raise ValueError(f'Preprocessing produced {count} visual tokens, expected {tokens}')
        grid = (patch_grid[0] // qwen_vision.MERGE, patch_grid[1] // qwen_vision.MERGE)
        ids = self.prefix_ids + [self.image_pad_id] * count + list(suffix)
        rope = qwen_vision.mrope_positions(ids, self.image_pad_id, grid)
        with self.packet.open('wb') as stream:
            for key in ('patches', 'pos_idx', 'pos_w', 'positions', 'valid'):
                stream.write(inputs[key].tobytes())
        remote = b.remote + '/vision_inputs.bin'
        b.command('push', str(self.packet), remote)
        blob = b.rpc('collect_image', vision_signature=signature, input_path=remote, cap=len(inputs['valid']), visual_tokens=count,
                     ids=ids, rope_pos=rope.tolist(), image_start=len(self.prefix_ids), image_pad_id=self.image_pad_id,
                     include_embeddings=False)
        call = b.calls[-1]
        if call['position'] != len(ids) or call['kv_bytes'] != len(ids) * 1024 * 4:
            raise ValueError('Unexpected Qwen image acquisition geometry')
        active = np.frombuffer(blob[:call['kv_bytes']], dtype='<f4').reshape(len(ids), -1).copy()
        return active[len(self.prefix_ids):], list(grid)


FAMILIES = {'gemma': GemmaFamily, 'qwen': QwenFamily}


def render(reconstructor, features, candidate, path):
    
    native = reconstructor.reconstruct(features, candidate['grid'])
    image = F.interpolate(native, size=tuple(candidate['input_hw']), mode='bilinear', align_corners=False, antialias=True)
    to_image(image).save(path)


def rank(target, replayed, true_grid):
    
    half = target.shape[1] // 2
    parts = dict(k=slice(0, half), v=slice(half, None), kv=slice(None))
    distances = []
    for grid, kv in replayed:
        diff = kv.astype(np.float64) - target.astype(np.float64)
        distances.append(dict(grid=grid, **{m: float(np.sqrt(np.mean(diff[:, s] ** 2))) for m, s in parts.items()}))
    truth = next(d for d in distances if d['grid'] == list(true_grid))
    rankings = {}
    for m in METRICS:
        best = min(distances, key=lambda d: d[m])
        rankings[m] = dict(rank=1 + sum(d[m] < truth[m] for d in distances),
                           unique_correct=sum(d[m] <= truth[m] for d in distances) == 1, best_grid=best['grid'])
    return rankings, distances


def search(cohort, checkpoint, images, output, replay, limit=None, device=None, log=print):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    reconstructor = Reconstructor(checkpoint, device)
    data = Cohort(cohort, images)
    results = []
    for sample in data.samples(limit=limit):
        folder = output / 'samples' / sample.id
        folder.mkdir(parents=True, exist_ok=True)
        source, image_name = data.source_image(sample)
        with np.load(sample.capture, allow_pickle=False) as arrays:
            true_grid = recorded_grid(arrays, sample.metadata) or gemma_visual_grid(source.height, source.width)
            keys, values = visual_kv(arrays, sample.metadata, true_grid[0] * true_grid[1])
        target, tokens = np.concatenate([keys, values], axis=1), len(keys)
        original = folder / ('original' + PurePosixPath(image_name).suffix)
        original.write_bytes(data.images.read(image_name))
        control, control_grid = replay(original, tokens)
        if not np.array_equal(control, target) or control_grid != list(true_grid):
            raise ValueError(f'{sample.id}: replaying the original image does not reproduce the observed K/V')
        features = reconstructor.features(keys, values)
        replayed = []
        for candidate in type(replay).candidates(tokens):
            path = folder / f"candidate_{candidate['grid'][0]}x{candidate['grid'][1]}.png"
            render(reconstructor, features, candidate, path)
            kv, grid = replay(path, tokens)
            if grid != candidate['grid']:
                raise ValueError(f"{sample.id}: candidate {candidate['grid']} replayed as {grid}")
            replayed.append((grid, kv))
        rankings, distances = rank(target, replayed, true_grid)
        result = dict(id=sample.id, image=image_name, true_grid=list(true_grid), candidates=len(distances),
                      control_bit_exact=True, rankings=rankings, distances=distances)
        (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        log(f"{len(results)} {sample.id} true {list(true_grid)} " + ' '.join(f"{m}:rank{r['rank']}" for m, r in rankings.items()))
    summary = dict(count=len(results), candidates=sum(r['candidates'] for r in results),
                   **{m: dict(unique_correct=sum(r['rankings'][m]['unique_correct'] for r in results),
                              mean_rank=float(np.mean([r['rankings'][m]['rank'] for r in results]))) for m in METRICS})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=sorted(FAMILIES), required=True, help='Deployed model family of the cohort')
    p.add_argument('--cohort', type=Path, required=True, help='K/V cohort directory')
    p.add_argument('--checkpoint', type=Path, required=True, help='Reconstructor directory or release zip')
    p.add_argument('--images', type=Path, required=True, help='Original-image directory or release zip')
    p.add_argument('--inspection', type=Path, required=True, help='Inspection JSON of the deployed model')
    p.add_argument('--tokenizer', type=Path, help='Qwen tokenizer.json, used to look up the image-pad token')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--limit', type=int, help='Search only the first N samples')
    p.add_argument('--device', help='Torch device for the reconstructor')
    p.add_argument('--adb', default='adb')
    p.add_argument('--serial')
    p.add_argument('--package', default='com.example.annoymous')
    p.add_argument('--device-model', help='Absolute model path on the phone; defaults to the app files directory')
    p.add_argument('--no-install', action='store_true', help='Require the Gemma test APK to be installed already')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    inspection = json.loads(args.inspection.read_text(encoding='utf-8'))
    args.device_model = args.device_model or f"/sdcard/Android/data/{args.package}/files/{PurePosixPath(inspection['model_file']).name}"
    first = Cohort(args.cohort, args.images).samples(limit=1)[0]
    if first.metadata['model_sha256'] != inspection['model_sha256']:
        raise ValueError('Cohort was captured with a different model than the inspection describes')
    layer = first.metadata.get('layer_index', first.metadata.get('layer'))
    family = FAMILIES[args.model]
    workers = args.output.with_name(args.output.name + '_device')
    with family.open(args, inspection['model_sections'], layer, inspection['model_sha256'], workers) as backend:
        extra = (args.tokenizer or HERE.parents[2] / 'models/derived/qwen3_5_2b/tokenizer.json',) if family is QwenFamily else ()
        replay = family(backend, first.metadata['prefix_ids'], *extra)
        summary = search(args.cohort, args.checkpoint, args.images, args.output, replay, args.limit, args.device)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
