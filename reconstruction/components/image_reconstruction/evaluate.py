"""Evaluate a frozen image reconstructor on one recorded K/V cohort."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .capture import gemma_visual_grid, recorded_grid, visual_kv
from .checkpoint import Reconstructor
from .cohort import Cohort
from .metrics import EVALUATION_SIZE, VIEWS, score, to_image

METRICS = ('psnr', 'ssim', 'mse')


def evaluate(cohort, checkpoint, images, output, split=None, limit=None, device=None, resize='bilinear', pixels=None, log=print):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    view = VIEWS[resize]
    reconstructor = Reconstructor(checkpoint, device)
    data = Cohort(cohort, images)
    companion = Cohort(pixels, images) if pixels else None
    samples = data.samples(split, limit)
    (output / 'samples').mkdir(parents=True, exist_ok=True)
    _write(output / 'protocol.json', dict(
        cohort=str(data.root), images=str(data.images.path), pixels=str(companion.root) if companion else None, split=split, count=len(samples),
        reconstructor=reconstructor.describe(),
        inputs='Observed visual K/V rows in stored dtype, dequantized and standardized with the checkpoint package; no original pixels',
        grid='Recorded capture grid, otherwise derived from the source image size by the deployed preprocessing rule',
        targets=dict(source='EXIF-oriented original RGB', preprocessed='Model-specific preprocessed RGB stored with the capture, or with the same sample in the pixels cohort'),
        metrics=f'Mean per-image PSNR/SSIM/MSE on RGB in [0, 1] at {EVALUATION_SIZE}x{EVALUATION_SIZE}; SSIM 11x11 Gaussian sigma 1.5, valid window',
        resize=view.name))
    results, started = [], time.perf_counter()
    with (output / 'metrics.jsonl').open('w', encoding='utf-8') as stream:
        for sample in samples:
            result = _evaluate_sample(reconstructor, view, data, sample, output / 'samples' / sample.id, companion)
            results.append(result)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            log(f"{len(results)}/{len(samples)} {sample.id} " + ' '.join(f"{t}:{result[t]['psnr']:.3f}dB/{result[t]['ssim']:.4f}" for t in _targets(result)))
    summary = dict(count=len(results), seconds=time.perf_counter() - started,
                   **{t: {m: float(np.mean([r[t][m] for r in results])) for m in METRICS} for t in _targets(results[0])})
    _write(output / 'summary.json', summary)
    return summary


def _evaluate_sample(reconstructor, view, data, sample, folder, companion=None):
    source, image_name = data.source_image(sample)
    with np.load(sample.capture, allow_pickle=False) as arrays:
        grid = recorded_grid(arrays, sample.metadata) or gemma_visual_grid(source.height, source.width)
        keys, values = visual_kv(arrays, sample.metadata, grid[0] * grid[1])
        features = reconstructor.features(keys, values)
        preprocessed = arrays['pixels'].copy() if 'pixels' in arrays else None
    if preprocessed is None and companion is not None:
        preprocessed = _companion_pixels(companion, sample.id, grid, reconstructor.model.pixels_per_token)
    native = reconstructor.reconstruct(features, grid)
    prediction = view.prediction(native)
    folder.mkdir(parents=True, exist_ok=True)
    to_image(native).save(folder / 'recovered_native.png')
    to_image(prediction).save(folder / f'recovered{EVALUATION_SIZE}.png')
    result = dict(id=sample.id, image=image_name, capture=sample.capture.name, grid_hw=list(grid), visual_tokens=len(features))
    if preprocessed is not None:
        result['preprocessed'] = score(prediction, view.target(preprocessed))
    result['source'] = score(prediction, view.target(source))
    return result


def _companion_pixels(companion, sample_id, grid, pixels_per_token):
    
    capture = companion.capture(sample_id)
    if capture is None:
        raise FileNotFoundError(f'{sample_id} is missing from {companion.root}')
    with np.load(capture, allow_pickle=False) as arrays:
        pixels = arrays['pixels'].copy()
    if pixels.shape[:2] != (grid[0] * pixels_per_token, grid[1] * pixels_per_token):
        raise ValueError(f'{sample_id}: pixels {pixels.shape[:2]} do not match grid {grid}')
    return pixels


def _targets(result):
    return [t for t in ('preprocessed', 'source') if t in result]


def _write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True, help='Directory containing samples/<id>/layerN.npz')
    p.add_argument('--checkpoint', type=Path, required=True, help='Reconstructor directory or release zip')
    p.add_argument('--images', type=Path, required=True, help='Original-image directory or release zip')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--split', help='Keep only samples whose sample.json split matches, e.g. test')
    p.add_argument('--limit', type=int, help='Evaluate only the first N selected samples')
    p.add_argument('--device', help='Torch device; defaults to CUDA when available')
    p.add_argument('--resize', choices=sorted(VIEWS), default='bilinear', help='Resize protocol for the 336 x 336 comparison')
    p.add_argument('--pixels', type=Path, help='Cohort of the same model that stores preprocessed pixels for captures without them')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    summary = evaluate(args.cohort, args.checkpoint, args.images, args.output, args.split, args.limit, args.device, args.resize, args.pixels)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
