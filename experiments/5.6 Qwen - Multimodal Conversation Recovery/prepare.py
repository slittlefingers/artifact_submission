"""Convert collected layer-3 captures to recovery inputs; token IDs and image span stay in a separate evaluation folder."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New folder; receives observations/ and evaluation/')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'models/derived/qwen3_5_2b/tokenizer.json')
    parser.add_argument('--template', default='<|im_start|>user\n<|vision_start|>', help='Public text before the image')
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prefix = tokenizer.encode(args.template, add_special_tokens=False).ids
    pad, end = tokenizer.token_to_id('<|image_pad|>'), tokenizer.token_to_id('<|vision_end|>')
    for source in sorted(args.collection.glob('*/layer3.npz')):
        with np.load(source, allow_pickle=False) as z:
            kv = np.concatenate([z['k'], z['v']], axis=1).astype('<f4')
            grid = z['grid'].astype(int).tolist()
        text_start = len(prefix) + grid[0] * grid[1] + 1
        if len(grid) != 2 or min(grid) < 1 or not np.isfinite(kv).all() or len(kv) <= text_start:
            raise ValueError(f'Invalid capture: {source}')
        sample = source.parent.name
        observation = args.output / 'observations' / sample
        observation.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(observation / 'observed.npz', kv=kv)
        (observation / 'metadata.json').write_text(json.dumps(dict(
            id=sample, public_prefix=prefix, image_pad_id=pad, image_end_id=end,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), indent=2) + '\n')
        labels = args.output / 'evaluation'
        labels.mkdir(parents=True, exist_ok=True)
        (labels / f'{sample}.json').write_text(json.dumps(dict(
            id=sample, visual_tokens=grid[0] * grid[1], grid=grid, text_start=text_start), indent=2) + '\n')


if __name__ == '__main__':
    main()
