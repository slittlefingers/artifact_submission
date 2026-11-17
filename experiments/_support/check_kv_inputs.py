"""Check the supplied recovery cohorts; optionally verify every asset checksum."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / 'assets/kv'
    for cohort in json.loads((root / 'cohorts.json').read_text()):
        files = sorted((root / cohort['path']).glob(cohort['pattern']))
        if len(files) != cohort['count']:
            raise ValueError(f"{cohort['name']}: expected {cohort['count']}, found {len(files)}")
        for path in files:
            with np.load(path, allow_pickle=False) as arrays:
                for name in cohort['required_arrays']:
                    if name not in arrays or not np.isfinite(arrays[name]).all():
                        raise ValueError(f'{path}: invalid {name}')
        print(f"{cohort['name']}: {len(files)} inputs OK", flush=True)
    if args.hashes:
        for entry in json.loads((root / 'manifest.json').read_text()):
            path = root / entry['path']
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != entry['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
        print('All checksums match.')

if __name__ == '__main__':
    main()
