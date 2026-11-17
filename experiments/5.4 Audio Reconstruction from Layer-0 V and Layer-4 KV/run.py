"""Section 5.4: audio reconstruction from Gemma 4 E2B layer-0 V and layer-4 KV (one utterance each; --full for 1,000)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/_support'))
from sections import ROOT, finish, output_root, parser, python, read, run, table

SCRIPT = ROOT / 'reconstruction/components/experiment3_audio/audio_recovery/evaluate_token_duration.py'
PAPER = {0: (0.1795, 0.7706), 4: (0.2008, 0.8301)}


def main():
    p = parser(__doc__)
    p.add_argument('--limit', type=int, default=1, help='Utterances per layer in the demo')
    args = p.parse_args()
    out = output_root('5.4', args)
    rows, passed = [], True
    for layer in (0, 4):
        folder = out / f'layer{layer}'
        code = run(python(SCRIPT, '--layer', layer, '--output', folder, *([] if args.full else ['--limit', args.limit])),
                   out / f'layer{layer}.log')
        if code:
            rows.append((layer, '-', '-', '-', 'FAILED'))
            passed = False
            continue
        result = read(folder / 'results.json')
        recovered = next(v for k, v in result['spectral_summary'].items() if k.endswith('_recovered'))
        rows.append((layer, result['utterances'], f"{recovered['log_magnitude_rmse']:.4f}", f"{recovered['spectral_convergence']:.4f}",
                     '{} / {}'.format(*PAPER[layer])))
    table(('Layer', 'Utterances', 'Log-magnitude RMSE', 'Spectral convergence', 'Paper RMSE / SC (1,000)'), rows)
    return finish('5.4', passed, out)


if __name__ == '__main__':
    sys.exit(main())
