"""Section 5.3: frozen image reconstructors on the paper's recorded K/V cohorts (one image per cohort; --full for all)."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'reconstruction/components'), str(ROOT / 'experiments/_support')]
from image_reconstruction.evaluate import evaluate
from sections import finish, output_root, parser, table

EVALUATIONS = {e['name']: e for e in json.loads((HERE / 'evaluations.json').read_text(encoding='utf-8'))}


def run_one(name, output, limit, device=None, overrides=None):
    config = EVALUATIONS[name]
    paths = {f: (overrides or {}).get(f) or ROOT / config[f] for f in ('cohort', 'checkpoint', 'images')}
    return evaluate(**paths, output=output, split=config.get('split'), limit=limit, device=device,
                    resize=config.get('resize', 'bilinear'), pixels=ROOT / config['pixels'] if 'pixels' in config else None,
                    log=lambda line: None)


def cell(summary, target):
    return f"{summary[target]['psnr']:.2f} / {summary[target]['ssim']:.4f}" if target in summary else '-'


def paper(config):
    values = config.get('paper', {})
    return '; '.join(f"{t} {v['psnr']} / {v['ssim']}" for t, v in values.items()) or 'figure examples'


def main():
    p = parser(__doc__)
    p.add_argument('evaluation', nargs='?', choices=sorted(EVALUATIONS), help='Run only this evaluation')
    p.add_argument('--list', action='store_true', help='List the available evaluations')
    p.add_argument('--limit', type=int, default=1, help='Images per cohort in the demo')
    p.add_argument('--device', help='Torch device; defaults to CUDA when available')
    for field in ('cohort', 'checkpoint', 'images'):
        p.add_argument(f'--{field}', type=Path, help=f'Override the configured {field} path (single evaluation only)')
    args = p.parse_args()
    if args.list:
        for name, e in EVALUATIONS.items():
            print(f"{name:24s} {e['description']}")
        return 0
    out = output_root('5.3', args)
    names = [args.evaluation] if args.evaluation else list(EVALUATIONS)
    overrides = {f: getattr(args, f) for f in ('cohort', 'checkpoint', 'images')} if args.evaluation else None
    rows, passed = [], True
    for name in names:
        print(f'{name} ...', flush=True)
        try:
            summary = run_one(name, out / name, None if args.full else args.limit, args.device, overrides)
        except Exception as error:
            print(f'  failed: {error}', flush=True)
            rows.append((name, '-', '-', '-', 'FAILED'))
            passed = False
            continue
        rows.append((name, summary['count'], cell(summary, 'preprocessed'), cell(summary, 'source'), paper(EVALUATIONS[name])))
    table(('Evaluation', 'Images', 'Preprocessed PSNR/SSIM', 'Source PSNR/SSIM', 'Paper (all images)'), rows)
    print('Metrics are means at 336 x 336; the demo scores only the first image of each cohort.' if not args.full else '')
    return finish('5.3', passed, out)


if __name__ == '__main__':
    sys.exit(main())
