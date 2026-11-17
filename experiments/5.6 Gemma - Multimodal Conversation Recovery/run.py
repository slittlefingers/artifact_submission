"""Section 5.6 (Gemma): multimodal OmniDialog conversation recovery on the phone NPU (one dialogue, one token; --full for all 50 complete)."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'experiments/_support'))
from sections import ROOT, finish, output_root, parser, percent, python, read, require_phone, run, score_tokens, table

DATA = ROOT / 'assets/kv/5.6_gemma/omnidialog'
SCRIPT = ROOT / 'reconstruction/components/experiment9/run.py'
MODEL = 'gemma-4-E2B-it_Google_Tensor_G5.litertlm'


def main():
    p = parser(__doc__)
    p.add_argument('--dialogues', type=int, default=1, help='Dialogues in the demo')
    p.add_argument('--start', type=int, default=1, help='First dialogue in manifest order (one-based)')
    p.add_argument('--positions', type=int, default=1, help='Recovered positions per dialogue in the demo')
    args = p.parse_args()
    require_phone(MODEL)
    samples = read(DATA / 'manifest.json')['samples']
    if not 1 <= args.start <= len(samples) or args.dialogues < 1:
        p.error('--start must select an existing dialogue and --dialogues must be positive')
    first = 1 if args.full else args.start
    samples = samples if args.full else samples[first-1:first-1+args.dialogues]
    out = output_root('5.6_gemma', args)
    rows, passed = [], True
    for number, sample in enumerate(samples, first):
        folder = out / sample['id']
        limits = [] if args.full else ['--max-positions', args.positions]
        code = run(python(SCRIPT, '--sample-number', number, '--output', folder, *limits), out / f"{sample['id']}.log")
        if code:
            rows.append((sample['id'], '-', '-', '-', 'FAILED'))
            passed = False
            continue
        result = read(folder / 'recovery/result.json')
        labels = read(DATA / 'evaluation' / sample['id'] / f"turn{sample['turns']}_tokens.json")['all_runtime_ids']
        recovered, correct = score_tokens(result, dict(enumerate(labels)))
        ok = recovered > 0 and correct == recovered
        passed &= ok
        rows.append((sample['id'], recovered, percent(correct, recovered), len(result.get('media_segments', [])), 'ok' if ok else 'FAILED'))
    table(('Dialogue', 'Text tokens', 'Accuracy', 'Media spans', 'Status'), rows)
    print('Paper (50 dialogues, complete): 100% text-token accuracy and all 100 media spans identified exactly.')
    return finish('5.6 Gemma', passed, out)


if __name__ == '__main__':
    sys.exit(main())
