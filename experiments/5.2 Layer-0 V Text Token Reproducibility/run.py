"""Section 5.2: layer-0 V token recovery for Gemma 4 E2B, Llama 3.2 1B and Ministral 3 3B (one prompt each; --full for 1,000)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/_support'))
from sections import ROOT, finish, output_root, parser, percent, python, read, run, table

MODELS = ('gemma4_e2b', 'llama3_2_1b', 'ministral3_3b')
SCRIPT = ROOT / 'reconstruction/components/experiment1_layer0/evaluate_generic_vocabulary.py'


def main():
    p = parser(__doc__)
    p.add_argument('--limit', type=int, default=1, help='Prompts per model in the demo')
    args = p.parse_args()
    out = output_root('5.2', args)
    rows, passed = [], True
    for model in MODELS:
        code = run(python(SCRIPT, '--model', model, '--output', out / model, *([] if args.full else ['--limit', args.limit])),
                   out / f'{model}.log')
        if code:
            rows.append((model, '-', '-', '-', 'FAILED'))
            passed = False
            continue
        summary = read(out / model / 'summary.json')
        npu = summary['tables']['npu']
        passed &= npu['token_accuracy'] == 1 and npu['exact_recovery_rate'] == 1
        rows.append((model, summary['samples'], summary['input_positions'], percent(npu['token_accuracy'], 1),
                     percent(npu['exact_recovery_rate'], 1)))
    table(('Model', 'Prompts', 'Token positions', 'Token accuracy', 'Exact recovery'), rows)
    print('Paper (1,000 prompts per model): 100% token accuracy and 100% exact recovery for all three models.')
    return finish('5.2', passed, out)


if __name__ == '__main__':
    sys.exit(main())
