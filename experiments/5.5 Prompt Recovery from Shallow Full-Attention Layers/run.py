"""Section 5.5: prompt recovery on the phone NPU with KVidence and the Collision+ Modified baseline (one prompt, one token each; --full for 100 prompts)."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'experiments/_support'))
from sections import ROOT, finish, output_root, parser, percent, python, read, require_phone, run, score_tokens, table

RUNS = [('gemma', 'KVidence: V + probability + ridge', 'gemma.json'),
        ('gemma', 'KVidence: learned V + ridge', 'gemma_learned.json'),
        ('gemma', 'Collision+ Modified (baseline)', 'gemma_collision.json'),
        ('qwen', 'KVidence: V + probability + ridge', 'qwen.json'),
        ('qwen', 'KVidence: learned V + ridge', 'qwen_learned.json'),
        ('qwen', 'Collision+ Modified (baseline)', 'qwen_collision.json')]
MODELS = {'gemma': 'gemma-4-E2B-it_Google_Tensor_G5.litertlm', 'qwen': 'qwen3_5_2b_vl_spec_Google_Tensor_G5.litertlm'}
CLI = ROOT / 'reconstruction/run.py'


def score(config, folder):
    
    labels = (HERE / read(config)['input_manifest']).resolve().parent.parent / 'evaluation'
    recovered = correct = queries = 0
    for result_path in sorted(folder.glob('recovery/*/*/result.json')):
        result = read(result_path)
        truth = read(labels / f"{result['sample_id']}.json")
        ids = dict(enumerate(truth['known_first_turn_ids'])) if 'known_first_turn_ids' in truth else dict(zip(truth['positions'], truth['token_ids']))
        n, c = score_tokens(result, ids)
        recovered, correct, queries = recovered + n, correct + c, queries + result['candidate_queries']
    return recovered, correct, queries


def main():
    p = parser(__doc__)
    p.add_argument('--model', choices=sorted(MODELS), help='Run only this model')
    p.add_argument('--positions', type=int, default=1, help='Recovered tokens per prompt in the demo')
    args = p.parse_args()
    runs = [r for r in RUNS if args.model in (None, r[0])]
    require_phone(*{MODELS[r[0]] for r in runs})
    out = output_root('5.5', args)
    rows, passed = [], True
    for model, method, name in runs:
        config, folder = HERE / name, out / Path(name).stem
        limits = [] if args.full else ['--limit', 1, '--positions', args.positions]
        code = run(python(CLI, 'run-json', '--config', config, '--output', folder, *limits), out / f'{Path(name).stem}.log')
        recovered, correct, queries = score(config, folder) if not code else (0, 0, 0)
        ok = not code and recovered > 0 and correct == recovered
        passed &= ok
        rows.append((model, method, recovered, percent(correct, recovered), f'{queries / recovered:.1f}' if recovered else '-',
                     'ok' if ok else 'FAILED'))
    table(('Model', 'Method', 'Tokens', 'Accuracy', 'Queries/token', 'Status'), rows)
    print('Paper (100 LMSYS prompts): 100% accuracy for every method; KVidence needs about 131-135 queries/token, '
          'Collision+ Modified 1,668 (Gemma) and 1,943 (Qwen).')
    return finish('5.5', passed, out)


if __name__ == '__main__':
    sys.exit(main())
