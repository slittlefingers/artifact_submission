"""Section 5.6 (Qwen): PhotoChat conversation recovery on the phone NPU with the image span and grid inferred from K/V (one conversation, one token; --full for all 100 complete)."""
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'experiments/_support'), str(ROOT / 'reconstruction')]
from sections import finish, output_root, parser, percent, read, require_phone, score_tokens, table

DATA = ROOT / 'assets/kv/5.6_qwen'
MODEL = 'qwen3_5_2b_vl_spec_Google_Tensor_G5.litertlm'


def check(folder):
    
    sample = folder.name
    truth, selection = read(DATA / 'evaluation' / f'{sample}.json'), read(folder / 'image/selection.json')
    with np.load(DATA / 'collection' / sample / 'layer3.npz', allow_pickle=False) as z:
        ids = dict(enumerate(z['token_ids'].tolist()))
    result = read(folder / 'recovery/result.json')
    recovered, correct = score_tokens(result, ids)
    span_ok = selection['visual_tokens'] == truth['visual_tokens'] and selection['grid'] == truth['grid']
    return (sample, f"{selection['visual_tokens']} ({'ok' if selection['visual_tokens'] == truth['visual_tokens'] else truth['visual_tokens']})",
            f"{selection['grid'][0]}x{selection['grid'][1]} ({'ok' if selection['grid'] == truth['grid'] else 'true %dx%d' % tuple(truth['grid'])})",
            recovered, percent(correct, recovered)), span_ok and recovered > 0 and correct == recovered


def main():
    p = parser(__doc__)
    p.add_argument('--config', type=Path, default=HERE / 'config.json', help='Run configuration (paths relative to the artifact root)')
    p.add_argument('--conversations', type=int, default=1, help='Conversations in the demo')
    p.add_argument('--positions', type=int, default=1, help='Recovered text tokens per conversation in the demo')
    p.add_argument('--check', action='store_true', help='Validate paths only, without the phone')
    args = p.parse_args()
    os.chdir(ROOT)
    from kvrecon.adapters.qwen_verify.photochat import main as photochat
    if args.check:
        return photochat(['--config', str(args.config.resolve()), '--check'])
    require_phone(MODEL)
    out = output_root('5.6_qwen', args)
    limits = [] if args.full else ['--limit', str(args.conversations), '--positions', str(args.positions)]
    try:
        photochat(['--config', str(args.config.resolve()), '--output', str(out), *limits])
    except Exception as error:
        print(f'Recovery failed: {error}')
        return finish('5.6 Qwen', False, out)
    rows, passed = [], True
    for folder in sorted(f for f in out.iterdir() if (f / 'recovery/result.json').is_file()):
        row, ok = check(folder)
        rows.append(row)
        passed &= ok
    table(('Conversation', 'Visual tokens (vs recorded)', 'Grid (vs recorded)', 'Text tokens', 'Accuracy'), rows)
    print('Paper (100 conversations, complete): all 47,827 text-token IDs recovered correctly.')
    return finish('5.6 Qwen', passed and bool(rows), out)


if __name__ == '__main__':
    sys.exit(main())
