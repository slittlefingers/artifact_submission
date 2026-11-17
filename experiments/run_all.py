"""Run every evaluation section (quick demo by default, --full for the complete cohorts) and print one summary table."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [('5.2', '5.2 Layer-0 V Text Token Reproducibility', 'offline'),
            ('5.3', '5.3 Image Reconstruction from KV Caches', 'offline'),
            ('5.4', '5.4 Audio Reconstruction from Layer-0 V and Layer-4 KV', 'offline'),
            ('5.5', '5.5 Prompt Recovery from Shallow Full-Attention Layers', 'phone NPU'),
            ('5.6-gemma', '5.6 Gemma - Multimodal Conversation Recovery', 'phone NPU'),
            ('5.6-qwen', '5.6 Qwen - Multimodal Conversation Recovery', 'phone NPU')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full', action='store_true', help='Complete cohorts instead of the quick demo (hours to days)')
    p.add_argument('--only', nargs='+', choices=[s[0] for s in SECTIONS], help='Run only these sections')
    p.add_argument('--offline', action='store_true', help='Skip the sections that need the phone')
    args = p.parse_args()
    rows = []
    for key, folder, where in SECTIONS:
        if (args.only and key not in args.only) or (args.offline and where != 'offline'):
            continue
        print(f'\n===== {folder} ({where}) =====', flush=True)
        started = time.perf_counter()
        code = subprocess.run([sys.executable, str(ROOT / 'experiments' / folder / 'run.py'), *(['--full'] if args.full else [])],
                              cwd=ROOT).returncode
        rows.append((key, where, 'PASS' if code == 0 else 'FAIL', f'{time.perf_counter() - started:.0f} s'))
    print('\nSection    Runs on     Result  Time\n---------  ----------  ------  ------')
    for row in rows:
        print('{:<9}  {:<10}  {:<6}  {}'.format(*row))
    print('\nFor extraction from the bundled LiME captures: python kvidence.py experiment 5.1')
    return 0 if rows and all(r[2] == 'PASS' for r in rows) else 1


if __name__ == '__main__':
    sys.exit(main())
