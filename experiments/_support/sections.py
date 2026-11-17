
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'com.example.annoymous'
PHONE_MODELS = f'/sdcard/Android/data/{PACKAGE}/files'
SHOWN = re.compile(r'position=|visual_tokens=|FINISHED|Saved|Error|error:|Traceback')


def parser(description):
    
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--full', action='store_true', help='Run the complete cohort instead of the quick demo')
    p.add_argument('--output', type=Path, help='New output directory (default: outputs/<section>/<demo|full>-<time>)')
    return p


def output_root(section, args):
    
    if args.output:
        folder = args.output.resolve()
    else:
        folder = ROOT / 'outputs' / section / (('full-' if args.full else 'demo-') + time.strftime('%Y%m%d-%H%M%S'))
    if folder.exists() and any(folder.iterdir()):
        raise SystemExit(f'Refusing to overwrite {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run(command, log):
    
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(c) for c in command]
    env = dict(os.environ, MSYS_NO_PATHCONV='1', PYTHONUNBUFFERED='1')
    print('$ ' + ' '.join(f'"{c}"' if ' ' in c else c for c in command), flush=True)
    tail = []
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace', env=env)
        for line in process.stdout:
            stream.write(line)
            tail = (tail + [line.rstrip()])[-20:]
            if SHOWN.search(line):
                print('  ' + line.rstrip()[:200], flush=True)
        code = process.wait()
    if code:
        print('\n'.join('  | ' + t for t in tail), flush=True)
        print(f'  step failed with exit code {code}; full log: {log}', flush=True)
    return code


def python(*arguments):
    return [sys.executable, *arguments]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table(headers, rows):
    widths = [max(len(str(x)) for x in column) for column in zip(headers, *rows)]
    line = lambda row: '  '.join(str(x).ljust(w) for x, w in zip(row, widths))
    print('\n' + line(headers) + '\n' + '  '.join('-' * w for w in widths))
    for row in rows:
        print(line(row))


def finish(section, passed, output):
    
    print(f'\n{section}: {"PASS" if passed else "FAIL"}   (outputs: {output})')
    return 0 if passed else 1


def require_phone(*models):
    
    try:
        devices = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        raise SystemExit('adb is not available; install Android platform-tools and add adb to PATH')
    if not any(line.strip().endswith('device') and not line.startswith('List') for line in devices.splitlines()):
        raise SystemExit('No phone attached; connect the Pixel 10 and enable USB debugging')
    listing = subprocess.run(['adb', 'shell', 'ls', PHONE_MODELS], capture_output=True, text=True, timeout=60).stdout.split()
    missing = [m for m in models if m not in listing]
    if missing:
        raise SystemExit(f'Missing on the phone ({PHONE_MODELS}): {", ".join(missing)}; see target_app/MODEL_PLACEMENT.md')


def score_tokens(result, truth):
    
    pairs = list(zip(result['recovered_token_positions'], result['recovered_token_ids']))
    return len(pairs), sum(truth.get(int(p)) == int(t) for p, t in pairs)


def percent(correct, total):
    return f'{100 * correct / total:.1f}%' if total else '-'
