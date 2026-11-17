"""Configure redacted absolute paths in a separate working copy."""
import argparse
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mapping', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
dest = a.output.resolve()
if dest == root or root in dest.parents or dest.exists():
    p.error('Output must be a new directory outside this release.')
mapping = json.loads(a.mapping.read_text(encoding='utf-8'))
for key, value in mapping.items():
    if not key.startswith('LOCAL_PATH_') or not isinstance(value, str) or not value:
        p.error('All mapping values must be non-empty paths.')
    if any(c in value for c in ('"', "'", '\n', '\r')):
        p.error('Paths containing quotes or newlines are unsupported.')
shutil.copytree(root, dest, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
for file in dest.rglob('*'):
    if file.suffix not in {'.py', '.json', '.cc', '.cpp', '.h', '.sh', '.ps1', '.bat', '.md'}:
        continue
    text = file.read_text(encoding='utf-8')
    for key, value in mapping.items():
        text = text.replace('/PATH/TO/' + key, value.replace('\\', '/'))
    file.write_text(text, encoding='utf-8')
print('Configured working copy:', dest)
