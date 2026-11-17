
import ast
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
errors=[]
for p in root.rglob('*.py'):
    try: ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p.relative_to(root)))
    except SyntaxError as e: errors.append(str(e))
for p in root.rglob('*'):
    if not p.is_file() or (p.suffix.lower() not in {'.py','.json','.md','.txt','.java','.kt','.c','.cc','.cpp','.h','.inc','.sh','.ps1','.bat','.gradle','.properties','.toml','.xml'} and p.name not in {'Makefile','LICENSE','NOTICE'}):continue
    if 'assets' in p.relative_to(root).parts or p.stat().st_size > 8*1024*1024:continue
    try:text=p.read_text(encoding='utf-8')
    except UnicodeDecodeError:continue
    if re.search(r'(?i)[A-Z]:[\\/](?:Users|Confirmation)|@[\w.-]+\.edu\b|_bazel_[a-z]|/Users/[a-z]',text):
        errors.append('Local identity path: '+str(p.relative_to(root)))
if errors:raise SystemExit('\n'.join(errors))
print('Syntax and local-identity checks passed.')
