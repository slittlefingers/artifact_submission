"""Extract the bundled LiME captures into per-model reconstruction inputs."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check bundled inputs only')
    args = parser.parse_args()
    cases = json.loads((HERE/'config.json').read_text())
    extraction = ROOT/'extraction'
    for case in cases:
        folder = ROOT/case['input']
        for path in (folder/'mem.lime', folder/'vmlinux.btf', folder/'zsmalloc.btf', ROOT/case['model']):
            if not path.is_file(): raise FileNotFoundError(path)
        if not isinstance(case['count'], int) or case['count'] <= 0:
            raise ValueError('Missing verified position range: '+case['name'])
    if args.check:
        print('All LiME captures, models and position ranges are configured.')
        return
    destination = ROOT/'outputs/5.1'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    destination.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop('LIVE_PID', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = os.pathsep.join([str(extraction/'tools/volatility3-arm64'),str(extraction),str(ROOT/'reconstruction')])
    spec = importlib.util.spec_from_file_location('extraction_entry', extraction/'run.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    for case in cases:
        folder = ROOT/case['input']
        output = destination/case['name']
        output.mkdir()
        kvex = output/'extracted.kvex'
        print('Extracting '+case['name'], flush=True)
        subprocess.run([sys.executable, '-B', str(extraction/'tools/volatility3-arm64/vol.py'),
            '-q','-r','csv','-f',str(folder/'mem.lime'),'-p',str(extraction/'vol_plugin'),
            'kv_extract.KvExtract','--pid',str(case['pid']),'--backend','litert_namechain',
            '--va-bits',str(case['va_bits']),'--recover-dir',str(folder),
            '--models-dir',str(ROOT/'models/litertlm'),'--out',str(kvex)],env=env,check=True)
        wrapper.export_observation(SimpleNamespace(output=kvex,model=ROOT/case['model'],
            observation_output=output/'observations',prefix_json=folder/'public_prefix.json',
            slot_map_json=None,layer=None,start=case['start'],count=case['count'],linear_cache=True,bank=1))
    print('NPZ and metadata: '+str(destination), flush=True)

if __name__ == '__main__':
    main()
