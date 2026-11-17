"""Use the bundled BTF converter and ARM64 Volatility with kvextract."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
VOL=ROOT/'tools/volatility3-arm64'

DEFAULT_LAYERS={
    'af1082986639ecde7db95d91be6fe54f8b6b458104734c5bafc204e69d6852dc': [0,4],
    '9ec3ff7dcc6218cdf9356d4e4a64bcbf4af0e23cb519ac7c058c22701721984f': [3],
    '4e2dd9e376fd290ff40d2da70921b9f433f55aa1353cba18f8cf740a971e0847': [0],
    'fdae6b2951a7049007e2a3bd763a10706a317fdefa6944d71d8df510c14dc8a8': [0],
}


def selected_layers(model, layer):
    if layer is not None:
        if layer < 0:raise ValueError('Layer must be nonnegative')
        return [layer]
    sys.path.insert(0,str(ROOT.parent/'reconstruction'))
    from kvrecon.common import digest
    layers=DEFAULT_LAYERS.get(digest(model))
    if layers is None:
        raise ValueError('Unknown model build: specify --layer explicitly; automatic layers use the four bundled model hashes')
    return layers


def export_observation(a):
    sys.path.insert(0,str(ROOT.parent/'reconstruction'))
    from kvrecon.kvex_import import import_kvex
    prefix=json.loads(a.prefix_json.read_text(encoding='utf-8-sig'))
    slots=json.loads(a.slot_map_json.read_text(encoding='utf-8-sig')) if a.slot_map_json else None
    layers=getattr(a,'selected_layers',None) or selected_layers(a.model,a.layer)
    if a.layer is not None:
        result=import_kvex(a.output,a.model,a.observation_output,prefix,a.start,a.count,a.layer,
                          linear_cache=a.linear_cache,slot_map=slots,bank=a.bank)
        print(json.dumps(result,indent=2));return
                                                                                     
    destination=Path(a.observation_output)
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kv-observation-',dir=destination.parent) as tmp:
        staging=Path(tmp)/'observations';staging.mkdir()
        report={'layers':[],'skipped':[]}
        for layer in layers:
            if layers==[0,4] and layer==0 and (a.count>512 or (a.linear_cache and a.start+a.count>512)):
                report['skipped'].append({'layer':0,'reason':'Requested range is not verified within the retained 512-position sliding window. Use a separate layer-0 range and explicit slot map.'})
                continue
            result=import_kvex(a.output,a.model,staging/f'layer{layer}',prefix,a.start,a.count,layer,
                              linear_cache=a.linear_cache,slot_map=slots,bank=a.bank)
            result['output']=f'layer{layer}'
            report['layers'].append(result)
        (staging/'index.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        staging.rename(destination)
    print(json.dumps(dict(output=str(destination),**report),indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='step',required=True)
    s=sub.add_parser('symbols',help='Generate an ISF from matching BTF and symbol addresses')
    s.add_argument('--btf2json',type=Path,required=True)
    s.add_argument('--btf',type=Path,required=True);s.add_argument('--map',type=Path,required=True)
    s.add_argument('--banner',required=True);s.add_argument('--vabits',type=int,required=True)
    s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('extract',help='Run strict KV localization through bundled ARM64 Volatility')
    s.add_argument('--dump',type=Path,required=True);s.add_argument('--symbols',type=Path,required=True)
    s.add_argument('--pid',type=int,required=True);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--recover-dir',type=Path,default=ROOT/'support')
    s.add_argument('--va-bits',type=int,required=True)
    s.add_argument('--address-manifest',type=Path)
    s.add_argument('--raw-only',action='store_true',help='Export only KVEX; default also creates a reconstruction observation')
    s.add_argument('--model',type=Path,help='Exact deployed Tensor-G5 .litertlm bundle')
    s.add_argument('--layer',type=int,help='Optional override; defaults: Qwen 3, Gemma 0+4, Llama/Ministral 0 (zero-based)')
    s.add_argument('--prefix-json',type=Path,help='Known public template token IDs; [] for layer-0 lookup')
    s.add_argument('--start',type=int,help='First retained logical position to recover')
    s.add_argument('--count',type=int,help='Verified number of retained target positions')
    s.add_argument('--bank',type=int,default=1)
    s.add_argument('--observation-output',type=Path,help='Default: OUTPUT stem + .observation directory')
    mapping=s.add_mutually_exclusive_group()
    mapping.add_argument('--linear-cache',action='store_true',help='Confirm physical slots equal logical positions')
    mapping.add_argument('--slot-map-json',type=Path,help='Physical slot for each requested logical position')
    s=sub.add_parser('vol-help',help='Inspect available plugins without reading a dump')
    a=p.parse_args()
    if a.step=='symbols':
        for f in [a.btf2json,a.btf,a.map]:
            if not f.is_file():p.error('Missing input: '+str(f))
        if a.output.exists():p.error('Refusing to overwrite existing ISF')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='isf-',suffix='.tmp',dir=a.output.parent)
        try:
            with os.fdopen(fd,'wb') as f:
                subprocess.run([str(a.btf2json.resolve()),'--btf',str(a.btf.resolve()),'--map',str(a.map.resolve()),'--banner',a.banner,'--vabits',str(a.vabits)],stdout=f,check=True)
            profile=json.loads(Path(tmp).read_text(encoding='utf-8'))
            if not all(k in profile for k in ['metadata','symbols','user_types']):raise ValueError('Converter did not produce a valid ISF structure')
            Path(tmp).rename(a.output)
        finally:
            if Path(tmp).exists():Path(tmp).unlink()
        return
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(VOL),str(ROOT),env.get('PYTHONPATH','')])
    cmd=[sys.executable,str(VOL/'vol.py')]
    if a.step=='vol-help':cmd+=['-p',str(ROOT/'vol_plugin'),'--help']
    else:
        if not a.dump.is_file() or not a.symbols.is_dir():p.error('Dump file and symbol directory must exist')
        if a.pid<=0:p.error('PID must be positive')
        if a.output.exists():p.error('Refusing to overwrite extracted KV')
        if not a.raw_only:
            for name in ('model','prefix_json','start','count'):
                if getattr(a,name) is None:p.error('--'+name.replace('_','-')+' is required for automatic reconstruction output (or use --raw-only)')
            if not a.model.is_file() or not a.prefix_json.is_file():p.error('Model and public-prefix files must exist')
            if a.count<=0 or a.start<0:p.error('Invalid retained position range')
            try:a.selected_layers=selected_layers(a.model,a.layer)
            except ValueError as e:p.error(str(e))
            if not a.linear_cache and a.slot_map_json is None:p.error('Specify --linear-cache or --slot-map-json')
            a.observation_output=a.observation_output or a.output.with_suffix('.observation')
            if a.observation_output.exists():p.error('Refusing to overwrite reconstruction observation')
        if a.address_manifest and a.address_manifest.exists():p.error('Refusing to overwrite address manifest')
        for name in ['vmlinux.btf','zsmalloc.btf','core/kbtf.py','core/btf_offsets.py']:
            if not (a.recover_dir/name).is_file():p.error('Missing BTF support input: '+name)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        cmd+=['-f',str(a.dump.resolve()),'-s',str(a.symbols.resolve()),'-p',str(ROOT/'vol_plugin'),'kv_extract.KvExtract','--pid',str(a.pid),'--backend','litert_namechain','--recover-dir',str(a.recover_dir.resolve()),'--va-bits',str(a.va_bits),'--out',str(a.output.resolve())]
        if a.model:cmd+=['--models-dir',str(a.model.resolve().parent)]
        if a.address_manifest:cmd+=['--address-manifest',str(a.address_manifest.resolve())]
    subprocess.run(cmd,env=env,check=True)
    if a.step=='extract' and not a.raw_only:
        export_observation(a)

if __name__=='__main__':main()
