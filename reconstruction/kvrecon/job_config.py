
from dataclasses import fields
from pathlib import Path
import json
from .search import SearchConfig

class BoundaryRules:
    def __init__(self,sequences):
        self.sequences=sequences
        if not sequences or any(not isinstance(k,str) or not v or any(type(t) is not int or t<0 for t in v) for k,v in sequences.items()):
            raise ValueError('Boundary rules require named, nonempty token-ID sequences')
    def match(self,history,position):
        for name,sequence in self.sequences.items():
            if len(history)>=len(sequence) and list(history[-len(sequence):])==sequence:
                return dict(name=name,start_position=position-len(sequence)+1,end_position=position+1,token_ids=sequence,verified_by_kv=True)
        return None

def load_job(path):
    from tokenizers import Tokenizer
    path=Path(path).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    required={'schema_version','model','device','metadata','search','ridge','input_manifest','output'}
    if not required<=set(data) or set(data)-required-{'reference','learned_candidates','media'} or data['schema_version']!=1:raise ValueError('Unsupported reconstruction job schema')
    for key in ['input_manifest','output']:data[key]=(path.parent/data[key]).resolve()
    model=data['model']
    for key in ['tokenizer','manifest','embeddings']:model[key]=(path.parent/model[key]).resolve()
    if type(model['layer']) is not int or model['layer']<1:raise ValueError('This paired text experiment requires a deeper layer')
    tok=Tokenizer.from_file(str(model['tokenizer']));markers=data['metadata']['markers'];resolved={}
    for category in ['image','user','assistant','system']+(['audio'] if 'audio' in markers else []):
        entry=markers[category]
        for boundary in ['start','end']:
            value=entry[boundary];values=value if isinstance(value,list) else [value]
            if not values or any(not isinstance(v,str) or not v for v in values):raise ValueError('Markers must be nonempty strings')
            for index,value in enumerate(values):
                ids=tok.encode(value,add_special_tokens=False).ids
                if not ids:raise ValueError('Empty encoded marker')
                resolved[f'{category}.{boundary}'+(f'.{index}' if len(values)>1 else '')]=ids
    boundaries={}
    for name in data['metadata']['stop_before']:
        matches={k:v for k,v in resolved.items() if k==name or k.startswith(name+'.')}
        if not matches:raise ValueError('Unknown stop marker '+name)
        boundaries.update(matches)
                                                                    
    data['resolved_markers']=resolved;data['boundaries']=BoundaryRules(boundaries) if boundaries else None
    if 'media' in data and any(name in boundaries for name in ('image.start','audio.start')):
        raise ValueError('Media recovery cannot stop_before the same media start marker')
    data['public_prefix_ids']=tok.encode(data['metadata']['public_prefix'],add_special_tokens=False).ids
    if not data['public_prefix_ids']:raise ValueError('Explicit nonempty public prefix required')
    search=data['search'];methods=search['methods']
    allowed={'guided','guided_kv','v_nearest','v_probability','collision_plus','collision_plus_modified'}
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=allowed:raise ValueError('Choose unique supported search methods')
    options={k:v for k,v in search.items() if k!='methods'}
    if set(options)-{f.name for f in fields(SearchConfig)} or 'method' in options or 'boundary_token_ids' in options:raise ValueError('Unknown search parameter; boundary IDs are derived from metadata.markers')
                                                                             
                                                                        
    data['boundary_priority_ids']=list(dict.fromkeys(t for sequence in resolved.values() for t in sequence))
    merge_initial=options.pop('merge_initial_batch',False)
    reference_seeds=options.pop('reference_seeds',0)
    if type(merge_initial) is not bool:raise ValueError('merge_initial_batch must be boolean')
    if reference_seeds and not any(m in ('guided_kv','v_probability') for m in methods):raise ValueError('Reference seeds require guided_kv or v_probability')
    data['search_configs']={method:SearchConfig(method=method,reference_seeds=reference_seeds if method in ('guided_kv','v_probability') else 0,merge_initial_batch=merge_initial if method in ('guided','guided_kv','v_probability') else False,boundary_token_ids=tuple(data['boundary_priority_ids']) if method=='collision_plus_modified' else (),**options).validate() for method in methods}
    if 'learned_candidates' in data:
        learned=data['learned_candidates']
        if learned.get('mode') not in ('v','kv'):raise ValueError('Learned candidates require V or K+V')
        for key in ['checkpoint','quantization','training_protocol']:
            learned[key]=(path.parent/learned[key]).resolve()
        if methods not in (['guided_kv'],['v_nearest']) or options.get('probability_seeds',0)!=0:
            raise ValueError('Learned proposals require guided_kv or v_nearest without target-model probability seeds')
    if any(m in ('guided_kv','v_nearest','v_probability') for m in methods) and 'learned_candidates' not in data:
        ref=data.get('reference')
        if not isinstance(ref,dict) or 'table' not in ref or set(ref)-{'table','device','construction','block_size'}:raise ValueError('Reference search requires reference.table')
        if ref.get('construction','independent') not in ('independent','sequential_prefill'):
            raise ValueError('Unknown reference construction')
        if ref.get('construction')=='sequential_prefill' and (type(ref.get('block_size')) is not int or ref['block_size']<1):
            raise ValueError('Sequential reference requires a positive block_size')
        ref['table']=(path.parent/ref['table']).resolve()
    if 'learned_candidates' in data:
        for key in ('checkpoint','quantization','training_protocol'):
            data['learned_candidates'][key]=str((path.parent/data['learned_candidates'][key]).resolve())
    data['config_path']=path
    return data
