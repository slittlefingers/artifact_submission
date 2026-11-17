
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from kvrecon.search import SearchConfig


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def build_search_config(tokenizer_path, variant='multimodal'):
    





    if variant not in ('multimodal','token_surrogate'):
        raise ValueError('Unknown Experiment 9 search variant')
    tokenizer_path=Path(tokenizer_path).resolve()
    config_path=tokenizer_path.with_name('tokenizer_config.json')
    public=json.loads(config_path.read_text(encoding='utf8'))
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    unk_id=tokenizer.token_to_id(public['unk_token'])
    entries=[];priority=[]

    def add(name,text,single=False):
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        if not ids or (unk_id is not None and unk_id in ids):
            raise ValueError('Public marker contains an unknown token: '+name)
        if single and (len(ids)!=1 or tokenizer.token_to_id(text)!=ids[0]):
            raise ValueError('Expected one registered public special token: '+name)
        if any(tokenizer.id_to_token(i) is None for i in ids):
            raise ValueError('Public marker has an invalid token ID: '+name)
        entries.append(dict(name=name,text=text,token_ids=ids))
        for token in ids:
            if token not in priority:priority.append(token)
        return ids

    if variant=='token_surrogate':
        add('image.start',public['boi_token'],True)
        add('audio.start',public['boa_token'],True)
        if priority!=[255999,256000]:
            raise ValueError('Token-substitution region policy expects this Gemma deployment')
        template_path=None
    else:
        for name,key in [('bos','bos_token'),('eos','eos_token'),
                         ('turn.start','sot_token'),('turn.end','eot_token'),
                         ('image.start','boi_token'),('audio.start','boa_token')]:
            add(name,public[key],True)
                                                                                
                                                                              
        template_path=tokenizer_path.with_name('chat_template.jinja')
        template=template_path.read_text(encoding='utf8')
        if ("'<|turn>system\\n'" not in template or
                "'model' if message['role'] == 'assistant'" not in template or
                "'<|turn>' + role + '\\n'" not in template):
            raise ValueError('Public role template differs from the supported Gemma format')
        for role in ('system','user','model'):
            add('role.'+role,public['sot_token']+role+'\n')

    config=SearchConfig(method='guided_kv',probability_seeds=50,reference_seeds=78,
        random_seeds=0,batch_size=128,merge_initial_batch=True,proposals=128,
        epsilon=30. if variant=='token_surrogate' else 2.,distance_space='lsb',
        retain_query_cache=True,ridge_stagnation_rounds=2,stagnation_neighbors=127,
        max_queries=None,require_stagnation_for_acceptance=variant=='token_surrogate',
        boundary_token_ids=tuple(priority) if variant=='token_surrogate' else (),
        stagnation_token_ids=() if variant=='token_surrogate' else tuple(priority)).validate()
    if len(priority)+50>128:
        raise ValueError('Public markers would displace the fixed probability seed allocation')
    metadata=dict(variant=variant,source='Public tokenizer and canonical chat template only',
        entries=entries,token_ids=priority,token_strings=[tokenizer.id_to_token(i) for i in priority],
        marker_stage='initial' if variant=='token_surrogate' else 'stagnation_neighborhood',
        priority_count=len(priority) if variant=='token_surrogate' else 0,
        fallback_count=0 if variant=='token_surrogate' else len(priority),probability_candidates=50,
        expected_reference_candidates=128-50-(len(priority) if variant=='token_surrogate' else 0),physical_batch_size=128,
        labels_used=False,actual_prompt_contents_used=False,
        tokenizer_sha256=_sha(tokenizer_path),tokenizer_config_sha256=_sha(config_path),
        chat_template_sha256=_sha(template_path) if template_path else None,
        excluded_continuous_end_representations=dict(image=public['eoi_token'],audio=public['eoa_token']),
        note='Image/audio end positions use model constant embeddings in the multimodal path. Their similarly named vocabulary IDs are not assumed equivalent and are not added as marker candidates. Public candidates still compete by measured K/V distance; already measured IDs are reused.')
    return config,metadata
