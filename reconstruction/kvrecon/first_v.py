
import json
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from .search import kv_distance
from .common import dump, digest, kv_dtype, distance_values, resolve_distance_space


def table_identity(model_hash, manifest_hash, adapter, signature, ids):
    import hashlib
    return dict(schema=1,model_sha256=model_hash,manifest_sha256=manifest_hash,
                adapter=adapter,batch_signature=signature,
                token_ids_sha256=hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest())


def check_independence(b, spec, ids):
    
    split=spec['quantization'][0]['width']
    samples=np.asarray(ids)[np.unique(np.linspace(0,len(ids)-1,min(8,len(ids)),dtype=int))].tolist()
    def query():
        chunks=[b.query(samples[i:i+b.batch_capacity])[:,split:] for i in range(0,len(samples),b.batch_capacity)]
        return np.concatenate(chunks)
    b.reset(spec['prefix']);reference=query();checks=[]
    for prefix in ([],spec['prefix'],spec['prefix']+[int(ids[-1])],spec['prefix']*16):
        b.reset(prefix);actual=query()
        np.testing.assert_array_equal(actual,reference,err_msg='First V depends on context/position or native path')
        committed=b.commit(samples[0])[:,split:]
        np.testing.assert_array_equal(committed,reference[:1],err_msg='First V commit/query mismatch')
        checks.append(dict(prefix=prefix,maximum_absolute_error=0.))
    b.reset(spec['prefix'])
    return dict(passed=True,sample_ids=samples,checks=checks,
                note='Finite check, not proof of all-prefix numeric equivalence')


def _atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _backend_identity(b):
    if hasattr(b,'table_provenance'):
        return b.table_provenance
    device=Path(b.output)/'device.json' if hasattr(b,'output') else None
    if device and device.exists():
        info=json.loads(device.read_text(encoding='utf-8'))
                                                                                  
        keys=('model_sha256','runtime_sha256','test_apk_sha256','libraries','native_mode','batch_signature')
        return {key:info[key] for key in keys if key in info}
    return None


def build_table(path, b, spec, ids, identity, *, resume=False, chunk_size=4096):
    





    path=Path(path)
    if not spec.get('context_independent_by_architecture') or spec['layer']!=0:
        raise ValueError('Adapter must declare context-independent first text-layer V')
    if type(chunk_size) is not int or not 1<=chunk_size<=4096:
        raise ValueError('chunk-size must be between 1 and 4096')
    ids=np.asarray(ids,dtype=np.int64)
    if not len(ids) or len(np.unique(ids))!=len(ids) or min(ids)<0 or max(ids)>=b.vocab_size:
        raise ValueError('Invalid table token IDs')
    if path.exists() and not resume:raise FileExistsError('Preserve the existing table directory; use resume explicitly')
    quant=spec['quantization'][1];dtype=kv_dtype([quant]);shape=(len(ids),quant['width'])
    binding=dict(identity=identity,spec=spec,vocab_size=b.vocab_size,rows=len(ids),
        dtype=np.dtype(dtype).str,width=quant['width'],backend=_backend_identity(b))
                                                                                
    binding=json.loads(json.dumps(binding))
    checkpoint=path/'progress.json'
    if path.exists():
        if json.loads((path/'collection.json').read_text(encoding='utf-8'))!=binding:
            raise ValueError('Resume model/runtime/configuration identity mismatch')
        if (path/'metadata.json').exists():
            load_table(path,identity,spec,ids)
            return
        state=json.loads(checkpoint.read_text(encoding='utf-8'))
        if not np.array_equal(np.load(path/'ids.npy',allow_pickle=False),ids):raise ValueError('Resume token IDs mismatch')
        codes=np.load(path/'v.npy',mmap_mode='r+',allow_pickle=False)
        if codes.shape!=shape or codes.dtype!=dtype:raise ValueError('Resume table shape/dtype mismatch')
        end=0
        for chunk in state['chunks']:
            if chunk['start']!=end or not end<chunk['end']<=len(ids):raise ValueError('Invalid resume chunk range')
            actual=hashlib.sha256(codes[end:chunk['end']].tobytes()).hexdigest()
            if actual!=chunk['sha256']:raise ValueError('Completed V chunk integrity mismatch')
            end=chunk['end']
        if end!=state['completed_rows']:raise ValueError('Invalid resume checkpoint count')
        validation=state['validation']
        if not validation.get('passed') and end:raise ValueError('Invalid independence validation')
    else:
        path.mkdir(parents=True)
        _atomic_json(path/'collection.json',binding)
        np.save(path/'ids.npy',ids,allow_pickle=False)
        codes=np.lib.format.open_memmap(path/'v.npy',mode='w+',dtype=dtype,shape=shape)
        codes.flush()
                                                                                
        state=dict(complete=False,completed_rows=0,total_rows=len(ids),chunks=[],
            validation=dict(passed=False),build_seconds=0.)
        _atomic_json(checkpoint,state)
        validation=None
    started=time.perf_counter()
    try:
        if validation is None or not validation.get('passed'):
            validation=check_independence(b,spec,ids)
            state['validation']=validation
            _atomic_json(checkpoint,state)
        else:
                                                                          
            repeated=check_independence(b,spec,ids)
            if repeated!=validation:raise ValueError('Resume independence validation mismatch')
        split=spec['quantization'][0]['width']
        previous_seconds=state.get('build_seconds',0.)
        for start in range(state['completed_rows'],len(ids),chunk_size):
            chunk=ids[start:start+chunk_size].tolist()
            if callable(getattr(b,'first_v_table',None)):
                values=b.first_v_table(chunk)
            else:
                values=np.concatenate([b.query(chunk[i:i+b.batch_capacity])[:,split:] for i in range(0,len(chunk),b.batch_capacity)])
            if values.shape!=(len(chunk),quant['width']) or values.dtype!=codes.dtype or not np.isfinite(values).all():
                raise ValueError('Invalid native first V table output')
            codes[start:start+len(chunk)]=values;codes.flush()
            with (path/'v.npy').open('r+b') as stream:os.fsync(stream.fileno())
            state['chunks'].append(dict(start=start,end=start+len(chunk),sha256=hashlib.sha256(values.tobytes()).hexdigest()))
            state.update(completed_rows=start+len(chunk),build_seconds=previous_seconds+time.perf_counter()-started)
            _atomic_json(checkpoint,state)
            print(f"first_v_table={start+len(chunk)}/{len(ids)} elapsed={time.perf_counter()-started:.1f}s",flush=True)
        codes._mmap.close();codes=None
        device=Path(b.output)/'device.json' if hasattr(b,'output') else None
        _atomic_json(path/'metadata.json',dict(**identity,complete=True,spec=spec,vocab_size=b.vocab_size,
            rows=len(ids),full_vocabulary=len(ids)==b.vocab_size,quantization=[quant],
            validation=validation,build_seconds=state['build_seconds'],backend_identity=binding['backend'],
            device=json.loads(device.read_text()) if device and device.exists() else None,
            codes_sha256=digest(path/'v.npy'),ids_sha256=digest(path/'ids.npy')))
        state['complete']=True;_atomic_json(checkpoint,state)
        if (path/'failure.json').exists():(path/'failure.json').unlink()
    except Exception as e:
        dump(path/'failure.json',dict(error=str(e),complete=False))
        raise
    finally:
        if codes is not None:codes._mmap.close()


def collect_vocabulary_v(path, backend, spec, identity, *, resume=False, chunk_size=4096):
    




    ids=np.arange(backend.vocab_size,dtype=np.int64)
    build_table(path,backend,spec,ids,identity,resume=resume,chunk_size=chunk_size)
    return load_table(path,identity,spec,ids)[0]


def load_table(path, identity, spec, ids):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text())
    if (not meta.get('complete') or any(meta.get(k)!=v for k,v in identity.items()) or meta['spec']!=spec
            or meta['quantization']!=[spec['quantization'][1]] or not meta.get('validation',{}).get('passed')):
        raise ValueError('First V table identity/configuration mismatch or incomplete table')
    if digest(path/'v.npy')!=meta['codes_sha256'] or digest(path/'ids.npy')!=meta['ids_sha256']:
        raise ValueError('First V table integrity mismatch')
    stored_ids=np.load(path/'ids.npy',allow_pickle=False)
    codes=np.load(path/'v.npy',mmap_mode='r',allow_pickle=False)
    if not np.array_equal(stored_ids,ids) or codes.shape!=(len(ids),spec['quantization'][1]['width']) or codes.dtype!=kv_dtype(meta['quantization']):
        raise ValueError('First V table shape, dtype or ID mismatch')
    return meta,codes


def nearest_v(codes, ids, quant, target, epsilon, chunk_size=4096, distance_space="auto", distance_metric="rmse"):
    
    target=np.asarray(target,dtype=np.float64)
    if not np.isfinite(epsilon) or epsilon<=0 or target.shape!=(quant[0]['width'],) or not np.isfinite(target).all():
        raise ValueError('Invalid V matching threshold/target')
    passing=[];best_id=None;best_distance=float('inf');second=float('inf')
    for start in range(0,len(ids),chunk_size):
        values=distance_values(codes[start:start+chunk_size],quant,distance_space)
        if not np.isfinite(values).all():raise ValueError('Nonfinite V table')
        distances=kv_distance(values,target,distance_metric)
        order=np.argsort(distances,kind='stable')
        for j in order[:2]:
            distance=float(distances[j])
            if distance<best_distance:
                second=best_distance;best_distance=distance;best_id=int(ids[start+j])
            elif distance<second:
                second=distance
        passing.extend(np.asarray(ids[start:start+chunk_size])[distances<epsilon].tolist())
    return dict(nearest_token_id=best_id,minimum_distance=best_distance,distance_metric=distance_metric,
                minimum_l2=best_distance*np.sqrt(len(target)) if distance_metric=="rmse" else best_distance,
                second_distance=second if np.isfinite(second) else None,
                second_l2=(second*np.sqrt(len(target)) if distance_metric=="rmse" else second) if np.isfinite(second) else None,passing_ids=passing,
                accepted_token_id=passing[0] if len(passing)==1 else None,
                stop_reason='threshold' if len(passing)==1 else ('ambiguous' if passing else 'unresolved'))


def match_observation(output, codes, ids, table_meta, targets, positions, epsilon, max_positions=None, distance_space="auto", distance_metric="rmse"):
    distance_space=resolve_distance_space(distance_space,codes.dtype)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if max_positions is not None and max_positions<=0:raise ValueError('max-positions must be positive')
    count=len(positions) if max_positions is None else min(len(positions),max_positions)
    split=table_meta['spec']['quantization'][0]['width'];results=[];started=time.perf_counter()
    (output/'candidates').mkdir()
    with (output/'trace.jsonl').open('w') as f:
        for i in range(count):
            row=nearest_v(codes,ids,table_meta['quantization'],targets[i,split:],epsilon,distance_space=distance_space,distance_metric=distance_metric)
            passing=row.pop('passing_ids');file=f'candidates/{i:06d}.npy'
            np.save(output/file,np.asarray(passing,dtype=np.int64),allow_pickle=False)
            row.update(index=i,position=int(positions[i]),distance_space=distance_space,queries=0,table_rows_compared=len(ids),
                       passing_count=len(passing),passing_ids_file=file,passing_ids_preview=passing[:32],
                       accepted_distance=row['minimum_distance'] if row['accepted_token_id'] is not None else None,
                       accepted_l2=row['minimum_l2'] if row['accepted_token_id'] is not None else None)
            results.append(row['accepted_token_id']);f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            print(f"position={positions[i]} first_v={row['stop_reason']} nearest={row['nearest_token_id']} {distance_metric.upper()}={row['minimum_distance']}",flush=True)
    status='complete' if all(x is not None for x in results) else 'ambiguous_or_unresolved'
    if status=='complete' and count<len(positions):status='position_limit'
    result=dict(status=status,method='first_v',recovered_token_ids=results,target_positions=len(positions),
        attempted_positions=count,config=dict(method='first_v',epsilon=epsilon,distance_space=distance_space,distance_metric=distance_metric),online_seconds=time.perf_counter()-started,
        decoder_seconds=0.,target_cache_used_as_prefix=False,independent_positions=True,
        matching_scope='full_vocabulary' if table_meta['full_vocabulary'] else 'configured_candidate_pool',
        uniqueness_proven=False,correctness='Unique threshold match within the table; identity requires separate evaluation')
    dump(output/'result.json',result);return result
