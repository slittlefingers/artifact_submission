
from pathlib import Path
import json,time
import numpy as np
from .job_config import load_job
from .common import digest,dump
from .observation import load_observation
from .npu import NpuBackend
from .engine import reconstruct

def open_backend(job,output):
    sections=json.loads(job['model']['manifest'].read_text(encoding='utf-8'))['model_sections']
    m=job['model'];d=job['device']
    backend=NpuBackend;extra={}
    if d.get('qwen_recovery') and d.get('qwen_query_mode')=='verify':
        from .npu_standalone import StandaloneNpuBackend
        backend=StandaloneNpuBackend
        extra['native_executable']=Path(__file__).parent/'adapters/qwen_verify/qwen_verify_worker'
    return backend(sections,m['layer'],output,m['sha256'],adb=d['adb'],serial=d['serial'],package=d['package'],device_model=m['device_model'],batch_signature=m['batch_signature'],mask_token_id=m['mask_token_id'],install=d.get('install',True),native_mode=d.get('native_mode','optimized_v2'),qwen_recovery=d.get('qwen_recovery',False),qwen_query_mode=d.get('qwen_query_mode','decode'),**extra)

class QueryLog:
    def __init__(self,backend,folder,positions,codes):
        self.backend=backend;self.folder=Path(folder);self.codes={int(p):v for p,v in zip(positions,codes)};self.batches={}
        self.folder.mkdir(parents=True,exist_ok=True)
    def __getattr__(self,key):return getattr(self.backend,key)
    def query(self,ids):
        position=self.backend.position;start=time.perf_counter();result=self.backend.query(ids);seconds=time.perf_counter()-start
        n=self.batches.get(position,0)+1;self.batches[position]=n;k=self.backend.quant[0]['width'];target=self.codes[position]
        np.savez_compressed(self.folder/f'p{position:04d}_q{n:04d}.npz',token_ids=np.asarray(ids,dtype=np.int32),k=result[:,:k],v=result[:,k:],target_k=target[:k],target_v=target[k:],v_rmse_lsb=np.sqrt(np.mean((result[:,k:].astype(np.float64)-target[None,k:].astype(np.float64))**2,axis=1)),query_seconds=seconds)
        return result

def run_job(path,limit=None,max_positions=None,output=None):
    job=load_job(path);output=Path(output).resolve() if output else job['output'];output.mkdir(parents=True,exist_ok=True)
                                                                                    
    import sys,os
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            parts=Path(os.fsdecode(args[0])).resolve().parts
            if any(x in parts for x in ['private_eval','evaluation','dataset','datasets']):raise PermissionError('Recovery must not open source dialogue or evaluation labels')
    sys.addaudithook(guard)
    manifest=json.loads(job['input_manifest'].read_text(encoding='utf-8'));rows=manifest['samples']
    if limit is not None:rows=rows[:limit]
    protocol=dict(schema=1,job_config_sha256=digest(path),config=json.loads(Path(path).read_text(encoding='utf-8')),resolved_markers=job['resolved_markers'],public_prefix_ids=job['public_prefix_ids'],input_manifest_sha256=digest(job['input_manifest']),planned_samples=len(rows),full_dataset_samples=len(manifest['samples']),baseline='Collision+ probability-order adaptation with shared KV RMSE acceptance; not original CPA calibration',order='Alternate method order across samples',query_logging='Measured candidate K and V, IDs and V RMSE; adaptive queried candidates only',timing='Online includes ranking, ridge, candidate execution/transport/logging and commits; model and public-prefix setup reported separately',private_labels_allowed=False)
    protocol.update(boundary_priority_ids=job['boundary_priority_ids'],modified_policy='Prepend deduplicated constituent IDs of configured public image and role delimiters at every position, then probability order; same batch size and measured KV acceptance',order='Cyclic method order across samples')
    reference=None
    if 'learned_candidates' in job:
        from .learned_candidates import LearnedCandidates
        reference=LearnedCandidates(**job['learned_candidates'],layer=job['model']['layer'],model_sha256=job['model']['sha256'])
        protocol['learned_candidates']=reference.metadata
    elif any(m in ('guided_kv','v_nearest','v_probability') for m in job['search']['methods']):
        from .kv_reference import VReferenceIndex,reference_identity
        reference=VReferenceIndex(job['reference']['table'],reference_identity(job),job['reference'].get('device'))
        protocol['reference']=dict(table=str(reference.path),metadata_sha256=digest(reference.path/'metadata.json'),
            metric='V-only squared distance; initialization for guided_kv, full ranking for v_nearest, repeated masked selection for v_probability; real K/V gate unchanged',identity=reference.metadata['identity'])
    previous=output/'protocol.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8'))['job_config_sha256']!=protocol['job_config_sha256']:raise ValueError('Configuration changed; choose a new output directory')
    dump(previous,protocol)
    from .torch_surrogate import TorchRidge
    ridge=TorchRidge(job['model']['embeddings'],**job['ridge']) if any(m in ('guided','guided_kv') for m in job['search']['methods']) else None
    with open_backend(job,output/'worker') as backend:
        media_factory=None
        if 'media' in job:
            from .media.pipeline import build_pipeline
            media_factory=build_pipeline(job['media'],backend,job['config_path'].parent)
        prefix=job['public_prefix_ids'];backend.reset(prefix);a=backend.query([0,1]);again=backend.query([0]);assert np.array_equal(a[0],again[0])
        committed=backend.commit(0);commit_exact=bool(np.array_equal(a[0],committed[0]))
        if not backend.commit_revalidates_distance:assert commit_exact
        if not np.isfinite(committed).all():raise ValueError('Nonfinite committed K/V')
        backend.reset(prefix);assert np.array_equal(a[0],backend.query([0])[0])
        dump(output/'backend_validation.json',dict(repeat_exact=True,commit_exact=commit_exact,reset_exact=True,
             commit_revalidated_by_distance=backend.commit_revalidates_distance))
        for index,item in enumerate(rows):
            observation=(job['input_manifest'].parent/item['observation']).resolve()
            meta,codes,targets,positions,public=load_observation(observation,distance_space=job['search']['distance_space'])
            if 'learned_candidates' in job:reference.validate_quantization(meta['quantization'])
            assert public==prefix and meta['model_sha256']==job['model']['sha256'] and meta['layer_index']==job['model']['layer']
            base_methods=job['search']['methods'];offset=index%len(base_methods);methods=base_methods[offset:]+base_methods[:offset]
            for method in methods:
                folder=output/'recovery'/item['id']/method
                if (folder/'result.json').exists():continue
                if folder.exists():
                                                                                         
                    folder.rename(folder.with_name(method+'_interrupted_'+str(time.time_ns())))
                dump(output/'progress.json',dict(state='recovering',sample=item['id'],sample_index=index+1,total=len(rows),method=method))
                logged=QueryLog(backend,folder/'queries',positions,codes)
                result=reconstruct(logged,targets,positions,prefix,job['search_configs'][method],folder,ridge if method in ('guided','guided_kv') else None,max_positions=max_positions,live_events=True,boundaries=job['boundaries'],reference_initializer=reference if method in ('guided_kv','v_nearest','v_probability') else None,media_router=media_factory(positions) if media_factory else None)
                traces=[json.loads(line) for line in (folder/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
                result.update(candidate_queries=sum(r['queries'] for r in traces),candidate_batches=sum(len(r['stages']) for r in traces),ranking_seconds=sum(r['ranking_seconds']+r['probability_seconds'] for r in traces),method=method,sample_id=item['id'])
                dump(folder/'result.json',result);print('FINISHED',item['id'],method,result['status'],result['candidate_queries'],round(result['online_seconds'],2),flush=True)
    dump(output/'progress.json',dict(state='complete',completed=len(rows),total=len(rows),full_dataset_samples=len(manifest['samples'])))
