
from pathlib import Path
import json
import time
import numpy as np
from .common import quantization,kv_dtype


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    for retry in range(30):
        try:tmp.replace(path);return
        except PermissionError:
            if retry==29:raise
            time.sleep(.1)


def reference_identity(job):
    from .config import model_dimensions
    sections=json.loads(job['model']['manifest'].read_text(encoding='utf-8'))['model_sections']
    dims=model_dimensions(sections,job['model']['batch_signature'])
    identity=dict(layer_index=job['model']['layer'],reference_prefix_ids=job['public_prefix_ids'],
        reference_position=len(job['public_prefix_ids']),batch_signature=job['model']['batch_signature'],vocab_size=dims['vocab_size'],
        quantization=quantization(sections,job['model']['layer'],job['model']['batch_signature']))
    if job['device'].get('qwen_recovery'):
        identity.update(qwen_recovery=True,qwen_query_mode=job['device'].get('qwen_query_mode','decode'))
    if job.get('reference',{}).get('construction')=='sequential_prefill':
        identity.update(reference_construction='sequential_prefill',reference_block_size=job['reference']['block_size'])
    return identity


def build_reference(config,output,resume=False,checkpoint_batches=16):
    from .job_config import load_job
    from .json_job import open_backend
    job=load_job(config);out=Path(output).resolve();identity=reference_identity(job)
    if identity.get('reference_construction')=='sequential_prefill':
        raise ValueError('Sequential references require the dedicated causal-prefill builder')
    if out.exists() and not resume:raise FileExistsError('Use --resume to continue the existing table')
    out.mkdir(parents=True,exist_ok=True);meta_path=out/'metadata.json'
    n=identity['vocab_size'];kw,vw=[q['width'] for q in identity['quantization']];dtype=kv_dtype(identity['quantization'])
    if meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['identity']!=identity:raise ValueError('Reference identity mismatch')
        k=np.load(out/'k.npy',mmap_mode='r+');v=np.load(out/'v.npy',mmap_mode='r+')
        if meta['completed']==n:print('REFERENCE ALREADY COMPLETE',n,flush=True);return
    else:
        k=np.lib.format.open_memmap(out/'k.npy',mode='w+',dtype=dtype,shape=(n,kw))
        v=np.lib.format.open_memmap(out/'v.npy',mode='w+',dtype=dtype,shape=(n,vw))
        np.save(out/'token_ids.npy',np.arange(n,dtype=np.int32))
        meta=dict(identity=identity,completed=0);write_json(meta_path,meta)
    with open_backend(job,out/('worker_'+str(time.time_ns()))) as backend:
        batch=min(128,backend.batch_capacity);prefix=identity['reference_prefix_ids']
        backend.reset(prefix);initial_pos=backend.position
        ids=list(range(batch));a=backend.query(ids)
        if not np.array_equal(a,backend.query(ids[::-1])[::-1]):raise ValueError('Candidate order affects KV; independent-branch contract failed')
        for token in [0,1,batch//2,batch-1]:
            if not np.array_equal(a[token],backend.query([token])[0]):raise ValueError('Batched candidate differs from singleton')
        backend.reset(prefix)
        if not np.array_equal(a,backend.query(ids)):raise ValueError('Reset changed reference states')
        for offset in range(meta['completed'],n,batch):
            stop=min(offset+batch,n);codes=backend.query(list(range(offset,stop)))
            if codes.shape!=(stop-offset,kw+vw) or codes.dtype!=dtype or backend.position!=initial_pos:
                raise ValueError('Invalid reference query result')
            k[offset:stop]=codes[:,:kw];v[offset:stop]=codes[:,kw:]
            if stop==n or (stop//batch)%checkpoint_batches==0:
                k.flush();v.flush();meta['completed']=stop;write_json(meta_path,meta)
                print('REFERENCE',stop,'/',n,flush=True)
    print('REFERENCE COMPLETE',n,flush=True)


class VReferenceIndex:
    
    def __init__(self,path,identity=None,device=None):
        import torch
        self.path=Path(path);self.metadata=json.loads((self.path/'metadata.json').read_text(encoding='utf-8'))
        stored=self.metadata['identity'];self.n=stored['vocab_size']
        if self.metadata.get('completed',self.n)!=self.n:raise ValueError('Incomplete KV reference table')
        if identity is not None and any(identity.get(key)!=value for key,value in stored.items()):
            raise ValueError('Reference does not match model layer, prefix or quantization')
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.kw=stored['quantization'][0]['width'];self.q=stored['quantization'][1]
        values=np.load(self.path/'v.npy')
        if values.shape!=(self.n,self.q['width']):raise ValueError('Reference V shape mismatch')
        self.dtype=values.dtype
        torch.backends.cuda.matmul.allow_tf32=False
        self.v=torch.from_numpy(values.astype(np.float32)).to(self.device)
        self.norms=self.v.square().sum(-1)

    def select(self,target,available,count,distance_space):
        import torch
        from .search import lowest_scored_ids
        if count<=0:return np.array([],dtype=np.int64)
        target=np.asarray(target);available=np.asarray(available)
        if target.shape!=(self.kw+self.q['width'],) or available.shape!=(self.n,):raise ValueError('Reference lookup shape mismatch')
        y=target[self.kw:].astype(np.float64)
        if distance_space=='original' and self.dtype.kind in 'iu':y=y/self.q['scale']+self.q['zero_point']
        if distance_space not in ['lsb','original']:raise ValueError('Reference lookup requires resolved distance units')
        if not np.isfinite(y).all():raise ValueError('Nonfinite reference target')
        with torch.inference_mode():
            t=torch.as_tensor(y,dtype=torch.float32,device=self.device)
            scores=(self.norms+t.square().sum()-2*(self.v@t)).clamp_min(0).cpu().numpy()
        return lowest_scored_ids(np.flatnonzero(available),scores,min(count,int(available.sum())))
