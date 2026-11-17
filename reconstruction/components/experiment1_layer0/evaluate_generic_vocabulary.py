"""Full-vocabulary layer-0 V RMSE distributions and exact-ID recovery on all input tokens."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import json
from pathlib import Path
import numpy as np
from io_utils import sha,save

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
ASSETS=HERE.parent.parent/'assets'

def table_model(meta):
    
    prov=meta.get('provenance') if isinstance(meta.get('provenance'),dict) else {}
    return meta.get('model_sha256') or prov.get('model_sha256')

def table_scale(meta):
    
    quant=meta.get('quantization')
    if isinstance(quant,list) and quant:
        assert quant[0].get('zero_point',0)==0;return float(quant[0]['scale'])
    assert meta.get('zero_point',0)==0;return float(meta.get('scale',1.0))

def find_table(model_sha):
    
    found=[p for p in sorted(ASSETS.glob('first_v_*_npu_full')) if table_model(json.loads((p/'metadata.json').read_text()))==model_sha]
    if len(found)!=1:raise SystemExit(f'Expected one NPU table for model {model_sha}, found {len(found)}; pass --table')
    return found[0]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='Input group under assets/kv/5.2, e.g. gemma4_e2b')
    p.add_argument('--data',type=Path,help='Directory holding samples/<id>/layer0.npz (default: assets/kv/5.2/MODEL)')
    p.add_argument('--table',action='append',default=[],metavar='NAME=DIR',help='Reference V table; default: the NPU table matching the model hash')
    p.add_argument('--output',type=Path,help='Output directory (default: comparison/v_distributions_MODEL)')
    p.add_argument('--limit',type=int,help='Evaluate only the first N samples')
    args=p.parse_args()
    data=args.data or ROOT/'assets/kv/5.2'/args.model
    protocol=json.loads((data/'protocol.json').read_text())
    out=args.output or HERE/'comparison'/('v_distributions_'+args.model);out.mkdir(parents=True,exist_ok=True)
    tables=dict(x.split('=',1) for x in args.table) if args.table else {'npu':str(find_table(protocol['model_sha256']))}
    folders=sorted(x for x in (data/'samples').iterdir() if (x/'layer0.npz').is_file())[:args.limit]
    observations=[];labels=[];lengths=[]
    for folder in folders:
        valid=json.loads((folder/'validation.json').read_text());assert valid['passed'] and sha(folder/'layer0.npz')==valid['npz_sha256']
        with np.load(folder/'layer0.npz') as sample:
            ids=sample['input_token_ids'];assert len(ids)==len(sample['v'])
            observations.append(sample['v']);labels.extend(ids);lengths.append(len(ids))
    unique,inverse,counts=np.unique(np.column_stack([labels,np.concatenate(observations)]),axis=0,return_inverse=True,return_counts=True)
    inverse=inverse.reshape(-1)
    truth=unique[:,0].astype(int);queries=unique[:,1:].astype(np.float64);width=queries.shape[1]
    summary=dict(model=args.model,model_sha256=protocol['model_sha256'],samples=len(folders),input_positions=len(labels),dimensions=width,
        distance='RMSE in dequantized V units (table scale)',scope='All input positions, including chat-template tokens',tables={})
    for backend,root in tables.items():
        root=Path(root);meta=json.loads((root/'metadata.json').read_text())
        assert meta['full_vocabulary'] and meta['complete'] and sha(root/'v.npy')==meta.get('v_sha256',meta.get('codes_sha256'))
        if table_model(meta)!=protocol['model_sha256']:raise SystemExit(f'{root} was built for another model')
        scale=table_scale(meta)
        table=np.load(root/'v.npy').astype(np.float64);assert np.array_equal(np.load(root/'ids.npy'),np.arange(len(table)))
        assert table.shape[1]==width
        norms=np.einsum('ij,ij->i',table,table);bound=(np.sqrt(norms.max())+np.linalg.norm(queries,axis=1).max())/np.sqrt(width)*scale*1.001
        edges=np.linspace(0,bound,501);hist=np.zeros(500,np.int64);targets=np.empty(len(queries));good=np.zeros(len(queries),bool);exact=np.zeros(len(queries),bool)
        total=0;sum_d=0.;sum_d2=0.;other_min=np.inf;other_max=0.
        for start in range(0,len(queries),32):
            q=queries[start:start+32];squared=np.sum(q*q,axis=1)[:,None]+norms[None,:]-2*q@table.T
            for j,row in enumerate(squared):
                idx=start+j;tid=truth[idx];weight=int(counts[idx]);delta=table[tid]-queries[idx]
                targets[idx]=np.sqrt(np.dot(delta,delta)/width)*scale;exact[idx]=np.array_equal(table[tid],queries[idx])
                                                                                           
                tol=1e-10*max(1.,norms.max(),float(np.dot(queries[idx],queries[idx])))
                candidates=np.flatnonzero(row<=row.min()+tol)
                direct=np.sum((table[candidates]-queries[idx])**2,axis=1);winners=candidates[direct==direct.min()]
                good[idx]=len(winners)==1 and winners[0]==tid
                row[tid]=np.dot(delta,delta)
                distances=np.sqrt(np.maximum(row,0)/width)*scale;wrong=np.concatenate([distances[:tid],distances[tid+1:]])
                hist+=np.histogram(wrong,bins=edges)[0]*weight;total+=len(wrong)*weight
                sum_d+=float(wrong.sum())*weight;sum_d2+=float(wrong@wrong)*weight
                other_min=min(other_min,float(wrong.min()));other_max=max(other_max,float(wrong.max()))
            print(backend,min(start+32,len(queries)),'/',len(queries),flush=True)
        expanded=targets[inverse];correct=good[inverse];offset=0;prompts_correct=0
        for n in lengths:prompts_correct+=bool(correct[offset:offset+n].all());offset+=n
        assert hist.sum()==total==len(labels)*(len(table)-1)
        np.savez_compressed(out/(backend+'_distribution.npz'),edges=edges,other_counts=hist,target_distances=expanded,correct=correct)
        stats=dict(target_mean=float(expanded.mean()),target_std=float(expanded.std()),target_min=float(expanded.min()),target_max=float(expanded.max()),other_mean=sum_d/total,other_std=float(np.sqrt(max(0,sum_d2/total-(sum_d/total)**2))),other_min=other_min,other_max=other_max,other_count=total,unique_correct=int(correct.sum()),exact_vectors=int(exact[inverse].sum()),token_accuracy=float(correct.mean()),exact_recovery_rate=prompts_correct/len(folders),scale=scale,table=root.name,table_sha256=meta.get('v_sha256',meta.get('codes_sha256')))
        summary['tables'][backend]=stats;summary['vocabulary']=len(table)
        save(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
