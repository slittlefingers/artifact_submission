
from pathlib import Path
import numpy as np
from . import MediaUnavailable
from ..common import dump,digest
from ..search import kv_distance


class GemmaMediaReplay:
    def __init__(self,backend,settings,base):
        self.backend=backend;self.settings=settings;self.base=Path(base);self.ends={}

    def end_embedding(self,kind,out):
        if kind in self.ends:return self.ends[kind]
        b=self.backend
        from ..litertlm import FlatReader
        import hashlib
        section=b.sections['tf_lite_end_of_vision' if kind=='image' else 'tf_lite_end_of_audio']
        offset,length=section['offset'],section['length']
        if offset%4096:raise ValueError('Expected aligned model section')
        raw=b.command('exec-out','dd','if='+b.device_model,'bs=4096',
                      'skip='+str(offset//4096),'count='+str((length+4095)//4096),'status=none')[:length]
        if raw[4:8]!=b'TFL3':raise ValueError('Invalid end-embedding section')
        r=FlatReader(raw);root=r.scalar_at(0);graph=r.vector(root,2,tables=True)[0]
        if r.vector(graph,3,tables=True):raise ValueError('End representation is not constant in this model')
        tensor=r.vector(graph,0,tables=True)[r.vector(graph,2)[0]]
        buffer=r.vector(root,4,tables=True)[r.scalar(tensor,2)];vector=r.table(buffer,0)
        if r.scalar_at(vector)!=1536*4:raise ValueError('Unexpected end-embedding width')
        end=np.frombuffer(raw,dtype='<f4',count=1536,offset=vector+4).copy()
        if not np.isfinite(end).all():raise ValueError('Invalid end embedding')
        np.save(out/f'end_of_{kind}.npy',end)
        dump(out/f'end_of_{kind}_provenance.json',dict(model_sha256=b.model_sha256,section=section,
             section_sha256=hashlib.sha256(raw).hexdigest()))
        self.ends[kind]=end;return end

    def __call__(self,result,request):
        b=self.backend;kind=result['modality'];n=len(request['targets']);start=request['span']['start']
        if b.position!=start:raise MediaUnavailable('media_replay_prefix_position_mismatch')
        out=request['output'];end=self.end_embedding(kind,out)
        if kind=='audio':
            features,call=b.encode_audio_wav(result['artifact'])
            if len(features)!=n:raise MediaUnavailable(f'recovered_audio_token_count_mismatch: expected {n}, encoded {len(features)}')
            dump(out/'audio_encoder.json',call)
        elif kind=='image':
            blob=Path(result.get('replay_artifact',result['artifact'])).read_bytes()
            if len(blob)>900000:
                for offset in range(0,len(blob),512000):b.rpc('upload_collection_image',offset=offset,data_hex=blob[offset:offset+512000].hex())
                args={}
            else:args=dict(image_hex=blob.hex())
                                                                                  
            encoded=b.rpc('collect_image',collect_kv=False,position=b.position,**args)
            call=b.calls[-1];rows=call['image_rows']
            grid=[call['resized_height']//48,call['resized_width']//48]
            if rows!=n or grid!=result['grid_hw']:
                raise MediaUnavailable('recovered_image_preprocessing_grid_mismatch')
            features=np.frombuffer(encoded,dtype='<f4',offset=rows*768*4,count=rows*1536).reshape(rows,1536)
        else:raise ValueError('Unknown media modality')
                                                                               
        values=features;measured=[]
        for first in range(0,len(values),b.batch_capacity):
            chunk=np.asarray(values[first:first+b.batch_capacity],dtype='<f4')
            raw=b.rpc('collect_embeddings',rows=len(chunk),position=b.position,features_hex=chunk.tobytes().hex())
            measured.append(np.frombuffer(raw,dtype=b.dtype).reshape(len(chunk),b.width).copy())
        codes=np.concatenate(measured);np.save(out/'replayed_kv.npy',codes)
        distances=kv_distance(codes,request['targets'])
        report=dict(status='complete',end_consumed=False,media_rmse=float(np.mean(distances)),
            grid_score_rmse=float(np.sqrt(np.mean((codes.astype(np.float64)-request['targets'])**2))),
            prefix_source=f'Recovered {kind} through deployed encoder; no observed KV imported',
            artifact_sha256=digest(result['artifact']))
        dump(out/'replay.json',report);return report

    def finish_end(self,request):
        
        b=self.backend;kind=request['modality'];position=request['position']
        if b.position!=position:raise MediaUnavailable('media_end_prefix_position_mismatch')
        end=self.end_embedding(kind,request['output'])
        raw=b.rpc('collect_embeddings',rows=1,position=b.position,features_hex=end.astype('<f4').tobytes().hex())
        codes=np.frombuffer(raw,dtype=b.dtype).reshape(1,b.width).copy()
                                                                               
        b.rpc('prefix_embedding_logits',position=b.position,features_hex=end.astype('<f4').tobytes().hex())
        distance=float(kv_distance(codes[0],request['target'],request['config'].metric))
        passed=distance<request['config'].epsilon
        result=dict(status='complete',position=position,modality=kind,representation='model_constant_embedding',
            accepted_distance=distance,accepted_codes=codes[0].tolist(),threshold_passed=passed,approximate=not passed,
            queries=1,stop_reason='proposed_media_end_replayed',
            boundary_identity_verified=False,
            note='End position was proposed by V similarity; replay residual is logged, not proof of identity.')
        dump(request['output']/'end_replay.json',result);return result


def create(backend,settings,base):return GemmaMediaReplay(backend,settings,base)
