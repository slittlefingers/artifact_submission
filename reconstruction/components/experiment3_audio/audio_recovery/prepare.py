
import json,hashlib,io,time,random,collections
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from common import SOURCE,OUT,SR,HOP,read_wave,magnitude
SEED=20261101

def inspect(task):
 row,entry,role=task;directory=SOURCE/'data'/row['split']/'samples'/row['id'];path=SOURCE/'dataset'/row['audio'];raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['audio_sha256']
 meta=json.loads((directory/'sample.json').read_text());assert meta['audio_sha256']==row['audio_sha256'] and meta['generated_tokens']==0 and meta['capture_stage']=='after_prefill'
 assert meta['id']==row['id']
 validation=json.loads((directory/'validation.json').read_text());assert validation['passed'] and entry['passed']
 for layer in ['0','4']:
  assert hashlib.sha256((directory/f'layer{layer}.npz').read_bytes()).hexdigest()==entry['layers'][layer]['npz_sha256']
 with np.load(directory/'layer0.npz',allow_pickle=False) as z:
  q=z['v'];pos=z['positions'];mod=z['modality_positions'];ids=z['modality_sentinel_ids'];audio=mod[ids==-2];indices=np.searchsorted(pos,audio)
  assert np.array_equal(pos[indices],audio) and q.dtype==np.int8 and q.shape[1]==256
  v=q[indices].copy();assert len(v)>0 and len(v)==len(set(indices.tolist()))
 waveform=read_wave(path);assert abs(len(waveform)/SR-row['duration_seconds'])<1/SR
 torch.set_num_threads(1);logs=torch.log1p(magnitude(torch.from_numpy(waveform))).numpy().astype('float32')
 cache=OUT/'cache'/f"{row['id']}.npz";np.savez(cache,v=v,log_magnitude=logs)
 result={k:row[k] for k in ['id','split','audio','audio_sha256','speaker_id','duration_seconds']};result.update({'role':role,'cache':str(cache),'waveform':str(path),'samples':len(waveform),'frames':logs.shape[1],'tokens':len(v)})
 if role!='train':return result,None
 vf=v.astype('float64');lf=logs.astype('float64')
 return result,{'vsum':vf.sum(0),'vsq':(vf*vf).sum(0),'vn':len(vf),'lsum':lf.sum(1),'lsq':(lf*lf).sum(1),'ln':lf.shape[1]}

def main():
 OUT.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True);started=time.time();source={};entries={}
 for split in ['train-clean-100','test-clean']:
  manifest=SOURCE/'dataset'/f'{split}.jsonl';source[split]=[json.loads(line) for line in manifest.read_text().splitlines()];protocol=json.loads((SOURCE/'data'/split/'protocol.json').read_text())
  assert hashlib.sha256(manifest.read_bytes()).hexdigest()==protocol['manifest_sha256']
  assert protocol['capture_stage']=='after_prefill' and protocol['generated_tokens']==0
  entries[split]={r['id']:r for r in json.loads((SOURCE/'data'/split/'index.json').read_text())};assert len(entries[split])==len(source[split])
 speakers=sorted({r['speaker_id'] for r in source['train-clean-100']});random.Random(SEED).shuffle(speakers)
 counts=collections.Counter(r['speaker_id'] for r in source['train-clean-100']);val_speakers=set();n=0
 for speaker in speakers:
  if n>=1000:break
  val_speakers.add(speaker);n+=counts[speaker]
 test_speakers={r['speaker_id'] for r in source['test-clean']};assert not (test_speakers & set(speakers))
 tasks=[];hash_roles={}
 for split,rows in source.items():
  for row in rows:
   role='test' if split=='test-clean' else 'val' if row['speaker_id'] in val_speakers else 'train'
   assert row['audio_sha256'] not in hash_roles,'Duplicate source waveform';hash_roles[row['audio_sha256']]=role
   tasks.append((row,entries[split][row['id']],role))
 splits={k:[] for k in ['train','val','test']};vs=np.zeros(256);vss=np.zeros(256);vn=0;ls=np.zeros(257);lss=np.zeros(257);ln=0
 with ProcessPoolExecutor(max_workers=6) as pool:
  for i,(row,stats) in enumerate(pool.map(inspect,tasks,chunksize=8),1):
   splits[row['role']].append(row)
   if stats:
    vs+=stats['vsum'];vss+=stats['vsq'];vn+=stats['vn'];ls+=stats['lsum'];lss+=stats['lsq'];ln+=stats['ln']
   if i%500==0:print('AUDIO_AUDITED',i,'/',len(tasks),'seconds',round(time.time()-started,1),flush=True)
 vm=vs/vn;vstd=np.sqrt(np.maximum(vss/vn-vm**2,1e-8));lm=ls/ln;lstd=np.sqrt(np.maximum(lss/ln-lm**2,1e-6))
 np.savez(OUT/'stats.npz',vmean=vm.astype('float32'),vstd=vstd.astype('float32'),logmean=lm.astype('float32'),logstd=lstd.astype('float32'))
 (OUT/'splits.json').write_text(json.dumps(splits,indent=2));chosen=sorted(splits['test'],key=lambda r:r['id'])[:6];(OUT/'fixed_case_ids.json').write_text(json.dumps([r['id'] for r in chosen],indent=2))
 audit={'counts':{k:len(v) for k,v in splits.items()},'speakers':{k:len({r['speaker_id'] for r in v}) for k,v in splits.items()},'hours':{k:sum(r['duration_seconds'] for r in v)/3600 for k,v in splits.items()},'normalization':'train-only per-channel V256 and per-frequency log1p magnitude statistics','split':'official test-clean1000 untouched; validation reserves whole speakers from train-clean-100; train/val/test speakers disjoint','capture':'layer0 INT8 V audio sentinel -2 only; -4 end marker and all text excluded; layer0/4 capture file hashes and original WAV SHA checked','labels':'transcript is not cached, used as input, or used as training target','spectrogram':'16kHz mono; FFT512, hop160, Hann512, center=True, linear257-bin magnitude; no Mel inversion','duration_side_information':'known original recording length sets output spectrogram length; no per-record energy normalization','seconds':time.time()-started}
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2));print('AUDIO_DATA_READY',json.dumps(audit),flush=True)
if __name__=='__main__':main()
