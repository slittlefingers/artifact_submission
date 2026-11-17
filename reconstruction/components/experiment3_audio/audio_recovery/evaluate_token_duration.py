"""Frozen layer-0 V or layer-4 K/V audio recovery; duration inferred solely from the audio-token count."""
import argparse,hashlib,importlib.util,io,json,sys,time,wave,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;CODE=HERE.parents[1];ROOT=CODE.parents[1]
sys.path.insert(0,str(HERE))
from common import SR,N_FFT,HOP,write_wave,magnitude
from griffin_compat import griffinlim
ARCHITECTURES={0:HERE/'audio_model.py',4:CODE/'experiment3_audio/audio_chat_layer4/transfer/audio_layer4_kv_recovery/audio_model.py'}

def load_module(name,path):
 
 sys.path.insert(0,str(Path(path).parent))
 try:
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
 finally:sys.path.pop(0)
class Package:
 
 def __init__(self,path):self.path=Path(path);self.zip=zipfile.ZipFile(self.path) if self.path.suffix=='.zip' else None
 def bytes(self,name):return self.zip.read(name) if self.zip else (self.path/name).read_bytes()
class Recordings:
 
 def __init__(self,path):self.path=Path(path);self.zip=zipfile.ZipFile(self.path) if self.path.suffix=='.zip' else None
 def read(self,name):
  raw=self.zip.read('recordings/test-clean/'+name) if self.zip else (self.path/'test-clean'/name).read_bytes()
  return raw
def decode_wave(raw):
 with wave.open(io.BytesIO(raw),'rb') as f:
  assert f.getnchannels()==1 and f.getframerate()==SR and f.getsampwidth()==2
  return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype('float32')/32768.
def griffin(logs,length,seed):
 torch.manual_seed(seed)
 return griffinlim(torch.expm1(logs.clamp(0,8)),window=torch.hann_window(N_FFT,device=logs.device),n_fft=N_FFT,hop_length=HOP,win_length=N_FFT,power=1.,n_iter=32,momentum=.99,length=int(length),rand_init=True)
def sisdr(pred,target):
 pred=pred-pred.mean();target=target-target.mean();projection=(pred*target).sum()/target.square().sum().clamp_min(1e-10)*target
 return float(10*torch.log10((projection.square().sum()+1e-10)/((pred-projection).square().sum()+1e-10)))
def spectrum_metrics(pred,truth):
 rmse=float((pred-truth).square().mean().sqrt());pm=torch.expm1(pred.clamp(0,8));tm=torch.expm1(truth);sc=float(torch.linalg.vector_norm(pm-tm)/torch.linalg.vector_norm(tm).clamp_min(1e-6))
 return {'log_magnitude_rmse':rmse,'spectral_convergence':sc}
def pad_to(a,length):return F.pad(a,(0,length-a.shape[-1]))
def aligned_metrics(pred,truth):
 n=max(pred.shape[-1],truth.shape[-1]);return spectrum_metrics(pad_to(pred,n),pad_to(truth,n))
def audio_codes(path,layer):
 
 with np.load(path) as z:
  pos=z['modality_positions'][z['modality_sentinel_ids']==-2];ix=np.searchsorted(z['positions'],pos)
  assert np.array_equal(z['positions'][ix],pos) and z['v'].dtype==np.int8
  return z['v'][ix] if layer==0 else np.concatenate([z['k'][ix],z['v'][ix]],axis=1)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--layer',type=int,choices=sorted(ARCHITECTURES),required=True)
 p.add_argument('--checkpoint',type=Path,help='Model archive or directory with best.pt and stats.npz (default: assets/model_gemma_audio_layerL.zip)')
 p.add_argument('--samples',type=Path,default=ROOT/'assets/kv/5.4/librispeech_test',help='Directory with index.json and samples/<id>/')
 p.add_argument('--recordings',type=Path,default=ROOT/'assets/data_librispeech.zip',help='data_librispeech.zip or its extracted recordings/ directory')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
 a=p.parse_args();device=a.device;OUT=a.output;OUT.mkdir(parents=True,exist_ok=True)
 def save(name,data):
  q=OUT/name;t=q.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(q)
 torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 package=Package(a.checkpoint or ROOT/f'assets/model_gemma_audio_layer{a.layer}.zip');recordings=Recordings(a.recordings)
 rows=json.loads((a.samples/'index.json').read_text(encoding='utf8'))[:a.limit]
 stats=np.load(io.BytesIO(package.bytes('stats.npz')))
 vm=torch.tensor(stats['vmean'],device=device);vs=torch.tensor(stats['vstd'],device=device);lm=torch.tensor(stats['logmean'],device=device)[:,None];ls=torch.tensor(stats['logstd'],device=device)[:,None]
 ck=torch.load(io.BytesIO(package.bytes('best.pt')),map_location='cpu',weights_only=True)
 module=load_module(f'audio_model_layer{a.layer}',ARCHITECTURES[a.layer]);model=module.AudioReconstructor(checkpointing=False).to(device).eval();model.load_state_dict(ck['model']);epoch=ck['epoch'];del ck
 label='V' if a.layer==0 else 'KV'
 save('protocol.json',dict(method=f'Fixed frozen checkpoint, duration estimated only from valid audio {label} count N',layer=a.layer,estimated_samples='640*N at16kHz (40ms/token)',estimated_stft_frames='floor(estimated_samples/160)+1 =4*N+1',inference_side_information='No reference duration, samples or frames used by reconstructor or Griffin-Lim',evaluation_alignment='Common 10ms frame/time axes, right zero-padding shorter spectrum or waveform to union length',seed='SHA256(id)-derived',samples=len(rows),checkpoint=str(a.checkpoint or f'assets/model_gemma_audio_layer{a.layer}.zip')))
 (OUT/'audio').mkdir(exist_ok=True);records=[];start=time.monotonic()
 autocast=torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else torch.autocast('cpu',dtype=torch.bfloat16)
 with torch.inference_mode():
  for i,r in enumerate(rows):
   sid=r['id'];folder=a.samples/'samples'/sid;path=folder/f'layer{a.layer}.npz'
   assert r['passed'] and hashlib.sha256(path.read_bytes()).hexdigest()==r['layers'][str(a.layer)]['npz_sha256']
   x=torch.tensor(audio_codes(path,a.layer).astype('float32'),device=device)[None]
   n=x.shape[1];samples=n*640;frames=samples//160+1
   seed=int(hashlib.sha256(sid.encode()).hexdigest()[:8],16);gen=torch.Generator(device=device).manual_seed(seed)
   shuffled=x[:,torch.randperm(n,generator=gen,device=device)]
   with autocast:
    pred=model((x-vm)/vs,frames)[0].float()*ls+lm
    wrong=model((shuffled-vm)/vs,frames)[0].float()*ls+lm
   pred=pred.clamp(0,8);wrong=wrong.clamp(0,8)
   wave_out=griffin(pred,samples,seed);assert len(wave_out)==samples
   clipping=write_wave(OUT/'audio'/f'{sid}_recovered.wav',wave_out.cpu().numpy())
   sample=json.loads((folder/'sample.json').read_text(encoding='utf8'));raw=recordings.read(sample['audio_name'])
   assert hashlib.sha256(raw).hexdigest()==sample['audio_sha256']
   reference=torch.tensor(decode_wave(raw),device=device);truth=torch.log1p(magnitude(reference));length=max(len(wave_out),len(reference))
   record=dict(id=sid,tokens=n,estimated_samples=samples,estimated_frames=frames,estimated_duration_seconds=samples/SR,reference_duration_seconds=len(reference)/SR,duration_error_seconds=(samples-len(reference))/SR,methods={label+'_recovered':aligned_metrics(pred,truth),'within_utterance_shuffled_'+label:aligned_metrics(wrong,truth),'training_mean_spectrum':aligned_metrics(lm.expand(-1,frames),truth)},recovered_waveform_si_sdr_db=sisdr(pad_to(wave_out,length),pad_to(reference,length)),clipping_fraction=clipping)
   records.append(record)
   if (i+1)%25==0:save('progress.json',dict(state='running',completed=i+1,total=len(rows),elapsed_seconds=time.monotonic()-start));print('EVALUATED',i+1,round(time.monotonic()-start,1),flush=True)
 summary={method:{metric:float(np.mean([r['methods'][method][metric] for r in records])) for metric in ['log_magnitude_rmse','spectral_convergence']} for method in records[0]['methods']}
 err=np.array([r['duration_error_seconds'] for r in records])
 result=dict(layer=a.layer,utterances=len(records),best_epoch=epoch,spectral_summary=summary,recovered_waveform_si_sdr_db_mean=float(np.mean([r['recovered_waveform_si_sdr_db'] for r in records])),duration_error_seconds=dict(mean=float(err.mean()),mae=float(np.abs(err).mean()),max_absolute=float(np.abs(err).max())),elapsed_seconds=time.monotonic()-start)
 save('per_utterance.json',records);save('results.json',result);save('progress.json',dict(state='complete',**result));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
