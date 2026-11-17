
import json,time,random,math,hashlib,argparse,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from common import OUT,HERE
from audio_model import AudioReconstructor
SEED=20261101

def atomic_save(value,path):
 temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)
def write_json(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
class AudioData(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with np.load(r['cache']) as z:v=torch.from_numpy(z['v'].astype('float32'));target=torch.from_numpy(z['log_magnitude'].copy())
  return v,target

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(6);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 splits=json.loads((OUT/'splits.json').read_text());stats=np.load(OUT/'stats.npz');vm=torch.tensor(stats['vmean'],device='cuda');vs=torch.tensor(stats['vstd'],device='cuda');lm=torch.tensor(stats['logmean'],device='cuda')[None,:,None];ls=torch.tensor(stats['logstd'],device='cuda')[None,:,None]
 model=AudioReconstructor().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01);steps_per_epoch=math.ceil(len(splits['train'])/4);total=steps_per_epoch*3
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/100,1.) if step<100 else max((total-step)/max(total-100,1),0.))
 config={'architecture':'V256->4096, one causal GQA/SwiGLU Transformer,1536->3072->768 MLP,4 temporal subframes/token,1D UNet128/256/512,257-frequency head','parameters':sum(p.numel() for p in model.parameters()),'initialize':None,'input':'layer0 V only, audio positions; no K, transcript, original waveform, hidden embedding or true phase in inference','target':'normalized log1p linear-STFT magnitude, FFT512 hop160 at16kHz','loss':'standardized log-magnitude MSE +0.1 spectral convergence','epochs':3,'batch':1,'gradient_accumulation':4,'effective_batch':4,'lr':3e-5,'warmup':100,'schedule':'linear decay','selection':'validation raw log-magnitude RMSE','seed':SEED,'split_sha256':hashlib.sha256((OUT/'splits.json').read_bytes()).hexdigest(),'decoder':'32-iteration Griffin-Lim; random phase with deterministic per-case seed, no target phase','duration_side_information':'known recording length determines output frame count','limits':'phase is not recovered by this network; waveform similarity may be low despite magnitude recovery; three-epoch feasibility run, not converged ceiling'}
 print('CONFIG',json.dumps(config),flush=True)
 def losses(v,target):
  with torch.autocast('cuda',dtype=torch.bfloat16):pred=model((v-vm)/vs,target.shape[-1])
  pred=pred.float();normalized_target=(target-lm)/ls;mse=F.mse_loss(pred,normalized_target);logpred=pred*ls+lm
  mag=torch.expm1(logpred.clamp(0,8));truth=torch.expm1(target);sc=torch.linalg.vector_norm(mag-truth)/torch.linalg.vector_norm(truth).clamp_min(1e-6)
  return mse+.1*sc,pred,sc
 if a.smoke:
  row=max(splits['train'],key=lambda r:r['tokens']);v,target=AudioData([row])[0];v=v.cuda()[None];target=target.cuda()[None];model.train();loss,_,_=losses(v,target);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(grad);optimizer.step();torch.cuda.synchronize();print('AUDIO_SMOKE_OK',json.dumps({'tokens':row['tokens'],'frames':row['frames'],'loss':loss.item(),'peak_gb':torch.cuda.max_memory_allocated()/2**30}),flush=True);return
 if (OUT/'last.pt').exists() and not a.resume:raise RuntimeError('Run exists, use --resume')
 write_json(OUT/'config.json',config);epoch0=0;step=0;best=math.inf
 if a.resume:
  ck=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);epoch0=ck['epoch'];step=ck['step'];best=ck['best'];del ck
 started=time.time()
 for epoch in range(epoch0+1,4):
  rows=list(splits['train']);random.Random(SEED+epoch).shuffle(rows);loader=DataLoader(AudioData(rows),batch_size=1,num_workers=4,pin_memory=True);model.train();optimizer.zero_grad(set_to_none=True);total_loss=0.;tick=time.time()
  for micro,(v,target) in enumerate(loader,1):
   v=v.cuda(non_blocking=True);target=target.cuda(non_blocking=True);loss,_,_=losses(v,target);assert torch.isfinite(loss);(loss/4).backward();total_loss+=loss.item()
   if micro%4==0 or micro==len(loader):
    if micro%4:
     for parameter in model.parameters():
      if parameter.grad is not None:parameter.grad.mul_(4/(micro%4))
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
   if micro%100==0:
    status={'state':'training','epoch':epoch,'utterance':micro,'utterances':len(loader),'step':step,'mean_loss':total_loss/micro,'elapsed_seconds':time.time()-started,'epoch_seconds':time.time()-tick,'peak_gb':torch.cuda.max_memory_allocated()/2**30};write_json(OUT/'status.json',status);print('PROGRESS',json.dumps(status),flush=True)
  train_loss=total_loss/len(loader);model.eval();val_mse=0.;val_sc=0.;val_nmse=0.
  with torch.inference_mode():
   for v,target in DataLoader(AudioData(splits['val']),batch_size=1,num_workers=4):
    v=v.cuda();target=target.cuda();_,pred,sc=losses(v,target);val_mse+=float(F.mse_loss(pred*ls+lm,target));val_nmse+=float(F.mse_loss(pred,(target-lm)/ls));val_sc+=float(sc)
  metric={'epoch':epoch,'train_loss':train_loss,'val_log_rmse':math.sqrt(val_mse/len(splits['val'])),'val_normalized_mse':val_nmse/len(splits['val']),'val_spectral_convergence':val_sc/len(splits['val']),'seconds':time.time()-tick,'step':step}
  improved=metric['val_log_rmse']<best;best=min(best,metric['val_log_rmse']);state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'step':step,'best':best,'config':config,'metrics':metric};atomic_save(state,OUT/'last.pt')
  if improved:atomic_save({'model':model.state_dict(),'epoch':epoch,'config':config,'metrics':metric},OUT/'best.pt')
  with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
  print('EPOCH',json.dumps(metric),flush=True);write_json(OUT/'status.json',{'state':'training',**metric})
 write_json(OUT/'training_complete.json',{'epochs':3,'best_val_log_rmse':best,'seconds':time.time()-started});print('AUDIO_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
