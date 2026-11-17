
import numpy as np
import torch
class TorchRidge:
 def __init__(self,path,max_anchors=512,lambda_per_anchor=.01,score_chunk_size=8192):
  if type(max_anchors) is not int or max_anchors<2 or lambda_per_anchor<=0 or score_chunk_size<1:raise ValueError('Invalid ridge parameters')
  torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
  self.e=torch.from_numpy(np.load(path)).cuda();self.stats={'fit_calls':0};self.max_anchors=max_anchors;self.lambda_per_anchor=lambda_per_anchor;self.score_chunk_size=score_chunk_size
 def neighbors(self,center,count):
  
  with torch.inference_mode():
   if not hasattr(self,'neighbor_norm'):self.neighbor_norm=self.e.square().sum(1).sqrt().clamp_min(1e-12)
   distances=(1-(self.e@self.e[center])/(self.neighbor_norm*self.neighbor_norm[center])).cpu().numpy()
  ids=np.arange(len(distances));ids=ids[ids!=center]
  from .search import lowest_scored_ids
  return lowest_scored_ids(ids,distances,count)
 def scores(self,queried,measured,target,distances):
  with torch.inference_mode():
   q=np.asarray(queried);keep=np.arange(len(q))
   if len(keep)>self.max_anchors:
    nearest=self.max_anchors//2;remaining=self.max_anchors-nearest
    near=np.argsort(distances,kind='stable')[:nearest];rest=np.setdiff1d(keep,near);keep=np.r_[near,rest[np.linspace(0,len(rest)-1,remaining,dtype=int)]]
   x=self.e[torch.as_tensor(q[keep],device='cuda')];mean=x.mean(0);scale=x.std(0).clamp_min(1e-4);x=(x-mean)/scale
   y=torch.as_tensor(np.asarray(measured)[keep],dtype=torch.float32,device='cuda');ym=y.mean(0);ys=y.std(0).clamp_min(1e-4);y=(y-ym)/ys
   xm=x.mean(0);yb=y.mean(0);xc=x-xm;yc=y-yb
   dual=torch.linalg.solve(xc@xc.T+torch.eye(len(x),device='cuda')*(self.lambda_per_anchor*len(x)),yc)
   b=dual.double()*ys.double();t=torch.as_tensor(target,dtype=torch.float64,device='cuda')-(ym.double()+yb.double()*ys.double())
   orth,upper=torch.linalg.qr(b.T,mode='reduced');projected=t@orth;constant=((t-projected@orth.T)**2).sum()
   weights=(xc/scale).T;offset=(mean/scale+xm)@xc.T;result=[]
   for chunk in self.e.split(self.score_chunk_size):
    g=chunk@weights-offset;res=g@upper.T.float()-projected.float();result.append((res.square().sum(-1)+constant.float()).cpu())
   self.stats['fit_calls']+=1
   return torch.cat(result).numpy()
