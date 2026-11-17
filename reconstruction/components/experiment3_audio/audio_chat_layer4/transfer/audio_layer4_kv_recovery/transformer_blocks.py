


import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class RMSNorm(nn.Module):
 def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
 def forward(self,x):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).to(x.dtype)*self.weight

def rotary(x,pos):
 d=x.shape[-1];freq=torch.exp(torch.arange(0,d,2,device=x.device,dtype=torch.float32)*(-math.log(500000.)/d));theta=pos.float()[:,None]*freq
 cos=torch.cat([theta.cos(),theta.cos()],-1)[None,None];sin=torch.cat([theta.sin(),theta.sin()],-1)[None,None];a,b=x.chunk(2,-1)
 return (x*cos.to(x.dtype)+torch.cat([-b,a],-1)*sin.to(x.dtype))
class CausalLlamaBlock(nn.Module):
 def __init__(self,width=4096,intermediate=14336,heads=32,kv_heads=8):
  super().__init__();self.width=width;self.heads=heads;self.kv_heads=kv_heads;self.dim=width//heads
  self.norm1=RMSNorm(width);self.norm2=RMSNorm(width)
  self.q=nn.Linear(width,width,bias=False);self.k=nn.Linear(width,kv_heads*self.dim,bias=False);self.v=nn.Linear(width,kv_heads*self.dim,bias=False);self.o=nn.Linear(width,width,bias=False)
  self.gate=nn.Linear(width,intermediate,bias=False);self.up=nn.Linear(width,intermediate,bias=False);self.down=nn.Linear(intermediate,width,bias=False)
 def forward(self,x):
  b,n,c=x.shape;z=self.norm1(x);q=self.q(z).view(b,n,self.heads,self.dim).transpose(1,2);k=self.k(z).view(b,n,self.kv_heads,self.dim).transpose(1,2);v=self.v(z).view(b,n,self.kv_heads,self.dim).transpose(1,2)
  pos=torch.arange(n,device=x.device);q=rotary(q,pos);k=rotary(k,pos)
                                                                                   
  k=k.repeat_interleave(self.heads//self.kv_heads,1);v=v.repeat_interleave(self.heads//self.kv_heads,1)
  a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
  x=x+self.o(a.transpose(1,2).reshape(b,n,c));z=self.norm2(x)
  return x+self.down(F.silu(self.gate(z))*self.up(z))
