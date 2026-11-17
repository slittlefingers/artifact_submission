
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformer_blocks import CausalLlamaBlock,RMSNorm
                                                                                          
class Conv1DBlock(nn.Module):
 def __init__(self,a,b):
  super().__init__();self.net=nn.Sequential(nn.Conv1d(a,b,3,padding=1),nn.GroupNorm(16,b),nn.SiLU(),nn.Conv1d(b,b,3,padding=1),nn.GroupNorm(16,b),nn.SiLU())
 def forward(self,x):return self.net(x)
class TemporalUNet(nn.Module):
 def __init__(self):
  super().__init__();self.e0=Conv1DBlock(128,128);self.e1=Conv1DBlock(128,256);self.e2=Conv1DBlock(256,512);self.d1=Conv1DBlock(512+256,256);self.d0=Conv1DBlock(256+128,128)
 def forward(self,x):
  a=self.e0(x);b=self.e1(F.avg_pool1d(a,2,ceil_mode=True));c=self.e2(F.avg_pool1d(b,2,ceil_mode=True));d=self.d1(torch.cat([F.interpolate(c,size=b.shape[-1],mode='linear',align_corners=False),b],1));return self.d0(torch.cat([F.interpolate(d,size=a.shape[-1],mode='linear',align_corners=False),a],1))
class AudioReconstructor(nn.Module):
 def __init__(self,checkpointing=True):
  super().__init__();self.input=nn.Linear(1024,4096);self.block=CausalLlamaBlock();self.norm=RMSNorm(4096);self.aligned=nn.Linear(4096,1536);self.inverse_align=nn.Sequential(nn.Linear(1536,3072),nn.GELU(),nn.Linear(3072,768));self.unmerge=nn.Linear(768,4*128);self.unet=TemporalUNet();self.head=nn.Conv1d(128,257,3,padding=1);self.checkpointing=checkpointing
 def forward(self,v,frames):
  b,n,_=v.shape;z=self.input(v)
  z=checkpoint(self.block,z,use_reentrant=False) if self.training and self.checkpointing else self.block(z)
  z=self.unmerge(self.inverse_align(self.aligned(self.norm(z)))).view(b,n,4,128).permute(0,3,1,2).reshape(b,128,n*4)
  z=checkpoint(self.unet,z,use_reentrant=False) if self.training and self.checkpointing else self.unet(z)
  prediction=self.head(z)
  return F.interpolate(prediction,size=int(frames),mode='linear',align_corners=False)
