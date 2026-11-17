
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .common import digest

class LearnedCandidates:
    def __init__(self, checkpoint, quantization, model, mode, layer, model_sha256,
                 training_protocol, device='cuda'):
        self.path=Path(checkpoint)
        self.quant=json.loads(Path(quantization).read_text(encoding='utf-8'))
        provenance=json.loads(Path(training_protocol).read_text(encoding='utf-8'))
        if provenance['model_sha256']!=model_sha256 or provenance['layer']!=layer:
            raise ValueError('Classifier training model/layer differs from replay model')
        ck=torch.load(self.path,map_location='cpu',weights_only=False)
        cfg=ck['config']
        if cfg['model']!=model or cfg['mode']!=mode or cfg['layer_index']!=layer or mode not in ('v','kv'):
            raise ValueError('Classifier mode/model/layer mismatch')
        self.mode=mode;self.device=device;self.mean=np.asarray(ck['mean']);self.std=np.asarray(ck['std'])
        self.width=sum(q['width'] for q in self.quant)
        self.vocab_size=ck['state']['net.4.weight'].shape[0]
        width=self.width if mode=='kv' else self.quant[1]['width']
        self.net=nn.Sequential(nn.Linear(width,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,self.vocab_size,bias=False))
        self.net.load_state_dict({k.removeprefix('net.'):v for k,v in ck['state'].items()})
        self.net.to(device).eval()
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or not (self.std>0).all():
            raise ValueError('Invalid training normalization')
        self.last_target=None;self.last_order=None;self.last_logits=None;self.last_space=None
        self.metadata=dict(checkpoint_sha256=digest(self.path),mode=mode,model=model,layer=layer,
            model_sha256=model_sha256,training_protocol_sha256=digest(Path(training_protocol)),
            role='Proposal ranking only; full vocabulary remains searchable, acceptance requires measured K/V',
            identity=dict(vocab_size=self.vocab_size))
                                                                           
        with torch.inference_mode():
            for _ in range(3):self.net(torch.zeros((1,width),device=device)).cpu()

    def validate_quantization(self,quant):
        if quant!=self.quant:raise ValueError('Observation quantization differs from classifier training')

    def order(self,target,space='original'):
        target=np.asarray(target,dtype=np.float64)
        if target.shape!=(self.width,) or not np.isfinite(target).all():raise ValueError('Invalid observed KV')
        if self.last_space==space and self.last_target is not None and np.array_equal(target,self.last_target):
            return self.last_order
        if space not in ('original','lsb'):raise ValueError(space)
        parts=[];offset=0
        for key,q in zip(['k','v'],self.quant):
            value=target[offset:offset+q['width']].astype(np.float32);offset+=q['width']
            if self.mode!='kv' and key!=self.mode:continue
            if space=='lsb':value=(value-q['zero_point'])*q['scale']
            parts.append(value)
        x=(np.concatenate(parts)-self.mean)/self.std
        with torch.inference_mode():
            logits=self.net(torch.from_numpy(x)[None].to(self.device))[0].float().cpu().numpy()
        if logits.shape!=(self.vocab_size,) or not np.isfinite(logits).all():raise ValueError('Invalid classifier logits')
        order=np.argsort(-logits,kind='stable')
        self.last_target=target.copy();self.last_space=space;self.last_order=order;self.last_logits=logits
        return order

    def select(self,target,available,count,space='original'):
        order=self.order(target,space)
        return order[np.asarray(available,dtype=bool)[order]][:count]

    def validate_runtime(self,output):
        info=json.loads((Path(output)/'device.json').read_text(encoding='utf-8'))
        if info['model_sha256']!=self.metadata['model_sha256']:raise ValueError('Unexpected replay model')
