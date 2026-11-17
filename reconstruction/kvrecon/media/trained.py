





from contextlib import nullcontext, contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from . import MediaUnavailable
from ..common import dump, digest


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class FrozenMediaDecoder:
    def __init__(self, package, modality, device='cuda', griffin_module=None,
                 samples_per_token=640, frames_per_token=4, sample_rate=16000):
        self.package=Path(package);self.modality=modality;self.device=device
        self.griffin_module=griffin_module
        self.samples_per_token=samples_per_token;self.frames_per_token=frames_per_token;self.sample_rate=sample_rate
        if modality not in ('image','audio'):raise ValueError('Unknown modality')
        if modality=='audio' and (sample_rate!=16000 or any(type(x) is not int or x<=0 for x in (samples_per_token,frames_per_token))):
            raise ValueError('This frozen audio package requires 16 kHz and positive token ratios')
        self.model=None

    @contextmanager
    def weight_file(self, name):
        folder=self.package if self.modality=='image' else self.package/'audio'
        if (folder/name).is_file():
            with (folder/name).open('rb') as stream:yield stream
        else:
            archive=Path(__file__).resolve().parents[3]/'assets'/f'model_gemma_{self.modality}_layer4.zip'
            with zipfile.ZipFile(archive) as bundle, bundle.open(name) as stream:
                yield stream

    def weight_digest(self, name):
        h=hashlib.sha256()
        with self.weight_file(name) as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        return h.hexdigest()

    def load(self):
        if self.model is not None:return
        import torch
        p=self.package
        if self.modality=='image':
            module=load_module('kvrecon_frozen_image',p/'model.py')
            model=module.NativeGridReconstructor(input_dim=1024,checkpointing=False)
            with self.weight_file('quantization.json') as stream:self.quant=json.load(stream)
            model_dir=p
        else:
                                                                               
                                                          
            previous=sys.modules.get('transformer_blocks')
            sys.modules['transformer_blocks']=load_module('kvrecon_audio_blocks',p/'transformer_blocks.py')
            try:module=load_module('kvrecon_frozen_audio',p/'audio_model.py')
            finally:
                if previous is None:sys.modules.pop('transformer_blocks',None)
                else:sys.modules['transformer_blocks']=previous
            model=module.AudioReconstructor(checkpointing=False);model_dir=p/'audio'
            if self.griffin_module is None:raise ValueError('Specify the tested Griffin-Lim implementation')
            self.griffin=load_module('kvrecon_griffin',self.griffin_module).griffinlim
            self.write_wave=load_module('kvrecon_audio_io',p/'audio_utils.py').write_wave
        checkpoint=model_dir/'best.pt'
        with self.weight_file('best.pt') as stream:ck=torch.load(stream,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);self.epoch=ck['epoch'];del ck
        self.model=model.to(self.device).eval()
        with self.weight_file('stats.npz') as stream, np.load(stream,allow_pickle=False) as z:self.stats={k:z[k].copy() for k in z.files}
        self.identity=dict(checkpoint_sha256=self.weight_digest('best.pt'),stats_sha256=self.weight_digest('stats.npz'),epoch=self.epoch)

    def decode(self, codes, output, grid=None):
        import torch
        from PIL import Image
        codes=np.asarray(codes)
        if codes.dtype!=np.int8 or codes.ndim!=2 or codes.shape[1]!=1024 or not len(codes):
            raise ValueError('Frozen layer-4 models require original INT8 [N,K512+V512]')
        if self.modality=='image' and (grid is None or len(grid)!=2 or any(type(x) is not int or x<1 for x in grid) or grid[0]*grid[1]!=len(codes)):
            raise ValueError('Image reconstruction requires a candidate grid with h*w=N')
        self.load();output=Path(output);output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'observed_media.npz',k=codes[:,:512],v=codes[:,512:])
        raw=codes.astype(np.float32);s=self.stats;n=len(codes)
        if self.modality=='image':
            raw=np.concatenate([(a-q['zero_point'])*q['scale'] for a,q in zip(np.split(raw,2,axis=1),self.quant)],axis=1)
            raw=(raw-s['mean'])/s['std']
        else:raw=(raw-s['vmean'])/s['vstd']
        x=torch.from_numpy(raw)[None].to(self.device)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if str(self.device).startswith('cuda') else nullcontext()
        with torch.inference_mode(),context:
            prediction=self.model(x,tuple(grid),None) if self.modality=='image' else self.model(x,self.frames_per_token*n+1)
        result=dict(status='decoded',modality=self.modality,tokens=n,model=self.identity,
                    original_media_used=False,original_duration_used=False)
        if self.modality=='image':
            array=(prediction[0].float().cpu().permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)
            artifact=output/'recovered.png';Image.fromarray(array).save(artifact)
            result.update(grid_hw=list(grid))
        else:
            logs=(prediction[0].float()*torch.as_tensor(s['logstd'],device=self.device)[:,None]+torch.as_tensor(s['logmean'],device=self.device)[:,None]).clamp(0,8)
            seed=int(hashlib.sha256(codes.tobytes()).hexdigest()[:8],16)
            with torch.random.fork_rng(devices=[torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []):
                torch.manual_seed(seed)
                wave=self.griffin(torch.expm1(logs),window=torch.hann_window(512,device=self.device),n_fft=512,hop_length=160,
                    win_length=512,power=1.,n_iter=32,momentum=.99,length=self.samples_per_token*n,rand_init=True)
            artifact=output/'recovered.wav';self.write_wave(artifact,wave.cpu().numpy())
            np.save(output/'log_magnitude.npy',logs.cpu().numpy())
            result.update(samples=self.samples_per_token*n,sample_rate=self.sample_rate,
                          duration_seconds=self.samples_per_token*n/self.sample_rate,seed=seed)
        result['artifact']=str(artifact.resolve());dump(output/'prediction.json',result)
        return result


class TrainedMediaHandler:
    def __init__(self, decoder, grid_selector=None, replay=None):
        self.decoder=decoder;self.grid_selector=grid_selector;self.replay=replay

    def finish_end(self,backend,request):
        replay=self.replay or getattr(backend,'replay_recovered_media',None)
        finish=getattr(replay,'finish_end',None)
        if not callable(finish):raise MediaUnavailable('continuous_media_end_replay_unavailable')
        return finish(request)

    def __call__(self,backend,request):
        if request['modality']!=self.decoder.modality:raise ValueError('Wrong modality handler')
        if request['config'].distance_space!='lsb':
            raise ValueError('Frozen INT8 media handler requires explicit LSB observations')
        raw=request['targets']
        if not np.isfinite(raw).all() or np.any(raw!=np.rint(raw)) or np.any(raw<-128) or np.any(raw>127):
            raise ValueError('Media target is not native INT8 codes')
        output=request['output'];output.mkdir(parents=True,exist_ok=True)
        dump(output/'boundary.json',dict(span=request['span'],proposal=request.get('boundary')))
        if hasattr(self.grid_selector,'recover'):
            return self.grid_selector.recover(backend,request,self.decoder,self.replay)
        grid=None
        if self.decoder.modality=='image':
            if self.grid_selector is None:
                raise MediaUnavailable('image_grid_unresolved: select a preprocessing-compatible factor pair before decoding')
            grid=self.grid_selector(backend,request)
        result=self.decoder.decode(raw.astype(np.int8),output,grid)
        replay=self.replay or getattr(backend,'replay_recovered_media',None)
        if not callable(replay):
            raise MediaUnavailable('media_decoded_replay_unavailable: '+result['artifact'])
        replay_result=replay(result,request)
        if not isinstance(replay_result,dict) or replay_result.get('status')!='complete':
            raise MediaUnavailable('recovered_media_replay_incomplete')
        result.update(replay_result)
        return result
