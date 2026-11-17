"""Qwen PhotoChat runner on the Qwen verify adapter; all search remains in kvrecon.engine."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from ...npu_standalone import StandaloneNpuBackend
from ...media import qwen_vision as vision
from ...search import SearchConfig
from ...common import dump


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def image_template(meta):
    
    before=list(meta["public_prefix"]);pad=meta["image_pad_id"];end=meta["image_end_id"]
    if any(type(x) is not int or x<0 for x in before+[pad,end]) or pad in before or pad==end:
        raise ValueError("Invalid public template IDs")
    return before,pad,end


class PhotoChatNpuBackend(StandaloneNpuBackend):
    
    def __init__(self,*args,observation,reconstructed_image,grid,**kwargs):
        kwargs.update(qwen_recovery=True,qwen_query_mode="verify")
        super().__init__(*args,**kwargs)
        folder=Path(observation);before,self.image_pad,end=image_template(read(folder/"metadata.json"))
        self.grid=tuple(grid);self.visual_start=len(before);self.visual_tokens=self.grid[0]*self.grid[1]
        self.public_prefix=before+[self.image_pad]*self.visual_tokens+[end]
        self.observed=load_observed(folder)
        if self.observed.shape[1]!=self.width or len(self.observed)<=len(self.public_prefix):
            raise ValueError("Invalid observed joint K/V")
        if len(self.observed)+63>self.dimensions["context_capacity"]:
            raise ValueError("Insufficient physical slots for verify_64 at the last text position")
        self.reconstructed_image=Path(reconstructed_image)
        self.validated_prefix_ready=False

    def __enter__(self):
        super().__enter__()
        ready=json.loads(self.read("ready.json"))
        if not ready.get("image_prefix") or not ready.get("selected_rows_restore"):
            self.__exit__(None,None,None)
            raise RuntimeError("Rebuild the worker with Qwen image-prefix support")
        return self

    def _restore(self,start,count):
        if start<0 or count<=0 or start+count>self.position:
            raise ValueError("Never restore unaccepted future rows")
        self.write("selected_rows.bin",self.observed[start:start+count].astype("<f4").tobytes())
        self.rpc("restore_selected_rows",start=start,count=count)

    def reset(self,prefix):
        if list(prefix)!=self.public_prefix:raise ValueError("Unexpected image prefix")
                                                                            
                                                                               
        if self.validated_prefix_ready:
            self.validated_prefix_ready=False
            if self.position!=len(prefix):raise RuntimeError("Validated image prefix changed")
            return
        gh,gw=self.grid
        with Image.open(self.reconstructed_image) as im:
            pixels=np.asarray(im.convert("RGB").resize((gw*32,gh*32),Image.Resampling.BICUBIC),dtype=np.float32)/255
        patches,grid=vision.patchify(((pixels-.5)/.5).transpose(2,0,1))
        sig,kw,n=vision.encoder_inputs(patches,grid)
        if n!=self.visual_tokens:raise ValueError("Unexpected vision token count")
        self.write("vision_inputs.bin",b"".join(kw[k].tobytes() for k in ("patches","pos_idx","pos_w","positions","valid")))
        self.pending_begin=False;self.pending_retain=None;self.selected_retention=False
        self.rpc("reset_image",prefix_ids=self.public_prefix,rope_pos=vision.mrope_positions(self.public_prefix,self.image_pad,self.grid).tolist(),image_start=self.visual_start,visual_tokens=n,image_pad_id=self.image_pad,cap=len(kw["valid"]),vision_signature=sig)
        if self.position!=len(prefix):raise RuntimeError("Image prefix not advanced")
        self._restore(0,self.position)

    def commit(self,token):
        position=self.position
        actual=super().commit(token)
        if self.position!=position+1:raise RuntimeError("Invalid commit advancement")
                                                                                     
        self._restore(position,1)
        return actual


def search_config(patience,markers):
                                                                                 
    return SearchConfig(method="guided_kv",epsilon=1e-12,batch_size=128,
        probability_seeds=50,random_seeds=0,reference_seeds=78,merge_initial_batch=True,
        proposals=128,max_queries=None,max_ridge_rounds=None,ridge_stagnation_rounds=patience,
        stagnation_neighbors=127,stagnation_token_ids=tuple(dict.fromkeys(markers)),
        distance_space="original",distance_metric="rmse",retain_query_cache=False).validate()


def load_observed(folder):
    with np.load(Path(folder)/"observed.npz",allow_pickle=False) as z:
        observed=np.asarray(z["kv"],dtype="<f4").copy()
    if observed.ndim!=2 or not np.isfinite(observed).all():raise ValueError(f"Invalid observed K/V: {folder}")
    return observed


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--check",action="store_true",help="Validate paths/schema without loading models or contacting a phone")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--positions",type=int,help="Optional pilot; omit to recover all text")
    parser.add_argument("--output",type=Path,help="Output directory; overrides the configured output")
    args=parser.parse_args(argv);cfg=read(args.config)
    if cfg.get("backend")!="npu":raise ValueError("Only NPU replay is supported")
    for key in ("inspection","worker","image_worker","end_references","tokenizer","embeddings"):
        if not Path(cfg[key]).is_file():raise FileNotFoundError(cfg[key])
    folders=sorted(p for p in Path(cfg["observations"]).iterdir() if p.is_dir() and (p/"metadata.json").is_file())
    if not folders:raise ValueError("No prepared observations")
    if args.limit is not None:
        if args.limit<1:raise ValueError("limit must be positive")
        folders=folders[:args.limit]
    for folder in folders:image_template(read(folder/"metadata.json"))
    inspection=read(cfg["inspection"])
    signatures=inspection["model_sections"]["tf_lite_prefill_decode"]["signatures"]
    if "verify_64" not in signatures:raise ValueError("verify_64 signature required")
    if "tf_lite_vision_encoder" not in inspection["model_sections"]:
        raise ValueError("Deployed vision encoder required")
    if not Path(cfg["reconstructor"]).is_dir():raise FileNotFoundError(cfg["reconstructor"])
    if not Path(cfg["reference"]).is_dir():raise FileNotFoundError(cfg["reference"])
    if args.check:
        print(f"Validated {len(folders)} observation manifests; no NPU execution performed.");return
    from tokenizers import Tokenizer
    from ...engine import reconstruct
    from ...kv_reference import VReferenceIndex
    from ...torch_surrogate import TorchRidge
    tokenizer=Tokenizer.from_file(cfg["tokenizer"])
    markers=[]
    for marker in cfg.get("public_markers",["<|im_start|>","<|im_end|>","<|endoftext|>","<|vision_start|>","<|vision_end|>","<|image_pad|>","<think>","</think>","system","user","assistant","\n"]):
        markers.extend(tokenizer.encode(marker,add_special_tokens=False).ids)
    search=search_config(cfg.get("patience",2),markers)
    reference=VReferenceIndex(cfg["reference"],cfg.get("reference_identity"),cfg.get("search_device","cuda"))
    ridge=TorchRidge(cfg["embeddings"],**cfg.get("ridge",{}))
    output=Path(args.output or cfg["output"]);spans={}
    for folder in folders:
        if (output/folder.name).exists():raise FileExistsError(f"Refusing overwrite: {output/folder.name}")
                                                                                   
    from .image_span import Reconstructor,open_replay,recover_span
    reconstructor=Reconstructor(cfg["reconstructor"],cfg.get("reconstructor_device"))
    with np.load(cfg["end_references"],allow_pickle=False) as z:end_references=z["image"].copy()
    prefix=image_template(read(folders[0]/"metadata.json"))[0]
    image_backend,replay=open_replay(cfg,inspection,output/"image_worker",prefix)
    with image_backend:
        for folder in folders:
            if image_template(read(folder/"metadata.json"))[0]!=prefix:raise ValueError("Mixed image templates")
            spans[folder]=recover_span(replay,reconstructor,load_observed(folder),len(prefix),end_references,output/folder.name/"image")
            print(f"{folder.name}: visual_tokens={spans[folder][0]} grid={spans[folder][1]}",flush=True)
                                                                  
    for folder in folders:
        dest=output/folder.name;_,grid,image=spans[folder]
        backend=PhotoChatNpuBackend(inspection["model_sections"],cfg.get("layer",3),dest/"worker",inspection["model_sha256"],native_executable=cfg["worker"],observation=folder,reconstructed_image=image,grid=grid,adb=cfg["adb"],serial=cfg.get("serial"),package=cfg["package"],device_model=cfg["device_model"],batch_signature="prefill_128",mask_token_id=0,install=False)
        with backend:
                                                                                            
            backend.reset(backend.public_prefix)
            ids=[15,16,17,198];position=backend.position
            a=backend.query(ids);b=backend.query(ids[::-1])[::-1]
            c=np.concatenate([backend.query([t]) for t in ids])
            delta=max(float(np.max(np.abs(a-b))),float(np.max(np.abs(a-c))))
            if backend.position!=position or delta>cfg.get("independence_atol",1e-4):
                raise RuntimeError("Independent NPU candidate verification failed")
            dump(dest/"backend_validation.json",dict(max_abs=delta,position_unchanged=True))
            backend.validated_prefix_ready=True
            positions=np.arange(len(backend.public_prefix),len(backend.observed))
            reconstruct(backend,backend.observed[positions],positions,backend.public_prefix,search,dest/"recovery",surrogate=ridge,reference_initializer=reference,max_positions=args.positions)

if __name__=="__main__":main()
