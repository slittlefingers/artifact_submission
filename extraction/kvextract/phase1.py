












from __future__ import annotations
import glob
import hashlib
import logging
import os
from typing import Dict, List, Optional
from .descriptor import Descriptor
from .manifest import Geometry

vollog = logging.getLogger(__name__)


                                                                                                   
                                                                                                     
                                                                                             
 
                                                                                   
                                                                                                     
                                                                                        
                                                                                              
                                                                                                  
 
                                                                                                     
                                                                                                 
                                                                                                   
                                                                                                
AICORE_TYPE_NEEDLES: Dict[str, List[bytes]] = {
    "main":  [b"kNanoV4MainModel"],
    "fast":  [b"kNanoV4FastModel"],
    "tiny":  [b"kNanoV4TinyModel"],
    "audio": [b"TinyAudioGemma", b"kTinyAudioGemma"],
}


def loaded_type_census(dump_path: str, needle_map: Dict[str, List[bytes]] = None,
                       probe_cap: Optional[int] = None, chunk: int = 256 << 20) -> Dict[str, int]:
    



    needle_map = needle_map or AICORE_TYPE_NEEDLES
    counts = {t: 0 for t in needle_map}
    if not dump_path or not os.path.isfile(dump_path):
        vollog.debug(f"loaded_type_census: dump not readable: {dump_path!r}")
        return counts
    needles = [(t, n) for t, ns in needle_map.items() for n in ns]
    maxlen = max(len(n) for _, n in needles)
    prev = b""
    pos = 0
    with open(dump_path, "rb") as f:
        while True:
            to_read = chunk
            if probe_cap is not None:
                if pos >= probe_cap:
                    break
                to_read = min(chunk, probe_cap - pos)
            buf = f.read(to_read)
            if not buf:
                break
            win = prev + buf
            for t, nd in needles:
                st = 0
                while True:
                    i = win.find(nd, st)
                    if i < 0:
                        break
                    counts[t] += 1
                    st = i + 1
            prev = win[-maxlen:]
            pos += len(buf)
    return counts


def loaded_types(census: Dict[str, int]) -> List[str]:
    
    return [t for t, c in census.items() if c > 0]


class Candidate:
    __slots__ = ("kind", "path", "geometry", "descriptor", "referenced")

    def __init__(self, kind, path, geometry=None, descriptor=None, referenced=False):
        self.kind = kind                                 
        self.path = path
        self.geometry = geometry                                        
        self.descriptor = descriptor                                          
        self.referenced = referenced                                                 

    def __repr__(self):
        g = self.geometry
        extra = (f"{g.n_layers}L kv={g.num_kv_heads} w={g.kv_width()}" if g else
                 f"{self.descriptor.meta.get('n_layers')}L" if self.descriptor else "?")
        return f"<{self.kind} {os.path.basename(self.path)} {extra} ref={self.referenced}>"


def discover_candidates(models_dir: str) -> List[Candidate]:
    
    out = []
    if not models_dir:
        return out
    if not os.path.isdir(models_dir):
        vollog.warning(f"--models-dir {models_dir!r} is not a directory (unset variable or wrong path?) "
                       f"-> no model configs loaded, AICore KV matching will fall back to shape-derived dims")
        return out
    for p in sorted(glob.glob(os.path.join(models_dir, "**", "*.litertlm"), recursive=True)):
        try:
            d = Descriptor.from_litertlm(p)
            out.append(Candidate("litert", p, descriptor=d))
        except Exception as e:
            vollog.debug(f"litertlm parse skipped {p}: {e}")
                                                                                                          
    seen = set()
    cfgs = (glob.glob(os.path.join(models_dir, "**", "*config*.pb"), recursive=True)
            + glob.glob(os.path.join(models_dir, "**", "*config*.binarypb"), recursive=True))
    for p in sorted(set(cfgs)):
        try:
            h = hashlib.sha1(open(p, "rb").read()).digest()
            if h in seen:
                continue
            seen.add(h)
            g = Geometry.from_binarypb(p)
            out.append(Candidate("aicore", p, geometry=g))
        except Exception as e:
            vollog.debug(f"binarypb parse skipped {p}: {e}")
    return out


def referenced_models(reader, task) -> set:
    

    import re
    from volatility3.plugins.linux import proc
    from .locators.base import vma_name
    refs = set()
    rx = re.compile(r"([0-9a-zA-Z_.\-]+\.litertlm|[0-9a-zA-Z_.\-]+\.tflite|aicore-models/files/[0-9a-f]{8,}|[\w.\-]*config[\w.\-]*\.pb)")
    try:
        ctx = getattr(reader, "_ctx", None)
    except Exception:
        ctx = None
    try:
        for v in proc.Maps.list_vmas(task):
            nm = vma_name(ctx, task, v) if ctx else None
            if not nm:
                continue
            for m in rx.finditer(nm):
                refs.add(os.path.basename(m.group(1).rstrip("/")))
    except Exception:
        pass
    return refs


def summarize(cands: List[Candidate]) -> str:
    return (f"{len(cands)} model(s): "
            + ", ".join(repr(c) for c in cands)) if cands else "no models parsed"
