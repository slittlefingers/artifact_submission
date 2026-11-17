
















from __future__ import annotations
from typing import Dict, List, Optional
from .descriptor import Descriptor

ACTIVE, CANDIDATE, ADAPTER, CACHED_ONLY = "ACTIVE", "CANDIDATE", "ADAPTER", "CACHED_ONLY"


class ModelRef:
    __slots__ = ("path", "sha1", "descriptor", "mapped", "resident", "role", "score", "notes")

    def __init__(self, path, sha1=None, descriptor=None, mapped=False, resident=False):
        self.path = path; self.sha1 = sha1; self.descriptor = descriptor
        self.mapped = mapped; self.resident = resident
        self.role = None; self.score = 0.0; self.notes = ""

    def __repr__(self):
        return f"<ModelRef {self.path!r} role={self.role} score={self.score:.2f} map={self.mapped} {self.notes}>"


def fingerprint(desc: Descriptor) -> dict:
    
    hds = set()
    for s in desc.specs.values():
        hds.add(s.shape[-1] if s.role == "k" else s.shape[-2])                                      
    return {"n_layers": desc.meta.get("n_layers", 0),
            "wide_layers": tuple(desc.meta.get("wide_layers", [])),
            "head_dims": frozenset(hds),
            "dtype": desc.meta.get("dtype"),
            "max_context": desc.meta.get("max_context", 0),
            "n_kv": len(desc.specs)}


def observed_kv_geometry(head_dims: List[int]) -> dict:
    


    return {"head_dims": frozenset(h for h in head_dims if h), "n_kv": len(head_dims)}


def _score(fp: dict, obs: dict) -> float:
    
    if not obs["head_dims"]:
        return 0.0
    inter = fp["head_dims"] & obs["head_dims"]
    if not inter:
        return 0.0
    cover = len(inter) / len(obs["head_dims"])                                               
    return round(cover, 3)


def label_models(refs: List[ModelRef], head_dims: List[int]) -> List[ModelRef]:
    

    obs = observed_kv_geometry(head_dims)
    for r in refs:
        r.score = _score(fingerprint(r.descriptor), obs) if r.descriptor else 0.0
                                                                                          
    for r in refs:
        if r.descriptor is not None and len(r.descriptor.specs) == 0:
            r.role = ADAPTER; r.notes = "no KV tensors (delta/adapter weights)"
    scored = [r for r in refs if r.role is None]
    best = max((r.score for r in scored), default=0.0)
    winners = [r for r in scored if r.score == best and best > 0.0 and r.mapped]
    for r in scored:
        if r in winners:
            r.role = ACTIVE
            if len(winners) > 1:
                r.notes = "ambiguous: same KV geometry as other candidate(s) (base vs fine-tune not resolvable from KV)"
        elif r.mapped or r.resident:
            r.role = CANDIDATE
            if r.score == best and best > 0 and not r.mapped:
                r.notes = "geometry fits but not mapped by this pid"
        else:
            r.role = CACHED_ONLY
    return refs


def build_catalog(litertlm_paths: List[str]) -> List[ModelRef]:
    

    out = []
    for p in litertlm_paths:
        r = ModelRef(p)
        try:
            r.descriptor = Descriptor.from_litertlm(p)
        except Exception as e:
            r.descriptor = Descriptor({}, {"model": p, "n_layers": 0, "wide_layers": []})
            r.notes = f"parse: {type(e).__name__}: {e}"
        out.append(r)
    return out
