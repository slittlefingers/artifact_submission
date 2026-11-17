










from __future__ import annotations
import re
from typing import List, Optional


                                                                                                 
def _pb_varint(b, i):
    s = r = 0
    while True:
        x = b[i]; i += 1; r |= (x & 0x7f) << s; s += 7
        if not (x & 0x80):
            return r, i


def _pb_is_msg(b):
    i, n = 0, len(b)
    if n == 0:
        return False
    try:
        while i < n:
            tag, i = _pb_varint(b, i); wt = tag & 7
            if (tag >> 3) == 0:
                return False
            if wt == 0:
                _, i = _pb_varint(b, i)
            elif wt == 2:
                ln, i = _pb_varint(b, i); i += ln
                if i > n:
                    return False
            elif wt == 5:
                i += 4
            elif wt == 1:
                i += 8
            else:
                return False
        return i == n
    except Exception:
        return False


def _pb_parse(b, path=""):
    
    out = []; i, n = 0, len(b)
    while i < n:
        tag, i = _pb_varint(b, i); wt = tag & 7; fn = tag >> 3
        if wt == 0:
            v, i = _pb_varint(b, i); out.append((f"{path}{fn}", "varint", v))
        elif wt == 2:
            ln, i = _pb_varint(b, i); seg = b[i:i + ln]; i += ln
            txt = seg.decode("latin1")
            if re.fullmatch(r"[\x20-\x7e]{2,}", txt):
                out.append((f"{path}{fn}", "str", txt))
            elif len(seg) > 1 and _pb_is_msg(seg):
                out += _pb_parse(seg, f"{path}{fn}.")
            else:
                out.append((f"{path}{fn}", "bytes", seg))
        elif wt == 5:
            out.append((f"{path}{fn}", "i32", int.from_bytes(b[i:i + 4], "little"))); i += 4
        elif wt == 1:
            out.append((f"{path}{fn}", "i64", int.from_bytes(b[i:i + 8], "little"))); i += 8
        else:
            break
    return out


def _pb_get(flat, exact_path, typ="varint"):
    for p, t, v in flat:
        if p == exact_path and t == typ:
            return v
    return None


class Geometry:
    
    def __init__(self, attn_pattern: str, sliding_window: int, head_dim: int,
                 kv_share_start: Optional[int] = None, num_kv_heads: int = 1,
                 num_q_heads: Optional[int] = None, name: str = "", source: Optional[str] = None):
        self.attn = attn_pattern                                                                           
        self.window = sliding_window
        self.head_dim = head_dim                                                                   
        self.kv_share_start = kv_share_start
        self.num_kv_heads = num_kv_heads                                                                  
        self.num_q_heads = num_q_heads                                                                
        self.name = name or self.sig()
        self.source = source                                                                                      

    def sig(self) -> str:
        

        return f"L{self.n_layers}.w{self.kv_width()}.win{self.window}"

    @classmethod
    def from_binarypb(cls, path: str, name: Optional[str] = None) -> "Geometry":
        





        flat = _pb_parse(open(path, "rb").read())
                                                                                                     
        cands = [(p, v) for p, t, v in flat if t == "str" and re.fullmatch(r"[LG]{8,}", v)]
        if not cands:
            raise ValueError(f"{path}: no attention pattern [LG]+ found")
        ppath, pat = max(cands, key=lambda pv: len(pv[1]))
        pmsg = ppath.rsplit(".", 1)[0]                                                     
        cfg = pmsg.rsplit(".", 1)[0] if "." in pmsg else ""                             
        pre = cfg + "." if cfg else ""
        window = _pb_get(flat, pmsg + ".2") or _pb_get(flat, pre + "4.15") or 512
        head_dim = _pb_get(flat, pre + "4.7") or 256
        num_kv = _pb_get(flat, pre + "4.9") or 2
        num_q = _pb_get(flat, pre + "4.4")
        kv_share = _pb_get(flat, pre + "20")
        import os
        return cls(pat, sliding_window=window, head_dim=head_dim, kv_share_start=kv_share,
                   num_kv_heads=num_kv, num_q_heads=num_q, name=name, source=os.path.basename(path))

    def kv_width(self) -> int:
        


        return self.num_kv_heads * self.head_dim

    def bufs_per_layer(self) -> int:
        
        return 2

    @property
    def n_layers(self) -> int:
        return len(self.attn)

    def scope(self, layer: int) -> str:
        return "global" if self.attn[layer] == "G" else "local"

    def global_layers(self) -> List[int]:
        return [i for i, c in enumerate(self.attn) if c == "G"]

    def local_layers(self) -> List[int]:
        return [i for i, c in enumerate(self.attn) if c != "G"]


                                                                                                   
                                                                                                    
                                                                                                    
                                                                             
def default_aicore_fast() -> Geometry:
    
    return Geometry("LLLLG" * 7, sliding_window=512, head_dim=256,
                    kv_share_start=15, num_kv_heads=1, num_q_heads=8, name="gemma4-fast")


def default_aicore_main() -> Geometry:
    
    return Geometry("LLLLLG" * 7, sliding_window=512, head_dim=256,
                    kv_share_start=24, num_kv_heads=2, num_q_heads=8, name="gemma4-main")


def aicore_geometries(config_paths=None):
    



    import logging
    geoms = []
    for p in (config_paths or []):
        try:
            geoms.append(Geometry.from_binarypb(p))
        except Exception as e:
            logging.getLogger(__name__).debug(f"from_binarypb({p}) skipped: {e}")
    return geoms


def expected_manifest(geom: Geometry) -> List[dict]:
    


    ent = []
    for L in range(geom.n_layers):
        sc = geom.scope(L)
        shared = geom.kv_share_start is not None and L >= geom.kv_share_start and sc == "local"
        for role in ("k", "v"):
            ent.append({"layer": L, "role": role, "scope": sc,
                        "cache": geom.window if sc == "local" else None, "shared": shared})
    return ent


def classify_scope(cache_size: int, window: int, global_max: int = 65536) -> str:
    



    if window <= 0:
        return "reject"
    if cache_size == window:
        return "local"
    if window < cache_size <= global_max:
        return "global"
    return "reject"


def classify_buffer(cache_size: int, geom: Geometry, global_max: int = 65536) -> str:
    
    return classify_scope(cache_size, geom.window, global_max)


def match_report(cache_sizes: List[int], geom: Geometry) -> dict:
    
    cls = [classify_buffer(cs, geom) for cs in cache_sizes]
    man = expected_manifest(geom)
    exp_g = sum(1 for e in man if e["scope"] == "global")
    exp_l = sum(1 for e in man if e["scope"] == "local")
    exp_l_unshared = sum(1 for e in man if e["scope"] == "local" and not e["shared"])
    return {
        "classes": cls,
        "n_layers": geom.n_layers, "window": geom.window, "head_dim": geom.head_dim,
        "num_kv_heads": geom.num_kv_heads, "bufs_per_layer": geom.bufs_per_layer(),
        "expected_global_slots": exp_g,                                              
        "expected_local_slots": exp_l,                                              
        "expected_local_unshared": exp_l_unshared,                         
        "got_global": cls.count("global"),
        "got_local": cls.count("local"),
        "rejected": cls.count("reject"),
    }
