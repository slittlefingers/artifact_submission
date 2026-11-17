


from __future__ import annotations
import json
import re
import struct
from typing import Dict, Optional

_TT = {0: "FLOAT32", 2: "INT32", 3: "UINT8", 4: "INT64", 7: "INT16", 9: "INT8", 17: "INT4"}
                                                                                               
                                                           
_EB = {"FLOAT32": 4, "INT32": 4, "UINT8": 1, "INT64": 8, "INT16": 2, "INT8": 1, "INT4": 0.5}


class TensorSpec:
    __slots__ = ("name", "layer", "role", "shape", "dtype", "nbytes", "scale", "zero_point")

    def __init__(self, name, layer, role, shape, dtype, nbytes, scale=None, zero_point=None):
        self.name = name; self.layer = layer; self.role = role
        self.shape = shape; self.dtype = dtype; self.nbytes = nbytes
        self.scale = scale; self.zero_point = zero_point


class Descriptor:
    NAME_RX = re.compile(r"(?:decode_|prefill_\d+_|verify_)?kv_cache_([kv])_(\d+)")

    def __init__(self, specs: Dict[str, TensorSpec], meta: dict):
        self.specs = specs
        self.meta = meta

    @classmethod
    def from_json(cls, path: str) -> "Descriptor":
        d = json.load(open(path)); specs = {}
        for L, e in d["layers"].items():
            for role in ("k", "v"):
                t = e[role]; nm = f"kv_cache_{role}_{L}"
                specs[nm] = TensorSpec(nm, int(L), role, t["shape"], t["dtype"], t["bytes"],
                                       t.get("scale"), t.get("zero_point"))
        return cls(specs, d)

    def base_name(self, raw_name: str) -> Optional[str]:
        m = self.NAME_RX.search(raw_name)
        return f"kv_cache_{m.group(1)}_{m.group(2)}" if m else None

    @classmethod
    def from_litertlm(cls, path: str) -> "Descriptor":
        



        f = open(path, "rb")
        head = f.read(32)
        if head[:8] != b"LITERTLM":
            raise ValueError(f"{path}: not a LITERTLM container")
        hend = struct.unpack_from("<Q", head, 24)[0]
        f.seek(0); hdr = f.read(hend)[32:]

        def _i32(b, p): return struct.unpack_from("<i", b, p)[0]
        def _u16(b, p): return struct.unpack_from("<H", b, p)[0]
        def _u32(b, p): return struct.unpack_from("<I", b, p)[0]
        def _u64(b, p): return struct.unpack_from("<Q", b, p)[0]
        def _fp(b, t, fid):
            vt = t - _i32(b, t)
            if _u16(b, vt) <= 4 + fid * 2: return None
            vo = _u16(b, vt + 4 + fid * 2); return None if vo == 0 else t + vo
        def _dr(b, p): return p + _u32(b, p)
        root = _dr(hdr, 0); sm = _dr(hdr, _fp(hdr, root, 1))
        vec = _dr(hdr, _fp(hdr, sm, 0)); nsec = _u32(hdr, vec)
        secs = []
        for i in range(nsec):
            so = _dr(hdr, vec + 4 + i * 4)
            bp, dp = _fp(hdr, so, 1), _fp(hdr, so, 3)
            secs.append((_u64(hdr, bp) if bp else 0, hdr[dp] if dp else 0))

        class _FB:
            def __init__(s, base): s.base = base
            def rd(s, p, n): f.seek(s.base + p); return f.read(n)
            def i32(s, p): return struct.unpack("<i", s.rd(p, 4))[0]
            def u32(s, p): return struct.unpack("<I", s.rd(p, 4))[0]
            def u16(s, p): return struct.unpack("<H", s.rd(p, 2))[0]
            def i8(s, p): return struct.unpack("<b", s.rd(p, 1))[0]
            def f32(s, p): return struct.unpack("<f", s.rd(p, 4))[0]
            def i64(s, p): return struct.unpack("<q", s.rd(p, 8))[0]
            def field(s, t, fid):
                vt = t - s.i32(t)
                if s.u16(vt) <= 4 + fid * 2: return None
                vo = s.u16(vt + 4 + fid * 2); return None if vo == 0 else t + vo
            def deref(s, p): return p + s.u32(p)
            def string(s, p):
                q = s.deref(p); n = s.u32(q); return s.rd(q + 4, n).decode("utf-8", "replace")
            def vec(s, p):
                v = s.deref(p); return v + 4, s.u32(v)

        specs = {}
        for base, dtyp in secs:
            if dtyp != 3:
                continue
            fb = _FB(base)
            if fb.rd(4, 4) != b"TFL3":
                continue
            mroot = fb.deref(0); sgf = fb.field(mroot, 2)
            if sgf is None:
                continue
            sgs, nsg = fb.vec(sgf); got = {}
            for si in range(nsg):
                sg = fb.deref(sgs + si * 4); tf = fb.field(sg, 0)
                if tf is None:
                    continue
                ts, ntt = fb.vec(tf)
                probe = None
                for ti in range(min(ntt, 4)):
                    t = fb.deref(ts + ti * 4); npos = fb.field(t, 3)
                    if npos: probe = fb.string(npos); break
                if not probe or not probe.startswith("decode_"):
                    continue
                for ti in range(ntt):
                    t = fb.deref(ts + ti * 4); npos = fb.field(t, 3)
                    if npos is None: continue
                    m = cls.NAME_RX.search(fb.string(npos))
                    if not m: continue
                    role, L = m.group(1), int(m.group(2))
                    shp = []; spos = fb.field(t, 0)
                    if spos is not None:
                        es, ec = fb.vec(spos)
                        shp = [struct.unpack("<i", fb.rd(es + j * 4, 4))[0] for j in range(ec)]
                    tp = fb.field(t, 1); ttype = _TT.get(fb.i8(tp) if tp is not None else 0, "INT8")
                    scale = zp = None
                    qp = fb.field(t, 4)
                    if qp is not None:
                        qt = fb.deref(qp)
                        sc = fb.field(qt, 2); zpf = fb.field(qt, 3)
                        if sc is not None:
                            es, ec = fb.vec(sc)
                            if ec: scale = fb.f32(es)
                        if zpf is not None:
                            es, ec = fb.vec(zpf)
                            if ec: zp = fb.i64(es)
                    nm = f"kv_cache_{role}_{L}"
                    if nm not in got:
                        ne = 1
                        for d in shp: ne *= (d if d > 0 else 1)
                        nb = int(ne * _EB.get(ttype, 1))                                     
                        got[nm] = TensorSpec(nm, L, role, shp, ttype, nb, scale, zp)
                if got:
                    specs = got; break
            if specs:
                break
        f.close()
        if not specs:
            raise ValueError(f"{path}: no decode kv_cache tensors found")
        layers = sorted({s.layer for s in specs.values()})
        wide = sorted({s.layer for s in specs.values() if s.role == "k" and s.shape[-1] == 512})
        k0 = specs.get("kv_cache_k_0")
        meta = {"model": path.replace("\\", "/").split("/")[-1],
                "dtype": (k0.dtype if k0 else next(iter(specs.values())).dtype),
                "n_layers": len(layers), "wide_layers": wide,
                "max_context": k0.shape[-2] if k0 else 0}
        return cls(specs, meta)

    @classmethod
    def gemma4e2b_default(cls) -> "Descriptor":
        

        wide = {4, 9, 14}; ctx = 4096; specs = {}
        meta = {"model": "gemma-4-E2B (built-in default)", "dtype": "INT8",
                "context": ctx, "n_layers": 15, "wide_layers": sorted(wide)}
        for L in range(15):
            hd = 512 if L in wide else 256
            nb = ctx * hd
            for role, shape in (("k", [1, 1, ctx, hd]), ("v", [1, 1, hd, ctx])):
                nm = f"kv_cache_{role}_{L}"
                specs[nm] = TensorSpec(nm, L, role, shape, "INT8", nb)
        return cls(specs, meta)
