







from __future__ import annotations
import hashlib
import logging
import struct
from collections import Counter
from typing import Dict
from volatility3.framework import exceptions
from volatility3.plugins.linux import proc
from .base import Locator, KvBuf, vma_name
from .. import classify, constants as C

vollog = logging.getLogger(__name__)


class LitertDmabufLocator(Locator):
    key = "litert_dmabuf"
    ROW_BYTES, EMB_NARROW, EMB_WIDE = 4096, 256, 512

    def default_descriptor(self):
        from ..descriptor import Descriptor
        return Descriptor.gemma4e2b_default()

    def detect(self, ctx, task) -> int:
        for vma in proc.Maps.list_vmas(task):
            if "scudo" in (vma_name(ctx, task, vma) or ""):
                return 25
        return 0

                                                                             
    def _tb_at(self, reader, h, plausible):
        
        if not (C.HEAP_LO <= h < C.HEAP_HI):
            return None
        obj = reader.read(h, C.TB_LEN)
        if not obj:
            return None
        sz = struct.unpack_from("<Q", obj, C.TB_SIZE)[0]
        return sz if plausible(sz) else None

    def _name_tb_candidates(self, reader, task, desc):
        




        import re
                                                                                                     
        rx = re.compile(rb'kv_cache_[kvcr]_([0-9]{1,2})(?![0-9])')
                                                                                                     
                                                                                                       
                                                                                                       
        desc_sizes = {s.nbytes for s in desc.specs.values()}
        def plausible(sz):
            return sz in desc_sizes or (0 < sz <= (64 << 20) and sz % 4096 == 0)
        name_va = {}; pages = {}
        for vma in proc.Maps.list_vmas(task):
            try:
                if "rw" not in vma.get_protection() or vma.vm_file != 0:
                    continue
                start = int(vma.vm_start); end = int(vma.vm_end)
            except exceptions.InvalidAddressException:
                continue
            if end - start > 256 << 20:
                continue
            va = start
            while va < end:
                b = reader.read(va, min(1 << 20, end - va))
                pages[va] = b
                for m in rx.finditer(b):
                    if int(m.group(1)) < 64:                                                                 
                        name_va[va + m.start()] = m.group().decode()
                va += len(b)
        cands = []; seen = set(); tb_sizes = {}; layout = {}

        def add(nm, h, sz, how):
            if (nm, h) in seen:
                return
            seen.add((nm, h)); cands.append((nm, h)); tb_sizes[h] = sz; layout.setdefault(nm, how)
                                                                                                
                                                                                                  
                                                                             
        for name_addr, nm in sorted(name_va.items()):
            key = name_addr - C.SSO_KEY_BACK
            tag = reader.read(key, 1)
            if not tag:
                continue
            t = tag[0]
            if (t & 1) != 0 or (t >> 1) != len(nm):                                                 
                continue
            p = reader.read(key + C.SSO_TB, 8)
            if not p:
                continue
            h = struct.unpack_from("<Q", p, 0)[0] & C.MTE_MASK
            sz = self._tb_at(reader, h, plausible)
            if sz is not None:
                add(nm, h, sz, "sso")
                                                                                                   
                                                                                                       
                                                                                                       
                                                                                                 
        if not cands:
            va_set = set(name_va)
            for base in sorted(pages):
                b = pages[base]
                for o in range(0, len(b) - 48, 8):
                    ptr = struct.unpack_from("<Q", b, o)[0] & C.MTE_MASK
                    if ptr not in va_set:
                        continue
                    nm = name_va[ptr]
                    if struct.unpack_from("<Q", b, o + 8)[0] != len(nm):
                        continue
                    for wi in range(C.BANKS):
                        h = struct.unpack_from("<Q", b, o + 16 + wi * 8)[0] & C.MTE_MASK
                        sz = self._tb_at(reader, h, plausible)
                        if sz is not None:
                            add(nm, h, sz, "ptrlen"); break
                                                                                                     
                                                                                       
        self._observed_sizes = set(tb_sizes.values())
        self._resident_names = set(name_va.values())
        self._name_layout = layout
        return cands

                                                       
    def _embs(self):
        return (self.EMB_NARROW, self.EMB_WIDE)

    def locate(self, ctx, task, reader, reasm, off, desc) -> Dict[str, KvBuf]:
        try:
            files = int(task.files)
        except exceptions.InvalidAddressException:
            files = 0
        sizes = {s.nbytes for s in desc.specs.values()}
        fdarr, maxfds = reasm.fd_table(files)
        vollog.info(f"litert_dmabuf[FALLBACK]: files={files:#x} fdarr={fdarr and hex(fdarr)} maxfds={maxfds}")
        if not fdarr:
            return {}
        fops = reasm.derive_fops(fdarr, maxfds, sizes=sizes)
        if not fops:
            vollog.warning("litert_dmabuf[FALLBACK]: dma_buf_fops not derivable")
            return {}
                                                          
        allbuf = []
        for fd, sz, db in reasm.enumerate(fdarr, maxfds, fops):
            if sz not in sizes:
                continue
            priv = reasm.priv(db)
            raw = reasm.reassemble(priv, sz, tol=True) if priv else None
            if raw:
                allbuf.append((fd, len(raw), raw))
                                                             
        row, embs = self.ROW_BYTES, self._embs()
        steps = [s for s in (classify.k_step(r, row, embs)
                             for (_, sz, r) in allbuf if sz == self.EMB_NARROW * row) if s]
        S = Counter(steps).most_common(1)[0][0] if steps else 0
                                                                
        reps = {}
        for fd, sz, raw in allbuf:
            c = classify.classify_kv(raw, S, row, embs)
            if not c:
                continue
            kind, emb = c
            key = (kind, emb, hashlib.md5(classify.content_key(raw, kind, emb, S, row)).digest())
            if key not in reps or fd < reps[key][0]:
                reps[key] = (fd, raw)
                                                      
        wide = set(desc.meta.get("wide_layers", [4, 9, 14]))
        nL = desc.meta.get("n_layers", 15)
        out = {}; assigned = 0
        for kind in ("k", "v"):
            for emb, is_wide in ((self.EMB_WIDE, True), (self.EMB_NARROW, False)):
                layers = [L for L in range(nL) if (L in wide) == is_wide]
                pool = sorted([(fd, r) for (kk, ee, _), (fd, r) in reps.items()
                               if kk == kind and ee == emb], key=lambda x: x[0])
                for i, L in enumerate(layers):
                    if i >= len(pool):
                        break
                    base = f"kv_cache_{kind}_{L}"
                    out[base] = KvBuf(base, raw=pool[i][1], src=("fd-order", pool[i][0]),
                                      size=len(pool[i][1]), hd=emb, note="UNVERIFIED fd-order fallback")
                    assigned += 1
        vollog.warning(f"litert_dmabuf[FALLBACK]: dma_bufs={len(allbuf)} S={S} classified_reps={len(reps)} "
                       f"assigned={assigned}/{nL * 2} (LAYER INDEX UNVERIFIED: fd-order != layer-order)")
        return out
