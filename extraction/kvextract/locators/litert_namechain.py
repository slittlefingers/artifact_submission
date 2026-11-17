










from __future__ import annotations
import logging
import struct
from typing import Dict
from .base import KvBuf
from .litert_dmabuf import LitertDmabufLocator
from .. import constants as C

vollog = logging.getLogger(__name__)


class LitertNameChainLocator(LitertDmabufLocator):
    key = "litert_namechain"

    def detect(self, ctx, task) -> int:
        return 0                                                                                          

    @staticmethod
    def _ptr(b, o):
        return struct.unpack_from("<Q", b, o)[0] & C.MTE_MASK

    def _tb_to_fd(self, reader, tb):
        
        obj = reader.read(tb, C.TB_LEN)
        if not obj:
            return None
        gb = reader.read(self._ptr(obj, C.TB_AHWB), C.GB_LEN)
        if not gb:
            return None
        nh = reader.read(self._ptr(gb, C.GB_NH), C.NH_LEN)
        if not nh:
            return None
        if struct.unpack_from("<i", nh, C.NH_VER)[0] != C.NH_VERSION_MAGIC:
            return None
        if struct.unpack_from("<i", nh, C.NH_NUMFDS)[0] < 1:
            return None
        return struct.unpack_from("<i", nh, C.NH_FD)[0]

    def _resolve(self, cands, reader, reasm, off, fdarr, maxfds, sizes):
        




        chosen = {}
        n = dict(occ=len(cands), nh=0, file=0, size=0, sg=0, conflicts=0)
        for nm, tb in cands:
            fd = self._tb_to_fd(reader, tb)                                   
            if fd is None or fd < 0 or fd >= maxfds:
                continue
            n["nh"] += 1
            fp = reasm.kr.u64(fdarr + fd * 8)                                  
            db = reasm.dmabuf_of_file(fp)                                                            
            if not db:
                continue
            n["file"] += 1
            sz = reasm.kr.u64(db + off['DB_SIZE'])
            if sz not in sizes:
                continue
            n["size"] += 1
            priv = reasm.priv(db)                                              
            raw = reasm.reassemble(priv, sz, exact=True) if priv else None
            if not raw:
                continue
            n["sg"] += 1
            if nm in chosen and chosen[nm][0] != db:
                n["conflicts"] += 1
            chosen[nm] = (db, fd, raw)
        return chosen, n

    def locate(self, ctx, task, reader, reasm, off, desc) -> Dict[str, KvBuf]:
        if off.get('DB_FILE') is None:
            vollog.warning("litert_namechain: kernel offset dma_buf.file unavailable (ISF/BTF); "
                           "cannot confirm dma_buf files, refusing to guess")
            return {}
        cands = self._name_tb_candidates(reader, task, desc)
                                                                                                 
                                                                                                       
        sizes = getattr(self, "_observed_sizes", None) or {s.nbytes for s in desc.specs.values()}
        want = getattr(self, "_resident_names", None) or {n for n in desc.specs}

        try:
            files = int(task.files)
        except Exception:
            files = 0
        fdarr, maxfds = reasm.fd_table(files)
        if not fdarr:
            vollog.warning("litert_namechain: fd table not derivable")
            return {}

        chosen, n = self._resolve(cands, reader, reasm, off, fdarr, maxfds, sizes)
        out = {}
        for nm, (db, fd, raw) in chosen.items():
            hd = desc.specs[nm].shape[-1] if nm in desc.specs and nm.startswith("kv_cache_k") else None
            out[nm] = KvBuf(nm, raw=raw, src=("name-chain", fd), size=len(raw), hd=hd,
                            note=f"name-chain fd={fd}")

        vollog.warning(f"litert_namechain: names={len(out)}/{len(want)} occurrences={n['occ']} "
                       f"-> native_handle={n['nh']} -> dma_buf={n['file']} -> size ok={n['size']} "
                       f"-> sg_table ok={n['sg']} conflicts={n['conflicts']}")
        return out
