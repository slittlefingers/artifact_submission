






from __future__ import annotations
import logging
import struct
from collections import Counter
from typing import Dict
from volatility3.framework.layers import scanners
from volatility3.plugins.linux import proc
from .base import Locator, KvBuf, vma_name
from .. import constants as C

vollog = logging.getLogger(__name__)


class LitertHostLocator(Locator):
    key = "litert_host"
    self_describing = True
    SIG = C.HOST_SIG
    ELEM, BT, SIZE, ADDR, HDR = C.HOST_ELEM, C.HOST_BT, C.HOST_SIZE, C.HOST_ADDR, C.HOST_HDR
    DIMS = 0x10
    MTE = C.MTE_MASK

    def __init__(self):
        self.row = 4096

    def detect(self, ctx, task) -> int:
        for v in proc.Maps.list_vmas(task):
            nm = vma_name(ctx, task, v) or ""
            if "xnnpack" in nm or nm.endswith("gemma-4-E2B-it.litertlm"):
                return 60
        return 0

    def _accept(self, hdr, seen):
        if not hdr or len(hdr) < self.HDR:
            return None
        if struct.unpack_from("<I", hdr, self.BT)[0] != 1:
            return None
        d0, d1, d2, d3 = struct.unpack_from("<4i", hdr, self.DIMS)
        if d0 != 1 or d1 != 1:
            return None
        size = struct.unpack_from("<Q", hdr, self.SIZE)[0]
        addr = struct.unpack_from("<Q", hdr, self.ADDR)[0] & self.MTE
        if d3 in (256, 512):
            role, hd, cx = "k", d3, d2
        elif d2 in (256, 512):
            role, hd, cx = "v", d2, d3
        else:
            return None
        if size != d2 * d3 or cx <= 0 or cx > (1 << 20) or not addr or addr in seen:
            return None
        return (addr, role, hd, cx, size)

    def locate(self, ctx, task, reader, reasm, off, desc) -> Dict[str, KvBuf]:
        pl = reader.pl
        nhit = 0; found = []; seen = set()
        for pa in pl.scan(context=ctx, scanner=scanners.BytesScanner(self.SIG)):
            nhit += 1
            base_pa = pa - self.ELEM
            if base_pa < 0:
                continue
            hdr = reader._rdpa(base_pa, self.HDR)
            acc = self._accept(hdr, seen)
            if not acc:
                continue
            addr, role, hd, cx, size = acc
            if reader.v2p(addr & ~0xfff) is None:
                continue
            raw = reader.read(addr, size)
            nz = len(raw) - raw.count(0)
            if nz < 4096:
                continue
            seen.add(addr)
            found.append((addr, role, hd, cx, size, raw, nz))
        vollog.warning(f"litert_host(phys): {nhit} sig hits -> {len(found)} Host-Int8 KV TensorBufferT "
                       f"(K={sum(1 for f in found if f[1]=='k')} V={sum(1 for f in found if f[1]=='v')})")

                                                                                          
        if getattr(reader, "zram", None):
            pairs = list(reader.iter_swapped_pages())
            zobj = getattr(reader, "zram_obj", None)
            if zobj is not None:
                try:
                    n = zobj.prime([o for _, o in pairs])
                    vollog.warning(f"litert_host(zram): priming {len(pairs)} swapped pages -> {n} decoded")
                except Exception as e:
                    vollog.warning(f"litert_host(zram): prime failed: {e}")
            zhit = zrec = 0
            for va, o in pairs:
                try:
                    pg = reader.zram(o)
                except Exception:
                    pg = None
                if not pg:
                    continue
                pos = pg.find(self.SIG)
                if pos < 0:
                    continue
                zhit += 1
                acc = self._accept(reader.read(va + pos - self.ELEM, self.HDR), seen)
                if not acc:
                    continue
                addr, role, hd, cx, size = acc
                raw = reader.read(addr, size)
                if len(raw) - raw.count(0) < 4096:
                    continue
                seen.add(addr); found.append((addr, role, hd, cx, size, raw, len(raw) - raw.count(0))); zrec += 1
            vollog.warning(f"litert_host(zram): swapped-page sig hits={zhit} -> recovered {zrec} more "
                           f"(total now {len(found)})")

        if found:
            self.row = Counter(f[3] for f in found).most_common(1)[0][0]
        out = {}
        for role in ("k", "v"):
            rl = sorted([f for f in found if f[1] == role])
            for i, (addr, r, hd, cx, size, raw, nz) in enumerate(rl):
                nm = f"kv_cache_{role}_{i}"
                out[nm] = KvBuf(nm, raw=raw, src=("host", hex(addr)), size=size, hd=hd,
                                note=f"hd={hd} ctx={cx} nz={nz} PROVISIONAL-layer")
        return out
