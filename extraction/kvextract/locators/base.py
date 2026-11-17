





from __future__ import annotations
import logging
from abc import abstractmethod
from typing import Dict
from volatility3.framework import exceptions
from volatility3.plugins.linux import proc

vollog = logging.getLogger(__name__)


class KvBuf:
    __slots__ = ("name", "raw", "src", "size", "note", "hd")

    def __init__(self, name, raw, src, size, note="", hd=None):
        self.name = name; self.raw = raw; self.src = src; self.size = size
        self.note = note; self.hd = hd                                                           


def vma_name(ctx, task, vma):
    try:
        return vma.get_name(ctx, task)
    except Exception:
        return None


class Locator:
    key = "base"
    self_describing = False

    @abstractmethod
    def detect(self, ctx, task) -> int: ...

    @abstractmethod
    def locate(self, ctx, task, reader, reasm, off, desc) -> Dict[str, KvBuf]: ...

    @staticmethod
    def scan(ctx, reader, task, needle, limit=None, max_vma=1 << 40, budget=16 << 30):
        

        WIN = 2 << 20
        hits = []; scanned = 0; nvma = 0; nrw = 0; rwbytes = 0; readmb = 0

        def mapped(va, span):
            for p in (va, va + span // 2, va + span - 0x1000):
                try:
                    if reader.v2p(p & ~0xfff) is not None:
                        return True
                except Exception:
                    pass
            return False

        for vma in proc.Maps.list_vmas(task):
            nvma += 1
            try:
                start = int(vma.vm_start); end = int(vma.vm_end)
                if "rw" not in vma.get_protection() or vma.vm_file != 0:
                    continue
            except exceptions.InvalidAddressException:
                continue
            if end - start > max_vma:
                continue
            nrw += 1; rwbytes += end - start
            va = start
            while va < end:
                n = min(WIN, end - va)
                if not mapped(va, n):
                    va += n; continue
                chunk = reader.read(va, n)
                readmb += n
                off = 0
                while True:
                    i = chunk.find(needle, off)
                    if i < 0:
                        break
                    hits.append(va + i); off = i + 1
                    if limit and len(hits) >= limit:
                        return hits
                va += n; scanned += n
                if scanned > budget:
                    vollog.warning(f"scan: BUDGET hit at {scanned/2**30:.1f}GB"); return hits
        vollog.debug(f"scan: VMAs={nvma} rw-anon={nrw} span={rwbytes/2**20:.0f}MB "
                     f"read={readmb/2**20:.0f}MB hits={len(hits)}")
        return hits
