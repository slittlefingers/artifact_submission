






from __future__ import annotations
import struct
from volatility3.framework import exceptions

DEF_PAGE_OFFSET = 0xFFFFFF8000000000                                                           
DEF_PHYS_OFFSET = 0x80000000                                                                   


class ProcReader:
    PA_MASK = 0x0000FFFFFFFFF000

    def __init__(self, ctx, phys_layer_name, pgd_va, zram=None,
                 page_offset=DEF_PAGE_OFFSET, phys_offset=DEF_PHYS_OFFSET):
        self.pl = ctx.layers[phys_layer_name]
        self.PAGE_OFFSET = page_offset
        self.PHYS_OFFSET = phys_offset
        self.pgd_pa = (pgd_va - page_offset + phys_offset) & ((1 << 64) - 1)
        self.zram = zram                                                         
        self.zram_obj = None                                                 

    def _rdpa(self, pa, n):
        try:
            return self.pl.read(pa, n, pad=False)
        except exceptions.InvalidAddressException:
            return None

    def _page(self, va):
        v = va & 0x7fffffffff
        i1 = (v >> 30) & 0x1ff; i2 = (v >> 21) & 0x1ff; i3 = (v >> 12) & 0x1ff
        b = self._rdpa(self.pgd_pa + i1 * 8, 8)
        if not b: return None
        e1 = struct.unpack("<Q", b)[0]
        if e1 & 3 == 1:
            return self._rdpa(((e1 & self.PA_MASK) & ~0x3fffffff) | (va & 0x3ffff000), 0x1000)
        if e1 & 3 != 3: return None
        b = self._rdpa((e1 & self.PA_MASK) + i2 * 8, 8)
        if not b: return None
        e2 = struct.unpack("<Q", b)[0]
        if e2 & 3 == 1:
            return self._rdpa(((e2 & self.PA_MASK) & ~0x1fffff) | (va & 0x1ff000), 0x1000)
        if e2 & 3 != 3: return None
        b = self._rdpa((e2 & self.PA_MASK) + i3 * 8, 8)
        if not b: return None
        e3 = struct.unpack("<Q", b)[0]
        if e3 & 1:
            return self._rdpa(e3 & self.PA_MASK, 0x1000)
        if e3 and self.zram:
            return self.zram((e3 >> 8) & ((1 << 50) - 1))
        return None

    def v2p(self, va):
        v = va & 0x7fffffffff
        i1 = (v >> 30) & 0x1ff; i2 = (v >> 21) & 0x1ff; i3 = (v >> 12) & 0x1ff
        b = self._rdpa(self.pgd_pa + i1 * 8, 8)
        if not b: return None
        e1 = struct.unpack("<Q", b)[0]
        if e1 & 3 == 1:
            return ((e1 & self.PA_MASK) & ~0x3fffffff) | (va & 0x3fffffff)
        if e1 & 3 != 3: return None
        b = self._rdpa((e1 & self.PA_MASK) + i2 * 8, 8)
        if not b: return None
        e2 = struct.unpack("<Q", b)[0]
        if e2 & 3 == 1:
            return ((e2 & self.PA_MASK) & ~0x1fffff) | (va & 0x1fffff)
        if e2 & 3 != 3: return None
        b = self._rdpa((e2 & self.PA_MASK) + i3 * 8, 8)
        if not b: return None
        e3 = struct.unpack("<Q", b)[0]
        return (e3 & self.PA_MASK) | (va & 0xfff) if e3 & 1 else None

    def v2p_diag(self, va):
        v = va & 0x7fffffffff
        i1 = (v >> 30) & 0x1ff; i2 = (v >> 21) & 0x1ff; i3 = (v >> 12) & 0x1ff
        b = self._rdpa(self.pgd_pa + i1 * 8, 8)
        if not b: return (None, "L1 table page absent from dump")
        e1 = struct.unpack("<Q", b)[0]
        if e1 & 3 == 1: return (((e1 & self.PA_MASK) & ~0x3fffffff) | (va & 0x3fffffff), "1GB block")
        if e1 & 3 != 3: return (None, f"L1 PTE not present (e1={e1:#x}) -> UNMAPPED")
        b = self._rdpa((e1 & self.PA_MASK) + i2 * 8, 8)
        if not b: return (None, "L2 table page absent from dump")
        e2 = struct.unpack("<Q", b)[0]
        if e2 & 3 == 1: return (((e2 & self.PA_MASK) & ~0x1fffff) | (va & 0x1fffff), "2MB block")
        if e2 & 3 != 3: return (None, f"L2 PTE not present (e2={e2:#x}) -> UNMAPPED")
        b = self._rdpa((e2 & self.PA_MASK) + i3 * 8, 8)
        if not b: return (None, "L3 table page absent from dump")
        e3 = struct.unpack("<Q", b)[0]
        if e3 & 1 == 0:
            return (None, f"L3 PTE not present (e3={e3:#x}) -> UNMAPPED/swapped")
        pa = (e3 & self.PA_MASK) | (va & 0xfff)
        probe = self._rdpa(pa & ~0xfff, 8)
        if probe is None:
            return (pa, f"MAPPED (PTE ok) but DATA PAGE ABSENT = TENSOR-PROTECTED/excluded")
        return (pa, f"MAPPED+PRESENT (e3={e3:#x})")

    def read(self, va, n, zero_fill=True):
        out = bytearray()
        while n > 0:
            off = va & 0xfff; c = min(n, 0x1000 - off)
            pg = self._page(va & ~0xfff)
            d = pg[off:off + c] if pg and len(pg) >= off + c else None
            if d is None:
                if not zero_fill:
                    return None
                d = b"\x00" * c
            out += d; va += c; n -= c
        return bytes(out)

    def find_page(self, target_page, user_only=True, cap=1 << 22):
        PA = self.PA_MASK; seen = 0
        for i1 in range(512):
            e1 = self._q(self.pgd_pa + i1 * 8)
            if not e1 or e1 & 3 != 3:
                continue
            l2 = e1 & PA
            for i2 in range(512):
                e2 = self._q(l2 + i2 * 8)
                if e2 is None:
                    continue
                if e2 & 3 == 1:
                    base = (e2 & PA) & ~0x1fffff
                    if base <= target_page < base + 0x200000:
                        return (i1 << 30) | (i2 << 21) | (target_page - base)
                    continue
                if e2 & 3 != 3:
                    continue
                l3 = e2 & PA
                for i3 in range(512):
                    e3 = self._q(l3 + i3 * 8)
                    if e3 and e3 & 1:
                        seen += 1
                        if (e3 & PA) == target_page:
                            return (i1 << 30) | (i2 << 21) | (i3 << 12)
                if seen > cap:
                    return None
        return None

    def iter_swapped_pages(self, cap=1 << 20):
        PA = self.PA_MASK; n = 0
        l1 = self._rdpa(self.pgd_pa, 4096)
        if not l1 or len(l1) < 4096:
            return
        e1s = struct.unpack_from("<512Q", l1, 0)
        for i1 in range(512):
            e1 = e1s[i1]
            if not e1 or e1 & 3 != 3:
                continue
            l2p = self._rdpa(e1 & PA, 4096)
            if not l2p or len(l2p) < 4096:
                continue
            e2s = struct.unpack_from("<512Q", l2p, 0)
            for i2 in range(512):
                e2 = e2s[i2]
                if e2 & 3 != 3:
                    continue
                l3p = self._rdpa(e2 & PA, 4096)
                if not l3p or len(l3p) < 4096:
                    continue
                e3s = struct.unpack_from("<512Q", l3p, 0)
                for i3 in range(512):
                    e3 = e3s[i3]
                    if e3 and not (e3 & 1):
                        n += 1
                        yield ((i1 << 30) | (i2 << 21) | (i3 << 12), (e3 >> 8) & ((1 << 50) - 1))
                        if n > cap:
                            return

    def _q(self, pa):
        b = self._rdpa(pa, 8)
        return struct.unpack("<Q", b)[0] if b else None
