


















from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

MTE_MASK = 0x00FFFFFFFFFFFFFF
PTR = 8


def strip(p: int) -> int:
    return p & MTE_MASK


class MemReader:
    

    def read(self, addr: int, size: int) -> bytes:
        raise NotImplementedError

    def u64(self, addr: int) -> int | None:
        b = self.read(addr, PTR)
        return struct.unpack("<Q", b)[0] if len(b) == PTR else None

    def mapped_words(self) -> Iterable[int]:
        
        raise NotImplementedError


@dataclass
class Layout:
    wrap_tb: int                                              
    stride: int                                       
    wrap_in_elem: int                                                                                  
    array_addr: int                              
    count: int                                                   
    host_ptrs: list = field(default_factory=list)                                           
    extra_wrap_offsets: list = field(default_factory=list)                                                                


def _host_from_tb(reader: MemReader, tb: int, regions: dict[int, int], scan: int = 0x100):
    

    if tb <= 0:
        return None
    blob = reader.read(tb, scan)
    for off in range(0, len(blob) - PTR + 1, PTR):
        v = strip(struct.unpack_from("<Q", blob, off)[0])
        if v in regions:
            return v, regions[v]
    return None


def _chases_to_dmabuf(reader: MemReader, wrapper: int, wrap_tb: int, regions: dict[int, int]):
    
    wrapper = strip(wrapper)
    if wrapper <= 0x10000:
        return None
    tb = reader.u64(wrapper + wrap_tb)
    if tb is None:
        return None
    return _host_from_tb(reader, strip(tb), regions)


def discover(reader: MemReader, regions: dict[int, int], *,
             min_count: int = 8, strides=range(0x18, 0x101, 8),
             wrap_tb_candidates=range(0, 0x101, 8)) -> Layout | None:
    







    wrap_tb_candidates = list(wrap_tb_candidates)

                                                                                                       
                                                                                                       
    per_wt: dict[int, dict[int, tuple]] = {wt: {} for wt in wrap_tb_candidates}                                    
    all_ptrs: dict[int, int] = {}                                                                                      
    for w in reader.mapped_words():
        wp = reader.u64(w)
        if wp is None:
            continue
        sp = strip(wp)
        if sp > 0x10000:
            all_ptrs[w] = sp
            for wt in wrap_tb_candidates:
                tb = reader.u64(sp + wt)
                if tb is None:
                    continue
                hr = _host_from_tb(reader, strip(tb), regions)
                if hr is not None:
                    per_wt[wt][sp] = hr

    wrap_tb, wrappers = max(((wt, d) for wt, d in per_wt.items()), key=lambda kv: len(kv[1]))
    if len(wrappers) < min_count:
        return None

                                                                                                       
    slotset = {w for w, sp in all_ptrs.items() if sp in wrappers}
    if len(slotset) < min_count:
        return None

                                                                                                     
                                                                                                       
                                                                                                         
    def best_progression(candidate_slots):
        best = (0, 0, 0)                           
        for S in strides:
            for s0 in candidate_slots:
                if (s0 - S) in candidate_slots:
                    continue                                                     
                c, x = 0, s0
                while x in candidate_slots:
                    c += 1; x += S
                if c > best[0]:
                    best = (c, s0, S)
        return best

    count, start, stride = best_progression(slotset)
    if count < min_count:
        return None

                                                                                                         
    role_starts = []
    for off in range(0, stride, 8):
        s0 = start + off
        c, x = 0, s0
        while x in slotset:
            c += 1; x += stride
        if c >= count - 1:                                                                            
            role_starts.append(off)
    wrap_in_elem = min(role_starts) if role_starts else 0
    array_addr = start - wrap_in_elem
    extra = [o - wrap_in_elem for o in role_starts if o - wrap_in_elem != 0]

    host_ptrs = []
    for i in range(count):
        a = start + i * stride
        wp = reader.u64(a)
        if wp is not None and strip(wp) in wrappers:
            host_ptrs.append((i, *wrappers[strip(wp)]))

    return Layout(wrap_tb=wrap_tb, stride=stride, wrap_in_elem=wrap_in_elem, array_addr=array_addr,
                  count=count, host_ptrs=host_ptrs, extra_wrap_offsets=extra)


                                                                                                    
                                                                                             
                                                                                                    
class BytesMem(MemReader):
    def __init__(self, base: int, size: int):
        self.base = base
        self.buf = bytearray(size)

    def _o(self, addr: int) -> int:
        return strip(addr) - self.base

    def write(self, addr: int, data: bytes):
        o = self._o(addr)
        self.buf[o:o + len(data)] = data

    def wptr(self, addr: int, val: int):
        self.write(addr, struct.pack("<Q", val))

    def read(self, addr: int, size: int) -> bytes:
        o = self._o(addr)
        if o < 0 or o + size > len(self.buf):
            return b""
        return bytes(self.buf[o:o + size])

    def mapped_words(self):
        return range(self.base, self.base + len(self.buf), PTR)


def _build(mem: BytesMem, *, kv_vector_at, stride, roles, wrap_tb, n_layers,
           regions, dmabuf_base, tb_base, wrap_base, host_in_tb=0x18):
    

                          
    hosts = []
    for i in range(n_layers * len(roles)):
        h = dmabuf_base + i * 0x2000
        regions[h] = 0x2000
        hosts.append(h)
                               
    hi = 0
    for li in range(n_layers):
        elem = kv_vector_at + 0x1000 + li * stride                                      
        for (rname, roff) in roles:
            tb = tb_base + hi * 0x200
            wrap = wrap_base + hi * 0x200
            mem.wptr(tb + host_in_tb, hosts[hi])                                                         
            mem.wptr(wrap + wrap_tb, tb)                            
            mem.wptr(elem + roff, wrap)                                            
            hi += 1
    begin = kv_vector_at + 0x1000
    end = begin + n_layers * stride
    mem.wptr(kv_vector_at + 0x00, begin)
    mem.wptr(kv_vector_at + 0x08, end)
    mem.wptr(kv_vector_at + 0x10, end)
    return begin


def _selftest():
    ok = True
    for name, roles, wrap_tb, stride, n in [
        ("v1_xs-like (K@0,V@8)", [("K", 0x00), ("V", 0x08)], 0x30, 0x30, 20),
        ("v4-like (single wrapper@0)", [("KV", 0x00)], 0x30, 0x30, 15),
        ("hypothetical v5 (stride 0x40, wrap_tb 0x18)", [("K", 0x00), ("V", 0x10)], 0x18, 0x40, 28),
    ]:
        base = 0x70_0000_0000
        mem = BytesMem(base, 0x40_0000)
        regions: dict[int, int] = {}
        _build(mem, kv_vector_at=base + 0x100, stride=stride, roles=roles, wrap_tb=wrap_tb,
               n_layers=n, regions=regions, dmabuf_base=0x7b_0000_0000,
               tb_base=base + 0x10_0000, wrap_base=base + 0x20_0000)
        lay = discover(mem, regions)
        got = lay is not None and lay.stride == stride and lay.wrap_tb == wrap_tb and lay.count == n
        print(f"[{'PASS' if got else 'FAIL'}] {name}: "
              + (f"recovered stride={hex(lay.stride)} wrap_tb={hex(lay.wrap_tb)} count={lay.count} "
                 f"extra_role_offsets={[hex(o) for o in lay.extra_wrap_offsets]}" if lay else "discover() returned None"))
        ok = ok and got
    print("\nALL PASS" if ok else "\nFAILED")
    return ok


def regions_from_maps(maps_text: str) -> dict[int, int]:
    
    out = {}
    for ln in maps_text.splitlines():
        if "/dmabuf" not in ln:
            continue
        rng = ln.split()[0]
        a, b = (int(x, 16) for x in rng.split("-"))
        out[a] = b - a
    return out


class FileRangeMem(MemReader):
    

    def __init__(self, maps_text: str, source):
        self.src = source                                                  
        self.ranges = []                                               
        self.heap = []                                                                          
        for ln in maps_text.splitlines():
            parts = ln.split()
            if len(parts) < 2 or "r" not in parts[1]:
                continue
            a, b = (int(x, 16) for x in parts[0].split("-"))
            self.ranges.append((a, b))
            name = parts[5] if len(parts) > 5 else ""
            if "rw" in parts[1] and ("/dmabuf" not in ln) and not name.endswith((".so", ".oat", ".art", ".dex", ".jar")) and "/dev/" not in name and (b - a) <= 0x10000000:
                self.heap.append((a, b))

    def read(self, addr: int, size: int) -> bytes:
        addr = strip(addr)
        for a, b in self.ranges:
            if a <= addr and addr + size <= b:
                try:
                    self.src.seek(addr); return self.src.read(size)
                except Exception:
                    return b""
        return b""

    def mapped_words(self):
        for a, b in self.heap:
            yield from range(a, b - PTR + 1, PTR)


class VolProcMem(MemReader):
    


    def __init__(self, proc_layer, vmas, dmabuf_regions):
        self.layer = proc_layer
        self.vmas = vmas                                                         
        self._regions = dmabuf_regions
        self.ranges = [(s, e) for s, e, *_ in vmas]
        self.heap = [(s, e) for s, e, anon, *_ in vmas if anon]

    def read(self, addr, size):
        addr = strip(addr)
        try:
            return self.layer.read(addr, size, pad=False)
        except Exception:
            return b""

    def mapped_words(self):
        for a, b in self.heap:
            yield from range(a, b - PTR + 1, PTR)


class CachedVolMem(MemReader):
    





    def __init__(self, proc_read, heap_vmas, dmabuf_vmas, *, per_region_cap=256 << 20, total_cap=3 << 30):
        self._read = proc_read
        self.regions = {s: sz for s, sz in dmabuf_vmas}
        self.blocks = []                                                  
        total = 0
        for s, e in sorted(heap_vmas):
            n = min(e - s, per_region_cap)
            if n <= 0 or total + n > total_cap:
                continue
            data = proc_read(s, n) or b""
            if data:
                self.blocks.append((s, data)); total += len(data)

    def _slice(self, addr, size):
        for s, data in self.blocks:
            if s <= addr and addr + size <= s + len(data):
                o = addr - s
                return data[o:o + size]
        return None

    def read(self, addr, size):
        addr = strip(addr)
        b = self._slice(addr, size)
        if b is not None:
            return b
        return self._read(addr, size) or b""                                                    

    def mapped_words(self):
        for s, data in self.blocks:
            for o in range(0, len(data) - PTR + 1, PTR):
                yield s + o

                                                         
    def u64(self, addr):
        addr = strip(addr)
        for s, data in self.blocks:
            if s <= addr and addr + PTR <= s + len(data):
                return struct.unpack_from("<Q", data, addr - s)[0]
        b = self._read(addr, PTR)
        return struct.unpack("<Q", b)[0] if b and len(b) == PTR else None


def _cli(maps_path: str, mem_path: str):
    maps = open(maps_path, "r", errors="replace").read()
    regions = regions_from_maps(maps)
    print(f"dma_buf regions from maps: {len(regions)}")
    with open(mem_path, "rb") as f:
        lay = discover(FileRangeMem(maps, f), regions)
    if lay is None:
        print("no KVCache layout discovered"); return 1
    print(f"DISCOVERED layout (no generation assumed):")
    print(f"  Layer array @ 0x{lay.array_addr:x}  stride=0x{lay.stride:x}  count={lay.count}")
    print(f"  wrapper offset in element = 0x{lay.wrap_in_elem:x}; sibling role offsets = {[hex(o) for o in lay.extra_wrap_offsets]}")
    print(f"  wrapper -> TensorBuffer at +0x{lay.wrap_tb:x}; host via TB content scan")
    for i, host, sz in lay.host_ptrs[:40]:
        print(f"    layer {i:3d}: host=0x{host:x} size={sz}")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:                                                          
        sys.exit(_cli(sys.argv[1], sys.argv[2]))
    sys.exit(0 if _selftest() else 1)
