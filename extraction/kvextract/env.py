





from __future__ import annotations
import struct
from typing import List, Tuple, Optional
from . import constants as C

MASK64 = (1 << 64) - 1


                                                                                
def phys_offset(ranges: List[Tuple[int, int]]) -> int:
    
    return min(r[0] for r in ranges)


def banks(ranges: List[Tuple[int, int]], gap: int = 0x100000000) -> List[Tuple[int, int]]:
    

    rs = sorted(ranges)
    out: List[List[int]] = []
    for lo, hi in rs:                                       
        if out and lo - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def page_offset(va_bits: int) -> int:
    

    return (-(1 << va_bits)) & MASK64


                                                                                 
class LimeImage:
    

    def __init__(self, path: str):
        self.path = path
        self.ranges: List[Tuple[int, int, int]] = []                                       
        self._index()

    def _index(self) -> None:
        with open(self.path, "rb") as f:
            while True:
                h = f.read(C.LIME_HDR)
                if len(h) < C.LIME_HDR:
                    break
                magic, _ver = struct.unpack_from("<II", h, 0)
                if magic != C.LIME_MAGIC:
                    break
                sa, ea, _ = struct.unpack_from("<QQQ", h, 8)
                self.ranges.append((sa, ea, f.tell()))
                f.seek(ea - sa + 1, 1)
        if not self.ranges:
            raise ValueError(f"{self.path}: no LiME ranges (bad magic?)")

    def phys_offset(self) -> int:
        return phys_offset([(s, e) for s, e, _ in self.ranges])

    def banks(self) -> List[Tuple[int, int]]:
        return banks([(s, e) for s, e, _ in self.ranges])

    def file_offset(self, pa: int) -> Optional[int]:
        
        for sa, ea, fo in self.ranges:
            if sa <= pa <= ea:
                return fo + (pa - sa)
        return None
