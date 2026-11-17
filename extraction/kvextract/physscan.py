










from __future__ import annotations
import re
import struct
from typing import Iterator, Tuple

from . import constants as C

NAME_RX = re.compile(rb"kv_cache_([kv])_([0-9]{1,2})(?![0-9])")
MERGED_RX = re.compile(rb"merged_\d+_[0-9x]+_(key|value)_cache_(\d+)")
Row = Tuple[str, str, str]


def run(ctx, kr) -> Iterator[Row]:
    from volatility3.framework.layers import scanners
    pl = kr.pl

                                                                                         
    kv_names = {}
    try:
        for pa in pl.scan(context=ctx, scanner=scanners.BytesScanner(b"kv_cache_")):
            win = kr._rdpa(pa, 24)
            if not win:
                continue
            m = NAME_RX.match(win) or NAME_RX.search(win)
            if m:
                kv_names.setdefault((m.group(1).decode(), int(m.group(2))), pa)
    except Exception as e:
        yield ("A kv_cache scan", "err", str(e)[:80])
    if kv_names:
        layers = sorted({l for _, l in kv_names})
        yield ("A: kv_cache names", str(len(kv_names)),
               f"litertlm/cpu: K/V layers={layers} (EXPLICIT order from name suffix)")

                                                                                               
    ELEM, DIMS, BT, SIZE, ADDR, HDR, MTE = C.HOST_ELEM, 0x10, C.HOST_BT, C.HOST_SIZE, C.HOST_ADDR, C.HOST_HDR, C.MTE_MASK
    cpu = []; seen = set()
    try:
        for pa in pl.scan(context=ctx, scanner=scanners.BytesScanner(C.HOST_SIG)):
            base = pa - ELEM
            if base < 0:
                continue
            hdr = kr._rdpa(base, HDR)
            if not hdr or len(hdr) < HDR or struct.unpack_from("<I", hdr, BT)[0] != 1:
                continue
            d0, d1, d2, d3 = struct.unpack_from("<4i", hdr, DIMS)
            if d0 != 1 or d1 != 1:
                continue
            size = struct.unpack_from("<Q", hdr, SIZE)[0]; addr = struct.unpack_from("<Q", hdr, ADDR)[0] & MTE
            if d3 in (256, 512):
                role, hd, cx = "k", d3, d2
            elif d2 in (256, 512):
                role, hd, cx = "v", d2, d3
            else:
                continue
            if size != d2 * d3 or cx <= 0 or cx > (1 << 20):
                continue
            k = (role, hd, cx, size, addr)
            if k in seen:
                continue
            seen.add(k); cpu.append(k)
    except Exception as e:
        yield ("B cpu TB scan", "err", str(e)[:80])
    for role, hd, cx, size, addr in cpu[:40]:
        yield (f"B: cpu TB {role}", hex(size),
               f"hd={hd} ctx={cx} host=0x{addr:x} (header@phys; data needs owner pgd if freed)")
    if cpu:
        yield ("B: cpu TB count", str(len(cpu)),
               f"K={sum(1 for g in cpu if g[0]=='k')} V={sum(1 for g in cpu if g[0]=='v')}")

                                                                                  
    mrg = {}
    try:
        for pa in pl.scan(context=ctx, scanner=scanners.BytesScanner(b"_cache_")):
            win = kr._rdpa(pa - 48, 120)
            if not win:
                continue
            m = MERGED_RX.search(win)
            if m:
                mrg.setdefault((m.group(1).decode(), int(m.group(2))), pa)
    except Exception as e:
        yield ("C aicore scan", "err", str(e)[:80])
    if mrg:
        layers = sorted({l for _, l in mrg})
        yield ("C: aicore merged_cache", str(len(mrg)),
               f"AICore names present (layers {min(layers)}..{max(layers)}); NO name->buffer ptr "
               f"-> identify by dma_buf/content+geometry, not names")

    if not kv_names and not cpu and not mrg:
        yield ("(nothing)", "0", "no KV anchors found in the physical layer")
