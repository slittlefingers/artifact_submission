


from __future__ import annotations
import struct
from collections import Counter


def k_step(raw: bytes, row: int = 4096):
    
    n = len(raw)
    if n % row:
        return None
    emb = n // row
    if emb not in (256, 512):
        return None
    z = b"\x00" * emb
    for r in range(row):
        if raw[r * emb:(r + 1) * emb] == z:
            return r if r > 0 else None
    return row


def read_kvex(path: str):
    

    b = open(path, "rb").read()
    if len(b) < 24 or b[:4] != b"KVEX":
        raise ValueError(f"{path}: not a KVEX blob")
    count = struct.unpack_from("<I", b, 20)[0]
    out = []; p = 24
    for _ in range(count):
        p += 1                                                        
        nl = struct.unpack_from("<H", b, p)[0]; p += 2
        name = b[p:p + nl].decode("latin1"); p += nl
        nb = struct.unpack_from("<Q", b, p)[0]; p += 8
        out.append((name, b[p:p + nb])); p += nb
    return out


def is_kvex(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"KVEX"
    except Exception:
        return False


def write_kvex(raws: dict, out: str, row: int = 4096):
    ks = [s for s in (k_step(raws[nm], row) for nm in raws if "_k_" in nm) if s]
    S = Counter(ks).most_common(1)[0][0] if ks else None

    def order(nm):
        tail = nm.rsplit("_", 1)[1]
        num = int("".join(c for c in tail if c.isdigit()) or "0")
        return (num, 0 if "_k_" in nm else 1)

    entries = list(raws.items())
    blob = bytearray(b"KVEX") + struct.pack("<IIII", 1, row, 0, 0) + struct.pack("<I", len(entries))
    for nm, data in sorted(entries, key=lambda e: order(e[0])):
        nb = nm.encode()
        blob += struct.pack("<B", 1) + struct.pack("<H", len(nb)) + nb + struct.pack("<Q", len(data)) + data
    with open(out, "wb") as f:
        f.write(blob)
    return S
