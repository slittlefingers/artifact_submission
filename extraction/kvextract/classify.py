






from __future__ import annotations
from typing import Optional, Tuple


def k_step(raw: bytes, row_bytes: int, embs: Tuple[int, ...]) -> Optional[int]:
    




    n = len(raw)
    if not row_bytes or n % row_bytes:
        return None
    emb = n // row_bytes
    if emb not in embs:
        return None
    last = len(raw.rstrip(b"\x00"))
    if last == 0 or last == n or last % emb:
        return None
    return last // emb


def classify_kv(raw: bytes, S: int, row_bytes: int, embs: Tuple[int, ...],
                k_dense: float = 0.5, k_tail_zero: float = 0.95,
                v_row_dense: float = 0.5, v_row_tail_zero: float = 0.90,
                v_min_rows: float = 0.4) -> Optional[Tuple[str, int]]:
    






    n = len(raw)
    if not row_bytes or n % row_bytes or S <= 0:
        return None
    emb = n // row_bytes
    if emb not in embs:
        return None
    kb = S * emb
    if 0 < kb < n:
        head, tail = raw[:kb], raw[kb:]
        if (kb - head.count(0)) / kb > k_dense and tail.count(0) / len(tail) > k_tail_zero:
            return ("k", emb)
    ok = 0
    for r in range(emb):
        row = raw[r * row_bytes:(r + 1) * row_bytes]
        h, t = row[:S], row[S:]
        if (S - h.count(0)) / max(1, S) > v_row_dense and \
           t.count(0) / max(1, len(t)) > v_row_tail_zero:
            ok += 1
    if ok > v_min_rows * emb:
        return ("v", emb)
    return None


def content_key(raw: bytes, kind: str, emb: int, S: int, row_bytes: int) -> bytes:
    
    if kind == "k":
        return raw[:S * emb]
    return b"".join(raw[r * row_bytes:r * row_bytes + S] for r in range(emb))


def causal_fill(raw: bytes, head_dim: int = 256) -> Optional[Tuple[str, int, int]]:
    


    n = len(raw)
    if not head_dim or n % head_dim:
        return None
    cs = n // head_dim
    if cs < 2:
        return None
    last = -1
    for r in range(cs):
        row = raw[r * head_dim:(r + 1) * head_dim]
        if row.count(0) != head_dim:                        
            if last == r - 1:
                last = r                                                
            else:
                return None                                              
    if 0 <= last < cs - 1:                                                    
        return ("k", last + 1, cs)
    return None


def looks_float32(raw: bytes) -> bool:
    



    n = (len(raw) // 4) * 4
    if n < 256:
        return False
    msb = raw[3:n:4][:8192]                                                                           
    if not msb:
        return False
                                                                                                        
    plausible = sum(1 for b in msb if b == 0 or 0x30 <= (b & 0x7f) <= 0x43)
    uniq = len(set(msb))
    return uniq <= 48 and plausible / len(msb) > 0.95


def aicore_layout(raw: bytes, head_dim: int, k_dense: float = 0.9,
                  v_row_agree: float = 0.6) -> Optional[Tuple[str, int, int]]:
    







    n = len(raw)
    if not head_dim or n % head_dim:
        return None
    cache = n // head_dim
    if cache < 2:
        return None
    if looks_float32(raw):                                                                                
        return None
                                                                                              
    fe = len(raw.rstrip(b"\x00"))
    if 0 < fe < n and fe % head_dim == 0:
        pref = raw[:fe]
        if (fe - pref.count(0)) / fe > k_dense:
            return ("k", cache, fe // head_dim)
                                                                                           
    fills = [len(raw[r * cache:(r + 1) * cache].rstrip(b"\x00")) for r in range(head_dim)]
    nz = [f for f in fills if f > 0]
    if nz:
        from collections import Counter
        modal, cnt = Counter(nz).most_common(1)[0]
        if cnt / head_dim > v_row_agree and 0 < modal < cache:
            return ("v", cache, modal)
    return None


def host_sig_match(header: bytes, sig: bytes) -> bool:
    
    off = 0x08
    return len(header) >= off + len(sig) and header[off:off + len(sig)] == sig
