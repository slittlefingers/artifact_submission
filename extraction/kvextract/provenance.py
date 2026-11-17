






from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Iterable


TOKEN_STRUCT_SIZE = 40                                                      
MAX_TOKEN_BATCH = 8

                                                                      
                                                                         
                                                                         
                                                                    
LITERT_NPU_PROFILES = {
    "d5a85b937c1251e327767b1d72ca4b9a6ea957d07aa50eacf41aef65e5b184e2": {
        "library": "liblitertlm_jni.so",
        "vtable_offset": 0x143E9A8,
        "current_step_offset": 0x10A8,
        "processed_tokens_offset": 0x10B8,
        "evidence": "installed ELF RTTI vtable slot 18=0x889bc8, slot 10=0x88f8ec",
    },
}


class TokenPageOverlay:
    





    def __init__(self, reader, pages: list[dict]):
        self.reader = reader
        self.pages = {}
        for row in pages:
            va = int(row["virtual_page"], 16)
            pa = int(row["physical_page"], 16)
            if va & 4095 or pa & 4095 or va in self.pages:
                raise ValueError("token page map requires unique 4 KiB aligned pages")
            raw = reader._rdpa(pa, 4096)
            if raw is None or hashlib.sha256(raw).hexdigest() != row["sha256"].lower():
                raise ValueError(f"token page map image hash mismatch at {pa:#x}")
            self.pages[va] = pa

    def v2p(self, va):
        clean = va & 0x7FFFFFFFFF                                        
        page = clean & ~4095
        return self.pages[page] + (clean & 4095) if page in self.pages else self.reader.v2p(va)

    def v2p_diag(self, va):
        clean = va & 0x7FFFFFFFFF
        if clean & ~4095 in self.pages:
            return self.v2p(va), "external pagemap, image SHA-256 verified"
        return self.reader.v2p_diag(va)

    def read(self, va, n, zero_fill=False):
        out = bytearray()
        while n:
            clean = va & 0x7FFFFFFFFF
            take = min(n, 4096 - (clean & 4095))
            page = clean & ~4095
            if page in self.pages:
                raw = self.reader._rdpa(self.pages[page] + (clean & 4095), take)
            else:
                raw = self.reader.read(va, take, zero_fill=False)
            if raw is None:
                return None if not zero_fill else bytes(out) + b"\x00" * n
            out.extend(raw)
            va += take
            n -= take
        return bytes(out)


def scan_executor_vtables(reader, regions: Iterable[tuple[int, int]],
                          library_base: int, library_sha256: str,
                          expected_step: int, vocab_size: int = 262144,
                          bos_id: int = 2, step_tolerance: int = 1):
    




    profile = LITERT_NPU_PROFILES.get(library_sha256.lower())
    if profile is None:
        return {"status": "unknown_library_build", "candidates": [],
                "library_sha256": library_sha256}
    vtable_va = library_base + profile["vtable_offset"]
    needle = struct.pack("<Q", vtable_va)
    objects = []
    bytes_read = 0
    bytes_examined = 0
    for start, end in regions:
        page_va = start & ~4095
        while page_va < end:
            lo = max(start, page_va)
            hi = min(end, page_va + 4096)
            bytes_examined += hi - lo
            page = reader.read(page_va, 4096, zero_fill=False)
            if page is not None:
                bytes_read += hi - lo
                pos = max(0, lo - page_va)
                while True:
                    pos = page.find(needle, pos, hi - page_va)
                    if pos < 0:
                        break
                    object_va = page_va + pos
                    if object_va % 8 == 0:
                        step_raw = reader.read(object_va + profile["current_step_offset"], 4,
                                               zero_fill=False)
                        current_step = struct.unpack("<I", step_raw)[0] if step_raw else None
                        header_va = object_va + profile["processed_tokens_offset"]
                        raw_header = reader.read(header_va, 24, zero_fill=False)
                        tokens_array_va = None
                        tokens_array_pa = None
                        tokens_array_mapping = None
                        if raw_header is not None:
                            tokens_array_va = struct.unpack_from("<Q", raw_header)[0]
                            tokens_array_pa = reader.v2p(tokens_array_va) if tokens_array_va else None
                            if tokens_array_va and hasattr(reader, "v2p_diag"):
                                _, tokens_array_mapping = reader.v2p_diag(tokens_array_va)
                        vector = None
                        if raw_header is not None and current_step is not None:
                            vector = parse_token_vector(reader, header_va, current_step,
                                                        vocab_size, bos_id, step_tolerance)
                            if vector is not None:
                                vector["evidence"] = (
                                    "exact installed-library vtable and member offset, "
                                    "executor current step, BOS, vocabulary bounds")
                                vector["status"] = "executor_offset_validated"
                        objects.append({
                            "executor_va": _hex(object_va),
                            "executor_pa": _hex(reader.v2p(object_va)),
                            "vtable_va": _hex(vtable_va),
                            "current_step": current_step,
                            "expected_kv_step": expected_step,
                            "processed_tokens_va": _hex(header_va),
                            "processed_tokens_pa": _hex(reader.v2p(header_va)),
                            "vector_header_readable": raw_header is not None,
                            "vector_header_words": [_hex(x) for x in struct.unpack("<QQQ", raw_header)]
                            if raw_header is not None else None,
                            "tokens_array_va": _hex(tokens_array_va),
                            "tokens_array_pa": _hex(tokens_array_pa),
                            "tokens_array_mapping": tokens_array_mapping,
                            "token_status": "validated" if vector is not None else (
                                "vector_header_unreadable" if raw_header is None else
                                "tokens_array_unmapped_in_snapshot" if tokens_array_va and tokens_array_pa is None else
                                "token_validation_failed"),
                            "token_vector": vector,
                        })
                    pos += 8
            page_va += 4096
    return {"status": "executor_vtable_scan_complete", "library_sha256": library_sha256,
            "library_base_va": _hex(library_base), "profile": profile,
            "bytes_read": bytes_read, "bytes_examined": bytes_examined,
            "objects": objects,
            "candidates": [o["token_vector"] for o in objects if o["token_vector"] is not None]}


def _hex(value):
    return None if value is None else hex(int(value))


def snapshot_kernel_field(kr, label: str, va: int | None, length: int):
    
    if va is None:
        return None
    if kr.PAGE_OFFSET <= va < kr.PAGE_END:
        pa = va - kr.PAGE_OFFSET + kr.PHYS_OFFSET
        raw = kr.rv(va, length)
        virtual = va
    elif 0 <= va < kr.PAGE_OFFSET:
        pa = va
        raw = kr._rdpa(pa, length)
        virtual = None
    else:
        pa = None
        raw = None
        virtual = va
    return {"name": label, "virtual_address": _hex(virtual),
            "physical_address": _hex(pa), "length": length,
            "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
            "value": _hex(int.from_bytes(raw, "little")) if raw and length in (4, 8) else None}


def kernel_chain_fields(kr, off: dict, task_va: int | None, fd_trace: dict, sg_trace: dict):
    
    def addr(value):
        return int(value, 16) if value else None

    fields = []
    specs = [
        ("task.files", task_va, off.get("OFF_FILES"), 8),
        ("task.mm", task_va, off.get("OFF_MM"), 8),
        ("files.fdt", addr(fd_trace.get("files_struct_va")), off.get("F_FDT"), 8),
        ("fdtable.max_fds", addr(fd_trace.get("fdtable_va")), off.get("FDT_MAXFDS"), 4),
        ("fdtable.fd", addr(fd_trace.get("fdtable_va")), off.get("FDT_FD"), 8),
        ("fd_entry", addr(fd_trace.get("fd_entry_va")), 0, 8),
        ("file.f_op", addr(fd_trace.get("file_va")), off.get("FILE_FOP"), 8),
        ("file.private_data", addr(fd_trace.get("file_va")), off.get("FILE_PRIV"), 8),
        ("dma_buf.size", addr(fd_trace.get("dma_buf_va")), off.get("DB_SIZE"), 8),
        ("dma_buf.priv", addr(fd_trace.get("dma_buf_va")), off.get("DB_PRIV"), 8),
        ("priv.sg_table", addr(sg_trace.get("sg_table_field_va")), 0, 16),
    ]
    for label, base, offset, length in specs:
        if base is not None and offset is not None:
            snap = snapshot_kernel_field(kr, label, base + offset, length)
            if snap is not None:
                fields.append(snap)
    seen = set()
    for segment in sg_trace.get("segments", []):
        va = addr(segment.get("scatterlist_va"))
        if va is not None and va not in seen:
            seen.add(va)
            snap = snapshot_kernel_field(kr, "scatterlist.entry", va, off["SC_SZ"])
            if snap is not None:
                fields.append(snap)
    return fields


def _segments(reader, va: int, size: int):
    
    out = []
    end = va + size
    while va < end:
        n = min(end - va, 4096 - (va & 4095))
        pa = reader.v2p(va)
        raw = reader.read(va, n, zero_fill=False) if pa is not None else None
        out.append({"virtual_address": _hex(va), "physical_address": _hex(pa),
                    "length": n, "sha256": hashlib.sha256(raw).hexdigest() if raw else None})
        va += n
    return out


def parse_token_vector(reader, outer_va: int, expected_step: int, vocab_size: int = 262144,
                       bos_id: int = 2, step_tolerance: int = 1):
    





    if expected_step <= 0 or (bos_id >= 0 and vocab_size <= bos_id):
        return None
    raw = reader.read(outer_va, 24, zero_fill=False)
    if raw is None or len(raw) != 24:
        return None
    begin, end, cap = struct.unpack("<QQQ", raw)
    if not (begin and begin % 8 == 0 and begin < end <= cap):
        return None
    if (end - begin) % TOKEN_STRUCT_SIZE or (cap - begin) % TOKEN_STRUCT_SIZE:
        return None
    batch = (end - begin) // TOKEN_STRUCT_SIZE
    if not 1 <= batch <= MAX_TOKEN_BATCH or (cap - begin) // TOKEN_STRUCT_SIZE > 64:
        return None
    items = []
    for i in range(batch):
        item_va = begin + i * TOKEN_STRUCT_SIZE
        item = reader.read(item_va, TOKEN_STRUCT_SIZE, zero_fill=False)
        if item is None or len(item) != TOKEN_STRUCT_SIZE:
            return None
        ids_va, ids_end, ids_cap, pending_va, control_va = struct.unpack("<QQQQQ", item)
        if not (ids_va and ids_va % 4 == 0 and ids_va < ids_end <= ids_cap):
            return None
        if (ids_end - ids_va) % 4 or (ids_cap - ids_va) % 4:
            return None
        count = (ids_end - ids_va) // 4
        if count < 8 or abs(count - expected_step) > step_tolerance:
            return None
        if (ids_cap - ids_va) // 4 > max(8192, expected_step * 4):
            return None
        data = reader.read(ids_va, count * 4, zero_fill=False)
        if data is None or len(data) != count * 4:
            return None
        ids = struct.unpack("<%dI" % count, data)
        if (bos_id >= 0 and ids[0] != bos_id) or any(t >= vocab_size for t in ids):
            return None
        if bool(pending_va) != bool(control_va):
            return None
        pending_id = None
        if pending_va:
            pending_raw = reader.read(pending_va, 4, zero_fill=False)
            if pending_raw is None or len(pending_raw) != 4:
                return None
            pending_id = struct.unpack("<I", pending_raw)[0]
            if pending_id >= vocab_size:
                return None
        items.append({"tokens_struct_va": _hex(item_va),
                      "token_ids_va": _hex(ids_va), "token_ids_end_va": _hex(ids_end),
                      "token_ids_capacity_va": _hex(ids_cap), "count": count,
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "physical_segments": _segments(reader, ids_va, len(data)),
                      "pending_token_object_va": _hex(pending_va),
                      "pending_token_id_address_va": _hex(pending_va) if pending_va else None,
                      "pending_token_id_address_pa": _hex(reader.v2p(pending_va)) if pending_va else None,
                      "pending_token_id_sha256": hashlib.sha256(pending_raw).hexdigest()
                      if pending_va else None})
    return {"processed_tokens_candidate_va": _hex(outer_va),
            "processed_tokens_candidate_pa": _hex(reader.v2p(outer_va)),
            "tokens_array_va": _hex(begin), "tokens_array_pa": _hex(reader.v2p(begin)),
            "batch_size": batch, "branches": items,
            "evidence": "nested libc++ vectors, K/V step, BOS, vocabulary bounds",
            "status": "candidate_not_symbol_confirmed"}


def scan_token_vectors(reader, regions: Iterable[tuple[int, int]], expected_step: int,
                       vocab_size: int = 262144, bos_id: int = 2,
                       budget_bytes: int = 1024 << 20, max_hits: int = 32,
                       step_tolerance: int = 1):
    
    hits = []
    scanned = 0
    examined = 0
    for start, end in regions:
        va = (start + 7) & ~7
        while va < end and scanned < budget_bytes:
            n = min(4096 - (va & 4095), end - va, budget_bytes - scanned)
            if n < 24:
                break
            page = reader.read(va, n, zero_fill=False)
            examined += n
            if page is not None:
                scanned += n
                for off in range(0, len(page) - 23, 8):
                    begin, stop, cap = struct.unpack_from("<QQQ", page, off)
                    if begin and begin % 8 == 0 and 40 <= stop - begin <= MAX_TOKEN_BATCH * 40 \
                            and stop <= cap and (stop - begin) % 40 == 0:
                        cand = parse_token_vector(reader, va + off, expected_step, vocab_size,
                                                  bos_id, step_tolerance)
                        if cand is not None:
                            hits.append(cand)
                            if len(hits) >= max_hits:
                                return {"candidates": hits, "bytes_read": scanned,
                                        "bytes_examined": examined, "truncated": True}
                                                                                      
                for off in range(max(0, ((len(page) - 23 + 7) // 8) * 8), len(page), 8):
                    outer = va + off
                    if outer + 24 > end:
                        continue
                    cand = parse_token_vector(reader, outer, expected_step, vocab_size,
                                              bos_id, step_tolerance)
                    if cand is not None:
                        hits.append(cand)
                        if len(hits) >= max_hits:
                            return {"candidates": hits, "bytes_read": scanned,
                                    "bytes_examined": examined, "truncated": True}
            va += n
    return {"candidates": hits, "bytes_read": scanned,
            "bytes_examined": examined, "truncated": scanned >= budget_bytes}


def write_manifest(path: str, manifest: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temp, path)
