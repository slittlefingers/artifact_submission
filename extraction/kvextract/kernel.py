






from __future__ import annotations
import hashlib
import struct
from collections import Counter
from volatility3.framework import exceptions

DEF_PAGE_OFFSET = 0xFFFFFF8000000000
DEF_PAGE_END = 0xFFFFFFC000000000
DEF_PHYS_OFFSET = 0x80000000
DEF_VMEMMAP = 0xfffffffdfe000000                                                                      
DEF_SP = 64                                                 


class KernelReader:
    def __init__(self, ctx, phys_layer_name, page_offset=DEF_PAGE_OFFSET, page_end=DEF_PAGE_END,
                 phys_offset=DEF_PHYS_OFFSET, vmemmap=DEF_VMEMMAP, struct_page=DEF_SP):
        self.pl = ctx.layers[phys_layer_name]
        self.PAGE_OFFSET = page_offset; self.PAGE_END = page_end; self.PHYS_OFFSET = phys_offset
        self.VMEMMAP_C = vmemmap; self.SP = struct_page

    def _rdpa(self, pa, n):
        try:
            return self.pl.read(pa, n, pad=False)
        except exceptions.InvalidAddressException:
            return None

    def rv(self, va, n):
        if not (self.PAGE_OFFSET <= va < self.PAGE_END):
            return None
        return self._rdpa(va - self.PAGE_OFFSET + self.PHYS_OFFSET, n)

    def u64(self, va):
        b = self.rv(va, 8); return struct.unpack("<Q", b)[0] if b else None

    def u32(self, va):
        b = self.rv(va, 4); return struct.unpack("<I", b)[0] if b else None

    def page_pa(self, page_va, off):
        return (((page_va - self.VMEMMAP_C) // self.SP) << 12) + off

    def read_page(self, page_va, off, n):
        return self._rdpa(self.page_pa(page_va, off), n)


class DmaBufReassembler:
    

    def __init__(self, kr: KernelReader, off: dict, record_addresses: bool = False):
        self.kr = kr
        self.off = off
        self.last_missing = 0
        self.record_addresses = record_addresses
        self.reset_trace()

    def reset_trace(self):
        self.fd_table_trace = {}
        self.fd_traces = {}
        self.priv_traces = {}

                        
    def fd_table(self, files):
        off = self.off
        if self.record_addresses:
            self.fd_table_trace = {"files_struct_va": hex(files) if files else None}
        if not files:
            return None, 0
        fdt = self.kr.u64(files + off['F_FDT'])
        if self.record_addresses:
            self.fd_table_trace["fdtable_va"] = hex(fdt) if fdt else None
        if not fdt:
            return None, 0
        maxfds = self.kr.u32(fdt + off['FDT_MAXFDS']) or 0
        fdarr = self.kr.u64(fdt + off['FDT_FD'])
        if self.record_addresses:
            self.fd_table_trace.update({"fd_array_va": hex(fdarr) if fdarr else None,
                                        "max_fds": min(maxfds, 65536)})
        return fdarr, min(maxfds, 65536)

    def derive_fops(self, fdarr, maxfds, sizes=None, lo=256 * 1024, hi=512 * 1024 * 1024):
        

        off = self.kr and self.off
        cnt = Counter()
        for q in range(maxfds):
            fp = self.kr.u64(fdarr + q * 8)
            if not fp:
                continue
            op = self.kr.u64(fp + off['FILE_FOP'])
            if not op or op < 0xFFFFFF8000000000:
                continue
            db = self.kr.u64(fp + off['FILE_PRIV'])
            if not db:
                continue
            s = self.kr.u64(db + off['DB_SIZE'])
            if s is None:
                continue
            ok = (s in sizes) if sizes else (lo <= s <= hi and s % 4096 == 0)
            if ok:
                cnt[op] += 1
        return cnt.most_common(1)[0][0] if cnt else 0

    def enumerate(self, fdarr, maxfds, fops):
        
        off = self.off; seen = set(); out = []
        for q in range(maxfds):
            fp = self.kr.u64(fdarr + q * 8)
            if not fp or self.kr.u64(fp + off['FILE_FOP']) != fops:
                continue
            db = self.kr.u64(fp + off['FILE_PRIV'])
            if not db or db in seen:
                continue
            sz = self.kr.u64(db + off['DB_SIZE'])
            if not sz:
                continue
            seen.add(db)
            out.append((q, sz, db))
            if self.record_addresses:
                self.fd_traces[q] = {"fd": q, "fd_entry_va": hex(fdarr + q * 8),
                                     "file_va": hex(fp), "file_fops_va": hex(fops),
                                     "dma_buf_va": hex(db), "size": sz}
        return out

    def dmabuf_of_file(self, fp):
        




        off = self.off
        if not fp or off.get('DB_FILE') is None:
            return None
        db = self.kr.u64(fp + off['FILE_PRIV'])
        if not db or self.kr.u64(db + off['DB_FILE']) != fp:
            return None
        return db

    def priv(self, db):
        value = self.kr.u64(db + self.off['DB_PRIV'])
        if self.record_addresses and value:
            self.priv_traces[value] = {"dma_buf_va": hex(db), "priv_va": hex(value)}
            for item in self.fd_traces.values():
                if item["dma_buf_va"] == hex(db):
                    item["priv_va"] = hex(value)
        return value

                             
    def reassemble(self, priv, size, tol=False, exact=False):
        




        if exact:
            tol = False
        off = self.off
        SC_SZ, SG_LEN, SG_OFF = off['SC_SZ'], off['SG_LEN'], off['SG_OFF']
        self.last_missing = 0
        best = None
        for o in range(0, 1024, 8):
            blk = self.kr.rv(priv + o, 16)
            if not blk:
                continue
            sgl = struct.unpack_from("<Q", blk, 0)[0]; nents = struct.unpack_from("<I", blk, 8)[0]
            orig = struct.unpack_from("<I", blk, 12)[0]
            if sgl < 0xFFFFFF8000000000 or not orig or orig >= 100000:
                continue
            if nents != orig and nents != 0:
                continue
            fl = self.kr.u32(sgl + SG_LEN)
            if not fl or fl > size:
                continue
            tot = self._sg_total(sgl, orig, SC_SZ, SG_LEN)
            if tot is None:
                continue
            if tot == size:
                best = (sgl, orig, tot, o); break
            if not exact and tot <= size and (best is None or tot > best[2]):
                best = (sgl, orig, tot, o)
        if not best:
            if self.record_addresses:
                self.priv_traces.setdefault(priv, {"priv_va": hex(priv)})["sg_status"] = "not_found"
            return None
        sgl, orig = best[0], best[1]
        trace = None
        if self.record_addresses:
            trace = self.priv_traces.setdefault(priv, {"priv_va": hex(priv)})
            trace.update({"sg_table_field_va": hex(priv + best[3]),
                          "sgl_va": hex(sgl), "orig_nents": orig,
                          "sg_total_bytes": best[2], "segments": []})
        buf = bytearray(); cur = sgl; rem = orig; i = 0; missing = 0
        while rem > 0 and i < 100000:
            ent = self.kr.rv(cur, SC_SZ)
            if ent is None:
                if tol:
                    break
                return None
            pl = struct.unpack_from("<Q", ent, 0)[0]
            if pl & 0x01:
                cur = (pl & ~0x03) & ((1 << 64) - 1); continue
            page = (pl & ~0x03) & ((1 << 64) - 1)
            o = struct.unpack_from("<I", ent, SG_OFF)[0]; ln = struct.unpack_from("<I", ent, SG_LEN)[0]
            pa = self.kr.page_pa(page, o)
            d = self.kr.read_page(page, o, ln)
            if d is None or len(d) < ln:
                if tol:
                    if trace is not None:
                        trace["segments"].append({"scatterlist_va": hex(cur),
                                                  "struct_page_va": hex(page),
                                                  "physical_address": hex(pa),
                                                  "kv_offset": len(buf), "length": ln,
                                                  "sha256": None, "status": "unreadable"})
                    buf += b"\x00" * ln; missing += ln; rem -= 1
                    cur = (cur + SC_SZ) & ((1 << 64) - 1); i += 1
                    if pl & 0x02:
                        break
                    continue
                return None
            if trace is not None:
                pos = 0
                while pos < ln:
                    chunk_len = min(ln - pos, 4096 - ((pa + pos) & 4095))
                    trace["segments"].append({"scatterlist_va": hex(cur),
                                              "struct_page_va": hex(page),
                                              "physical_address": hex(pa + pos),
                                              "kv_offset": len(buf) + pos,
                                              "length": chunk_len,
                                              "sha256": hashlib.sha256(d[pos:pos + chunk_len]).hexdigest(),
                                              "status": "readable"})
                    pos += chunk_len
            buf += d; rem -= 1; cur = (cur + SC_SZ) & ((1 << 64) - 1); i += 1
            if pl & 0x02:
                break
        if exact and len(buf) != size:
            return None
        if len(buf) < size:
            buf += b"\x00" * (size - len(buf))
        self.last_missing = missing
        if trace is not None:
            trace["missing_bytes"] = missing
            trace["sha256"] = hashlib.sha256(buf[:size]).hexdigest()
        return bytes(buf[:size])

    def _sg_total(self, sgl, orig, SC_SZ, SG_LEN):
        cur = sgl; rem = orig; tot = 0; i = 0
        while rem > 0 and i < 100000:
            ent = self.kr.rv(cur, SC_SZ)
            if ent is None:
                return None
            pl = struct.unpack_from("<Q", ent, 0)[0]
            if pl & 0x01:
                cur = (pl & ~0x03) & ((1 << 64) - 1); continue
            tot += struct.unpack_from("<I", ent, SG_LEN)[0]; rem -= 1
            cur = (cur + SC_SZ) & ((1 << 64) - 1); i += 1
            if pl & 0x02:
                break
        return tot
