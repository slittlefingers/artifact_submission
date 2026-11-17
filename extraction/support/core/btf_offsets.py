#!/usr/bin/env python3




import struct, sys

_TRAIL = {1: lambda v: 4, 3: lambda v: 12, 4: lambda v: v*12, 5: lambda v: v*12,
          6: lambda v: v*8, 13: lambda v: v*8, 14: lambda v: 4, 15: lambda v: v*12,
          17: lambda v: 4, 19: lambda v: v*12}
K_STRUCT, K_UNION = 4, 5

class BTF:
    def __init__(self, path, base=None):
        d = open(path, 'rb').read()
        magic, ver, flags, hdr_len, type_off, type_len, str_off, str_len = struct.unpack_from('<HBBIIIII', d, 0)
        assert magic == 0xEB9F, f"bad BTF magic {magic:#x}"
        self.d = d; self.sbase = hdr_len + str_off; self.str_len = str_len
        self.base = base                                                                         
        self.base_strlen = base.str_len if base else 0                                                 
        self.start_id = (base.ntypes if base else 0) + 1
        self.types = []                                                                                                 
        self.by_name = {}
        off = hdr_len + type_off; end = off + type_len
        while off + 12 <= end:
            name_off, info, sz = struct.unpack_from('<III', d, off)
            vlen = info & 0xffff; kind = (info >> 24) & 0x1f; kflag = (info >> 31) & 1; off += 12
            if kind in (K_STRUCT, K_UNION):
                name = self._str(name_off); mem = []
                for _ in range(vlen):
                    mno, mt, mo = struct.unpack_from('<III', d, off); off += 12
                    bo = (mo & 0xffffff) if kflag else mo; bf = ((mo >> 24) & 0xff) if kflag else 0
                    mem.append((self._str(mno), mt, bo, bf))
                self.types.append((kind, name, sz, mem))
                if name: self.by_name.setdefault(name, len(self.types) - 1 + self.start_id)
            else:
                self.types.append((kind, self._str(name_off), sz, None))
                off += _TRAIL.get(kind, lambda v: 0)(vlen)
        self.ntypes = (base.ntypes if base else 0) + len(self.types)

    def _str(self, o):
        if self.base and o < self.base_strlen:                                                           
            return self.base._str(o)
        o -= self.base_strlen
        e = self.d.index(b'\x00', self.sbase + o); return self.d[self.sbase + o:e].decode('latin1')

    def _type(self, tid):                                                                                
        if self.base and tid < self.start_id: return self.base._type(tid)
        return self.types[tid - self.start_id]

    def _resolve_struct(self, tid):                                                                 
        for _ in range(8):
            k, nm, sz, mem = self._type(tid)
            if k in (K_STRUCT, K_UNION): return (k, nm, sz, mem)
            if k in (8, 9, 10, 11, 18) and sz: tid = sz                                                                               
            else: return None
        return None

    def _walk(self, mem, base, out):
        for (mname, mtype, bo, bf) in mem:
            boff = base + bo // 8
            if mname:
                out.setdefault(mname, boff)
            else:                                                                     
                r = self._resolve_struct(mtype)
                if r: self._walk(r[3], boff, out)

    def offsets(self, name):
        tid = self.by_name.get(name)
        if tid is None: return None
        out = {}; self._walk(self._type(tid)[3], 0, out); return out

    def size(self, name):
        tid = self.by_name.get(name)
        return self._type(tid)[2] if tid is not None else None

    def bit(self, name, member):                                                                   
        tid = self.by_name.get(name)
        if tid is None: return None
        def rec(mem, base):
            for (m, mt, bo, bf) in mem:
                if m == member: return (base*8 + bo, bf)
                if not m:
                    r = self._resolve_struct(mt)
                    if r:
                        got = rec(r[3], base + bo // 8)
                        if got: return got
            return None
        return rec(self._type(tid)[3], 0)

if __name__ == '__main__':
    vml = BTF(sys.argv[1] if len(sys.argv) > 1 else 'vmlinux.btf')
    mod = BTF(sys.argv[2], base=vml) if len(sys.argv) > 2 else None
    def show(b, sname, mems):
        o = b.offsets(sname)
        if o is None: print(f"{sname:14s}: <not present>"); return
        print(f"{sname:14s}: " + "  ".join((f"{m}={o[m]:#x}" if m in o else f"{m}=?") for m in mems))
    show(vml, 'task_struct', ['comm','pid','real_parent','active_mm','mm','files','tasks'])
    show(vml, 'mm_struct',   ['pgd'])
    show(vml, 'file',        ['f_op','private_data'])
    show(vml, 'dma_buf',     ['size','priv'])
    show(vml, 'files_struct',['fdt']); show(vml, 'fdtable', ['max_fds','fd'])
    show(vml, 'xa_node',     ['slots'])
    show(vml, 'page',        ['private','index'])
    b2 = mod or vml
    show(b2, 'size_class',   ['size']); show(b2, 'zs_pool', ['size_class']); show(b2, 'zram', ['table','mem_pool'])
    if mod:
        bf = mod.bit('zspage', 'class'); print(f"zspage.class bitfield: {bf}")
