

import struct
from types import SimpleNamespace

from kvextract import constants as C
from kvextract.kernel import DmaBufReassembler
from kvextract.locators import litert_dmabuf as ld
from kvextract.locators.litert_namechain import LitertNameChainLocator

OFF = dict(F_FDT=0, FDT_MAXFDS=0, FDT_FD=8, FILE_FOP=0, FILE_PRIV=8,
           DB_SIZE=0, DB_FILE=8, DB_PRIV=16, SC_SZ=32, SG_LEN=16, SG_OFF=20)
SGL = 0xFFFFFF8000007000
PRIV = 0x6000


class FakeKernelReader:
    def __init__(self, bad_pages=()):
        self.mem = {}
        self.payload = bytes(range(256)) * 32                                       
        self.bad_pages = set(bad_pages)

    def put(self, va, data):
        self.mem.update((va + i, b) for i, b in enumerate(data))

    def rv(self, va, n):
        try:
            return bytes(self.mem[va + i] for i in range(n))
        except KeyError:
            return None

    def u64(self, va):
        raw = self.rv(va, 8)
        return struct.unpack('<Q', raw)[0] if raw else None

    def u32(self, va):
        raw = self.rv(va, 4)
        return struct.unpack('<I', raw)[0] if raw else None

    def page_pa(self, page, offset):
        return 0x90000000 + offset

    def read_page(self, page, offset, n):
        if page in self.bad_pages:
            return None
        return self.payload[offset:offset + n]


def _sg(kr, entries):
    
    kr.put(PRIV + 0x20, struct.pack('<QII', SGL, len(entries), len(entries)))
    for i, (page, ln, off) in enumerate(entries):
        link = page | (0x2 if i == len(entries) - 1 else 0)
        kr.put(SGL + i * 32, struct.pack('<Q8xII8x', link, ln, off))


                                                                                                    

def test_exact_reassembles_multi_segment_table_honouring_offsets():
    kr = FakeKernelReader()
    _sg(kr, [(0x8000, 2048, 0), (0x9000, 2048, 100)])
    raw = DmaBufReassembler(kr, OFF).reassemble(PRIV, 4096, exact=True)
    assert raw == kr.payload[0:2048] + kr.payload[100:2148]


def test_exact_rejects_partial_table_that_default_mode_pads():
    kr = FakeKernelReader()
    _sg(kr, [(0x8000, 4096, 0)])                                                              
    reasm = DmaBufReassembler(kr, OFF)
    padded = reasm.reassemble(PRIV, 8192)                                                   
    assert len(padded) == 8192 and padded[4096:] == b"\x00" * 4096
    assert reasm.reassemble(PRIV, 8192, exact=True) is None


def test_exact_rejects_unreadable_page_that_tolerant_mode_zero_fills():
    kr = FakeKernelReader(bad_pages={0x8000})
    _sg(kr, [(0x8000, 4096, 0)])
    reasm = DmaBufReassembler(kr, OFF)
    assert reasm.reassemble(PRIV, 4096, tol=True) == b"\x00" * 4096
    assert reasm.reassemble(PRIV, 4096, exact=True) is None
    assert reasm.reassemble(PRIV, 4096, tol=True, exact=True) is None                         


                                                                                                    

def test_dmabuf_of_file_requires_back_pointer():
    kr = FakeKernelReader()
    fp, db = 0xFFFFFF8000020000, 0xFFFFFF8000030000
    kr.put(fp + OFF['FILE_PRIV'], struct.pack('<Q', db))
    kr.put(db + OFF['DB_FILE'], struct.pack('<Q', fp))
    assert DmaBufReassembler(kr, OFF).dmabuf_of_file(fp) == db
    kr.put(db + OFF['DB_FILE'], struct.pack('<Q', fp + 0x100))                                   
    assert DmaBufReassembler(kr, OFF).dmabuf_of_file(fp) is None
    no_field = {k: v for k, v in OFF.items() if k != 'DB_FILE'}
    assert DmaBufReassembler(kr, no_field).dmabuf_of_file(fp) is None


                                                                                                    

class UserMem:
    def __init__(self):
        self.mem = {}

    def put(self, va, data):
        self.mem.update((va + i, b) for i, b in enumerate(data))

    def read(self, va, n, zero_fill=True):
        try:
            return bytes(self.mem[va + i] for i in range(n))
        except KeyError:
            return None


def _tensorbuffer(um, tb, fd, version=12):
    gb, nh = tb + 0x1000, tb + 0x2000
    obj = bytearray(C.TB_LEN); struct.pack_into('<Q', obj, C.TB_AHWB, gb); um.put(tb, obj)
    g = bytearray(C.GB_LEN); struct.pack_into('<Q', g, C.GB_NH, nh); um.put(gb, g)
    um.put(nh, struct.pack('<iiii', version, 1, 0, fd) + b"\x00" * (C.NH_LEN - 16))


class FakeReasm:
    
    FDARR = 0x1000

    def __init__(self, fd_file, file_db, db_size):
        q = {self.FDARR + fd * 8: fp for fd, fp in fd_file.items()}
        q.update({db + OFF['DB_SIZE']: sz for db, sz in db_size.items()})
        self.kr = SimpleNamespace(u64=q.get)
        self.file_db = file_db
        self.calls = []

    def dmabuf_of_file(self, fp):
        return self.file_db.get(fp)

    def priv(self, db):
        return db + 1

    def reassemble(self, priv, size, exact=False):
        self.calls.append(exact)
        return struct.pack('<Q', priv - 1) * (size // 8)


def test_resolve_walks_every_occurrence_and_keeps_last_valid():
    um = UserMem()
    _tensorbuffer(um, 0x5000000000, fd=3, version=0)                                      
    _tensorbuffer(um, 0x5000010000, fd=4)                                                        
    _tensorbuffer(um, 0x5000020000, fd=5)                                             
    _tensorbuffer(um, 0x5000030000, fd=6)                                                         
    _tensorbuffer(um, 0x5000040000, fd=7)                                                   
    _tensorbuffer(um, 0x5000050000, fd=8)                                            
    fd_file = {fd: 0xF000 + fd for fd in range(3, 9)}
    file_db = {0xF004: 0xD400, 0xF005: 0xD500, 0xF006: 0xD600, 0xF007: 0xD700}
    db_size = {0xD400: 64, 0xD500: 64, 0xD600: 64, 0xD700: 48}
    reasm = FakeReasm(fd_file, file_db, db_size)
    cands = [("kv_cache_k_3", 0x5000000000), ("kv_cache_k_3", 0x5000010000),
             ("kv_cache_v_3", 0x5000020000), ("kv_cache_v_3", 0x5000030000),
             ("kv_cache_c_1", 0x5000040000), ("kv_cache_r_1", 0x5000050000)]

    chosen, n = LitertNameChainLocator()._resolve(cands, um, reasm, OFF, FakeReasm.FDARR, 64, {64})

    assert set(chosen) == {"kv_cache_k_3", "kv_cache_v_3"}
    assert chosen["kv_cache_k_3"][:2] == (0xD400, 4)
    assert chosen["kv_cache_v_3"][:2] == (0xD600, 6)                                       
    assert n == dict(occ=6, nh=5, file=4, size=3, sg=3, conflicts=1)
    assert reasm.calls and all(reasm.calls)                                                


def test_candidates_keep_every_occurrence_of_a_name(monkeypatch):
    base, name = 0x6000000000, b"kv_cache_k_3"
    region = bytearray(0x2000)
    tbs = []
    for key_off, tb_off in ((0x100, 0x800), (0x200, 0x900)):
        region[key_off] = len(name) << 1                                           
        region[key_off + 1:key_off + 1 + len(name)] = name
        struct.pack_into('<Q', region, key_off + C.SSO_TB, base + tb_off)
        struct.pack_into('<Q', region, tb_off + C.TB_SIZE, 0x10000)
        tbs.append(base + tb_off)
    um = UserMem(); um.put(base, region)
    vma = SimpleNamespace(vm_start=base, vm_end=base + len(region), vm_file=0,
                          get_protection=lambda: "rw-")
    monkeypatch.setattr(ld.proc.Maps, "list_vmas", lambda task: [vma])

    loc = LitertNameChainLocator()
    cands = loc._name_tb_candidates(um, None, SimpleNamespace(specs={}))

    assert cands == [("kv_cache_k_3", tbs[0]), ("kv_cache_k_3", tbs[1])]
    assert loc._observed_sizes == {0x10000}
