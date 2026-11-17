
import mmap
import struct
import zlib
from pathlib import Path

TFLITE_MODEL = 3
SENTENCEPIECE_TOKENIZER = 4
HF_TOKENIZER_ZLIB = 6
EXTERNAL_WEIGHTS = 7


class FlatReader:
    def __init__(self, data):
        self.data = data

    def scalar_at(self, pos, fmt="<I"):
        return struct.unpack_from(fmt, self.data, pos)[0]

    def field(self, table, index):
        vt = table - self.scalar_at(table, "<i")
        entry = 4 + 2 * index
        offset = self.scalar_at(vt + entry, "<H") if entry < self.scalar_at(vt, "<H") else 0
        return table + offset if offset else 0

    def ref(self, pos):
        return pos + self.scalar_at(pos) if pos else 0

    def table(self, table, index):
        return self.ref(self.field(table, index))

    def scalar(self, table, index, fmt="<I"):
        pos = self.field(table, index)
        return self.scalar_at(pos, fmt) if pos else 0

    def string(self, table, index):
        pos = self.table(table, index)
        return bytes(self.data[pos + 4:pos + 4 + self.scalar_at(pos)]).decode() if pos else ""

    def vector(self, table, index, fmt="<i", tables=False):
        pos = self.table(table, index)
        if not pos:
            return []
        count = self.scalar_at(pos)
        if count >= 1000000:
            raise ValueError("Implausible flatbuffer vector length")
        width = struct.calcsize(fmt)
        return [self.ref(pos + 4 + 4 * i) if tables else self.scalar_at(pos + 4 + width * i, fmt)
                for i in range(count)]


class Bundle:
    

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.file = self.path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.data[:8] != b"LITERTLM":
            self.__exit__()
            raise ValueError(f"{self.path} is not a LiteRT-LM bundle")
        self.r = FlatReader(self.data)
        return self

    def __exit__(self, *exc):
        self.data.close()
        self.file.close()

    def sections(self):
        
        r = self.r
        root = 32 + r.scalar_at(32)
        for section in r.vector(r.table(root, 1), 0, tables=True):
            kind = r.scalar(section, 3, "<B")
            info = {r.string(x, 0): r.string(r.table(x, 2), 0)
                    for x in r.vector(section, 0, tables=True) if r.scalar(x, 1, "<B") == 9}
            begin = r.scalar(section, 1, "<Q")
            yield kind, begin, r.scalar(section, 2, "<Q") - begin, info

    def model_sections(self):
        
        r = self.r
        result = {}
        for kind, begin, length, info in self.sections():
            key = info.get("model_type", "unknown")
            if kind == EXTERNAL_WEIGHTS:
                result.setdefault(key, {})["external_weights"] = [begin, length]
            if kind != TFLITE_MODEL:
                continue
            if self.data[begin + 4:begin + 8] != b"TFL3":
                raise ValueError(f"Section {key} is not a TFLite flatbuffer")
            model = begin + r.scalar_at(begin)
            subgraphs = r.vector(model, 2, tables=True)
            signatures = {}
            for index, sig in enumerate(r.vector(model, 7, tables=True)):
                tensors = r.vector(subgraphs[r.scalar(sig, 4)], 0, tables=True)
                item = {"index": index}
                for slot, label in ((0, "inputs"), (1, "outputs")):
                    entries = []
                    for tensor_map in r.vector(sig, slot, tables=True):
                        t = tensors[r.scalar(tensor_map, 1)]
                        q = r.table(t, 4)
                        entries.append(dict(name=r.string(tensor_map, 0), shape=r.vector(t, 0),
                                            shape_signature=r.vector(t, 7), type=r.scalar(t, 1, "<B"),
                                            scale=r.vector(q, 2, "<f") if q else [],
                                            zero_point=r.vector(q, 3, "<q") if q else []))
                    item[label] = entries
                signatures[r.string(sig, 2)] = item
            result.setdefault(key, {}).update(offset=begin, length=length, signatures=signatures)
        return result

    def read(self, offset, length):
        return bytes(self.data[offset:offset + length])

    def tokenizers(self):
        
        found = {}
        for kind, begin, length, _ in self.sections():
            if kind == HF_TOKENIZER_ZLIB:
                size = struct.unpack("<Q", self.data[begin:begin + 8])[0]
                raw = zlib.decompress(self.data[begin + 8:begin + length])
                if len(raw) != size:
                    raise ValueError("Bundled HF tokenizer size mismatch")
                found["hf_json"] = raw
            elif kind == SENTENCEPIECE_TOKENIZER:
                found["sentencepiece"] = self.read(begin, length)
        return found
