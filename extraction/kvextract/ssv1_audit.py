
from __future__ import annotations

import json
import struct


def audit(reader, candidates_path: str):
    for line in open(candidates_path, encoding="utf-8"):
        candidate = json.loads(line)
        n = candidate["nact"]
        begin = candidate["begin"]
        data = reader.read(begin, n * 0x30, zero_fill=False)
        if data is None:
            yield candidate, -1, []
            continue
        rows = []
        for i in range(n):
            at = i * 0x30
            k, v = struct.unpack_from("<QQ", data, at)
            bpt = struct.unpack_from("<I", data, at + 0x10)[0]
            filled = struct.unpack_from("<Q", data, at + 0x28)[0]
            rows.append((i, k, v, bpt, filled))
        plausible = sum(k and v and bpt in (128, 256, 512, 1024) and filled < 100000
                        for _, k, v, bpt, filled in rows)
        yield candidate, plausible, rows
