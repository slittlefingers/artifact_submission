
















from __future__ import annotations
import logging
import os
import re
from typing import Dict, Optional, Tuple
from .manifest import _pb_varint, _pb_is_msg

vollog = logging.getLogger(__name__)
_HEX40 = re.compile(rb"^[0-9a-f]{40}$")
_TS = re.compile(r"[0-9]+(?:_[0-9]+)*$")                                                          


def _walk(b: bytes, sink):
    


    i, n = 0, len(b)
    fields: Dict[int, list] = {}
    while i < n:
        try:
            tag, i = _pb_varint(b, i)
        except Exception:
            break
        wt = tag & 7
        fn = tag >> 3
        if fn == 0:
            break
        if wt == 0:
            try:
                v, i = _pb_varint(b, i)
            except Exception:
                break
            fields.setdefault(fn, []).append(("v", v))
        elif wt == 2:
            try:
                ln, i = _pb_varint(b, i)
            except Exception:
                break
            seg = b[i:i + ln]
            i += ln
            if i > n:
                break
            fields.setdefault(fn, []).append(("b", seg))
            if len(seg) > 1 and _pb_is_msg(seg):
                _walk(seg, sink)
        elif wt == 5:
            i += 4
        elif wt == 1:
            i += 8
        else:
            break
    sink(fields)


class FileGroupStore:
    

    __slots__ = ("by_sha1",)

    def __init__(self, by_sha1: Dict[str, Tuple[Optional[str], Optional[str]]]):
        self.by_sha1 = by_sha1

    @classmethod
    def from_pb(cls, path: str) -> "FileGroupStore":
        data = open(path, "rb").read()
        out: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        def sink(fields):
            f2 = [v for t, v in fields.get(2, []) if t == "b" and _HEX40.match(v)]
            if not f2:
                return
            sha1 = f2[0].decode()
            name = next((v.decode("latin1") for t, v in fields.get(1, []) if t == "b"), None)
            lpath = next((v.decode("latin1") for t, v in fields.get(5, []) if t == "b"), None)
                                                                             
            if sha1 not in out or (lpath and not out[sha1][1]):
                out[sha1] = (name, lpath)

        _walk(data, sink)
        return cls(out)

    def resolve(self, sha1: str) -> Optional[Tuple[Optional[str], Optional[str]]]:
        return self.by_sha1.get(sha1)

    @staticmethod
    def family(logical_path: Optional[str]) -> Optional[str]:
        



        if not logical_path or "/" not in logical_path:
            return None
        segs = [s for s in re.split(r"[\\/]+", logical_path) if s and not _TS.fullmatch(s)]
        return segs[0] if segs else None

    def family_of(self, sha1: str) -> Optional[str]:
        r = self.resolve(sha1)
        if not r:
            return None
        return self.family(r[1]) or self.family(r[0])


def discover(models_dir: Optional[str], explicit: Optional[str] = None) -> Optional[str]:
    

    if explicit and os.path.isfile(explicit):
        return explicit
    if not models_dir:
        return None
    d = os.path.abspath(models_dir)
    for _ in range(5):
        cand = os.path.join(d, "filegroupstore.pb")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None
