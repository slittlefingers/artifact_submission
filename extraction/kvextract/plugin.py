



from __future__ import annotations
import glob
import hashlib
import json
import logging
import os
import sys
import urllib.parse
from collections import Counter
from volatility3.framework import renderers, interfaces, exceptions
from volatility3.framework.configuration import requirements
from volatility3.framework.interfaces import plugins
from volatility3.plugins.linux import psscan, proc

from . import env, offsets as offmod, modelmatch, phase1, filegroup, physscan, ssv1_audit, provenance
from .reader import ProcReader
from .kernel import KernelReader, DmaBufReassembler, DEF_VMEMMAP
from .descriptor import Descriptor
from .kvex import write_kvex, k_step
from .locators import LOCATORS, vma_name

vollog = logging.getLogger(__name__)


class KvExtract(plugins.PluginInterface):
    

    _required_framework_version = (2, 0, 0)
    _version = (1, 0, 0)

    @classmethod
    def get_requirements(cls):
        return [
            requirements.ModuleRequirement(name="kernel", description="Linux kernel",
                                           architectures=["AArch64", "Intel32", "Intel64"]),
            requirements.PluginRequirement(name="psscan", plugin=psscan.PsScan, version=(1, 0, 0)),
            requirements.PluginRequirement(name="proc", plugin=proc.Maps, version=(1, 0, 0)),
            requirements.ListRequirement(name="pid", description="Target PID(s)",
                                         element_type=int, optional=True),
            requirements.StringRequirement(name="backend", optional=True,
                                           description="Force a locator: %s" % "|".join(l.key for l in LOCATORS)),
            requirements.StringRequirement(name="recover-dir", optional=True,
                                           description="deadbox 'recover' dir holding core/kbtf.py + *.btf (kernel offsets, zram)"),
            requirements.StringRequirement(name="descriptor", optional=True,
                                           description="Prebuilt KV descriptor JSON (L3)"),
            requirements.StringRequirement(name="out", optional=True,
                                           description="Write recovered KV as a KVEX blob"),
            requirements.StringRequirement(name="address-manifest", optional=True,
                                           description="Write KV fd/dma-buf/sg_table/physical-page addresses and "
                                                       "version-gated LiteRT-LM token-object provenance as JSON"),
            requirements.StringRequirement(name="token-library", optional=True,
                                           description="Local copy of the exact liblitertlm_jni.so mapped by the target; "
                                                       "SHA-256 gates executor field offsets"),
            requirements.StringRequirement(name="token-page-map", optional=True,
                                           description="Optional same-session VA->PA page map for token pages "
                                                       "whose user page tables were filled by acquisition; "
                                                       "each physical page must SHA-256 match the LiME image"),
            requirements.BooleanRequirement(name="token-heuristic-scan", default=False, optional=True,
                                            description="Diagnostic only: scan anonymous VMAs for generic token vector shapes"),
            requirements.IntRequirement(name="token-vocab-size", default=262144, optional=True,
                                        description="LiteRT-LM token candidate validation: model vocabulary size"),
            requirements.IntRequirement(name="token-bos-id", default=2, optional=True,
                                        description="LiteRT-LM token candidate validation: BOS token ID (-1 disables BOS gate for diagnostics)"),
            requirements.IntRequirement(name="token-scan-mb", default=1024, optional=True,
                                        description="Max readable RW-anon MiB to inspect for token vectors"),
            requirements.IntRequirement(name="token-step-tolerance", default=1, optional=True,
                                        description="Diagnostic: allowed difference between token vector length and KV step"),
            requirements.StringRequirement(name="models-dir", optional=True,
                                           description="Phase-1: dir of on-disk .litertlm models to identify/match + pull quant from"),
            requirements.BooleanRequirement(name="dequant", default=False, optional=True,
                                            description="Phase-1: write a per-tensor scale/zero_point sidecar from the matched model"),
            requirements.IntRequirement(name="va-bits", default=39, optional=True,
                                        description="Kernel VA_BITS (PAGE_OFFSET derivation)"),
            requirements.BooleanRequirement(name="zram", default=False, optional=True,
                                            description="Decode zram-swapped pages (needs recover-dir + live EH oracle)"),
            requirements.BooleanRequirement(name="diag", default=False, optional=True,
                                            description="Emit per-task diagnostics"),
            requirements.BooleanRequirement(name="autolayers", default=False, optional=True,
                                            description="auto-discover the GemExecutor KVCache struct layout from "
                                            "memory (no per-generation offsets) and print per-layer host ptrs = layer order"),
            requirements.BooleanRequirement(name="vma-dump", default=False, optional=True,
                                            description="Print target VMAs for offline dma_buf and heap pointer analysis"),
            requirements.BooleanRequirement(name="triage", default=False, optional=True,
                                            description="Per-process backend/PID dispatch: which LLM runtime runs "
                                                        "(aicore/litertlm), detect scores + evidence (mapped libs/model)"),
            requirements.BooleanRequirement(name="phys-scan", default=False, optional=True,
                                            description="Process/fd-INDEPENDENT physical hard-scan for KV anchors "
                                                        "(kv_cache_* / CPU TensorBufferT header / merged_*_cache_*); "
                                                        "survives closed fd / torn-down model"),
            requirements.StringRequirement(name="ssv1-candidates", optional=True,
                                           description="JSONL physical SSV1 vector signatures to test in this PID"),
        ]

                       
    def _phys_layer(self, kernel):
        kl = self.context.layers[kernel.layer_name]
        return kl.config.get("memory_layer") or kl.dependencies[0]

    def _dump_path(self, kernel):
        seen = set(); stack = [self._phys_layer(kernel)]
        while stack:
            nm = stack.pop()
            if nm in seen or nm not in self.context.layers:
                continue
            seen.add(nm)
            lyr = self.context.layers[nm]
            loc = lyr.config.get("location")
            if loc:
                return urllib.parse.unquote(urllib.parse.urlparse(loc).path).lstrip("/")
            stack += list(getattr(lyr, "dependencies", []))
        return ""

    def _tasks(self, kernel):
        pids = self.config.get("pid", None)
        for task in psscan.PsScan.scan_tasks(self.context, self.config["kernel"], kernel.layer_name):
                                                                                          
                                                                                                     
                                      
            try:
                pid = int(task.pid)
            except exceptions.InvalidAddressException:
                pid = -1
            if pids and pid not in pids:
                continue
            yield pid, task

    def _pick(self, task):
        forced = self.config.get("backend")
        if forced:
            for l in LOCATORS:
                if l.key == forced:
                    return l
            raise ValueError(f"unknown --backend {forced}")
        scored = sorted(((l.detect(self.context, task), l) for l in LOCATORS),
                        key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored[0][0] > 0 else None

    def _find_mapped_paths(self, task):
        
        found = {}
        try:
            for v in proc.Maps.list_vmas(task):
                nm = vma_name(self.context, task, v) or ""
                if nm.endswith(".litertlm") or nm.endswith(".tflite") or "aicore-models/files/" in nm:
                    found[os.path.basename(nm.rstrip("/"))] = nm
        except Exception:
            pass
        return found

    def _resolve_mapped_identities(self, mapped):
        


        fgs = getattr(self, "_fgs", None)
        rows = []
        if fgs is None:
            return rows, set()
        fams = set()
        for bn, pth in mapped.items():
            if "aicore-models/files/" not in pth:
                continue
            r = fgs.resolve(bn)
            if not r:
                continue
            name, lpath = r
            fam = fgs.family(lpath) or fgs.family(name)
            if fam:
                fams.add(fam)
            rows.append(("loaded-file", bn[:16] + "...", fam or "?", (lpath or name or "")[-52:]))
        return rows, fams

    def _phase1(self, task, head_dims):
        
        mapped = self._find_mapped_paths(task)
        rows = [("model", bn, "mapped", pth[-48:]) for bn, pth in mapped.items()]
        id_rows, id_fams = self._resolve_mapped_identities(mapped)
        rows += id_rows
        if id_fams:
            rows.append(("loaded-models", "== from VMA file ids ==", ",".join(sorted(id_fams)),
                         "resolved via filegroupstore.pb"))
        models_dir = self.config.get("models-dir")
        if not models_dir:
            return None, rows
        cat = modelmatch.build_catalog(sorted(glob.glob(os.path.join(models_dir, "*.litertlm"))))
        for r in cat:
            bn = os.path.basename(r.path)
            r.mapped = bn in mapped or any(m.endswith(bn) for m in mapped)
                                                                                                         
        obs = modelmatch.observed_kv_geometry(head_dims)
        for r in cat:
            r.score = modelmatch._score(modelmatch.fingerprint(r.descriptor), obs) if r.descriptor else 0.0
        pid_maps_model = any(k.endswith((".litertlm", ".tflite")) or "aicore-models" in k for k in mapped)
        if not any(r.mapped for r in cat) and pid_maps_model and cat:
            top = max(cat, key=lambda r: r.score)
            if top.score > 0:
                top.mapped = True
                top.notes = (top.notes + " | " if top.notes else "") + "matched by geometry (mapped filename differs)"
        modelmatch.label_models(cat, head_dims)
        active = None
        for r in sorted(cat, key=lambda x: -x.score):
            rows.append(("model-match", os.path.basename(r.path), r.role or "-",
                         f"score={r.score:.2f} map={r.mapped}" + (f" | {r.notes}" if r.notes else "")))
            if r.role == modelmatch.ACTIVE and active is None:
                active = r.descriptor
        return active, rows

    def _setup_zram(self, kernel, recover_dir):
        path = self._dump_path(kernel)
        core = os.path.join(recover_dir, "core")
        if core not in sys.path:
            sys.path.insert(0, core)
        prev = sys.argv; sys.argv = ["x", path, "-"]
        try:
            import path_b_extract as P, kbtf, bootstrap, zdecode, eh_bridge
            BO = kbtf.load(os.path.join(recover_dir, "vmlinux.btf"),
                           os.path.join(recover_dir, "zsmalloc.btf"))
            for k in ('OFF_TASKS', 'OFF_COMM', 'OFF_PID', 'OFF_FILES', 'F_FDT', 'FDT_MAXFDS', 'FDT_FD',
                      'FILE_FOP', 'FILE_PRIV', 'DB_SIZE', 'DB_PRIV', 'SC_SZ', 'SG_LEN', 'SG_OFF'):
                if BO.get(k) is not None:
                    setattr(P, k, BO[k])
            KOFF, IT_VA, SWAPPER, _ = bootstrap.bootstrap(P, BO, locate_zram=False)
            P.KOFF = KOFF; P.INIT_TASK_VA = IT_VA
            ZRAM = eh_bridge.locate_zram_by_disksize(P)
            if ZRAM is None:
                vollog.warning("zram: struct zram not located"); return None
            O = eh_bridge.EHOracle(); O.setup()
            Z = zdecode.Zram(P, zram_pa=ZRAM, swapper_pa=SWAPPER, offsets=BO, oracle=O)
            vollog.warning(f"zram ready: zram={ZRAM:#x} [EH oracle]")
            return Z
        except Exception as e:
            vollog.warning(f"zram setup failed: {type(e).__name__}: {e}")
            return None
        finally:
            sys.argv = prev

                    
    def _generator(self):
        manifest_path = self.config.get("address-manifest")
        if manifest_path and len(self.config.get("pid") or []) != 1:
            raise ValueError("--address-manifest requires exactly one --pid")
        kernel = self.context.modules[self.config["kernel"]]
        phys = self._phys_layer(kernel)
        recover = self.config.get("recover-dir")
                                                                                                   
                                                                                                     
                                                                                                       
        off = offmod.offsets_from_isf(kernel)
        if not off:
            if not recover:
                raise ValueError("could not read kernel offsets from the ISF; pass --recover-dir with vmlinux.btf")
            off = offmod.load_kernel_offsets(recover)
        if self.config.get("zram") and not recover:
            raise ValueError("--zram needs --recover-dir (core/*.py + zsmalloc.btf)")
        sp = offmod.struct_page_size(off)
                                                              
        page_offset = env.page_offset(int(self.config.get("va-bits", 39)))
        phys_offset = 0x80000000
        try:
            phys_offset = env.LimeImage(self._dump_path(kernel)).phys_offset()
        except Exception:
            pass
        vollog.info(f"geometry: PHYS_OFFSET={phys_offset:#x} PAGE_OFFSET={page_offset:#x} "
                    f"sizeof(page)={sp} vmemmap={DEF_VMEMMAP:#x}")

        desc = Descriptor.from_json(self.config["descriptor"]) if self.config.get("descriptor") else None
                                                                                                  
        cands = phase1.discover_candidates(self.config.get("models-dir"))
        if cands:
            vollog.warning(f"phase1: {phase1.summarize(cands)}")
        litert_desc = next((c.descriptor for c in cands if c.kind == "litert"), None)
        self._fgs = None
        zobj = self._setup_zram(kernel, recover) if self.config.get("zram", False) else None
        decode_swap = zobj.decode_swap if zobj is not None else None

        kr = KernelReader(self.context, phys, page_offset=page_offset, phys_offset=phys_offset,
                          vmemmap=DEF_VMEMMAP, struct_page=sp)
        reasm = DmaBufReassembler(kr, off, record_addresses=bool(manifest_path))

        if self.config.get("phys-scan", False):
                                                                                                
            for t, b, s in physscan.run(self.context, kr):
                yield (0, (0, "phys-scan", "phys_scan", t, b, s))
            return

        for pid, task in self._tasks(kernel):
            try:
                comm = str(task.comm.cast("string", max_length=16, errors="replace"))
            except Exception:
                comm = "?"
            try:
                mm = task.mm
                pgd_va = int(mm.pgd) if mm else 0
            except exceptions.InvalidAddressException:
                pgd_va = 0
            if not pgd_va:
                yield (0, (pid, comm, "-", "(no mm/pgd)", "", "")); continue
            reader = ProcReader(self.context, phys, pgd_va, zram=decode_swap,
                                page_offset=page_offset, phys_offset=phys_offset)
            reader.zram_obj = zobj

            if self.config.get("vma-dump", False):
                count = 0
                try:
                    for v in proc.Maps.list_vmas(task):
                        try:
                            start, end = int(v.vm_start), int(v.vm_end)
                            protection = v.get_protection()
                            name = vma_name(self.context, task, v) or ""
                        except Exception:
                            continue
                        count += 1
                        yield (0, (pid, comm, "vma", f"{start:#x}-{end:#x}", protection, name))
                except Exception as error:
                    yield (0, (pid, comm, "vma", "walk interrupted", str(count), type(error).__name__))
                continue

            ssv1_path = self.config.get("ssv1-candidates")
            if ssv1_path:
                for cand, plausible, rows in ssv1_audit.audit(reader, ssv1_path):
                    info = [(i, bpt, filled) for i, _k, _v, bpt, filled in rows]
                    yield (0, (pid, comm, "ssv1_audit", hex(cand["file_offset"]),
                               str(plausible), str(info)[:160]))
                continue

            if self.config.get("autolayers", False):
                                                                                                  
                from . import kvcache_struct as kcs
                heap, dmabuf = [], []
                for v in proc.Maps.list_vmas(task):
                    try:
                        s = int(v.vm_start); e = int(v.vm_end)
                        prot = v.get_protection(); has_file = v.vm_file != 0
                    except exceptions.InvalidAddressException:
                        continue
                    if not (0 < e - s <= (256 << 20)):
                        continue
                    nm = vma_name(self.context, task, v) or ""
                    if "dmabuf" in nm:                                                              
                        dmabuf.append((s, e - s))
                    elif "rw" in prot and not has_file:                                                     
                        heap.append((s, e))
                mem = kcs.CachedVolMem(lambda a, n: reader.read(a, n, zero_fill=False), heap, dmabuf)
                lay = kcs.discover(mem, mem.regions)
                if lay is None:
                    yield (0, (pid, comm, "autolayers", "no layout discovered",
                               f"heap_vmas={len(heap)} dmabuf_vmas={len(dmabuf)}", ""))
                else:
                    roles = ["0x0"] + [hex(o) for o in lay.extra_wrap_offsets]
                    yield (0, (pid, comm, "autolayers", f"stride={hex(lay.stride)} wrap_tb={hex(lay.wrap_tb)}",
                               f"layers={lay.count}", f"role_offsets={roles}"))
                    for i, host, sz in lay.host_ptrs:
                        yield (0, (pid, comm, "autolayers", f"layer{i:02d}", hex(host), str(sz)))
                continue

            if self.config.get("diag", False):
                scores = {l.key: l.detect(self.context, task) for l in LOCATORS}
                yield (0, (pid, comm, "diag", "scores", hex(pgd_va), str(scores)))

            if self.config.get("triage", False):
                                                                                                   
                                                                                                    
                                                                                                    
                                                                                             
                scores = {}
                for l in LOCATORS:
                    try:
                        scores[l.key] = l.detect(self.context, task)
                    except Exception:
                        scores[l.key] = 0
                best = max(scores, key=scores.get); bestv = scores[best]
                ev = set()
                try:
                    for v in proc.Maps.list_vmas(task):
                        nm = vma_name(self.context, task, v) or ""
                        for key in ("libgoogle3", "liblitert", "libedgetpu", "xnnpack", "scudo",
                                    ".litertlm", "aicore", "dma_heap", "edgetpu"):
                            if key in nm:
                                ev.add(key)
                except Exception as e:
                    ev.add("vma-walk-err:%s" % type(e).__name__)
                backend = ("aicore" if "libgoogle3" in ev
                           else "litertlm" if (ev & {"liblitert", "libedgetpu", "xnnpack", ".litertlm"})
                           else (best if bestv > 0 else "-"))
                                                                                               
                                                                                               
                                                                                                   
                                                                                                 
                cmdline = ""
                try:
                    a0 = int(mm.arg_start); n = int(mm.arg_end) - a0
                    if 0 < n <= 4096:
                        raw = reader.read(a0, n, zero_fill=False) or b""
                        cmdline = raw.split(b"\x00", 1)[0].decode("latin-1", "replace")
                except Exception:
                    pass
                yield (0, (pid, comm, "triage", backend, str(bestv),
                           f"top={best} scores={scores} evidence={sorted(ev)} cmdline={cmdline!r}"))
                continue

            loc = self._pick(task)
            if loc is None:
                yield (0, (pid, comm, "-", "(no locator matched)", "", "")); continue
            vollog.info(f"pid {pid} ({comm}): locator = {loc.key}")
            d = desc or litert_desc
            if d is None and hasattr(loc, "default_descriptor"):
                d = loc.default_descriptor()
            if d is None and not getattr(loc, "self_describing", False):
                yield (0, (pid, comm, loc.key, "(detection only)", "", "no descriptor")); continue
            if manifest_path:
                reasm.reset_trace()
            try:
                bufs = loc.locate(self.context, task, reader, reasm, off, d)
            except NotImplementedError as e:
                yield (0, (pid, comm, loc.key, "(WIP)", "", str(e)[:80])); continue

            recov = {b.name: b.raw for b in bufs.values() if getattr(b, "raw", None)}
            resolved_names = {nm: nm for nm in bufs}
            for nm in sorted(bufs):
                b = bufs[nm]
                if getattr(b, "raw", None):
                    nz = len(b.raw) - b.raw.count(0)
                    st = f"RECOVERED {len(b.raw)}B nz={nz} src={b.src}" + (f" [{b.note}]" if b.note else "")
                else:
                    st = f"located src={b.src}" + (f" [{b.note}]" if b.note else "")
                yield (0, (pid, comm, loc.key, nm, hex(b.size), st))
            if recov:
                yield (0, (pid, comm, loc.key, "== SUMMARY ==", str(len(recov)),
                           f"{len(recov)} buffers recovered"))
                                                                                           
                head_dims = [b.hd for b in bufs.values() if getattr(b, "raw", None) and getattr(b, "hd", None)]
                active, mrows = self._phase1(task, head_dims)
                for a, b, c, dd in mrows:
                    yield (0, (pid, comm, a, b, c, dd))
                out = self.config.get("out")
                if out:
                    S = write_kvex(recov, out, row=getattr(loc, "row", 4096))
                    yield (0, (pid, comm, loc.key, "== KVEX ==", str(S), f"wrote -> {out} (K step={S})"))
                    if self.config.get("dequant", False) and active is not None:
                        quant = {}
                        for nm in recov:
                            base = active.base_name(nm) or nm
                            spec = active.specs.get(base)
                            if spec and spec.scale is not None:
                                quant[nm] = {"scale": spec.scale, "zero_point": spec.zero_point,
                                             "layer": spec.layer, "role": spec.role}
                        side = out + ".quant.json"
                        json.dump({"model": active.meta.get("model"), "note": "dequant: value = int8 * scale + zero_point; layer labels inherit the locator's reliability (UNVERIFIED/PROVISIONAL where noted)",
                                   "tensors": quant}, open(side, "w"), indent=1)
                        yield (0, (pid, comm, "dequant", "== QUANT ==", str(len(quant)),
                                   f"scale/zp for {len(quant)} tensors -> {side}"))

            if manifest_path:
                steps = [s for nm, raw in recov.items() if "_k_" in nm
                         for s in [k_step(raw, row=getattr(loc, "row", 4096))] if s]
                step = Counter(steps).most_common(1)[0][0] if steps else None
                items = []
                for name, b in sorted(bufs.items()):
                    src = getattr(b, "src", None)
                    fd = src[1] if isinstance(src, tuple) and len(src) > 1 \
                        and src[0] in ("fd", "fd-order") and isinstance(src[1], int) else None
                    fd_trace = reasm.fd_traces.get(fd, {}) if fd is not None else {}
                    priv_hex = fd_trace.get("priv_va")
                    priv_trace = reasm.priv_traces.get(int(priv_hex, 16), {}) if priv_hex else {}
                    chain_context = {**reasm.fd_table_trace, **fd_trace}
                    try:
                        task_for_fields = int(task.vol.offset)
                    except Exception:
                        task_for_fields = None
                    items.append({"source_name": name, "resolved_name": resolved_names.get(name, name),
                                  "source": list(src) if isinstance(src, tuple) else src,
                                  "label_note": getattr(b, "note", ""),
                                  "size": b.size, "sha256": hashlib.sha256(b.raw).hexdigest() if b.raw else None,
                                  "fd_chain": fd_trace, "sg_chain": priv_trace,
                                  "structure_fields": provenance.kernel_chain_fields(
                                      kr, off, task_for_fields, chain_context, priv_trace)})
                token_scan = {"status": "unsupported_backend", "candidates": []}
                if loc.key == "litert_dmabuf" and step:
                    regions = []
                    try:
                        library_bases = []
                        zero_offset_bases = []
                        for v in proc.Maps.list_vmas(task):
                            try:
                                start, end = int(v.vm_start), int(v.vm_end)
                                name = vma_name(self.context, task, v) or ""
                                if name.endswith("/liblitertlm_jni.so"):
                                    library_bases.append(start - int(v.vm_pgoff) * 4096)
                                    if int(v.vm_pgoff) == 0:
                                        zero_offset_bases.append(start)
                                if "rw" in v.get_protection() and v.vm_file == 0 and 0 < end - start <= 512 << 20:
                                    regions.append((start, end))
                            except Exception:
                                continue
                        lib_path = self.config.get("token-library")
                        if lib_path:
                            with open(lib_path, "rb") as lib_file:
                                lib_sha = hashlib.file_digest(lib_file, "sha256").hexdigest()
                            bases = sorted(set(library_bases))
                                                                                  
                                                                                       
                            if len(set(zero_offset_bases)) == 1:
                                token_reader = reader
                                page_map_path = self.config.get("token-page-map")
                                page_map = None
                                if page_map_path:
                                    with open(page_map_path, "r", encoding="utf-8") as f:
                                        page_map = json.load(f)
                                    if page_map.get("pid") != pid or page_map.get("library_sha256", "").lower() != lib_sha:
                                        raise ValueError("token page map PID/library mismatch")
                                    token_reader = provenance.TokenPageOverlay(reader, page_map["pages"])
                                token_scan = provenance.scan_executor_vtables(
                                    token_reader, regions, zero_offset_bases[0], lib_sha, step,
                                    vocab_size=int(self.config.get("token-vocab-size", 262144)),
                                    bos_id=int(self.config.get("token-bos-id", 2)),
                                    step_tolerance=max(0, int(self.config.get("token-step-tolerance", 1))))
                                if page_map is not None:
                                    objects = token_scan.get("objects", [])
                                    if not any(o["executor_va"] == page_map.get("executor_va")
                                               and o["current_step"] == page_map.get("current_step")
                                               for o in objects):
                                        raise ValueError("token page map executor/step mismatch")
                                    if "token_ids_sha256" in page_map:
                                        expected_pending_hash = hashlib.sha256(
                                            int(page_map["pending_token_id"]).to_bytes(4, "little")
                                        ).hexdigest() if "pending_token_id" in page_map else None
                                        if not any(
                                            branch["count"] == page_map.get("token_count")
                                            and branch["sha256"] == page_map["token_ids_sha256"]
                                            and (expected_pending_hash is None or
                                                 branch["pending_token_id_sha256"] == expected_pending_hash)
                                            for vector in token_scan["candidates"]
                                            for branch in vector["branches"]
                                        ):
                                            raise ValueError("token page map token count/hash mismatch")
                                    token_scan["external_page_map"] = {
                                        "path": page_map_path, "page_count": len(page_map["pages"]),
                                        "evidence": "live pagemap supplied; every LiME physical page SHA-256 verified",
                                    }
                            else:
                                token_scan = {"status": "library_mapping_ambiguous", "candidates": [],
                                              "library_bases": [hex(x) for x in bases],
                                              "zero_offset_bases": [hex(x) for x in sorted(set(zero_offset_bases))],
                                              "library_sha256": lib_sha}
                        elif self.config.get("token-heuristic-scan", False):
                            token_scan = provenance.scan_token_vectors(
                                reader, regions, step,
                                vocab_size=int(self.config.get("token-vocab-size", 262144)),
                                bos_id=int(self.config.get("token-bos-id", 2)),
                                budget_bytes=int(self.config.get("token-scan-mb", 1024)) << 20,
                                step_tolerance=max(0, int(self.config.get("token-step-tolerance", 1))))
                            token_scan["status"] = "heuristic_candidate_scan_complete"
                        else:
                            token_scan = {"status": "exact_token_library_required", "candidates": []}
                        token_scan["rw_anon_regions"] = len(regions)
                        token_scan["mapped_library_bases"] = [hex(x) for x in sorted(set(library_bases))]
                    except Exception as e:
                        token_scan = {"status": "scan_error", "error": f"{type(e).__name__}: {e}",
                                      "candidates": []}
                elif loc.key == "litert_dmabuf":
                    token_scan = {"status": "no_kv_step", "candidates": []}
                try:
                    task_va = int(task.vol.offset)
                except Exception:
                    task_va = None
                task_is_virtual = task_va is not None and page_offset <= task_va
                document = {"schema_version": 1, "scope": "one LiME image, one process",
                            "pid": pid, "comm": comm, "backend": loc.key,
                            "image_path": self._dump_path(kernel),
                            "page_offset": hex(page_offset), "phys_offset": hex(phys_offset),
                            "task_struct_va": hex(task_va) if task_is_virtual else None,
                            "task_struct_pa": hex(task_va - page_offset + phys_offset)
                            if task_is_virtual else (hex(task_va) if task_va is not None else None),
                            "mm_pgd_va": hex(pgd_va),
                            "fd_table": reasm.fd_table_trace,
                            "kv_buffers": items, "token_vector_scan": token_scan}
                provenance.write_manifest(manifest_path, document)
                yield (0, (pid, comm, loc.key, "== ADDRESSES ==", str(len(items)),
                           f"wrote -> {manifest_path}; token status={token_scan['status']}; "
                           f"executor objects={len(token_scan.get('objects', []))}; "
                           f"validated token vectors={len(token_scan['candidates'])}"))

    def run(self):
        return renderers.TreeGrid(
            [("PID", int), ("COMM", str), ("Locator", str),
             ("Tensor", str), ("Bytes", str), ("Status", str)],
            self._generator())
