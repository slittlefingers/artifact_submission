





from __future__ import annotations
import os
import sys
from typing import Dict, Optional

                                                                 
KERNEL_KEYS = ("OFF_FILES", "F_FDT", "FDT_MAXFDS", "FDT_FD", "FILE_FOP", "FILE_PRIV",
               "DB_SIZE", "DB_FILE", "DB_PRIV", "SC_SZ", "SG_LEN", "SG_OFF")


def load_kernel_offsets(recover_dir: str,
                        vmlinux_btf: Optional[str] = None,
                        zsmalloc_btf: Optional[str] = None) -> Dict[str, int]:
    




    core = os.path.join(recover_dir, "core")
    if core not in sys.path:
        sys.path.insert(0, core)
    import kbtf  # noqa: E402  (deadbox module, found via recover_dir/core)
    vmlinux_btf = vmlinux_btf or os.path.join(recover_dir, "vmlinux.btf")
    zsmalloc_btf = zsmalloc_btf or os.path.join(recover_dir, "zsmalloc.btf")
    off = kbtf.load(vmlinux_btf, zsmalloc_btf)
    return off


                                                                                                  
                                                                                                   
                                                          
_ISF_FIELDS = {
    "OFF_TASKS": ("task_struct", "tasks"), "OFF_COMM": ("task_struct", "comm"),
    "OFF_PID": ("task_struct", "pid"), "OFF_FILES": ("task_struct", "files"),
    "OFF_MM": ("task_struct", "mm"), "OFF_REAL_PARENT": ("task_struct", "real_parent"),
    "MM_PGD": ("mm_struct", "pgd"),
    "FILE_FOP": ("file", "f_op"), "FILE_PRIV": ("file", "private_data"),
    "DB_SIZE": ("dma_buf", "size"), "DB_FILE": ("dma_buf", "file"), "DB_PRIV": ("dma_buf", "priv"),
    "F_FDT": ("files_struct", "fdt"), "FDT_MAXFDS": ("fdtable", "max_fds"), "FDT_FD": ("fdtable", "fd"),
    "SG_OFF": ("scatterlist", "offset"), "SG_LEN": ("scatterlist", "length"),
    "SP_PRIVATE": ("page", "private"), "SP_INDEX": ("page", "index"), "XA_SLOTS": ("xa_node", "slots"),
    "MM_MT": ("mm_struct", "mm_mt"), "MA_ROOT": ("maple_tree", "ma_root"),
    "ARANGE_SLOT": ("maple_arange_64", "slot"), "LEAF_SLOT": ("maple_range_64", "slot"),
    "VM_START": ("vm_area_struct", "vm_start"), "VM_END": ("vm_area_struct", "vm_end"),
    "VM_ANON": ("vm_area_struct", "anon_name"), "AVN_NAME": ("anon_vma_name", "name"),
}
_ISF_SIZES = {"SC_SZ": "scatterlist"}


def offsets_from_isf(vmlinux_module) -> Dict[str, int]:
    


    out: Dict[str, int] = {}
    for key, (struct, field) in _ISF_FIELDS.items():
        try:
            m = vmlinux_module.get_type(struct).members
            if field in m:
                out[key] = m[field][0]
        except Exception:
            pass
    for key, struct in _ISF_SIZES.items():
        try:
            out[key] = vmlinux_module.get_type(struct).size
        except Exception:
            pass
    return out


def struct_page_size(off: Dict[str, int], default: int = 64) -> int:
    
    for k in ("SIZEOF_PAGE", "STRUCT_PAGE_SIZE", "SP"):
        if off.get(k):
            return off[k]
    return default
