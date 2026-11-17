#!/usr/bin/env python3



import btf_offsets

def load(vmlinux_btf, module_btf=None):
    vml = btf_offsets.BTF(vmlinux_btf)
    mod = btf_offsets.BTF(module_btf, base=vml) if module_btf else vml
    def M(b, s, m):
        o = b.offsets(s); return o.get(m) if o else None
    O = dict(
                                 
        OFF_TASKS=M(vml,'task_struct','tasks'), OFF_COMM=M(vml,'task_struct','comm'),
        OFF_PID=M(vml,'task_struct','pid'),     OFF_FILES=M(vml,'task_struct','files'),
        OFF_MM=M(vml,'task_struct','mm'),       OFF_REAL_PARENT=M(vml,'task_struct','real_parent'),
        MM_PGD=M(vml,'mm_struct','pgd'),
                                   
        FILE_FOP=M(vml,'file','f_op'),          FILE_PRIV=M(vml,'file','private_data'),
        DB_SIZE=M(vml,'dma_buf','size'),        DB_PRIV=M(vml,'dma_buf','priv'),
        DB_FILE=M(vml,'dma_buf','file'),
        F_FDT=M(vml,'files_struct','fdt'),      FDT_MAXFDS=M(vml,'fdtable','max_fds'), FDT_FD=M(vml,'fdtable','fd'),
                     
        SG_OFF=M(vml,'scatterlist','offset'),   SG_LEN=M(vml,'scatterlist','length'), SC_SZ=vml.size('scatterlist'),
                                                    
        SP_PRIVATE=M(vml,'page','private'),     SP_INDEX=M(vml,'page','index'),       XA_SLOTS=M(vml,'xa_node','slots'),
                               
        SC_SIZE=M(mod,'size_class','size'),     POOL_SC=M(mod,'zs_pool','size_class'),
                                                                                                   
        MM_MT=M(vml,'mm_struct','mm_mt'),       MA_ROOT=M(vml,'maple_tree','ma_root'),
        ARANGE_SLOT=M(vml,'maple_arange_64','slot'), LEAF_SLOT=M(vml,'maple_range_64','slot'),
        VM_START=M(vml,'vm_area_struct','vm_start'), VM_END=M(vml,'vm_area_struct','vm_end'),
        VM_ANON=M(vml,'vm_area_struct','anon_name'), AVN_NAME=M(vml,'anon_vma_name','name'),
    )
    bf = mod.bit('zspage', 'class')
    if bf: O['ZSPAGE_CLASS_SHIFT'], O['ZSPAGE_CLASS_BITS'] = bf
    return O

if __name__ == '__main__':
    import sys, json
    print(json.dumps({k: (hex(v) if isinstance(v, int) else v)
                      for k, v in load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None).items()}, indent=1))
