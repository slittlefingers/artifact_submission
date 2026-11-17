#define _GNU_SOURCE
#include <elf.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <android/log.h>

                                                                             
                                                                               
                                                                                
static ssize_t ghost_bulk_read(int fd,void* data,size_t count){
 if(count<=0x7ffff000UL)return syscall(__NR_read,fd,data,count);
 __android_log_print(ANDROID_LOG_INFO,"GhostIO","Split large read: %zu bytes",count);
 size_t done=0;while(done<count){size_t n=count-done;if(n>(1UL<<30))n=1UL<<30;
  ssize_t got=syscall(__NR_read,fd,(char*)data+done,n);
  if(got<0){if(errno==EINTR)continue;return done?(ssize_t)done:-1;}if(!got)break;done+=(size_t)got;}
 return (ssize_t)done;
}
static ssize_t ghost_bulk_pread64(int fd,void* data,size_t count,off64_t offset){
 if(count<=0x7ffff000UL||offset<0)return syscall(__NR_pread64,fd,data,count,offset);
 struct timespec started,now;clock_gettime(CLOCK_MONOTONIC,&started);
 __android_log_print(ANDROID_LOG_INFO,"GhostIO","Split large pread64: %zu bytes fd=%d offset=%lld",count,fd,(long long)offset);
 size_t done=0;while(done<count){size_t n=count-done;if(n>(1UL<<30))n=1UL<<30;
  ssize_t got=syscall(__NR_pread64,fd,(char*)data+done,n,offset+(off64_t)done);
  if(got<0){if(errno==EINTR)continue;__android_log_print(ANDROID_LOG_ERROR,"GhostIO","pread64 failed errno=%d done=%zu",errno,done);return done?(ssize_t)done:-1;}if(!got)break;done+=(size_t)got;
  clock_gettime(CLOCK_MONOTONIC,&now);double elapsed=(now.tv_sec-started.tv_sec)+(now.tv_nsec-started.tv_nsec)/1e9;
  __android_log_print(ANDROID_LOG_INFO,"GhostIO","pread64 progress %zu/%zu elapsed=%.3fs",done,count,elapsed);}
 return (ssize_t)done;
}
static uintptr_t address(uintptr_t base,uintptr_t value){return value<base?base+value:value;}
static int replace_slot(void** slot,void* target){
 long page=sysconf(_SC_PAGESIZE);uintptr_t lo=(uintptr_t)slot&~((uintptr_t)page-1);int prot=PROT_READ;
 FILE* maps=fopen("/proc/self/maps","r");if(maps){char line[1024],p[5];unsigned long a,b;
  while(fgets(line,sizeof(line),maps))if(sscanf(line,"%lx-%lx %4s",&a,&b,p)==3&&(uintptr_t)slot>=a&&(uintptr_t)slot<b){prot=(p[0]=='r'?PROT_READ:0)|(p[1]=='w'?PROT_WRITE:0)|(p[2]=='x'?PROT_EXEC:0);break;}fclose(maps);}
 if(mprotect((void*)lo,page,prot|PROT_WRITE))return 0;
 __atomic_store_n(slot,target,__ATOMIC_RELEASE);
 if(mprotect((void*)lo,page,prot))return 0;return 1;
}
int ghost_fix_vendor_bulk_io(void* vendor_symbol){
 static int installed=0;if(installed)return installed;
 Dl_info info={0};if(!vendor_symbol||!dladdr(vendor_symbol,&info)||!info.dli_fbase)return 0;
 uintptr_t base=(uintptr_t)info.dli_fbase;Elf64_Ehdr* eh=(Elf64_Ehdr*)base;
 if(memcmp(eh->e_ident,ELFMAG,SELFMAG)||eh->e_ident[EI_CLASS]!=ELFCLASS64||eh->e_machine!=EM_AARCH64)return 0;
 Elf64_Phdr* ph=(Elf64_Phdr*)(base+eh->e_phoff);Elf64_Dyn* dyn=NULL;size_t nd=0;
 for(int i=0;i<eh->e_phnum;i++)if(ph[i].p_type==PT_DYNAMIC){dyn=(Elf64_Dyn*)(base+ph[i].p_vaddr);nd=ph[i].p_memsz/sizeof(*dyn);break;}
 if(!dyn)return 0;Elf64_Sym* syms=NULL;const char* names=NULL;Elf64_Rela* relas=NULL;size_t bytes=0;int rela=0;
 for(size_t i=0;i<nd&&dyn[i].d_tag!=DT_NULL;i++){uintptr_t ptr=address(base,dyn[i].d_un.d_ptr);
  switch(dyn[i].d_tag){case DT_SYMTAB:syms=(Elf64_Sym*)ptr;break;case DT_STRTAB:names=(const char*)ptr;break;case DT_JMPREL:relas=(Elf64_Rela*)ptr;break;case DT_PLTRELSZ:bytes=dyn[i].d_un.d_val;break;case DT_PLTREL:rela=dyn[i].d_un.d_val==DT_RELA;break;}}
 if(!syms||!names||!relas||!rela||bytes/sizeof(*relas)>100000)return 0;
 for(size_t i=0;i<bytes/sizeof(*relas);i++){
  if(ELF64_R_TYPE(relas[i].r_info)!=R_AARCH64_JUMP_SLOT)continue;
  const char* name=names+syms[ELF64_R_SYM(relas[i].r_info)].st_name;void* target=NULL;
  if(!strcmp(name,"read"))target=(void*)ghost_bulk_read;
  if(!strcmp(name,"pread64"))target=(void*)ghost_bulk_pread64;
  if(target)installed+=replace_slot((void**)(base+relas[i].r_offset),target);
 }
 __android_log_print(ANDROID_LOG_INFO,"GhostIO","Vendor bulk-read compatibility imports: %d",installed);return installed;
}
