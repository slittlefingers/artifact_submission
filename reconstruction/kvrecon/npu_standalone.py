




import base64
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from .common import digest, dump
from .npu import NpuBackend


class StandaloneNpuBackend(NpuBackend):
    def __init__(self, *args, native_executable, **kwargs):
        super().__init__(*args, **kwargs)
        if self.transport != 'socket':
            raise ValueError('Standalone worker requires socket transport')
        self.native_executable=Path(native_executable).resolve()
        if not self.native_executable.is_file():
            raise FileNotFoundError(self.native_executable)
        self.remote='/data/local/tmp/'+self.name
                                                                                               
        self.rpc_timeout=kwargs.get('rpc_timeout',900)
        self.remote_executable=self.remote+'/worker'

    def write(self, filename, data):
                                                                                
        self.command('shell','-T',shlex.join(['sh','-c',
            'base64 -d > '+shlex.quote(self.remote+'/'+filename)]),input=base64.b64encode(data))

    def read(self, filename):
        return base64.b64decode(self.command('shell','-T',shlex.join(['base64',self.remote+'/'+filename])))

    def __enter__(self):
        started=time.perf_counter()
        try:
            package=self.shell('pm','path',self.package).decode().strip().removeprefix('package:')
            libs=package.rsplit('/',1)[0]+'/lib/arm64'
            actual=self.shell('sha256sum',self.device_model,timeout=240).decode().split()[0]
            if actual != self.model_sha256:
                raise ValueError('Staged model differs from the configured exact file hash')
            self.shell('mkdir','-p',self.remote)
            self.command('push',str(self.native_executable),self.remote_executable)
            self.shell('chmod','755',self.remote_executable)
            executable_hash=digest(self.native_executable)
            if self.shell('sha256sum',self.remote_executable).decode().split()[0]!=executable_hash:
                raise ValueError('Native worker transfer hash mismatch')
            job=dict(model_sections=self.sections,layer=self.layer,
                batch_signature=self.batch_signature,mask_token_id=self.mask_token_id,
                transport=self.transport,socket_name=self.name,socket_auth=self.socket_auth,
                native_mode=self.native_mode,qwen_recovery=self.qwen_recovery,
                qwen_query_mode=self.qwen_query_mode)
            self.write('job.json',json.dumps(job).encode())
            arguments=[self.remote_executable,self.device_model,libs,self.remote,self.remote+'/job.json']
            command='echo $$ > '+shlex.quote(self.remote+'/pid')+'; exec '+shlex.join(['env','LD_LIBRARY_PATH='+libs]+arguments)
            self.stream=(self.output/'native.log').open('wb')
            self.proc=subprocess.Popen(self.adb+['shell','-T',shlex.join(['sh','-c',command])],
                stdout=self.stream,stderr=subprocess.STDOUT)
            ready=self._wait('ready.json',None,180)
            self.selected_retention_supported=bool(ready.get('selected_retention',False))
            self.selected_retention=False
            self.pending_retain=None
            if ready.get('protocol')!=2 or ready.get('transport')!='socket':
                raise ValueError('Worker RPC protocol mismatch')
            self.forward_port=int(self.command('forward','tcp:0','localabstract:'+self.name).strip())
            self.connection=socket.create_connection(('127.0.0.1',self.forward_port),timeout=self.rpc_timeout)
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.ready=True
            if self.qwen_recovery:
                self.physical_batch_capacity=64 if self.qwen_query_mode=='verify' else 1
                self.rpc('query_mode',mode=self.qwen_query_mode)
            for key in ('vocab_size','batch_capacity','context_capacity'):
                if ready[key]!=self.dimensions[key]:
                    raise ValueError('Worker/model geometry mismatch: '+key)
            self.setup_seconds=time.perf_counter()-started
            dump(self.output/'device.json',dict(model_sha256=actual,
                runtime_sha256=self.shell('sha256sum',libs+'/liblitert-lm.so').decode().split()[0],
                native_executable=str(self.native_executable),native_executable_sha256=executable_hash,
                model_path=self.device_model,transport='socket',rpc_protocol=2,
                standalone=True,qwen_recovery=self.qwen_recovery,qwen_query_mode=self.qwen_query_mode,
                physical_batch_capacity=getattr(self,'physical_batch_capacity',self.batch_capacity),
                setup_seconds=self.setup_seconds,**self.dimensions))
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _terminate_owned_worker(self):
        try:
            pid=self.read('pid').decode().strip()
            if pid.isdigit():
                actual=self.shell('cat','/proc/'+pid+'/cmdline').split(b'\0')[0].decode()
                if actual==self.remote_executable:
                    self.shell('kill','-TERM',pid)
        except subprocess.CalledProcessError:
            pass

    def __exit__(self, typ, value, tb):
        error=None
        try:
            if self.ready and not self.rpc_broken and self.proc and self.proc.poll() is None:
                self.rpc('stop')
                self.proc.wait(timeout=30)
            elif self.proc and self.proc.poll() is None:
                self._terminate_owned_worker()
                self.proc.wait(timeout=10)
        except Exception as e:
            error=str(e)
            self._terminate_owned_worker()
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        finally:
            if self.connection:
                self.connection.close();self.connection=None
            if self.forward_port is not None:
                try:self.command('forward','--remove',f'tcp:{self.forward_port}')
                except Exception as e:error=error or str(e)
                self.forward_port=None
            if self.stream:
                self.stream.close()
            self.ready=False
        if self.proc and self.proc.poll() not in (None,0):
            error=error or 'Native worker returned '+str(self.proc.returncode)
        if error:
            dump(self.output/'shutdown_error.json',dict(error=error,remote_directory=self.remote))
            if typ is None:
                raise RuntimeError(error)
