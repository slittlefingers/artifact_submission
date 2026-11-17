




import base64
import json
import secrets
import socket
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid
import numpy as np
from .common import WORKSPACE, dump, digest, quantization, dequantize, kv_dtype
from .config import model_dimensions
from .transport import exchange

PKG = "com.example.annoymous"
MODEL_NAME = "gemma-4-E2B-it_Google_Tensor_G5.litertlm"


class NpuBackend:
    package=PKG
    def __init__(self, sections, layer, output, model_sha256, adb=None, install=True, serial=None,
                 device_model=None,batch_signature="prefill_128",mask_token_id=1437,transport="socket",native_mode="optimized_v2",package=PKG,qwen_recovery=False,qwen_query_mode="decode"):
        if not package or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._' for c in package):
            raise ValueError('Invalid Android package')
        self.package=package
        if native_mode not in ("reference","optimized","optimized_v2"):
            raise ValueError("Unknown native execution mode")
        self.native_mode=native_mode
        self.qwen_recovery=bool(qwen_recovery)
        if qwen_query_mode not in ('decode','exact_prefill','verify'):
            raise ValueError('Unknown Qwen query mode')
        self.qwen_query_mode=qwen_query_mode
        self.commit_revalidates_distance=self.qwen_recovery and qwen_query_mode=='verify'
        self.query_winner_reuse_supported=self.commit_revalidates_distance
        if self.commit_revalidates_distance:
            signatures=sections['tf_lite_prefill_decode']['signatures']
            if 'verify_64' not in signatures:
                raise ValueError('verify mode requires a compiled verify_64 signature')
            names={t['name'] for t in signatures['verify_64']['outputs']}
            if not {f'kv_cache_k_{layer}',f'kv_cache_v_{layer}'}<=names:
                raise ValueError('verify signature must expose selected-layer K/V outputs')
        if transport not in ("socket", "file"):
            raise ValueError("Unknown NPU transport")
        self.transport=transport;self.connection=None;self.forward_port=None
        self.socket_auth=secrets.token_hex(32);self.rpc_broken=False;self.pending_begin=False
        self.adb = [adb or str(Path.home()/"Library/Android/sdk/platform-tools/adb")]
        if serial:
            self.adb += ["-s", serial]
        self.sections, self.layer = sections, layer
        self.device_model=device_model or f"/sdcard/Android/data/{self.package}/files/{MODEL_NAME}"
        if not self.device_model.startswith("/"):
            raise ValueError("Device model path must be absolute")
        self.batch_signature=batch_signature;self.mask_token_id=mask_token_id
        self.dimensions=model_dimensions(sections,batch_signature)
        self.vocab_size=self.dimensions["vocab_size"];self.batch_capacity=self.dimensions["batch_capacity"]
        if not 0<=mask_token_id<self.vocab_size:
            raise ValueError("Mask helper token is outside the model vocabulary")
        self.output = Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.model_sha256, self.install = model_sha256, install
        self.name = "kvrecon-" + uuid.uuid4().hex[:16]
        self.remote = "files/" + self.name
        self.seq = 0; self.position = 0; self.proc = None; self.stream = None; self.ready = False
        self.quant = quantization(sections,layer,batch_signature)
        if self.quant!=quantization(sections,layer,"decode"):
            raise ValueError("This backend requires matching dereconstruction/components/prefill KV quantization metadata")
        self.width = sum(q["width"] for q in self.quant)
        self.dtype=kv_dtype(self.quant)
        self.calls = []; self.decoder_seconds = 0.

    def command(self, *args, **kwargs):
        return subprocess.run(self.adb+list(args),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=kwargs.pop("timeout",120),**kwargs).stdout

    def shell(self, *args, **kwargs):
        return self.command("shell",shlex.join(args),**kwargs)

    def write(self, filename, data):
                                                                                
        path = self.remote + "/" + filename
        self.command("shell","-T",shlex.join(["run-as",self.package,"sh","-c","base64 -d > "+shlex.quote(path)]),input=base64.b64encode(data))

    def read(self, filename):
                                                                                        
        return base64.b64decode(self.command("shell","-T",shlex.join(["run-as",self.package,"base64",self.remote+"/"+filename])))

    def __enter__(self):
        started = time.perf_counter()
        try:
            apk = WORKSPACE/"target_app/source/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
            packaged_apk = WORKSPACE/"target_app/ghost-androidTest.apk"
            if self.package == PKG and packaged_apk.is_file():
                apk = packaged_apk
            expected_apk=digest(apk)
            installed_path=self.shell("pm","path",self.package+".test").decode().strip().removeprefix("package:")
            installed_hash=self.shell("sha256sum",installed_path).decode().split()[0] if installed_path else None
            if self.install and installed_hash!=expected_apk:
                self.command("install","-r","-t",str(apk))
                installed_path=self.shell("pm","path",self.package+".test").decode().strip().removeprefix("package:")
                installed_hash=self.shell("sha256sum",installed_path).decode().split()[0]
            if installed_hash!=expected_apk:
                raise ValueError("Installed test APK differs from the local build; rebuild/install before --no-install")
            package = self.shell("pm","path",self.package).decode().strip().removeprefix("package:")
            libs = package.rsplit("/",1)[0]+"/lib/arm64"
            model = self.device_model
                                                                            
                                                                         
            hash_args=("run-as",self.package,"sha256sum",model) if model.startswith('/data/') else ("sha256sum",model)
            actual = self.shell(*hash_args,timeout=180).decode().split()[0]
            if actual != self.model_sha256:
                raise ValueError("Installed model differs from observation model")
            runtime = self.shell("run-as",self.package,"sha256sum",libs+"/liblitert-lm.so").decode().split()[0]
            libraries={}
            for name in ['libLiteRtDispatch_GoogleTensor.so','libedgetpu_litert.so']:
                try:libraries[name]=self.shell('run-as',self.package,'sha256sum',libs+'/'+name).decode().split()[0]
                except subprocess.CalledProcessError:libraries[name]=None
            self.shell("run-as",self.package,"mkdir","-p",self.remote)
            self.write("job.json",json.dumps(dict(model_sections=self.sections,layer=self.layer,
                batch_signature=self.batch_signature,mask_token_id=self.mask_token_id,
                transport=self.transport,socket_name=self.name,socket_auth=self.socket_auth,native_mode=self.native_mode,qwen_recovery=self.qwen_recovery,qwen_query_mode=self.qwen_query_mode)).encode())
            args=["am","instrument","-w","-r","-e","class",PKG+".ReconstructionDeviceTest","-e","reconstruction_job",self.name,"-e","reconstruction_model",model,self.package+".test/androidx.test.runner.AndroidJUnitRunner"]
            dump(self.output/"device.json",dict(job=self.name,model_sha256=actual,runtime_sha256=runtime,
                test_apk_sha256=installed_hash,libraries=libraries,command=args,transport=self.transport,
                rpc_protocol=2,ranking_location="device" if self.transport=="socket" else "host",native_mode=self.native_mode,
                model_path=model,batch_signature=self.batch_signature,**self.dimensions))
            if self.qwen_recovery:
                record=json.loads((self.output/'device.json').read_text())
                record.update(qwen_recovery=True,qwen_query_mode=self.qwen_query_mode,physical_batch_capacity=64 if self.qwen_query_mode=='verify' else 1)
                dump(self.output/'device.json',record)
            self.stream=(self.output/"instrumentation.log").open("wb")
            self.proc=subprocess.Popen(self.adb+["shell",shlex.join(args)],stdout=self.stream,stderr=subprocess.STDOUT)
            ready=self._wait("ready.json",None,180)
            self.selected_retention_supported=bool(ready.get('selected_retention',False))
            self.selected_retention=False
            self.pending_retain=None
            if ready.get("protocol")!=2 or ready.get("transport")!=self.transport:
                raise ValueError("Worker RPC protocol mismatch")
            if self.transport=="socket":
                self.forward_port=int(self.command("forward","tcp:0","localabstract:"+self.name).strip())
                self.connection=socket.create_connection(("127.0.0.1",self.forward_port),timeout=180)
                self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.ready=True
            if self.qwen_recovery:
                self.physical_batch_capacity=64 if self.qwen_query_mode=='verify' else 1
                self.rpc('query_mode',mode=self.qwen_query_mode)
            for key in ("vocab_size","batch_capacity","context_capacity"):
                if ready[key]!=self.dimensions[key]:
                    raise ValueError("Device signature does not match configured manifest: "+key)
            self.setup_seconds=time.perf_counter()-started
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _wait(self, file, seq, timeout=180):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:
                value=json.loads(self.read(file))
            except subprocess.CalledProcessError as e:
                if b"No such file" not in e.stderr:
                    raise
                if self.proc and self.proc.poll() is not None:
                    raise RuntimeError("Device worker exited; inspect native/instrumentation logs") from e
                time.sleep(.03);continue
            if seq is None or value.get("seq")==seq:
                if not value.get("ok"):
                    raise RuntimeError(value.get("error","Worker failed"))
                return value
            time.sleep(.03)
        raise TimeoutError("Worker response timeout")

    def rpc(self, op, **kwargs):
        if self.rpc_broken:
            raise RuntimeError("RPC state is uncertain; restart recovery instead of replaying commands")
        started=time.perf_counter();self.seq+=1
        command=dict(seq=self.seq,op=op,**kwargs)
        try:
            if self.transport=="socket":
                reply,payload=exchange(self.connection,dict(command,auth=self.socket_auth))
            else:
                self.write("command.tmp",json.dumps(command).encode())
                try:
                    self.shell("run-as",self.package,"mv",self.remote+"/command.tmp",self.remote+"/command.json")
                except subprocess.CalledProcessError as e:
                                                                                    
                    dump(self.output/f"publish_error_{self.seq}.json",dict(seq=self.seq,
                        stderr=e.stderr.decode(errors='replace'),returncode=e.returncode))
                reply=self._wait("response.json",self.seq)
                payload=self.read("payload.bin") if reply["payload_bytes"] else b""
                if len(payload)!=reply["payload_bytes"]:
                    raise ValueError("Truncated RPC payload")
        except BaseException:
            self.rpc_broken=True
            raise
        self.position=reply["position"];self.decoder_seconds+=reply["decoder_seconds"]
        row=dict(op=op,host_seconds=time.perf_counter()-started,**reply)
        self.calls.append(row)
        with (self.output/"rpc.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        return payload

    def encode_audio_wav(self, path):
        
        blob=Path(path).read_bytes();position=self.position
        if not 0<len(blob)<=16*1024*1024:raise ValueError('WAV must fit the audio upload limit')
        for offset in range(0,len(blob),512000):
            self.rpc('upload_collection_audio',offset=offset,data_hex=blob[offset:offset+512000].hex())
        payload=self.rpc('encode_audio',position=position)
        metadata=dict(self.calls[-1]);rows=metadata['audio_rows']
        if self.position!=position or metadata['prefix_advanced']:
            raise RuntimeError('Audio encoding unexpectedly advanced the text prefix')
        if rows<=0 or len(payload)!=rows*1536*4:raise ValueError('Invalid encoded audio size')
        features=np.frombuffer(payload,dtype='<f4').reshape(rows,1536).copy()
        if not np.isfinite(features).all():raise ValueError('Nonfinite audio embeddings')
        return features,metadata

    def reset(self, prefix):
        self.pending_begin=False
        self.pending_retain=None
        self.selected_retention=False
        self.rpc("reset",prefix_ids=list(map(int,prefix)))

    def probabilities(self):
        if self.transport=="socket":
            ids=np.frombuffer(self.rpc("ranking"),dtype="<i4").astype(np.int64)
            if ids.shape!=(self.vocab_size,) or np.any(ids<0) or np.any(ids>=self.vocab_size) or np.unique(ids).size!=self.vocab_size:
                raise ValueError("Invalid full-vocabulary ranking")
            return ids
        v=np.frombuffer(self.rpc("logits"),dtype="<f4").copy()
        if v.shape!=(self.vocab_size,) or not np.isfinite(v).all():
            raise ValueError("Invalid logits")
        return np.argsort(-v,kind="stable")

    def query(self, ids):
        ids=list(map(int,ids))
        if not ids or len(ids)>self.batch_capacity or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=self.vocab_size:
            raise ValueError(f"Query must contain 1..{self.batch_capacity} unique original IDs")
        extra={}
        if getattr(self,'selected_retention',False):
            extra['selected_retention']=True
            if self.pending_retain is not None:extra['retain_id']=self.pending_retain
        data=self.rpc("query",ids=ids,position=self.position,begin_candidates=self.pending_begin,**extra)
        self.pending_retain=None
        self.last_query_ids=ids
        self.pending_begin=False
        return np.frombuffer(data,dtype=self.dtype).reshape(len(ids),self.width).copy()

    def values(self, codes):
        return dequantize(codes,self.quant)

    cache_acceptance_policy='retained query KV; logits recomputed by decode because prefill exports no logits'

    def begin_candidates(self):
        self.selected_retention=False
        self.pending_retain=None
        if self.transport=="socket":
            self.pending_begin=True
        else:
            self.rpc('begin_candidates')

    def begin_selected_candidates(self):
        




        self.begin_candidates()
        self.selected_retention=getattr(self,'selected_retention_supported',False) and self.transport=='socket'
        self.last_query_ids=[]

    def retain_candidate(self, token):
        if not getattr(self,'selected_retention',False):return
        token=int(token)
        if token not in self.last_query_ids or self.pending_begin:
            raise ValueError('Cannot retain a token outside the latest query')
        self.pending_retain=token

    def accept_query(self, token):
        if self.pending_begin:
            raise ValueError('No candidates queried in the new batch')
        extra={}
        if getattr(self,'selected_retention',False) and self.pending_retain is not None:
            extra['retain_id']=self.pending_retain
        data=self.rpc('accept_query',id=int(token),position=self.position,**extra)
        self.pending_retain=None
        return np.frombuffer(data,dtype=self.dtype).reshape(1,self.width).copy()

    def query_bulk(self, ids):
        ids=list(map(int,ids))
        if not ids or len(ids)>4096 or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=self.vocab_size:
            raise ValueError('Bulk query requires 1..4096 unique original IDs')
        self.pending_begin=False
        data=self.rpc('query_bulk',ids=ids,position=self.position)
        return np.frombuffer(data,dtype=self.dtype).reshape(len(ids),self.width).copy()

    def first_v_table(self, ids):
        ids=list(map(int,ids))
        if self.layer!=0 or not ids or len(ids)>4096 or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=self.vocab_size:
            raise ValueError("First-layer V chunk requires 1..4096 unique original IDs")
        data=self.rpc("first_v_table",ids=ids)
        return np.frombuffer(data,dtype=self.dtype).reshape(len(ids),self.quant[1]["width"]).copy()

    def commit(self, token):
        self.pending_begin=False
        self.pending_retain=None
        data=self.rpc("commit",id=int(token),position=self.position)
        return np.frombuffer(data,dtype=self.dtype).reshape(1,self.width).copy()

    def commit_cached_qwen(self, token):
        




        self.pending_begin=False
        self.pending_retain=None
        data=self.rpc("commit_cached",id=int(token),position=self.position)
        return np.frombuffer(data,dtype=self.dtype).reshape(1,self.width).copy()

    def collect_tokens(self, ids):
        






        ids=list(map(int,ids))
        if (not ids or len(ids)>self.batch_capacity or min(ids)<0 or max(ids)>=self.vocab_size
                or self.position+self.batch_capacity>self.dimensions['context_capacity']):
            raise ValueError('Causal prefill block/padding exceeds compiled capacity')
        self.pending_begin=False
        self.pending_retain=None
        data=self.rpc('collect_tokens',ids=ids,layers=[self.layer])
        return np.frombuffer(data,dtype=self.dtype).reshape(len(ids),self.width).copy()

    def commit_embeddings(self, features, ple_id=0):
        




        values=np.asarray(features,dtype=np.float32)
        if (values.ndim!=2 or not len(values) or values.shape[1]!=self.dimensions['embedding_dim']
                or not np.isfinite(values).all() or self.position+len(values)>self.dimensions['context_capacity']
                or type(ple_id) is not int or not 0<=ple_id<self.vocab_size):
            raise ValueError('Invalid continuous input block or PLE ID')
        self.pending_begin=False
        rows=[]
        for start in range(0,len(values),8):
            chunk=values[start:start+8]
            data=self.rpc('commit_embeddings',embeddings=chunk.tolist(),ple_id=ple_id,position=self.position)
            rows.append(np.frombuffer(data,dtype=self.dtype).reshape(len(chunk),self.width).copy())
        return np.concatenate(rows)

    def _vision_call(self, op, latent):
        




        values=np.asarray(latent,dtype='<f4')
        if (values.ndim!=2 or not 1<=len(values)<=min(128,self.batch_capacity)
                or values.shape[1]!=768 or not np.isfinite(values).all()
                or 'tf_lite_vision_adapter' not in self.sections):
            raise ValueError('Expected 1..128 finite Gemma vision latent rows of width 768')
        if op=='vision_query':self.pending_begin=False
        return self.rpc(op,rows=len(values),latent_hex=values.tobytes().hex(),position=self.position)

    def project_vision(self, latent):
        data=self._vision_call('vision_project',latent)
        return np.frombuffer(data,dtype='<f4').reshape(len(latent),1536).copy()

    def query_vision(self, latent):
        data=self._vision_call('vision_query',latent)
        return np.frombuffer(data,dtype=self.dtype).reshape(len(latent),self.width).copy()

    def query_embeddings(self, features):
        




        values=np.asarray(features,dtype='<f4')
        if (values.ndim!=2 or not 1<=len(values)<=min(128,self.batch_capacity)
                or values.shape[1]!=1536 or self.dimensions['embedding_dim']!=1536
                or not np.isfinite(values).all()):
            raise ValueError('Expected 1..128 finite Gemma continuous rows of width 1536')
        self.pending_begin=False
        data=self.rpc('query_embeddings',rows=len(values),features_hex=values.tobytes().hex(),position=self.position)
        return np.frombuffer(data,dtype=self.dtype).reshape(len(values),self.width).copy()

    def __exit__(self, typ, value, tb):
        error=None
        if self.ready and not self.rpc_broken and self.proc and self.proc.poll() is None:
            try:
                self.rpc("stop");self.proc.wait(timeout=30)
            except Exception as e:
                error=str(e)
                self.proc.terminate()
                                                                                    
        if self.connection:
            self.connection.close();self.connection=None
        if self.forward_port is not None:
            try:
                self.command("forward","--remove",f"tcp:{self.forward_port}")
            except Exception as e:
                error=error or str(e)
            self.forward_port=None
        if self.proc and self.proc.poll() is None:
            try:self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:self.proc.terminate()
        if self.stream:
            self.stream.close()
        try:
            (self.output/"native.log").write_bytes(self.read("native.log"))
        except Exception as e:
            error=error or str(e)
        if error:
            dump(self.output/"shutdown_error.json",dict(error=error,remote_directory=self.remote))
            if typ is None:
                raise RuntimeError("Worker shutdown or log retrieval failed: "+error)
        elif self.proc and (self.proc.returncode or "OK (1 test)" not in (self.output/"instrumentation.log").read_text()):
            if typ is None:
                raise RuntimeError("Worker instrumentation failed; inspect logs")
