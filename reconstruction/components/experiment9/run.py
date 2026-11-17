




import argparse
from dataclasses import asdict
import json,os,sys,time,shutil,traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;CODE=ROOT.parent
sys.path.insert(0,str(CODE.parent))
from kvrecon.job_config import load_job
from kvrecon.json_job import open_backend,QueryLog
from kvrecon.kv_reference import VReferenceIndex
from kvrecon.torch_surrogate import TorchRidge
from kvrecon.search import SearchConfig,kv_distance
from kvrecon.engine import reconstruct
from kvrecon.media.pipeline import build_pipeline
from kvrecon.common import digest


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    for attempt in range(30):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=CODE.parents[1]/'outputs/5.6_gemma',help='New output directory')
    parser.add_argument('--sample-number',type=int,default=1,help='One-based entry in the fixed manifest')
    parser.add_argument('--variant',choices=['multimodal','token_surrogate'],default='multimodal')
    parser.add_argument('--data',type=Path,default=CODE.parents[1]/'assets/kv/5.6_gemma/omnidialog',help='OmniDialog cache root with manifest.json and samples/')
    parser.add_argument('--job',type=Path,default=CODE.parents[1]/'experiments/5.6 Gemma - Multimodal Conversation Recovery/gemma.json',help='Reconstruction job config supplying model/device/embeddings/ridge')
    parser.add_argument('--tokenizer',type=Path,default=CODE.parents[1]/'models/tokenizers/gemma4_e2b/tokenizer.json',help='Public Gemma tokenizer; tokenizer_config.json and chat_template.jinja sit beside it')
    parser.add_argument('--image-package',type=Path,default=CODE/'experiment2_image_layer4/transfer/gemma_layer4_kv_image',help='Frozen layer-4 image decoder package')
    parser.add_argument('--audio-package',type=Path,default=CODE/'experiment3_audio/audio_chat_layer4/transfer/audio_layer4_kv_recovery',help='Frozen layer-4 audio decoder package')
    parser.add_argument('--max-positions',type=int,help='Optional pilot; omit to recover the whole conversation')
    args=parser.parse_args()
    if args.sample_number<1:raise ValueError('Sample number must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    token_surrogate=args.variant=='token_surrogate'
    started=time.perf_counter();state=dict(state='preparing',pid=os.getpid(),run=out.name,variant=args.variant)
    def progress(**values):
        state.update(values);state['elapsed_seconds']=time.perf_counter()-started
        save(out/'progress.json',state)
    progress()
    try:
        data=Path(args.data);first=read(data/'manifest.json')['samples'][args.sample_number-1]
        sid=first['id'];source=data/first['path']/f"turn{first['turns']}_layer4_active.npz"
        job=load_job(args.job)
        prefix=[2]                                                                     
        with np.load(source,allow_pickle=False) as z:
            selected=z['positions']>=len(prefix);positions=z['positions'][selected].copy()
            codes=np.concatenate([z['k'][selected],z['v'][selected]],1).copy()
        if codes.dtype!=np.int8 or not np.array_equal(positions,np.arange(1,len(positions)+1)):
            raise ValueError('Need a contiguous full layer-4 INT8 observation after BOS')
        np.savez_compressed(out/'observation.npz',codes=codes,positions=positions,public_prefix_ids=prefix)
        targets=codes.astype(np.float64)
        from search_settings import build_search_config
        config,priority_metadata=build_search_config(args.tokenizer,args.variant)
        table=CODE.parent/'assets/kv_reference/gemma4_e2b_layer4_chatprefix_prefill_selected'
        metadata=read(table/'metadata.json');identity=metadata['identity']
        if identity['layer_index']!=job['model']['layer']:raise ValueError('Wrong reference layer')
        shutil.copy2(data/'boundary_pipeline/run01/end_references.npz',out/'end_references.npz')
        shutil.copy2(data/'boundary_pipeline/run01/reference_provenance.json',out/'end_reference_provenance.json')
        settings=dict(model_sha256=job['model']['sha256'],layer=4,k_width=512,distance_space='lsb',
            end_reference=str(out/'end_references.npz'),token_limits=dict(image=[1,280],audio=[1,750]),
            rules={kind:dict(start=start,end=None,soft=soft,handler='kvrecon.media.trained:TrainedMediaHandler')
                   for kind,start,soft in [('image',255999,258880),('audio',256000,258881)]},
            replay_factory='kvrecon.media.gemma_replay:create',
            image=dict(package=str(args.image_package),device='cuda',
                grid_selector_factory='kvrecon.media.grid_collision:create'),
            audio=dict(package=str(args.audio_package),device='cuda',
                griffin_module=str(CODE/'experiment3_audio/audio_recovery/griffin_compat.py'),samples_per_token=640,frames_per_token=4))
        if token_surrogate:settings=None
        save(out/'config.json',dict(case=sid,variant=args.variant,search=asdict(config),public_priority=priority_metadata,public_prefix_ids=prefix,media=settings,
            token_surrogate_regions=dict(start_ids=dict(image=255999,audio=256000),token_limits=dict(image=[1,280],audio=[1,750]),
                end_reference=str(out/'end_references.npz'),text_max_rmse_lsb=30,media_max_rmse_lsb=None) if token_surrogate else None,
            model={k:str(v) if isinstance(v,Path) else v for k,v in job['model'].items()},device=job['device']))
        save(out/'protocol.json',dict(case=sid,selection='Entry selected by fixed manifest order',sample_number=args.sample_number,turns=first['turns'],
            source_cache=str(source),source_sha256=digest(source),observed_positions=len(positions),public_prefix_ids=prefix,
            requested_scope='Whole available conversation cache, including role markers and generated replies; no assistant-start stop',
            variant=args.variant,search_revision='media_aware_marker_first_v3',public_priority=priority_metadata,
            text='Sequential positions; probability50 and unqueried V-nearest78 in the initial128; ridge128 thereafter. Public template/role/media markers have no initial priority.',
            stopping='Before reconstructed media: joint K/V RMSE<2 required, without a query budget. After media: threshold acceptance or two non-improving ridge rounds followed by fallback verification without improvement. Query unmeasured public markers first, then fill the remaining128 batch slots with unmeasured embedding neighbors; reuse prior measurements.',
            cache_advancement='On supported selected-retention backends, preserve each global-best measured KV across search rounds and append the selected slice directly; advancement must match the recorded query codes exactly.',
            media='Infer end from V; reconstruct interior K/V; replay interior; resume at end position',
            image_grid='Enumerate preprocessing-compatible ordered factors of observed N; select minimum joint K/V replay RMSE from the same recovered prefix',
            audio_length='640 samples per valid audio token at 16 kHz; estimated, not original duration',
            limitations=['Boundary locations and image grids are inferred, not guaranteed',
                         'Approximate media and locally stopped text recovery can propagate errors'],
            labels_used_for_search=False,original_media_used=False,reference_table=str(table),reference_identity=identity))
        if token_surrogate:
            protocol=read(out/'protocol.json')
            protocol.update(variant=args.variant,search_revision='token_surrogate_text30_v1',
                text='Every position searches vocabulary IDs; image/audio starts first, then probability50 and V-nearest to fill128; ridge128 thereafter.',
                stopping='Text: joint K/V RMSE<30 LSB AND two non-improving ridge rounds AND fixed127 embedding neighbors give no improvement. In inferred media interiors and continuous end positions: local stagnation accepts nearest measured without an absolute distance ceiling; RMSE<2 may accept immediately.',
                media='No image/audio decoder or encoder: inferred continuous positions are replaced by measured vocabulary-token states. Start markers come from accepted tokens; ends come from observed-V reference scan, never true spans.',
                image_grid=None,audio_length=None,
                limitations=['Continuous positions have no true vocabulary ID; substitutes are not recovered images or waveforms.',
                    'Media spans and resulting per-position thresholds are hypotheses inferred from recovered starts and observed V.',
                    'Prefix substitutions and local stopping may propagate errors.'])
            save(out/'protocol.json',protocol)
        snapshot=out/'source_snapshot';snapshot.mkdir()
        paths=[Path(__file__),ROOT/'search_settings.py',CODE.parent/'kvrecon/engine.py',CODE.parent/'kvrecon/search.py']
        paths+=list((CODE.parent/'kvrecon/media').glob('*.py'))
        for path in paths:shutil.copy2(path,snapshot/path.name)
        save(out/'source_hashes.json',{str(p):digest(p) for p in paths})
        progress(state='loading',case=sid,total_positions=len(positions),position=0)
        def guard(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,Path)):
                p=Path(os.fsdecode(args[0]))
                if any(x in p.parts for x in ('private_eval','evaluation','dataset','datasets')) or p.name in ('validation.json','sample.json'):
                    raise PermissionError('Recovery cannot read source media/dialogue labels')
        sys.addaudithook(guard)
        reference=VReferenceIndex(table,identity,device='cuda');ridge=TorchRidge(job['model']['embeddings'],**job['ridge'])
        with open_backend(job,out/'worker') as backend:
                                                                                      
            backend.reset(identity['reference_prefix_ids'])
            probe_ids=np.linspace(0,backend.vocab_size-1,128,dtype=np.int64)
            measured=backend.query(probe_ids.tolist())
            expected=np.concatenate([np.load(table/'k.npy',mmap_mode='r')[probe_ids],np.load(table/'v.npy',mmap_mode='r')[probe_ids]],1)
            exact=bool(np.array_equal(measured,expected))
            save(out/'reference_probe.json',dict(ids=probe_ids.tolist(),exact=exact,
                max_kv_rmse=float(kv_distance(measured,expected).max()),scope='128 public probes; table is candidate proposal only'))
            if not exact:raise ValueError('Public reference probe mismatch; stopped before recovery')
            router=build_pipeline(settings,backend,ROOT)(positions) if settings is not None else None
            policy=None
            if token_surrogate:
                from kvrecon.media.token_regions import TokenSurrogateRegionPolicy
                from kvrecon.media.boundaries import VBoundaryLocator
                with np.load(out/'end_references.npz',allow_pickle=False) as refs:
                    references={kind:refs[kind].copy() for kind in ('image','audio')}
                locator=VBoundaryLocator(references,512,dict(image=[1,280],audio=[1,750]),'lsb')
                policy=TokenSurrogateRegionPolicy(config,targets,positions,locator,output=out/'token_regions.json')
            def on_event(event):
                kind=event['type']
                if kind=='position_start':
                    extra=dict(token_region_policy=policy.status) if policy is not None else {}
                    progress(state='recovering',position=event['position'],current_queries=0,**extra)
                elif kind=='batch':progress(current_queries=event['queries'],phase=event['phase'])
                elif kind=='position_end':progress(last_completed_position=event['position'],last_distance=event.get('accepted_distance'),status=event['status'])
                elif kind=='media_end':progress(last_completed_position=event['position'],phase='media_end',status=event['status'])
            logged=QueryLog(backend,out/'recovery/queries',positions,codes)
            result=reconstruct(logged,targets,positions,prefix,config,out/'recovery',ridge,
                max_positions=args.max_positions,
                media_router=router,boundaries=None,reference_initializer=reference,live_events=True,event_callback=on_event,
                position_config_provider=policy)
        progress(state='complete' if result['status']=='complete' else 'stopped',status=result['status'],
            covered_positions=result['covered_positions'],attempted_positions=result['attempted_positions'],online_seconds=result['online_seconds'])
        print(json.dumps(state),flush=True)
    except BaseException as error:
        progress(state='failed',error=str(error));save(out/'error.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
