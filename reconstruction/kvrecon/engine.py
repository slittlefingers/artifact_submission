import json
from pathlib import Path
import time
from dataclasses import asdict
import numpy as np
from .common import dump
from .search import kv_distance, recover_position, acceptance, measured_values
from .media import MediaUnavailable


def _reconstruct(backend, targets, positions, prefix, config, output, surrogate=None, max_positions=None, candidate_ids=None, media_router=None, emit=None, boundaries=None, reference_initializer=None, position_config_provider=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/"trace.jsonl").exists() or (output/"result.json").exists():
        raise FileExistsError("Refusing to overwrite an existing recovery")
    if media_router and any(media_router.kind(t) or media_router.kind(t,'soft') for t in prefix):
        raise ValueError('Media in public prefix requires representation replay; include its start marker in recovery targets instead')
    positions=np.asarray(positions,dtype=np.int64)
    base_config=config
    media_replay_succeeded=False
    media_stopping_transitions=[]
    if position_config_provider is not None:
        if not callable(position_config_provider) or media_router is not None:
            raise ValueError('Position search policy requires a callable and no media decoder router')
    if emit:
        emit(dict(type="start", config=asdict(config), target_positions=len(positions),
            planned_positions=min(len(positions), max_positions) if max_positions else len(positions)))
    started=time.perf_counter();backend.reset(prefix);prefix_seconds=time.perf_counter()-started
    recovered=[];recovered_positions=[];media_segments=[];approximate_positions=[];status="complete";count=len(positions)
    expected_end=None;pending_media_end=None;index=0;attempted=0;covered=0;boundary=None;media_end_positions=[]
    if max_positions is not None:
        if max_positions<=0:
            raise ValueError("max_positions must be positive")
        count=min(count,max_positions)
    started=time.perf_counter();decoder_start=backend.decoder_seconds
    with (output/"trace.jsonl").open("w") as f:
        while index<count:
            position=int(positions[index])
            attempted+=1;next_index=index+1
            if backend.position!=position:
                raise ValueError("Reconstructed prefix position disagrees with target")
            if position_config_provider is not None:
                previous_id=recovered[-1] if recovered else (prefix[-1] if prefix else None)
                previous_position=recovered_positions[-1] if recovered_positions else position-1
                config=position_config_provider(position,previous_id,previous_position).validate()
            else:
                config=base_config
            config=config.validate().for_media_phase(media_replay_succeeded).validate()
            stopping_phase=('after_media' if media_replay_succeeded else 'strict_pre_media') if config.media_aware_stopping else 'legacy'
            if emit:emit(dict(type="position_start", index=index, position=position,
                             stopping_phase=stopping_phase,media_replay_succeeded=media_replay_succeeded))
            if pending_media_end is not None:
                tick=time.perf_counter()
                try:
                    end_result=media_router.finish_end(backend,pending_media_end,targets[index],config,
                        output/'media'/str(pending_media_end['start']-1))
                    row=dict(end_result,index=index,type='media_end',accepted_token_id=None,
                        search_seconds=time.perf_counter()-tick,distance_metric=config.metric,
                        probability_seconds=0.,ranking_seconds=0.,stages=[])
                    covered+=1;media_end_positions.append(position)
                    if row.get('approximate'):approximate_positions.append(position)
                    pending_media_end['end_replay']=end_result
                    pending_media_end=None
                except MediaUnavailable as error:
                    status='unresolved_media';row=dict(index=index,position=position,type='media_end',
                        accepted_token_id=None,media_error=str(error),search_seconds=time.perf_counter()-tick,
                        queries=0,probability_seconds=0.,ranking_seconds=0.,stages=[])
                row.update(stopping_phase=stopping_phase,media_replay_succeeded=media_replay_succeeded,
                           config=asdict(config))
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                if emit:emit(dict(type='media_end',index=index,position=position,status=status,result=row))
                print(f"position={position} media_end RMSE={row.get('accepted_distance')}",flush=True)
                if status!='complete':break
                index+=1
                continue
            tick=time.perf_counter()
            needs_probability=not (config.method in ('exhaustive','v_nearest') or
                (config.method=='guided_kv' and config.probability_seeds==0))
            order=backend.probabilities() if needs_probability else np.arange(backend.vocab_size)
            probability_seconds=time.perf_counter()-tick
            row=recover_position(backend,targets[index],order,index,config,surrogate,candidate_ids,on_batch=emit,reference_initializer=reference_initializer,media_replay_succeeded=media_replay_succeeded)
            row.update(position=position,probability_seconds=probability_seconds)
            if row.get('approximate'):approximate_positions.append(position)
            if row["accepted_token_id"] is None:
                status="unresolved"
            elif boundaries and (boundary:=boundaries.match(prefix+recovered+[row['accepted_token_id']],position)):
                boundary['verified_by_kv']=not any(boundary['start_position']<=p<boundary['end_position'] for p in approximate_positions)
                                                                                        
                                                                                                  
                status='boundary_reached';row['boundary']=boundary
            elif media_router and media_router.kind(row['accepted_token_id'],'soft'):
                status='unresolved_media'
                row['media_error']='Soft media slots cannot be committed as text token IDs'
            elif expected_end is not None and row['accepted_token_id']!=expected_end:
                status='unresolved_media'
                row['media_error']='Recovered media end marker disagrees with the selected model rule'
            elif media_router and position in media_router.spans and media_router.kind(row['accepted_token_id'])!=media_router.spans[position]['modality']:
                status='unresolved_media'
                row['media_error']='Recovered marker disagrees with supplied media span'
            else:
                tick=time.perf_counter()
                retained=config.method=='collision_plus_cpa' or config.retain_query_cache
                commit=backend.accept_query if retained else backend.commit
                codes=commit(row["accepted_token_id"])
                actual=measured_values(backend,codes,config)[0]
                revalidate=bool(getattr(backend,'commit_revalidates_distance',False)) and not retained
                if revalidate and not np.isfinite(actual).all():
                    raise ValueError('Nonfinite verify/decode commit K/V')
                distance=float(kv_distance(actual,targets[index],config.metric))
                row.update(commit_seconds=time.perf_counter()-tick,commit_distance=distance,commit_l2=float(np.linalg.norm(actual-targets[index])),
                    commit_matches_query=bool(np.array_equal(codes[0],np.asarray(row["accepted_codes"],dtype=codes.dtype))))
                                                                                
                valid,components=acceptance(actual,targets[index],config)
                if config.method=='collision_plus_cpa':row['accepted_kv_l2']=components.tolist()
                                                                             
                                                                              
                                                                             
                                                                               
                row['commit_revalidated_by_distance']=revalidate
                mismatch=((not valid and not row.get('approximate',False)) if revalidate else
                    ((not valid and not row.get('approximate',False)) or
                     ((retained or row.get('approximate',False)) and not row['commit_matches_query'])))
                if mismatch:
                    status="commit_mismatch"
                else:
                    recovered.append(row["accepted_token_id"])
                    recovered_positions.append(position);covered+=1;expected_end=None
                    if media_router and media_router.kind(row['accepted_token_id']):
                        try:
                            span=media_router.spans.get(position)
                            if span and span['end']>int(positions[count-1]):
                                raise MediaUnavailable('media_position_limit: whole span and end marker must fit the position budget')
                            segment=media_router.recover(backend,row['accepted_token_id'],position,
                                targets[:count],positions[:count],config,output/'media'/str(position))
                            resume=segment.get('resume_position',segment['end'])
                            if config.media_aware_stopping:
                                                                               
                                                                             
                                                                               
                                                                               
                                if segment.get('status')!='complete' or backend.position!=resume or resume<=segment['start']:
                                    raise MediaUnavailable('media_replay_not_completed_for_stopping_transition')
                                if not media_replay_succeeded:
                                    transition=dict(marker_position=position,resume_position=int(resume),
                                        modality=segment['modality'],from_phase='strict_pre_media',to_phase='after_media')
                                    media_stopping_transitions.append(transition)
                                    row['stopping_transition']=transition
                                media_replay_succeeded=True
                            media_segments.append(segment);row['media']=segment
                            next_index=int(np.searchsorted(positions,resume))
                            covered+=resume-segment['start']
                            expected_end=None if segment.get('end_consumed') else media_router.rules[segment['modality']]['end']
                            if not segment.get('end_consumed') and expected_end is None:
                                pending_media_end=segment
                        except MediaUnavailable as error:
                            status='unresolved_media';row['media_error']=str(error)
            f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
            if emit:
                emit(dict(type="position_end", index=index, position=position,
                    accepted_token_id=row["accepted_token_id"], accepted_l2=row["accepted_l2"],
                    accepted_distance=row["accepted_distance"],distance_metric=config.metric,
                    commit_l2=row.get("commit_l2"),commit_distance=row.get("commit_distance"), committed=position in recovered_positions,
                    queries=row["queries"], stop_reason=row["stop_reason"], status=status,
                    stopping_phase=stopping_phase,media_replay_succeeded=media_replay_succeeded,
                    search_seconds=row["search_seconds"], stages=row["stages"]))
            print(f"position={position} accepted={row['accepted_token_id']} queries={row['queries']} {config.metric.upper()}={row['accepted_distance']}",flush=True)
            if status!="complete":
                break
            index=next_index
    if status=="complete" and count<len(positions):
        status="position_limit"
    result=dict(status=status,recovered_token_ids=recovered,target_positions=len(positions),
        attempted_positions=attempted,covered_positions=covered,config=asdict(base_config),
        position_config_provider_active=position_config_provider is not None,
        media_aware_stopping=base_config.media_aware_stopping,media_replay_succeeded=media_replay_succeeded,
        stopping_phase=('after_media' if media_replay_succeeded else 'strict_pre_media') if base_config.media_aware_stopping else 'legacy',
        media_stopping_transitions=media_stopping_transitions,
        recovered_token_positions=recovered_positions,media_segments=media_segments,media_end_positions=media_end_positions,boundary=boundary,
        public_prefix_tokens=len(prefix),public_prefix_seconds=prefix_seconds,
        online_seconds=time.perf_counter()-started,decoder_seconds=backend.decoder_seconds-decoder_start,
        own_recovered_prefix=True,target_cache_used_as_prefix=False,approximate_positions=approximate_positions,
        correctness="May include explicitly marked nearest-measured stagnation fallbacks; exact identity requires separate evaluation",uniqueness_proven=False)
    dump(output/"result.json",result)
    if emit:emit(dict(type="finish", result=result))
    return result


def reconstruct(backend, targets, positions, prefix, config, output, surrogate=None,
                max_positions=None, candidate_ids=None, media_router=None, live_events=False,boundaries=None,reference_initializer=None,event_callback=None,position_config_provider=None):
    
    from contextlib import nullcontext
    from .events import EventLog
    context=EventLog(Path(output)/"events.jsonl") if live_events else nullcontext(None)
    with context as emit:
        if event_callback is not None:
            logged_emit=emit
            def emit(event):
                if logged_emit:logged_emit(event)
                event_callback(event)
        return _reconstruct(backend, targets, positions, prefix, config, output, surrogate,
            max_positions, candidate_ids, media_router, emit,boundaries,reference_initializer,position_config_provider)
