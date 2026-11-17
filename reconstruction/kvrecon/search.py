
from dataclasses import dataclass, asdict, replace
import time
import numpy as np
from .common import code_values, resolve_distance_space


@dataclass(frozen=True)
class SearchConfig:
    method: str = "guided"
    epsilon: float = 0.125
    batch_size: int = 128
    probability_seeds: int = 50
    random_seeds: int = 150
    proposals: int = 20
    seed: int = 20261019
    max_queries: int | None = None
    kv_thresholds: tuple[float, float] | None = None
    k_width: int | None = None
    stop_partition: int = 1
    distance_space: str = "auto"
    distance_metric: str = "rmse"
    boundary_token_ids: tuple[int, ...] = ()
    merge_initial_batch: bool = False
    reference_seeds: int = 0
    retain_query_cache: bool = False
    ridge_stagnation_rounds: int | None = None
    stagnation_neighbors: int = 0
    stagnation_token_ids: tuple[int, ...] = ()
    require_stagnation_for_acceptance: bool = False
    max_ridge_rounds: int | None = None
    media_aware_stopping: bool = False

    def for_media_phase(self, media_replay_succeeded=False):
        




        if not self.media_aware_stopping:
            return self
        if media_replay_succeeded:
            return replace(self, ridge_stagnation_rounds=self.ridge_stagnation_rounds or 2)
        return replace(self, max_queries=None, max_ridge_rounds=None,
                       ridge_stagnation_rounds=None, stagnation_neighbors=0,
                       stagnation_token_ids=(), require_stagnation_for_acceptance=False)

    @property
    def metric(self):
        return "l2" if self.method == "collision_plus_cpa" else self.distance_metric

    def space_for(self, dtype):
                                                                    
        return "original" if self.method == "collision_plus_cpa" else resolve_distance_space(self.distance_space, dtype)

    def validate(self):
        if type(self.retain_query_cache) is not bool:
            raise ValueError('retain_query_cache must be boolean')
        if type(self.merge_initial_batch) is not bool:
            raise ValueError('merge_initial_batch must be boolean')
        guided=self.method in ('guided','guided_kv')
        if type(self.media_aware_stopping) is not bool or (self.media_aware_stopping and not guided):
            raise ValueError('media_aware_stopping requires a boolean and a ridge method')
        if self.max_ridge_rounds is not None and (not guided or type(self.max_ridge_rounds) is not int or self.max_ridge_rounds<1):
            raise ValueError('max_ridge_rounds requires a positive integer and a ridge method')
        if type(self.require_stagnation_for_acceptance) is not bool:
            raise ValueError('require_stagnation_for_acceptance must be boolean')
        if self.require_stagnation_for_acceptance and (not guided or self.ridge_stagnation_rounds is None or self.stagnation_neighbors < 1):
            raise ValueError('Gated acceptance requires ridge patience and neighborhood verification')
        if type(self.stagnation_neighbors) is not int or self.stagnation_neighbors<0 or (self.stagnation_neighbors and self.ridge_stagnation_rounds is None):
            raise ValueError('stagnation_neighbors requires a nonnegative integer and ridge_stagnation_rounds')
        if self.ridge_stagnation_rounds is not None and (not guided or type(self.ridge_stagnation_rounds) is not int or self.ridge_stagnation_rounds<1):
            raise ValueError('ridge_stagnation_rounds requires a positive integer and a ridge method')
        if any(type(t) is not int or t<0 for t in self.stagnation_token_ids) or len(set(self.stagnation_token_ids))!=len(self.stagnation_token_ids):
            raise ValueError('Stagnation token IDs must be unique nonnegative integers')
        if self.stagnation_token_ids and (not guided or self.ridge_stagnation_rounds is None or self.stagnation_neighbors<1):
            raise ValueError('Stagnation token fallback requires ridge patience and neighborhood verification')
        total_seeds=self.probability_seeds+self.random_seeds+self.reference_seeds
        if type(self.reference_seeds) is not int or self.reference_seeds<0:
            raise ValueError('reference_seeds must be a nonnegative integer')
        if (self.method in ('guided_kv','v_probability')) != (self.reference_seeds>0):
            raise ValueError('Positive reference_seeds are required exclusively for guided_kv or v_probability')
        if self.method=='v_probability' and (self.probability_seeds<=0 or self.random_seeds or
                not self.merge_initial_batch or total_seeds!=self.batch_size or self.boundary_token_ids):
            raise ValueError('v_probability requires a full merged probability/V batch without random or boundary priority')
        if self.merge_initial_batch and (not (guided or self.method=='v_probability') or total_seeds>self.batch_size):
            raise ValueError('Merged guided seeds must fit in one candidate batch')
        if self.distance_metric not in ("rmse", "l2"):
            raise ValueError("Unknown KV distance metric")
        if self.distance_space not in ("auto", "lsb", "original"):
            raise ValueError("Unknown KV distance space")
        if self.method not in ("guided","guided_kv","v_nearest","v_probability","collision_plus","collision_plus_modified","collision_plus_cpa","exhaustive"):
            raise ValueError("Unknown search method")
        if self.method=='v_nearest' and (self.probability_seeds or self.random_seeds or self.reference_seeds or self.merge_initial_batch):
            raise ValueError('v_nearest uses full V ranking, with no seed stages')
        if any(type(t) is not int or t<0 for t in self.boundary_token_ids) or len(set(self.boundary_token_ids))!=len(self.boundary_token_ids):
            raise ValueError('Boundary token IDs must be unique nonnegative integers')
        if guided and self.boundary_token_ids and (not self.merge_initial_batch or len(self.boundary_token_ids)>self.batch_size):
            raise ValueError('Guided boundary priority requires a merged initial batch that fits the public markers')
        if self.method=='collision_plus_modified' and not self.boundary_token_ids:
            raise ValueError('collision_plus_modified requires public boundary token IDs')
        if not np.isfinite(self.epsilon) or self.epsilon<=0 or self.batch_size<=0:
            raise ValueError("Invalid epsilon or batch size")
        if guided and (min(self.probability_seeds,self.random_seeds)<0 or total_seeds<1 or self.proposals<=0):
            raise ValueError("Seed counts must be nonnegative with a positive total; proposals must be positive")
        if guided and total_seeds<2:
            raise ValueError("Guided ridge search requires at least two requested initial candidates")
        if self.max_queries is not None and self.max_queries<=0:
            raise ValueError("An explicit resource budget must be positive")
        if self.stop_partition < 1:
            raise ValueError("stop_partition must be positive")
        if self.method=='collision_plus_cpa':
            if self.kv_thresholds is None or len(self.kv_thresholds)!=2 or not np.isfinite(self.kv_thresholds).all() or min(self.kv_thresholds)<0 or not self.k_width or self.k_width<1:
                raise ValueError("CPA requires calibrated K/V thresholds and K width")
        elif self.stop_partition!=1:
            raise ValueError("Probability pruning is only implemented for CPA")
        return self


def measured_values(backend, codes, config):
    return code_values(codes) if config.space_for(np.asarray(codes).dtype) == "lsb" else backend.values(codes)


def kv_distance(values, target, metric="rmse"):
    
    delta=np.asarray(values,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    if delta.shape[-1] == 0:raise ValueError("Empty KV vector")
    if metric not in ("rmse","l2"):raise ValueError("Unknown KV distance metric")
    distance=np.linalg.norm(delta,axis=-1)
    return distance/np.sqrt(delta.shape[-1]) if metric == "rmse" else distance


def acceptance(values, target, config):
    if config.method=='collision_plus_cpa':
        from .cpa import component_distances
        distances=component_distances(values,target,config.k_width)
        tau=np.asarray(config.kv_thresholds)
        return np.all(distances < tau+0.01*tau,axis=-1),distances
    distances=kv_distance(values,target,config.metric)
    return distances<config.epsilon,distances


def lowest_scored_ids(indices, scores, count):
    




    if count<=0:return indices[:0]
    values=scores[indices]
    if count>=len(indices):return indices[np.lexsort((indices,values))]
    boundary=np.partition(values,count-1)[count-1]
    below=indices[values<boundary]
    tied=indices[values==boundary][:count-len(below)]
    selected=np.concatenate((below,tied))
    return selected[np.lexsort((selected,scores[selected]))]


def recover_position(backend, target, order, index, config, surrogate=None, candidate_ids=None, on_batch=None, reference_initializer=None, media_replay_succeeded=False):
    config.validate()
    config=config.for_media_phase(media_replay_succeeded).validate()
    stopping_phase=('after_media' if media_replay_succeeded else 'strict_pre_media') if config.media_aware_stopping else 'legacy'
    n=backend.vocab_size;order=np.asarray(order,dtype=np.int64)
    if order.shape!=(n,) or not np.array_equal(np.sort(order),np.arange(n)):
        raise ValueError("Probability ordering must include the entire original vocabulary exactly once")
    target=np.asarray(target,dtype=np.float64)
    if target.ndim!=1 or not np.isfinite(target).all():
        raise ValueError("Invalid target KV")
    if config.batch_size>getattr(backend,"batch_capacity",128):
        raise ValueError("Requested batch exceeds the compiled model capacity")
    pool=np.arange(n) if candidate_ids is None else np.asarray(candidate_ids)
    if pool.ndim!=1 or pool.dtype.kind not in "iu" or not len(pool) or len(np.unique(pool))!=len(pool) or pool.min()<0 or pool.max()>=n:
        raise ValueError("Candidate pool must contain unique original model token IDs")
    started=time.perf_counter();available=np.zeros(n,dtype=bool);available[pool]=True
    probability_method=config.method in ('collision_plus','collision_plus_modified','collision_plus_cpa')
    guided=config.method in ('guided','guided_kv')
    if config.stagnation_token_ids and max(config.stagnation_token_ids)>=n:
        raise ValueError('Stagnation token ID outside model vocabulary')
    fallback_ids=[t for t in config.stagnation_token_ids if available[t]]
    priority=[]
    if config.method=='collision_plus_modified' or (guided and config.boundary_token_ids):
        if max(config.boundary_token_ids)>=n:raise ValueError('Boundary token ID outside model vocabulary')
        priority=[t for t in config.boundary_token_ids if available[t]]
    if config.method=='collision_plus_modified':
                                                                             
                                                                               
        is_priority=np.zeros(n,dtype=bool);is_priority[priority]=True
        order=np.r_[np.asarray(priority,dtype=np.int64),order[~is_priority[order]]]
    if config.method=='collision_plus_cpa':
                                                                        
                                                       
        ranked=order[available[order]];limit=len(ranked)//config.stop_partition
        stop=((limit+config.batch_size-1)//config.batch_size)*config.batch_size
        available[ranked[stop:]]=False
    queried=[];measured=[];distances=[];stages=[];ranking=0.;forward=0.
    distance_space="original" if config.method=='collision_plus_cpa' else config.distance_space
    top=np.array([],dtype=np.int64);random=np.array([],dtype=np.int64);reference_ids=np.array([],dtype=np.int64);reference_seconds=0.
    reference_order=None
    if config.method=='v_nearest':
        if reference_initializer is None:raise ValueError('v_nearest requires a verified KV reference index')
        tick=time.perf_counter()
        space=config.space_for(np.dtype(getattr(backend,'dtype',np.float64)))
        reference_order=np.asarray(reference_initializer.select(target,available,len(pool),space),dtype=np.int64)
        if reference_order.shape!=(len(pool),) or not np.array_equal(np.sort(reference_order),np.sort(pool)):
            raise ValueError('Reference ranking must contain the candidate pool exactly once')
        reference_seconds=time.perf_counter()-tick;ranking+=reference_seconds
    if guided:
                                                                            
                                                                            
                                                                            
        seed_pool=available.copy();seed_pool[priority]=False
        seed_room=config.batch_size-len(priority) if priority else config.probability_seeds+config.reference_seeds+config.random_seeds
        top=order[seed_pool[order]][:min(config.probability_seeds,seed_room)]
        seed_room-=len(top)
        if config.method=='guided_kv':
            if reference_initializer is None:raise ValueError('guided_kv requires a verified KV reference index')
            remaining_pool=seed_pool.copy();remaining_pool[top]=False
            tick=time.perf_counter()
            space=config.space_for(np.dtype(getattr(backend,'dtype',np.float64)))
            reference_count=min(config.reference_seeds,seed_room)
            reference_ids=np.asarray(reference_initializer.select(target,remaining_pool,reference_count,space),dtype=np.int64) if reference_count else np.array([],dtype=np.int64)
            reference_seconds=time.perf_counter()-tick;ranking+=reference_seconds
            expected=min(reference_count,int(remaining_pool.sum()))
            if (reference_ids.shape!=(expected,) or len(np.unique(reference_ids))!=expected
                    or np.any(reference_ids<0) or np.any(reference_ids>=n) or not remaining_pool[reference_ids].all()):
                raise ValueError('Reference returned duplicate, excluded, or invalid candidate IDs')
        seed_room-=len(reference_ids)
        rest=np.setdiff1d(pool,np.r_[np.asarray(priority,dtype=np.int64),top,reference_ids])
        random=np.random.default_rng(config.seed+index).choice(rest,min(config.random_seeds,len(rest),seed_room),replace=False)
    initial=[("probability",top.tolist()),("random",random.tolist())]
    if config.method=='guided_kv':initial=[('probability',top.tolist()),('reference',reference_ids.tolist()),('random',random.tolist())]
    if config.merge_initial_batch:
        phase='probability_reference_random' if config.method=='guided_kv' else 'probability_random'
        if priority:phase='boundary_'+phase
        initial=[(phase,priority+top.tolist()+reference_ids.tolist()+random.tolist())]
    accepted=None;reason="probability_pruned" if config.stop_partition>1 else "exhausted";passing_ids=[];stage_passing=[];accepted_codes=None;accepted_distance=None
    nearest_id=None;nearest_codes=None;nearest_distance=float('inf');stagnant=0;ridge_rounds=0;approximate=False
    neighbor_pending=False;neighbor_checks=[]
                                                                             
                                                                        
                                                                           
                                                                             
                                                                            
    persistent_selected=bool(config.retain_query_cache and guided and
        config.ridge_stagnation_rounds is not None and
        getattr(backend,'selected_retention_supported',False) and
        getattr(backend,'transport','socket')=='socket' and
        config.batch_size<=getattr(backend,'physical_batch_capacity',getattr(backend,'batch_capacity',0)) and
        callable(getattr(backend,'begin_selected_candidates',None)) and
        callable(getattr(backend,'retain_candidate',None)))
    persistent_started=False;cache_restore_queries=0
    while available.any() or (neighbor_pending and config.require_stagnation_for_acceptance):
        previous_best=nearest_distance
        remaining=int(available.sum())
        if config.max_queries is not None:
            remaining=min(remaining,config.max_queries-len(queried))
        if remaining<=0 and not (neighbor_pending and config.require_stagnation_for_acceptance and not available.any()):
            reason="resource_budget";break
        tick=time.perf_counter()
        if neighbor_pending:
            phase='neighborhood'
            provider=getattr(surrogate,'neighbors',None)
            if not callable(provider):raise ValueError('Neighborhood checking requires surrogate.neighbors(center, count)')
            center=nearest_id
            fixed=np.asarray(provider(center,min(config.stagnation_neighbors,n-1)),dtype=np.int64)
            if fixed.ndim!=1 or len(fixed)!=min(config.stagnation_neighbors,n-1) or len(np.unique(fixed))!=len(fixed) or np.any(fixed<0) or np.any(fixed>=n) or center in fixed:
                raise ValueError('Invalid fixed embedding neighborhood')
                                                                             
                                                                             
            in_pool=np.isin(fixed,pool);fixed=fixed[in_pool]
            neighbor_count=len(fixed)
                                                                          
                                                                           
            fixed=np.asarray(list(dict.fromkeys(fixed.tolist()+fallback_ids)),dtype=np.int64)
            chosen=fixed[available[fixed]].tolist()
            if len(chosen)>remaining:
                reason='resource_budget_before_neighborhood';break
            neighbor_checks.append(dict(center=center,neighbors=neighbor_count,
                fallback_token_ids=fallback_ids,verification_candidates=len(fixed),
                new_queries=len(chosen),reused_queries=len(fixed)-len(chosen)))
            if not chosen:
                neighbor_checks[-1]['improved']=False
                ranking+=time.perf_counter()-tick
                if not config.require_stagnation_for_acceptance or nearest_distance < config.epsilon:
                    reason='ridge_stagnation_neighborhood';approximate=True;break
                neighbor_checks[-1]['acceptance_blocked_by_distance']=True
                neighbor_pending=False;stagnant=0
                continue
        elif config.method=='exhaustive':
            phase='exhaustive';chosen=np.flatnonzero(available)[:min(config.batch_size,remaining)].tolist()
        elif config.method=='v_nearest':
            phase='v_nearest';chosen=reference_order[available[reference_order]][:min(config.batch_size,remaining)].tolist()
        elif config.method=='v_probability':
            if reference_initializer is None:raise ValueError('v_probability requires a verified KV reference index')
            phase='probability_reference'
            prob=order[available[order]][:min(config.probability_seeds,remaining)]
            rest=available.copy();rest[prob]=False
            count=min(config.reference_seeds,remaining-len(prob))
            ref_tick=time.perf_counter()
            space=config.space_for(np.dtype(getattr(backend,'dtype',np.float64)))
            refs=np.asarray(reference_initializer.select(target,rest,count,space),dtype=np.int64)
            reference_seconds+=time.perf_counter()-ref_tick
            if (refs.shape!=(count,) or len(np.unique(refs))!=count or np.any(refs<0)
                    or np.any(refs>=n) or not rest[refs].all()):
                raise ValueError('Reference returned duplicate, excluded, or invalid candidate IDs')
            if not queried:top=prob.copy();reference_ids=refs.copy()
            chosen=prob.tolist()+refs.tolist()
        elif probability_method:
            phase="probability";chosen=order[available[order]][:min(config.batch_size,remaining)].tolist()
            if priority and any(available[t] for t in priority):phase='boundary_priority'
        elif initial:
            phase,chosen=initial.pop(0)
            if not chosen:
                continue
        else:
            if surrogate is None:
                raise ValueError("Guided search requires a surrogate")
            phase="ridge";scores=np.asarray(surrogate.scores(queried,np.asarray(measured),target,np.asarray(distances)))
            if config.metric=="rmse":scores=scores/len(target)                                        
            if scores.shape!=(n,) or not np.isfinite(scores).all():
                raise ValueError("Surrogate must score all original candidates finitely")
            indices=np.flatnonzero(available)
            chosen=lowest_scored_ids(indices,scores,min(config.proposals,remaining)).tolist()
        ranking+=time.perf_counter()-tick
        chosen=chosen[:remaining]
        stage_passing=[]
        retain_selected=None
        if config.retain_query_cache and config.method!='collision_plus_cpa':
            begin=getattr(backend,'begin_candidates',None)
            if not callable(begin) or not callable(getattr(backend,'accept_query',None)):
                raise ValueError('Retained-query advancement requires backend candidate-cache support')
            if persistent_selected:
                if not persistent_started:
                    backend.begin_selected_candidates();persistent_started=True
                retain_selected=backend.retain_candidate
            else:
                                                                           
                                                                                
                if config.batch_size<=getattr(backend,'physical_batch_capacity',getattr(backend,'batch_capacity',0)):
                    lazy=getattr(backend,'begin_selected_candidates',None)
                    keep=getattr(backend,'retain_candidate',None)
                    if callable(lazy) and callable(keep):begin=lazy;retain_selected=keep
                begin()
        for start in range(0,len(chosen),config.batch_size):
            ids=chosen[start:start+config.batch_size]
            if len(set(ids))!=len(ids) or not available[ids].all():
                raise ValueError("Duplicate or already queried candidate")
            tick=time.perf_counter()
            if config.method=='collision_plus_cpa':
                query=getattr(backend,'query_retained',None)
                if not callable(query):raise ValueError("CPA backend must retain candidate cache for acceptance")
                codes=query(ids)
            else:codes=backend.query(ids)
            values=measured_values(backend,codes,config)
            distance_space=config.space_for(np.asarray(codes).dtype)
            forward+=time.perf_counter()-tick
            if values.shape!=(len(ids),len(target)) or not np.isfinite(values).all():
                raise ValueError("Malformed real candidate KV")
            d=kv_distance(values,target,config.metric)
            closest=int(np.argmin(d))
            if float(d[closest])<nearest_distance:
                nearest_id=ids[closest];nearest_codes=codes[closest].copy();nearest_distance=float(d[closest])
                if persistent_selected:
                                                                               
                                                                          
                    tick=time.perf_counter();retain_selected(nearest_id);forward+=time.perf_counter()-tick
            queried.extend(ids)
            if guided:measured.extend(values)
            distances.extend(d.tolist());available[ids]=False
            mask,component_distances=acceptance(values,target,config)
            passing=np.flatnonzero(mask)
            stages.append(dict(phase=phase,queries=len(queried),valid_candidates=len(ids),physical_slots=getattr(backend,"physical_batch_capacity",getattr(backend,"batch_capacity",128)),
                minimum_distance=float(d.min()),distance_metric=config.metric,passing_ids=[ids[i] for i in passing]))
            if config.method=='collision_plus_cpa':
                stages[-1]['kv_distances']=component_distances.tolist()
            if on_batch is not None:
                on_batch(dict(type="batch", index=index, batch=len(stages), phase=phase,
                    distance_space=config.space_for(np.asarray(codes).dtype),distance_metric=config.metric,
                    queries=len(queried), token_ids=list(ids), distances=d.tolist(),
                    passing_ids=[ids[i] for i in passing],
                    kv_distances=component_distances.tolist() if config.method=='collision_plus_cpa' else None,
                    elapsed_seconds=time.perf_counter()-started))
            if len(passing) and not config.require_stagnation_for_acceptance:
                best=int(passing[0] if probability_method else passing[np.argmin(d[passing])])
                stage_passing.extend(ids[i] for i in passing)
                if accepted_distance is None or float(d[best])<accepted_distance:
                    if retain_selected is not None and not persistent_selected:
                        tick=time.perf_counter();retain_selected(ids[best]);forward+=time.perf_counter()-tick
                    accepted=ids[best];accepted_distance=float(d[best]);accepted_codes=codes[best].tolist()
                    passing_ids=[ids[i] for i in passing]
                reason="threshold"
                                                                             
                                                   
                if probability_method:
                    break
        if phase=='ridge':
                                                                            
                                                                            
                                                                         
            ridge_rounds+=1
        if accepted is not None:
            break
        if phase=='ridge':
            stagnant=0 if nearest_distance<previous_best else stagnant+1
                                                                              
                                                                         
                                                           
            if config.max_ridge_rounds is not None and ridge_rounds>=config.max_ridge_rounds:
                reason='ridge_round_budget';break
            if config.ridge_stagnation_rounds is not None and stagnant>=config.ridge_stagnation_rounds:
                if config.stagnation_neighbors:neighbor_pending=True
                else:reason='ridge_stagnation';approximate=True;break
        elif phase=='neighborhood':
            improved=nearest_distance<previous_best
            neighbor_checks[-1]['improved']=improved
            neighbor_pending=False
            if improved:stagnant=0
            elif not config.require_stagnation_for_acceptance or nearest_distance < config.epsilon:
                reason='ridge_stagnation_neighborhood';approximate=True;break
            else:
                neighbor_checks[-1]['acceptance_blocked_by_distance']=True
                stagnant=0
    if (reason!='ridge_round_budget' and not available.any() and accepted is None and
            config.ridge_stagnation_rounds is not None and nearest_id is not None and not config.require_stagnation_for_acceptance):
        reason='exhausted_nearest_measured';approximate=True
    if approximate:
        accepted=nearest_id;accepted_distance=nearest_distance
        if persistent_selected or (not config.retain_query_cache and
                getattr(backend,'query_winner_reuse_supported',False)):
                                                                               
                                                                              
                                                                           
                                                                           
                                                                      
            accepted_codes=nearest_codes.tolist()
        else:
                                                                              
                                                                            
            tick=time.perf_counter()
            if config.retain_query_cache:backend.begin_candidates()
            codes=backend.query([nearest_id])
            if not np.array_equal(codes[0],nearest_codes):
                raise ValueError('Stagnation winner changed on independent remeasurement')
            forward+=time.perf_counter()-tick
            accepted_codes=codes[0].tolist();cache_restore_queries=1
            queried.append(accepted);distances.append(accepted_distance)
            stages.append(dict(phase='fallback_replay',queries=len(queried),valid_candidates=1,
                physical_slots=getattr(backend,'physical_batch_capacity',getattr(backend,'batch_capacity',128)),
                minimum_distance=accepted_distance,distance_metric=config.metric,passing_ids=[]))
    return dict(index=index,distance_space=distance_space,distance_metric=config.metric,accepted_token_id=accepted,accepted_distance=accepted_distance,
        accepted_l2=(accepted_distance*np.sqrt(len(target)) if config.metric=="rmse" else accepted_distance) if accepted_distance is not None else None,
        accepted_codes=accepted_codes,passing_ids_in_accepting_batch=passing_ids,
        passing_ids_in_accepting_stage=stage_passing,
        uniqueness_proven=False,approximate=approximate,threshold_passed=accepted is not None and not approximate,
        distance_threshold_passed=accepted_distance is not None and accepted_distance < config.epsilon,
        acceptance_gate_passed=accepted is not None and config.require_stagnation_for_acceptance,
        ridge_rounds=ridge_rounds,stagnant_ridge_rounds=stagnant,neighborhood_checks=neighbor_checks,
        stop_reason=reason,queries=len(queried),queried_token_ids=queried,
        unique_candidate_queries=len(set(queried)),cache_restore_queries=cache_restore_queries,
        persistent_selected_retention=persistent_selected,
        distances=distances,stages=stages,ranking_seconds=ranking,query_transport_seconds=forward,
        search_seconds=time.perf_counter()-started,candidate_pool_size=len(pool),
        requested_initial_candidates=config.probability_seeds+config.random_seeds+config.reference_seeds if guided or config.method=='v_probability' else 0,
        effective_reference_seeds=len(reference_ids),reference_candidate_ids=reference_ids.tolist(),reference_seconds=reference_seconds,
        reference_top_ids=reference_order[:config.batch_size].tolist() if reference_order is not None else [],
        accepted_reference_rank=int(np.flatnonzero(reference_order==accepted)[0])+1 if reference_order is not None and accepted is not None else None,
        effective_probability_seeds=len(top),effective_random_seeds=len(random),boundary_priority_ids=priority,
        stagnation_fallback_ids=fallback_ids,config=asdict(config),
        stopping_phase=stopping_phase,media_replay_succeeded=bool(media_replay_succeeded))
