
import numpy as np


def compare_v(candidate_v, token_ids, memory_v, *, chunk_size=4096, on_result=None):
    






    candidates = np.asarray(candidate_v)
    ids = np.asarray(token_ids)
    observed = np.asarray(memory_v, dtype=np.float64)
    if observed.ndim == 1:
        observed = observed[None, :]
    if (candidates.ndim != 2 or not len(candidates) or candidates.shape[1] == 0
            or observed.ndim != 2 or observed.shape[1] != candidates.shape[1]):
        raise ValueError('Candidate and memory V must have matching nonzero widths')
    if (ids.ndim != 1 or len(ids) != len(candidates) or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Provide one unique nonnegative integer token ID per candidate')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if not np.isfinite(observed).all():
        raise ValueError('Nonfinite memory V')
    results = []
    for target in observed:
        best = float('inf'); second = float('inf'); nearest = []; second_id = None
                                                                                
        top = []
        for start in range(0, len(ids), chunk_size):
            values = candidates[start:start+chunk_size].astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite candidate V')
            delta = values-target
            squared = np.sum(delta*delta, axis=1)
            if not np.isfinite(squared).all():
                raise ValueError('L2 overflow')
            local_min = float(squared.min())
            local_ids = ids[start:start+len(values)]
            if local_min < best:
                best = local_min
                nearest = local_ids[squared == best].tolist()
            elif local_min == best:
                nearest.extend(local_ids[squared == best].tolist())
            order = np.lexsort((local_ids, squared))[:2]
            top = sorted(top+[(float(squared[j]), int(local_ids[j])) for j in order])[:2]
        nearest = sorted(int(i) for i in nearest)
        if len(top) > 1:
            second, second_id = top[1]
        row = dict(nearest_token_id=nearest[0], nearest_token_ids=nearest,
            unique_nearest=len(nearest)==1,
            unique_nearest_token_id=nearest[0] if len(nearest)==1 else None,
            minimum_l2=float(np.sqrt(best)), second_token_id=second_id,
            second_l2=float(np.sqrt(second)) if len(top)>1 else None,
            l2_margin=float(np.sqrt(second)-np.sqrt(best)) if len(top)>1 else None,
            exact_match=best==0, candidates_compared=len(ids), dimensions=candidates.shape[1])
        results.append(row)
        if on_result is not None:
            on_result(len(results)-1, row)
    return results
