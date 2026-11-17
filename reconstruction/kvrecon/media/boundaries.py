
import numpy as np
from . import MediaUnavailable


class VBoundaryLocator:
    def __init__(self, references, k_width, limits, distance_space='lsb'):
        self.references = {kind: np.asarray(v, dtype=np.float64) for kind, v in references.items()}
        self.k_width, self.limits, self.distance_space = k_width, limits, distance_space
        if type(k_width) is not int or k_width < 1:
            raise ValueError('Positive K width required')
        for kind, values in self.references.items():
            if kind not in ('image', 'audio') or values.ndim != 2 or not len(values) or not np.isfinite(values).all():
                raise ValueError('End-marker references must be finite [references,V width] arrays')
            lo, hi = limits[kind]
            if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi:
                raise ValueError('Specify valid media token-count bounds')

    def locate(self, modality, marker_position, targets, positions, config):
        if modality not in self.references:
            raise MediaUnavailable('media_end_reference_missing: '+modality)
                                                                         
        if config.distance_space != self.distance_space:
            raise ValueError('Boundary reference distance space must match explicit search distance space')
        positions = np.asarray(positions)
        if not np.array_equal(positions, np.arange(positions[0], positions[0]+len(positions))):
            raise ValueError('Boundary scan requires contiguous observations')
        start = marker_position+1
        lo, hi = self.limits[modality]
        indices = np.flatnonzero((positions >= start+lo) & (positions <= start+hi))
        if not len(indices):
            raise MediaUnavailable('media_end_outside_observation')
        values = np.asarray(targets, dtype=np.float64)[indices, self.k_width:]
        refs = self.references[modality]
        if values.shape[1] != refs.shape[1] or not np.isfinite(values).all():
            raise ValueError('Boundary V dimension/values mismatch')
                                                                            
        distances = np.concatenate([np.sqrt(((block[:,None]-refs[None])**2).mean(-1)).min(1)
                                    for block in np.array_split(values, max(1,(len(values)+63)//64))])
        order = np.argsort(distances, kind='stable')
        best = int(order[0]); end = int(positions[indices[best]])
        return dict(modality=modality, marker_position=marker_position, start=start, end=end,
                    source='V-nearest end-marker proposal; no true end position supplied'), dict(
                        method='minimum V RMSE over future candidate positions',
                        candidate_positions=len(indices), end_position=end,
                        distance=float(distances[best]),
                        runner_up_distance=float(distances[order[1]]) if len(order)>1 else None,
                        distance_space=self.distance_space, verified=False,
                        note='Similarity selects a boundary proposal; it does not prove marker identity.')
