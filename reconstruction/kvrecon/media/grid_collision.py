





from pathlib import Path
import math
import numpy as np
from . import MediaUnavailable
from ..common import dump


def factor_grids(n, capacity=280):
    if not 0<n<=capacity:raise MediaUnavailable('visual_token_count_outside_encoder_capacity')
                                                                              
                                                                                  
    return [(h,n//h) for h in range(1,n+1)
            if n%h==0 and (h+1)*(n//h+1)>capacity]


def preprocessed_grid(height,width):
    factor=np.sqrt(np.float32(2520*256)/(np.float32(width)*np.float32(height)))
    h=int(np.floor(factor*np.float32(height)/np.float32(48)))
    w=int(np.floor(factor*np.float32(width)/np.float32(48)))
    if h==0:return (1,min(int(np.floor(np.float32(width)/height)),280))
    if w==0:return (min(int(np.floor(np.float32(height)/width)),280),1)
    return h,w


def replay_dimensions(grid):
    





    h,w=grid
    if preprocessed_grid(h*48,w*48)==(h,w):return h*48,w*48
    low=max(w*w/280,280/(h+1)**2)
    high=min((w+1)**2/280,280/h**2)
    if low>=high:raise MediaUnavailable('grid_has_no_preprocessing_aspect')
    ratio=(low+high)/2
    for scale in (1,2,4,8):
        height=max(48*h,math.ceil(48*w/ratio))*scale
        width=round(height*ratio)
        if preprocessed_grid(height,width)==(h,w):return height,width
    raise MediaUnavailable('cannot_render_candidate_grid')


class GridCollision:
    def recover(self,backend,request,decoder,replay):
        if not callable(replay):raise MediaUnavailable('grid_collision_requires_media_replay')
        out=Path(request['output']);n=len(request['targets'])
        grids=factor_grids(n);rows=[];best=None
        if not grids:raise MediaUnavailable('no_preprocessing_compatible_grid')
        backend.rpc('save_media_prefix',position=backend.position)
        try:
            for h,w in grids:
                backend.rpc('restore_media_prefix')
                folder=out/'grids'/f'{h}x{w}'
                candidate=decoder.decode(np.asarray(request['targets'],dtype=np.int8),folder,[h,w])
                height,width=replay_dimensions((h,w))
                if (height,width)!=(48*h,48*w):
                    from PIL import Image
                    artifact=folder/'replay.png'
                    with Image.open(candidate['artifact']) as image:
                        image.resize((width,height),Image.Resampling.BICUBIC).save(artifact)
                    candidate['replay_artifact']=str(artifact.resolve())
                candidate['replay_dimensions_hw']=[height,width]
                report=replay(candidate,dict(request,output=folder))
                score=report['grid_score_rmse']
                if not np.isfinite(score):raise ValueError('Nonfinite grid score')
                rows.append(dict(grid_hw=[h,w],rmse=score,artifact=candidate['artifact'],
                                 replay_dimensions_hw=[height,width]))
                dump(out/'grid_trials.json',dict(tokens=n,metric='joint K+V RMSE (LSB)',
                     original_dimensions_used=False,candidates=rows))
                if best is None or score<best[0]:best=(score,candidate)
                                                                                  
            backend.rpc('restore_media_prefix')
            result=best[1];result.update(replay(result,request))
            if not np.isclose(result['grid_score_rmse'],best[0],rtol=0,atol=1e-6):
                raise RuntimeError('Winning grid replay changed from its measured trial')
            result['grid_selection']=dict(method='ordered factors + replay collision',
                grid_hw=result['grid_hw'],candidates=len(rows),metric='joint K+V RMSE (LSB)',
                original_dimensions_used=False,original_grid_used=False,
                verified_original=False,score=best[0],
                runner_up=min((r['rmse'] for r in rows if r['grid_hw']!=result['grid_hw']),default=None))
            dump(out/'grid_selection.json',result['grid_selection'])
            dump(out/'prediction.json',result)
            return result
        except BaseException:
            backend.rpc('restore_media_prefix')
            raise
        finally:backend.rpc('release_media_prefix')


def create(backend,settings,base):return GridCollision()
