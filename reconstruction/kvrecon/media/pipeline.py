
from importlib import import_module
from pathlib import Path
import numpy as np
from . import MediaRouter
from .boundaries import VBoundaryLocator
from .trained import FrozenMediaDecoder, TrainedMediaHandler


def factory(path):
    module,name=path.split(':',1)
    return getattr(import_module(module),name)


def build_pipeline(settings, backend, base_path):
    base=Path(base_path)
    if settings['model_sha256']!=backend.model_sha256 or settings['layer']!=backend.layer or backend.layer!=4 or backend.width!=1024 or np.dtype(backend.dtype)!=np.dtype('int8'):
        raise ValueError('Frozen media packages and end references must match the layer-4 INT8 model')
    path=lambda p:(base/p).resolve()
    reference_path=path(settings['end_reference'])
    with np.load(reference_path,allow_pickle=False) as z:
        references={kind:z[kind].copy() for kind in settings['rules']}
    locator=VBoundaryLocator(references,settings['k_width'],settings['token_limits'],settings['distance_space'])
    replay=factory(settings['replay_factory'])(backend,settings,base) if settings.get('replay_factory') else None
    handlers={}
    for kind,rule in settings['rules'].items():
        if kind not in ('image','audio') or type(rule.get('start')) is not int or rule['start']<0:
            raise ValueError('Media rules require a real nonnegative start token ID')
        opts=settings[kind]
        decoder=FrozenMediaDecoder(path(opts['package']),kind,opts.get('device','cuda'),
            griffin_module=path(opts['griffin_module']) if kind=='audio' else None,
            samples_per_token=opts.get('samples_per_token',640),frames_per_token=opts.get('frames_per_token',4))
        selector=None
        if kind=='image':
            if opts.get('grid_selector_factory'):
                selector=factory(opts['grid_selector_factory'])(backend,opts,base)
            elif opts.get('grid_hw'):
                if not opts.get('grid_source'):raise ValueError('Document the source of fixed grid metadata')
                grid=list(opts['grid_hw']);selector=lambda b,r,g=grid:g
        handlers[kind]=TrainedMediaHandler(decoder,selector,replay)
                                                                         
    return lambda positions:MediaRouter(settings['rules'],[],positions,locator,handlers)
