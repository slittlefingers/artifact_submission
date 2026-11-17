
from contextlib import nullcontext
import json

import numpy as np
import torch

from .model import NativeGridReconstructor
from .storage import Store

STATISTICS_KEYS = (('mean', 'std'), ('vmean', 'vstd'))


class Reconstructor:
    

    def __init__(self, package, device=None):
        self.store = Store(package)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        with self.store.open('best.pt') as stream:
            state = torch.load(stream, map_location='cpu', weights_only=True)
        weights = state['model']
        self.epoch = state.get('epoch')
        self.input_dim = weights['input.weight'].shape[1]
        self.merge = round((weights['inverse_merger.weight'].shape[0] // 256) ** 0.5)
        self.model = NativeGridReconstructor(self.input_dim, self.merge, checkpointing=False)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()
        self.mean, self.std = self._statistics()
        self.quantization = json.loads(self.store.read('quantization.json')) if 'quantization.json' in self.store else None
        self.checkpoint_sha256 = self.store.sha256('best.pt')

    @property
    def precision(self):
        return 'CUDA BF16 autocast' if self.device.type == 'cuda' else 'FP32'

    def features(self, keys, values):
        
        if self.input_dim == values.shape[1]:
            parts = [(values, 1)]
        elif self.input_dim == keys.shape[1] + values.shape[1]:
            parts = [(keys, 0), (values, 1)]
        else:
            raise ValueError(f'Checkpoint expects {self.input_dim} features; capture has K{keys.shape[1]}/V{values.shape[1]}')
        x = np.concatenate([self._dequantize(array, index) for array, index in parts], axis=1)
        if not np.isfinite(x).all():
            raise ValueError('Non-finite visual K/V')
        return x

    def reconstruct(self, features, grid):
        
        normalized = ((features - self.mean) / self.std).astype(np.float32, copy=False)
        x = torch.from_numpy(normalized)[None].to(self.device)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
        with torch.inference_mode(), autocast:
            image = self.model(x, tuple(grid)).float()
        px = self.model.pixels_per_token
        if image.shape != (1, 3, grid[0] * px, grid[1] * px) or not torch.isfinite(image).all():
            raise ValueError(f'Unexpected reconstruction {tuple(image.shape)} for grid {tuple(grid)}')
        return image

    def describe(self):
        return dict(checkpoint=str(self.store.path), checkpoint_sha256=self.checkpoint_sha256, epoch=self.epoch,
                    input_dim=self.input_dim, merge=self.merge, pixels_per_token=self.model.pixels_per_token,
                    quantization=self.quantization, precision=self.precision, device=str(self.device))

    def _statistics(self):
        with np.load(self.store.buffer('stats.npz'), allow_pickle=False) as stats:
            for mean_key, std_key in STATISTICS_KEYS:
                if mean_key in stats and std_key in stats:
                    mean, std = stats[mean_key], stats[std_key]
                    break
            else:
                raise KeyError(f'stats.npz must contain one of {STATISTICS_KEYS}')
        if mean.shape != (self.input_dim,) or std.shape != (self.input_dim,) or not np.all(std > 0):
            raise ValueError('Normalization statistics do not match the checkpoint input')
        return mean, std

    def _dequantize(self, array, index):
        if self.quantization is None:
            return array.astype(np.float32)
        spec = self.quantization[index]
        if array.dtype != np.dtype(spec['dtype']) or array.shape[1] != spec['width']:
            raise ValueError(f"Expected stored {spec['dtype']}[{spec['width']}] K/V, got {array.dtype}{list(array.shape)}")
        return (array.astype(np.float32) - spec['zero_point']) * spec['scale']
