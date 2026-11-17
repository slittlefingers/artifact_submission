
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from image_reconstruction.model import NativeGridReconstructor as _NativeGridReconstructor


class NativeGridReconstructor(_NativeGridReconstructor):
    def __init__(self, input_dim=1024, checkpointing=True):
        super().__init__(input_dim=input_dim, merge=2, checkpointing=checkpointing)
