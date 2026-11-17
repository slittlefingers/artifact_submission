
import math

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

EVALUATION_SIZE = 336
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
SSIM_C1, SSIM_C2 = 0.01 ** 2, 0.03 ** 2


def to_tensor(image):
    
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def resize(x):
    return F.interpolate(x.float(), size=(EVALUATION_SIZE, EVALUATION_SIZE), mode='bilinear', align_corners=False, antialias=True)


def ssim(a, b):
    
    t = torch.arange(SSIM_WINDOW, dtype=torch.float32) - SSIM_WINDOW // 2
    g = torch.exp(-t * t / (2 * SSIM_SIGMA ** 2))
    g /= g.sum()
    window = (g[:, None] * g[None, :])[None, None].expand(3, 1, SSIM_WINDOW, SSIM_WINDOW)
    blur = lambda x: F.conv2d(x, window, groups=3)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    score = ((2 * ma * mb + SSIM_C1) * (2 * cov + SSIM_C2)) / ((ma * ma + mb * mb + SSIM_C1) * (va + vb + SSIM_C2))
    return float(score.mean())


def score(prediction, target):
    mse = float(F.mse_loss(prediction, target))
    return dict(psnr=-10 * math.log10(max(mse, 1e-15)), ssim=ssim(prediction, target), mse=mse)


def to_image(x):
    
    return Image.fromarray((x[0].detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8))


class BilinearView:
    

    name = 'torch bilinear antialias on the float reconstruction'

    @staticmethod
    def prediction(native):
        return resize(native).cpu()

    @staticmethod
    def target(image):
        return resize(to_tensor(image))


class BicubicView:
    

    name = 'PIL bicubic on the saved 8-bit reconstruction'

    @staticmethod
    def prediction(native):
        return BicubicView.target(to_image(native))

    @staticmethod
    def target(image):
        image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        return to_tensor(image.resize((EVALUATION_SIZE, EVALUATION_SIZE), Image.Resampling.BICUBIC))


VIEWS = {'bilinear': BilinearView, 'bicubic': BicubicView}
