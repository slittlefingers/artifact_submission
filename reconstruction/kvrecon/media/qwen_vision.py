
import numpy as np
from PIL import Image
PATCH, MERGE, TPS = 16, 2, 2
FACTOR = PATCH * MERGE
POS_SIDE = 48
CAPS = (256, 576, 1024)

def target_size(w, h, max_patches=CAPS[-1], min_patches=CAPS[0]):
    max_px, min_px = max_patches * PATCH * PATCH, min_patches * PATCH * PATCH
    hb, wb = round(h / FACTOR) * FACTOR, round(w / FACTOR) * FACTOR
    if hb * wb > max_px:
        beta = np.sqrt(h * w / max_px)
        hb, wb = max(FACTOR, int(np.floor(h / beta / FACTOR)) * FACTOR), max(FACTOR, int(np.floor(w / beta / FACTOR)) * FACTOR)
    elif hb * wb < min_px:
        beta = np.sqrt(min_px / (h * w))
        hb, wb = int(np.ceil(h * beta / FACTOR)) * FACTOR, int(np.ceil(w * beta / FACTOR)) * FACTOR
    return wb, hb

def patchify(x):
    _, h, w = x.shape
    gh, gw = h // PATCH, w // PATCH
    x = np.broadcast_to(x, (TPS, 3, h, w))
    x = x.reshape(TPS, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return np.ascontiguousarray(x.reshape(gh * gw, 3 * TPS * PATCH * PATCH)), (gh, gw)

def preprocess(img, max_patches=CAPS[-1]):
    img = img.convert('RGB')
    x = np.asarray(img.resize(target_size(*img.size, max_patches), Image.Resampling.BICUBIC), np.float32) / 255.0
    return patchify(((x - 0.5) / 0.5).transpose(2, 0, 1))

def _axis_taps(index, size):
    src = index * (POS_SIDE - 1) / max(size - 1, 1)
    floor = np.floor(src)
    off = np.arange(2)
    taps = np.clip(floor[:, None].astype(np.int64) + off, 0, POS_SIDE - 1)
    weights = np.clip(1 - np.abs(src[:, None] - floor[:, None] - off), 0, None)
    return taps, weights

def grid_inputs(gh, gw):
    i = np.arange(gh * gw)
    blocks_w = gw // MERGE
    row = (i // (MERGE * MERGE * blocks_w)) * MERGE + (i // MERGE) % MERGE
    col = ((i // (MERGE * MERGE)) % blocks_w) * MERGE + i % MERGE
    ht, hw = _axis_taps(row.astype(np.float32), gh)
    wt, ww = _axis_taps(col.astype(np.float32), gw)
    pos_idx = (ht[:, :, None] * POS_SIDE + wt[:, None, :]).reshape(-1, 4)
    pos_w = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
    return pos_idx.astype(np.int32), pos_w.astype(np.float32), np.stack([row, col], -1).astype(np.int32)

def pick_cap(n):
    for cap in CAPS:
        if n <= cap:
            return cap
    raise ValueError(f'{n} patches exceed the largest capacity {CAPS[-1]}')

def encoder_inputs(patches, grid, cap=None):
    n = patches.shape[0]
    cap = cap or pick_cap(n)
    pos_idx, pos_w, positions = grid_inputs(*grid)
    pad = lambda a: np.concatenate([a, np.zeros((cap - n, *a.shape[1:]), a.dtype)])
    valid = np.zeros(cap, np.float32)
    valid[:n] = 1
    kw = dict(patches=pad(patches.astype(np.float32)), pos_idx=pad(pos_idx), pos_w=pad(pos_w), positions=pad(positions), valid=valid)
    return f'vision_encoder_{cap}', kw, n // (MERGE * MERGE)

def mrope_positions(ids, image_pad_id, grid):
    
    ids=np.asarray(ids)
    places=np.flatnonzero(ids == image_pad_id)
    gh,gw=grid
    if len(places)!=gh*gw or not np.all(np.diff(places)==1):
        raise ValueError('Expected one contiguous image token span matching the merged grid')
    start,end=int(places[0]),int(places[-1])+1
    out=np.zeros((3,len(ids)),np.int32)
    out[:,:start]=np.arange(start)
    out[0,start:end]=start
    out[1,start:end]=start+np.repeat(np.arange(gh),gw)
    out[2,start:end]=start+np.tile(np.arange(gw),gh)
    out[:,end:]=np.arange(len(ids)-end)+start+max(gh,gw)
    return out
