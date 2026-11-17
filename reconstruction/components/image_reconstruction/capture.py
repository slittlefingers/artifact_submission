
import numpy as np

GEMMA_PIXEL_BUDGET = 2520 * 256
GEMMA_PIXELS_PER_TOKEN = 48
GEMMA_MAX_TOKENS_PER_SIDE = 280
IMAGE_SENTINEL_ID = -1
FIRST_TURN_IMAGE_START = 5
LATER_TURN_IMAGE_OFFSET = 4


def gemma_visual_grid(height, width):
    
    factor = np.sqrt(np.float32(GEMMA_PIXEL_BUDGET) / np.float32(height * width))
    gh = int(np.floor(factor * np.float32(height) / GEMMA_PIXELS_PER_TOKEN))
    gw = int(np.floor(factor * np.float32(width) / GEMMA_PIXELS_PER_TOKEN))
    if gh == 0:
        gh, gw = 1, min(int(np.floor(np.float32(width) / height)), GEMMA_MAX_TOKENS_PER_SIDE)
    elif gw == 0:
        gh, gw = min(int(np.floor(np.float32(height) / width)), GEMMA_MAX_TOKENS_PER_SIDE), 1
    return gh, gw


def recorded_grid(arrays, metadata):
    
    if 'grid' in arrays:
        return tuple(int(x) for x in arrays['grid'])
    if 'grid' in metadata:
        return tuple(int(x) for x in metadata['grid'])
    return None


def visual_kv(arrays, metadata, count):
    
    if 'visual_k' in arrays:
        keys, values = arrays['visual_k'], arrays['visual_v']
        if len(keys) != count or len(values) != count:
            raise ValueError(f'Capture holds {len(keys)} visual rows, grid needs {count}')
        return keys, values
    positions = image_positions(arrays, metadata, count)
    if positions is None:
        return arrays['k'], arrays['v']
    stored = arrays['positions']
    index = np.searchsorted(stored, positions)
    if len(positions) != count or np.any(index >= len(stored)) or not np.array_equal(stored[index], positions):
        raise ValueError('Image positions are not present in the capture')
    return arrays['k'][index], arrays['v'][index]


def image_positions(arrays, metadata, count):
    
    if 'image_positions' in metadata:
        return np.asarray(metadata['image_positions'])
    if 'modality_sentinel_ids' in arrays:
        return arrays['modality_positions'][arrays['modality_sentinel_ids'] == IMAGE_SENTINEL_ID]
    if 'turns' in metadata:
        start = conversation_image_start(metadata['turns'])
        return np.arange(start, start + count)
    if len(arrays['k']) == count:
        return None
    raise ValueError('Capture contains non-image rows but no image position metadata')


def conversation_image_start(turns):
    
    image_turn = next(i for i, turn in enumerate(turns) if turn['user'].get('has_image'))
    if image_turn == 0:
        return FIRST_TURN_IMAGE_START
    return turns[image_turn - 1]['state']['step'] + LATER_TURN_IMAGE_OFFSET
