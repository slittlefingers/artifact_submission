
import json
import math
import struct
from pathlib import Path

import numpy as np

from .common import digest
from .litertlm import Bundle
from .observation import save_canonical_observation


def read_entries(path):
    
    entries = {}
    with Path(path).open('rb') as f:
        def take(n):
            data = f.read(n)
            if len(data) != n:
                raise ValueError('Truncated KVEX')
            return data
        header = take(24)
        if header[:4] != b'KVEX':
            raise ValueError('Not a KVEX file')
        version, row, flags, reserved, count = struct.unpack('<IIIII', header[4:])
        if version != 1 or count > 10000:
            raise ValueError('Unsupported KVEX version or tensor count')
        for _ in range(count):
            bank, length = struct.unpack('<BH', take(3))
            name = take(length).decode('ascii')
            size, = struct.unpack('<Q', take(8))
            if size > Path(path).stat().st_size - f.tell():
                raise ValueError('Invalid KVEX tensor length')
            key = (bank, name)
            if key in entries:
                raise ValueError(f'Duplicate KVEX tensor: {key}')
            entries[key] = take(size)
        if f.read(1):
            raise ValueError('Unexpected trailing KVEX bytes')
    return entries, dict(version=version, row=row, flags=flags)


def canonical_rows(entries, sections, layer, slots, bank=1):
    
    inputs = sections['tf_lite_prefill_decode']['signatures']['decode']['inputs']
    tensors = {t['name']: t for t in inputs}
    vectors, quant, layout = [], [], []
    for part, axis in (('k', 2), ('v', 3)):
        name = f'kv_cache_{part}_{layer}'
        if name not in tensors or (bank, name) not in entries:
            raise ValueError(f'Missing retained tensor {name} in model or KVEX bank {bank}')
        spec, raw = tensors[name], entries[bank, name]
        shape = spec['shape']
        if len(shape) != 4 or shape[0] != 1 or any(d <= 0 for d in shape):
            raise ValueError(f'Unsupported Tensor-G5 cache shape: {name}: {shape}')
        if spec['type'] == 0:
            dtype, scale, zero = np.dtype('<f4'), 1., 0
        elif spec['type'] == 9 and len(spec['scale']) == len(spec['zero_point']) == 1:
            dtype = np.dtype('i1')
            scale, zero = spec['scale'][0], spec['zero_point'][0]
        else:
            raise ValueError(f'Unsupported cache dtype/quantization: {name}')
        if len(raw) != math.prod(shape) * dtype.itemsize:
            raise ValueError(f'{name}: KVEX length {len(raw)} does not match deployed model shape {shape}')
                                                                                            
                                                                                                
        for marker in (b'TENSOR PROTECTED', b'LEMON FAIL READ'):
            if marker in raw:
                raise ValueError(f'{name}: LEMON {marker.decode()} filler; reacquire this buffer')
        if np.any(slots < 0) or np.any(slots >= shape[axis]):
            raise ValueError(f'{name}: slot mapping exceeds cache capacity')
        a = np.frombuffer(raw, dtype=dtype).reshape(shape)
        rows = np.moveaxis(a, axis, 0)[slots].reshape(len(slots), -1).copy()
        if not np.isfinite(rows).all():
            raise ValueError(f'{name}: nonfinite observed values')
        vectors.append(rows)
        quant.append(dict(width=rows.shape[1], scale=scale, zero_point=zero,
                          dtype='float32' if dtype.kind == 'f' else 'int8'))
        layout.append(dict(name=name, bank=bank, shape=shape, sequence_axis=axis,
                           dtype=str(dtype), order='C'))
    if vectors[0].dtype != vectors[1].dtype:
        raise ValueError('Mixed K/V storage types are not supported')
    return np.concatenate(vectors, axis=1), quant, layout


def import_kvex(kvex, model, output, prefix, start, count, layer, *,
                linear_cache=False, slot_map=None, bank=1):
    if type(start) is not int or type(count) is not int or start < 0 or count <= 0 or layer < 0:
        raise ValueError('Require nonnegative start/layer and positive count')
    if not isinstance(prefix, list) or any(type(t) is not int or t < 0 for t in prefix):
        raise ValueError('Public prefix must be a JSON array of nonnegative token IDs')
    if layer != 0 and (not prefix or start != len(prefix)):
        raise ValueError('Deep-layer targets must immediately follow a nonempty public prefix')
    if linear_cache == (slot_map is not None):
        raise ValueError('Specify either verified linear-cache mapping or an explicit slot map')
    positions = np.arange(start, start + count, dtype=np.int64)
    if slot_map is not None:
        if not isinstance(slot_map, list) or len(slot_map) != count or any(type(s) is not int for s in slot_map):
            raise ValueError('Slot map must contain one integer physical slot per requested position')
        slots = np.asarray(slot_map, dtype=np.int64)
    else:
        slots = positions.copy()
    if len(set(slots.tolist())) != count:
        raise ValueError('Duplicate cache slots cannot represent distinct retained positions')
    entries, header = read_entries(kvex)
    with Bundle(model) as bundle:
        sections = bundle.model_sections()
    codes, quant, layout = canonical_rows(entries, sections, layer, slots, bank)
    source = dict(kind='KVEX extracted memory image', kvex_sha256=digest(kvex),
                  kvex_header=header, layout=layout, physical_slots=slots.tolist(),
                  position_mapping='explicit slot map' if slot_map is not None else 'caller-confirmed linear cache',
                  note='Position range and public prefix supplied by examiner; no target token IDs used. '
                       'KVEX v1 may have been tail-cleaned by its writer; preserve the original memory image.')
    save_canonical_observation(output, codes, positions, prefix, layer, quant, digest(model), source)
    return dict(output=str(output), layer=layer, positions=count, width=codes.shape[1], dtype=str(codes.dtype))


def add_parser(sub):
    p = sub.add_parser('prepare-kvex', help='Convert extracted KVEX using the exact deployed Tensor-G5 model bundle')
    p.add_argument('--kvex', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix-json', type=Path, required=True, help='Public template token IDs only; [] allowed at layer 0')
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--count', type=int, required=True, help='Verified retained position count; never inferred from zeros')
    p.add_argument('--layer', type=int, required=True, help='Zero-based KV-producing layer')
    p.add_argument('--bank', type=int, default=1)
    mapping = p.add_mutually_exclusive_group(required=True)
    mapping.add_argument('--linear-cache', action='store_true', help='Confirm physical slot equals position for the requested range')
    mapping.add_argument('--slot-map-json', type=Path, help='Physical slots in requested logical-position order')


def run(args):
    prefix = json.loads(args.prefix_json.read_text(encoding='utf-8-sig'))
    slots = json.loads(args.slot_map_json.read_text(encoding='utf-8-sig')) if args.slot_map_json else None
    print(json.dumps(import_kvex(args.kvex, args.model, args.output, prefix, args.start,
                                args.count, args.layer, linear_cache=args.linear_cache,
                                slot_map=slots, bank=args.bank), indent=2))
