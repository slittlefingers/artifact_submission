import struct
import tempfile
import unittest
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from kvrecon.kvex_import import canonical_rows, read_entries
from kvrecon.observation import load_observation


class KVEXImportTests(unittest.TestCase):
    def fixture(self, dtype, heads, width):
        inputs, entries, expected = [], {}, []
        for part, axis in (('k', 2), ('v', 3)):
            shape = [1, heads, 5, width] if part == 'k' else [1, heads, width, 5]
            raw = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
            name = f'kv_cache_{part}_0'
            inputs.append(dict(name=name, shape=shape, type=0 if dtype == '<f4' else 9,
                               scale=[0.25], zero_point=[-3]))
            entries[1, name] = raw.tobytes()
            expected.append(np.stack([raw[:, :, s, :].ravel() if part == 'k'
                                      else raw[:, :, :, s].ravel() for s in (4, 1)]))
        sections = {'tf_lite_prefill_decode': {'signatures': {'decode': {'inputs': inputs}}}}
        return entries, sections, np.concatenate(expected, axis=1)

    def test_axes_heads_and_slot_order(self):
        for dtype, heads, width in [('i1', 1, 3), ('<f4', 2, 3), ('<f4', 8, 4)]:
            entries, sections, expected = self.fixture(dtype, heads, width)
            rows, quant, _ = canonical_rows(entries, sections, 0, np.array([4, 1]))
            np.testing.assert_array_equal(rows, expected)
            self.assertEqual(quant[0]['scale'], .25 if dtype == 'i1' else 1.)

    def test_missing_truncated_and_filler(self):
        entries, sections, _ = self.fixture('<f4', 2, 3)
        for replacement in (b'bad', b'TENSOR PROTECTED' + entries[1, 'kv_cache_k_0'][16:]):
            bad = dict(entries); bad[1, 'kv_cache_k_0'] = replacement
            with self.assertRaises(ValueError):
                canonical_rows(bad, sections, 0, np.array([1]))
        entries.pop((1, 'kv_cache_k_0'))
        with self.assertRaises(ValueError):
            canonical_rows(entries, sections, 0, np.array([1]))

    def test_strict_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.kvex'
            header = b'KVEX' + struct.pack('<IIIII', 1, 4096, 1, 0, 1)
            entry = struct.pack('<BH', 1, 1) + b'x' + struct.pack('<Q', 3) + b'abc'
            path.write_bytes(header + entry)
            self.assertEqual(read_entries(path)[0], {(1, 'x'): b'abc'})
            for data in (header + entry[:-1], header + entry + b'junk',
                         b'KVEX' + struct.pack('<IIIII', 1, 4096, 1, 0, 2) + entry + entry):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_entries(path)

    def test_extractor_writer_to_reconstruction(self):
        artifact = Path(__file__).resolve().parents[2]
        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod
        writer = module('kvex_writer_test', artifact / 'extraction/kvextract/kvex.py')
        wrapper = module('extract_run_test', artifact / 'extraction/run.py')
        entries, sections, expected = self.fixture('<f4', 2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            kvex = tmp/'x.kvex'
                                                                                                
            raw = {name: data for (_, name), data in entries.items()}
            raw['kv_cache_c_1'] = b'\x00' * 256 + b'x' * 256
            writer.write_kvex(raw, kvex)
            parsed, header = read_entries(kvex)
            self.assertEqual(header['flags'], 0)
            for name, data in raw.items(): self.assertEqual(parsed[1, name], data)
            (tmp/'prefix.json').write_text('[]')
            (tmp/'slots.json').write_text('[4,1]')
            (tmp/'model').write_bytes(b'model fixture')
            args = SimpleNamespace(output=kvex, model=tmp/'model', observation_output=tmp/'obs',
                                   prefix_json=tmp/'prefix.json', slot_map_json=tmp/'slots.json',
                                   start=0, count=2, layer=0, linear_cache=False, bank=1)
            with patch('kvrecon.kvex_import.Bundle') as bundle:
                bundle.return_value.__enter__.return_value.model_sections.return_value = sections
                wrapper.export_observation(args)
            meta, rows, _, positions, prefix = load_observation(tmp/'obs')
            np.testing.assert_array_equal(rows, expected)
            self.assertEqual(positions.tolist(), [0, 1])
            self.assertEqual(prefix, [])
                                                                              
            for part in ('k','v'):
                raw[f'kv_cache_{part}_4'] = raw[f'kv_cache_{part}_0']
            inputs=sections['tf_lite_prefill_decode']['signatures']['decode']['inputs']
            inputs.extend([dict(t,name=t['name'].replace('_0','_4')) for t in list(inputs)])
            writer.write_kvex(raw,kvex)
            (tmp/'prefix.json').write_text('[1]')
            args.layer=None;args.selected_layers=[0,4];args.start=1
            args.observation_output=tmp/'auto'
            with patch('kvrecon.kvex_import.Bundle') as bundle:
                bundle.return_value.__enter__.return_value.model_sections.return_value=sections
                wrapper.export_observation(args)
            for layer in (0,4):
                meta,rows,*_=load_observation(tmp/'auto'/f'layer{layer}')
                self.assertEqual(meta['layer_index'],layer)
                np.testing.assert_array_equal(rows,expected)
                                                                                            
            for sha,layers in wrapper.DEFAULT_LAYERS.items():
                with patch('kvrecon.common.digest',return_value=sha):
                    self.assertEqual(wrapper.selected_layers(tmp/'renamed-model',None),layers)
            with patch('kvrecon.common.digest',return_value='unknown'):
                with self.assertRaises(ValueError):wrapper.selected_layers(tmp/'model',None)
            self.assertEqual(wrapper.selected_layers(tmp/'model',3),[3])


if __name__ == '__main__':
    unittest.main()
