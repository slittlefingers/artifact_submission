
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.engine import reconstruct
from kvrecon.job_config import load_job
from kvrecon.media import MediaRouter, MediaUnavailable
from kvrecon.search import SearchConfig, recover_position


class Backend:
    vocab_size = 24
    batch_capacity = 2
    dtype = np.dtype('float32')
    decoder_seconds = 0.
    query_winner_reuse_supported = True

    def __init__(self, values=None):
        self.position = 1
        self.values_by_position = values or {}
        self.history = []
        self.commits = []

    def reset(self, prefix):
        self.position = len(prefix)

    def probabilities(self):
        return np.arange(self.vocab_size)

    def query(self, ids):
        self.history.append((self.position, list(ids)))
        values = {0: 8., **self.values_by_position.get(self.position, {})}
        return np.asarray([[values.get(int(t), 10.)] for t in ids], dtype=np.float32)

    def values(self, codes):
        return codes.astype(np.float64)

    def commit(self, token):
        values = {0: 8., **self.values_by_position.get(self.position, {})}
        result = np.array([[values.get(token, 10.)]], dtype=np.float32)
        self.commits.append((self.position, token))
        self.position += 1
        return result


class Ridge:
    def __init__(self):
        self.neighbor_calls = 0

    def scores(self, *_args):
        return np.arange(Backend.vocab_size, dtype=float)

    def neighbors(self, *_args):
        self.neighbor_calls += 1
        raise AssertionError('No neighborhood was requested for this test')


def config(**changes):
    values = dict(method='guided', epsilon=.125, batch_size=2,
                  probability_seeds=2, random_seeds=0, proposals=2,
                  distance_space='original', media_aware_stopping=True,
                  ridge_stagnation_rounds=2)
    values.update(changes)
    return SearchConfig(**values)


def run(backend, cfg, count, router=None):
    with tempfile.TemporaryDirectory() as folder:
        result = reconstruct(backend, np.zeros((count, 1)), np.arange(1, count+1),
                             [23], cfg, folder, Ridge(), media_router=router)
        rows = [json.loads(line) for line in (Path(folder)/'trace.jsonl').read_text().splitlines()]
    return result, rows


def router(kind='image', handler_mode='complete', end_consumed=True):
    rules = {kind: dict(start=1, end=22 if end_consumed else None,
                       soft=21, handler='synthetic:replay')}
    spans = [dict(modality=kind, marker_position=1, start=2, end=3, source='test structure')]

    class Handler:
        def __call__(self, backend, request):
            if handler_mode == 'raise':
                raise MediaUnavailable('replay unavailable')
            if handler_mode == 'decoded_only':
                return dict(status='decoded', artifact='recovered.png')
            if handler_mode != 'no_advance':
                backend.position = request['span']['end'] + int(end_consumed)
            return dict(status='complete', end_consumed=end_consumed)

        def finish_end(self, backend, request):
            backend.position += 1
            return dict(status='complete', position=request['position'],
                        queries=1, accepted_distance=4., approximate=True,
                        threshold_passed=False, stop_reason='proposed_media_end_replayed')

    return MediaRouter(rules, spans, np.arange(1, 7), handlers={kind: Handler()})


class MediaAwareStoppingTests(unittest.TestCase):
    def test_strict_phase_ignores_budgets_patience_and_fallbacks(self):
        cfg = config(max_queries=2, max_ridge_rounds=1,
                     stagnation_neighbors=3, stagnation_token_ids=(23,),
                     require_stagnation_for_acceptance=True)
        b = Backend({1: {10: .0625}}); ridge = Ridge()
        row = recover_position(b, [0.], b.probabilities(), 0, cfg, ridge)
        self.assertEqual(row['accepted_token_id'], 10)
        self.assertEqual(row['ridge_rounds'], 5)
        self.assertEqual(row['stop_reason'], 'threshold')
        self.assertFalse(row['approximate'])
        self.assertEqual(row['stopping_phase'], 'strict_pre_media')
        self.assertEqual(row['neighborhood_checks'], [])
        self.assertEqual(ridge.neighbor_calls, 0)
        for key in ('max_queries', 'max_ridge_rounds', 'ridge_stagnation_rounds'):
            self.assertIsNone(row['config'][key])
        self.assertEqual(row['config']['stagnation_neighbors'], 0)
        self.assertEqual(row['config']['stagnation_token_ids'], ())
        self.assertFalse(row['config']['require_stagnation_for_acceptance'])
        self.assertEqual(cfg.max_queries, 2)                                  
        self.assertEqual(cfg.stagnation_neighbors, 3)

    def test_strict_exhaustion_never_accepts_nearest_or_equal_threshold(self):
        b = Backend({1: {0: .125}})
        result, rows = run(b, config(max_queries=2, max_ridge_rounds=1), 2)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['attempted_positions'], 1)
        self.assertEqual(result['recovered_token_ids'], [])
        self.assertEqual(b.commits, [])
        self.assertEqual(rows[0]['queries'], b.vocab_size)
        self.assertIsNone(rows[0]['accepted_token_id'])
        self.assertFalse(rows[0]['approximate'])
        self.assertNotEqual(rows[0]['stop_reason'], 'exhausted_nearest_measured')

    def test_no_media_router_remains_strict_at_every_position(self):
        b = Backend({1: {1: .0625}, 2: {14: .0625}})
        result, rows = run(b, config(), 2)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['recovered_token_ids'], [1, 14])
        self.assertEqual(rows[1]['ridge_rounds'], 7)
        self.assertTrue(all(r['stopping_phase'] == 'strict_pre_media' for r in rows))
        self.assertFalse(result['media_replay_succeeded'])
        self.assertEqual(result['media_stopping_transitions'], [])

    def test_media_start_marker_requires_strict_threshold(self):
        b = Backend({1: {1: .25}})
        result, rows = run(b, config(), 6, router())
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(b.commits, [])
        self.assertEqual(result['media_segments'], [])
        self.assertEqual(rows[0]['queries'], b.vocab_size)
        self.assertFalse(result['media_replay_succeeded'])

    def test_successful_image_or_audio_replay_enables_default_patience_persistently(self):
        for kind in ('image', 'audio'):
            with self.subTest(kind=kind):
                b = Backend({1: {1: .0625}, 5: {2: .0625}})
                result, rows = run(b, config(ridge_stagnation_rounds=None), 6, router(kind))
                self.assertEqual(result['status'], 'complete')
                self.assertEqual([r['position'] for r in rows], [1, 4, 5, 6])
                self.assertEqual([r['stopping_phase'] for r in rows],
                                 ['strict_pre_media', 'after_media', 'after_media', 'after_media'])
                self.assertEqual(rows[0]['stop_reason'], 'threshold')
                self.assertFalse(rows[0]['media_replay_succeeded'])
                self.assertTrue(result['media_replay_succeeded'])
                self.assertEqual(len(result['media_stopping_transitions']), 1)
                self.assertEqual(result['media_stopping_transitions'][0]['resume_position'], 4)
                for row in (rows[1], rows[3]):
                    self.assertEqual(row['ridge_rounds'], 2)
                    self.assertEqual(row['stop_reason'], 'ridge_stagnation')
                    self.assertTrue(row['approximate'])
                    self.assertEqual(row['neighborhood_checks'], [])
                    self.assertEqual(row['config']['stagnation_token_ids'], [])
                self.assertEqual(rows[2]['stop_reason'], 'threshold')
                self.assertEqual(rows[2]['ridge_rounds'], 1)
                self.assertFalse(rows[2]['approximate'])

    def test_post_media_custom_patience_preserves_requested_neighbors_and_markers(self):
        cfg = config(ridge_stagnation_rounds=4, stagnation_neighbors=3,
                     stagnation_token_ids=(23,))
        effective = cfg.for_media_phase(True)
        self.assertEqual(effective.ridge_stagnation_rounds, 4)
        self.assertEqual(effective.stagnation_neighbors, 3)
        self.assertEqual(effective.stagnation_token_ids, (23,))
        b = Backend({1: {1: .0625}})
        _, rows = run(b, config(ridge_stagnation_rounds=4), 4, router())
        self.assertEqual(rows[-1]['ridge_rounds'], 4)

    def test_media_end_replay_keeps_post_media_state(self):
        b = Backend({1: {1: .0625}})
        result, rows = run(b, config(), 4, router(end_consumed=False))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(rows[1]['type'], 'media_end')
        self.assertEqual(rows[1]['stopping_phase'], 'after_media')
        self.assertEqual(rows[2]['stopping_phase'], 'after_media')
        self.assertEqual(rows[2]['ridge_rounds'], 2)

    def test_failed_or_decoded_only_media_never_relaxes_stopping(self):
        for failure in ('raise', 'decoded_only', 'no_advance'):
            with self.subTest(failure=failure):
                b = Backend({1: {1: .0625}})
                result, rows = run(b, config(), 6, router(handler_mode=failure))
                self.assertEqual(result['status'], 'unresolved_media')
                self.assertFalse(result['media_replay_succeeded'])
                self.assertEqual(result['stopping_phase'], 'strict_pre_media')
                self.assertEqual(result['media_stopping_transitions'], [])
                self.assertEqual(len(rows), 1)

    def test_disabled_mode_preserves_legacy_local_acceptance(self):
        b = Backend()
        result, rows = run(b, config(media_aware_stopping=False), 1)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(rows[0]['stop_reason'], 'ridge_stagnation')
        self.assertTrue(rows[0]['approximate'])
        self.assertEqual(rows[0]['stopping_phase'], 'legacy')

    def test_strict_verify_query_cannot_bypass_actual_decode_commit_check(self):
        class VerifyBackend(Backend):
            commit_revalidates_distance = True

            def commit(self, token):
                super().commit(token)
                return np.array([[.25]], dtype=np.float32)

        b = VerifyBackend({1: {0: .0625}})
        result, rows = run(b, config(), 2)
        self.assertEqual(rows[0]['stop_reason'], 'threshold')
        self.assertEqual(rows[0]['accepted_distance'], .0625)
        self.assertEqual(rows[0]['commit_distance'], .25)
        self.assertEqual(result['status'], 'commit_mismatch')
        self.assertEqual(result['recovered_token_ids'], [])
        self.assertEqual(result['attempted_positions'], 1)

    def test_field_validation_and_json_run_job_parsing(self):
        for value in (None, 1, 'true'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config(media_aware_stopping=value).validate()
        with self.assertRaises(ValueError):
            config(method='collision_plus', ridge_stagnation_rounds=None).validate()
        self.assertFalse(SearchConfig().media_aware_stopping)
        self.assertTrue(SearchConfig(**json.loads(json.dumps(asdict(config())))).media_aware_stopping)
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Tokenizer(WordLevel({'public': 0, 'marker': 1}, unk_token='public')).save(str(root/'tokenizer.json'))
            data = dict(schema_version=1, model=dict(layer=4, tokenizer='tokenizer.json',
                        manifest='model.json', embeddings='embeddings.npy'), device={}, ridge={},
                        input_manifest='samples.json', output='out',
                        metadata=dict(public_prefix='public', stop_before=[], markers={
                            name: dict(start='marker', end='marker') for name in ('image','audio','user','assistant','system')}),
                        search=dict(methods=['guided'], **{k: v for k, v in asdict(config()).items()
                                                          if k not in ('method', 'boundary_token_ids')}))
            (root/'job.json').write_text(json.dumps(data), encoding='utf8')
            job = load_job(root/'job.json')
            cfg = job['search_configs']['guided']
            self.assertTrue(cfg.media_aware_stopping)
            self.assertEqual(cfg.ridge_stagnation_rounds, 2)
            result, rows = run(Backend(), cfg, 1)
            self.assertEqual(result['status'], 'unresolved')
            self.assertEqual(rows[0]['stopping_phase'], 'strict_pre_media')


if __name__ == '__main__':
    unittest.main()
