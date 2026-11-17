import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kvrecon.engine import reconstruct
from kvrecon.search import SearchConfig, recover_position


class Backend:
    vocab_size = 24
    batch_capacity = 2
    physical_batch_capacity = 2
    dtype = np.dtype('float32')
    selected_retention_supported = True
    transport = 'socket'
    decoder_seconds = 0.

    def __init__(self, marker_distance=9., corrupt_commit=False):
        self.marker_distance = marker_distance
        self.corrupt_commit = corrupt_commit
        self.history = []
        self.retained = []
        self.begins = 0
        self.reset([])

    def reset(self, prefix):
        self.position = len(prefix)
        self.saved = {}
        self.live = {}
        self.pending_retain = None
        self.selected_retention = False

    def probabilities(self):
        return np.arange(self.vocab_size)

    def begin_candidates(self):
        self.begins += 1
        self.saved = {}
        self.pending_retain = None
        self.selected_retention = False

    def begin_selected_candidates(self):
        self.begin_candidates()
        self.selected_retention = self.selected_retention_supported

    def _flush_selection(self):
        if self.pending_retain is not None:
            token = self.pending_retain
            self.saved = {token: self.live[token].copy()}
            self.pending_retain = None

    def query(self, ids):
                                                                              
                                                                             
        self._flush_selection()
        self.history.append(list(ids))
        values = np.asarray([[{0: 5., 1: 3., 22: self.marker_distance}.get(i, 9.)]
                             for i in ids], dtype=np.float32)
        if len(ids) == 1:
            values += .25
        self.live = {i: row.copy() for i, row in zip(ids, values)}
        if not self.selected_retention:
            self.saved.update(self.live)
        return values

    def values(self, codes):
        return codes.astype(float)

    def retain_candidate(self, token):
        if self.selected_retention:
            assert token in self.live
            self.retained.append(token)
            self.pending_retain = token

    def accept_query(self, token):
        self._flush_selection()
        self.position += 1
        return self.saved[token][None, :] + (1. if self.corrupt_commit else 0.)


class Reference:
    def select(self, target, available, count, space):
        return np.flatnonzero(available)[:count]


class Surrogate:
    def scores(self, *args):
        return np.arange(24, dtype=float)

    def neighbors(self, center, count):
        return np.array([20, 21])


def config(**changes):
    args = dict(method='guided_kv', batch_size=2, probability_seeds=0,
                random_seeds=0, reference_seeds=2, proposals=2,
                merge_initial_batch=True, distance_space='original',
                retain_query_cache=True, ridge_stagnation_rounds=2,
                stagnation_neighbors=2, stagnation_token_ids=(22, 23))
    args.update(changes)
    return SearchConfig(**args)


class Tests(unittest.TestCase):
    def recover(self, backend, **changes):
        return recover_position(backend, [0], np.arange(24), 0, config(**changes),
                                Surrogate(), reference_initializer=Reference())

    def test_early_above_threshold_winner_survives_later_stages_without_replay(self):
        backend = Backend()
        result = self.recover(backend)
        self.assertEqual(result['accepted_token_id'], 1)
        self.assertEqual(result['accepted_codes'], [3.])
        self.assertEqual(backend.retained, [1])
        self.assertEqual(backend.begins, 1)
        self.assertTrue(result['persistent_selected_retention'])
        self.assertTrue(result['approximate'])
        self.assertEqual(result['cache_restore_queries'], 0)
        self.assertEqual(result['queries'], sum(map(len, backend.history)))
        self.assertEqual(result['queries'], result['unique_candidate_queries'])
        self.assertNotIn('fallback_replay', [x['phase'] for x in result['stages']])
        self.assertNotIn([1], backend.history)
                                                                                
        np.testing.assert_array_equal(backend.accept_query(1), [[3.]])

    def test_fallback_marker_improvement_replaces_saved_winner_and_resumes(self):
        backend = Backend(marker_distance=2.)
        result = self.recover(backend)
        self.assertEqual(result['accepted_token_id'], 22)
        self.assertEqual(backend.retained, [1, 22])
        self.assertEqual([x['improved'] for x in result['neighborhood_checks']], [True, False])
        self.assertEqual(result['ridge_rounds'], 4)
        self.assertEqual(result['cache_restore_queries'], 0)
        self.assertEqual(backend.begins, 1)
        np.testing.assert_array_equal(backend.accept_query(22), [[2.]])

    def test_fast_threshold_acceptance_flushes_last_selection(self):
        backend = Backend()
        result = self.recover(backend, epsilon=4)
        self.assertFalse(result['approximate'])
        self.assertEqual(result['stop_reason'], 'threshold')
        self.assertEqual(backend.history, [[0, 1]])
        self.assertEqual(backend.retained, [1])
        np.testing.assert_array_equal(backend.accept_query(1), [[3.]])

    def test_engine_keeps_exact_commit_check_for_approximate_winner(self):
        for corrupt, expected in ((False, 'complete'), (True, 'commit_mismatch')):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as path:
                backend = Backend(corrupt_commit=corrupt)
                result = reconstruct(backend, np.zeros((1, 1)), [0], [], config(),
                                     path, Surrogate(), reference_initializer=Reference())
                row = json.loads((Path(path) / 'trace.jsonl').read_text().splitlines()[0])
                self.assertEqual(result['status'], expected)
                self.assertEqual(row['commit_matches_query'], not corrupt)
                self.assertEqual(row['cache_restore_queries'], 0)

    def test_legacy_capability_does_not_enable_unbounded_cross_stage_retention(self):
        backend = Backend()
        backend.selected_retention_supported = False
        with self.assertRaisesRegex(ValueError, 'changed on independent remeasurement'):
            self.recover(backend)
        self.assertGreater(backend.begins, 1)
        self.assertEqual(backend.history[-1], [1])

    def test_nonretained_verify_winner_reuse_requires_explicit_capability(self):
        backend = Backend()
        backend.query_winner_reuse_supported = True
        result = self.recover(backend, retain_query_cache=False)

        self.assertTrue(result['approximate'])
        self.assertEqual(result['accepted_token_id'], 1)
        self.assertEqual(result['accepted_codes'], [3.])
        self.assertFalse(result['persistent_selected_retention'])
        self.assertEqual(backend.retained, [])
        self.assertEqual(backend.begins, 0)
        self.assertEqual(backend.position, 0)
        self.assertEqual(result['cache_restore_queries'], 0)
        self.assertEqual(result['queries'], sum(map(len, backend.history)))
        self.assertEqual(result['queries'], result['unique_candidate_queries'])
        self.assertNotIn('fallback_replay', [s['phase'] for s in result['stages']])
        self.assertNotIn([1], backend.history)

                                                                            
                                                                          
                                                                              
        legacy = Backend()
        with self.assertRaisesRegex(ValueError, 'changed on independent remeasurement'):
            self.recover(legacy, retain_query_cache=False)
        self.assertEqual(legacy.history[:-1], backend.history)
        self.assertEqual(legacy.history[-1], [1])
        self.assertEqual(sum(ids.count(1) for ids in legacy.history), 2)
        self.assertEqual(legacy.position, 0)


if __name__ == '__main__':
    unittest.main()
