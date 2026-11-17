
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.search import SearchConfig, recover_position


class Backend:
    vocab_size = 100
    batch_capacity = 2
    dtype = np.dtype('float32')
                                                                            
                                                                           
    query_winner_reuse_supported = True

    def __init__(self, values=None):
        self.distances = {0: 8., **(values or {})}
        self.history = []

    def query(self, ids):
        self.history.append(list(ids))
        return np.asarray([[self.distances.get(token, 10.)] for token in ids], dtype=np.float32)

    def values(self, codes):
        return codes.astype(np.float64)


class Ridge:
    def __init__(self):
        self.calls = 0

    def scores(self, *args):
        self.calls += 1
        return np.arange(Backend.vocab_size, dtype=float)

    def neighbors(self, *_args):
        raise AssertionError('Pure-text patience must not run neighbourhood verification')


def recover(values=None, **changes):
    kwargs = dict(method='guided', epsilon=.125, batch_size=2,
                  probability_seeds=2, random_seeds=0, proposals=2,
                  distance_space='original', retain_query_cache=False,
                  max_ridge_rounds=None, ridge_stagnation_rounds=15,
                  stagnation_neighbors=0, stagnation_token_ids=(),
                  require_stagnation_for_acceptance=False)
    kwargs.update(changes)
    backend = Backend(values)
    ridge = Ridge()
    result = recover_position(backend, [0.], np.arange(backend.vocab_size),
                              0, SearchConfig(**kwargs), ridge)
    return backend, ridge, result


class RidgePatienceTests(unittest.TestCase):
    def test_fifteen_consecutive_nonimproving_rounds_accept_measured_best(self):
        backend, ridge, result = recover()
        self.assertEqual(result['stop_reason'], 'ridge_stagnation')
        self.assertEqual(result['ridge_rounds'], 15)
        self.assertEqual(result['stagnant_ridge_rounds'], 15)
        self.assertEqual(ridge.calls, 15)
        self.assertEqual(result['queries'], 2 + 15*2)
        self.assertEqual(len(backend.history), 1 + 15)
        self.assertEqual(result['accepted_token_id'], 0)
        self.assertEqual(result['accepted_distance'], 8.)
        self.assertEqual(result['accepted_codes'], [8.])
        self.assertTrue(result['approximate'])
        self.assertFalse(result['threshold_passed'])
        self.assertFalse(result['distance_threshold_passed'])
        self.assertEqual(result['cache_restore_queries'], 0)
        self.assertEqual(result['neighborhood_checks'], [])

    def test_improvement_resets_patience_and_allows_more_than_fifteen_total_rounds(self):
                                                                           
                                                                          
        backend, ridge, result = recover({14: 4.})
        self.assertEqual(result['stop_reason'], 'ridge_stagnation')
        self.assertEqual(result['ridge_rounds'], 22)
        self.assertEqual(result['stagnant_ridge_rounds'], 15)
        self.assertEqual(ridge.calls, 22)
        self.assertEqual(result['queries'], 2 + 22*2)
        self.assertEqual(backend.history[7], [14, 15])
        self.assertEqual(result['accepted_token_id'], 14)
        self.assertEqual(result['accepted_distance'], 4.)
        self.assertTrue(result['approximate'])
        self.assertIsNone(result['config']['max_ridge_rounds'])
        self.assertEqual(result['neighborhood_checks'], [])

    def test_improvement_on_the_fifteenth_round_restarts_the_full_patience(self):
                                                                          
                                                                            
        _, ridge, result = recover({30: 4.})
        self.assertEqual(result['ridge_rounds'], 30)
        self.assertEqual(result['stagnant_ridge_rounds'], 15)
        self.assertEqual(ridge.calls, 30)
        self.assertEqual(result['accepted_token_id'], 30)
        self.assertEqual(result['stop_reason'], 'ridge_stagnation')
        self.assertEqual(result['neighborhood_checks'], [])

    def test_threshold_acceptance_is_immediate_for_seeds_or_ridge(self):
        for token, rounds in ((0, 0), (6, 3), (30, 15)):
            with self.subTest(token=token):
                _, ridge, result = recover({token: .0625})
                self.assertEqual(result['stop_reason'], 'threshold')
                self.assertEqual(result['accepted_token_id'], token)
                self.assertEqual(result['ridge_rounds'], rounds)
                self.assertEqual(ridge.calls, rounds)
                self.assertEqual(result['queries'], 2 + rounds*2)
                self.assertFalse(result['approximate'])
                self.assertTrue(result['threshold_passed'])
                self.assertEqual(result['neighborhood_checks'], [])

    def test_equal_distance_does_not_reset_consecutive_patience(self):
                                                                           
                                                                           
        _, ridge, result = recover({14: 8.})
        self.assertEqual(result['ridge_rounds'], 15)
        self.assertEqual(result['stagnant_ridge_rounds'], 15)
        self.assertEqual(ridge.calls, 15)
        self.assertEqual(result['accepted_token_id'], 0)
        self.assertEqual(result['stop_reason'], 'ridge_stagnation')


if __name__ == '__main__':
    unittest.main()
