import unittest

import numpy as np

from kvrecon.search import SearchConfig, recover_position


class Backend:
    vocab_size = 24
    batch_capacity = 2
    dtype = np.dtype('float32')

    def __init__(self):
        self.history = []

    def query(self, ids):
        self.history.extend(ids)
        return np.asarray([[{0: 5, 1: 6, 22: 3}.get(i, 9)] for i in ids], dtype=np.float32)

    def values(self, codes):
        return codes.astype(float)


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
                ridge_stagnation_rounds=2, stagnation_neighbors=2,
                stagnation_token_ids=(22, 23))
    args.update(changes)
    return SearchConfig(**args)


def recover(backend=None, **changes):
    backend = backend or Backend()
    events = []
    result = recover_position(backend, [0], np.arange(24), 0, config(**changes),
                              Surrogate(), on_batch=events.append,
                              reference_initializer=Reference())
    return backend, result, events


class Tests(unittest.TestCase):
    def test_marker_outside_neighbors_is_tested_only_after_stalls(self):
        backend, result, events = recover()
        self.assertEqual(events[0]['token_ids'], [0, 1])
        marker_event = next(x for x in events if 22 in x['token_ids'])
        self.assertEqual(marker_event['phase'], 'neighborhood')
        self.assertEqual(marker_event['token_ids'], [22, 23])
        self.assertTrue(all(len(x['token_ids']) <= 2 for x in events if x['phase']=='neighborhood'))
        self.assertEqual(result['boundary_priority_ids'], [])
        self.assertEqual(result['accepted_token_id'], 22)
        self.assertEqual(result['accepted_distance'], 3)
        self.assertEqual([x['improved'] for x in result['neighborhood_checks']], [True, False])
        self.assertEqual(result['ridge_rounds'], 4)
        self.assertEqual(result['stop_reason'], 'ridge_stagnation_neighborhood')
        self.assertTrue(result['approximate'])
        self.assertEqual(backend.history[-1], 22)                              

    def test_markers_reuse_seed_neighbor_and_previous_fallback_measurements(self):
        backend = Backend()
        backend.batch_capacity = 4
        backend, result, _ = recover(backend, batch_size=4, stagnation_token_ids=(1, 20, 22, 23))
        first, second = result['neighborhood_checks']
        self.assertEqual(first['neighbors'], 2)
        self.assertEqual(first['verification_candidates'], 5)
        self.assertEqual(first['new_queries'], 4)
        self.assertEqual(first['reused_queries'], 1)
        self.assertEqual(second['new_queries'], 0)
        self.assertEqual(second['reused_queries'], 5)
        self.assertEqual(len(backend.history[:-1]), len(set(backend.history[:-1])))

    def test_fast_acceptance_does_not_eagerly_query_fallback_markers(self):
        backend, result, _ = recover(epsilon=6)
        self.assertEqual(backend.history, [0, 1])
        self.assertEqual(result['accepted_token_id'], 0)
        self.assertEqual(result['neighborhood_checks'], [])

    def test_incomplete_marker_union_cannot_claim_local_verification(self):
        backend, result, _ = recover(max_queries=7)
        self.assertIsNone(result['accepted_token_id'])
        self.assertEqual(result['stop_reason'], 'resource_budget_before_neighborhood')
        self.assertNotIn(22, backend.history)

    def test_fallback_respects_candidate_domain(self):
        backend = Backend()
        result = recover_position(backend, [0], np.arange(24), 0, config(),
                                  Surrogate(), candidate_ids=np.array([0, 1, 2, 3, 4, 5, 20, 21, 23]),
                                  reference_initializer=Reference())
        self.assertNotIn(22, backend.history)
        self.assertEqual(result['stagnation_fallback_ids'], [23])
        self.assertEqual(result['accepted_token_id'], 0)

    def test_invalid_fallback_configuration_rejected(self):
        for changes in (dict(stagnation_token_ids=(1, 1)),
                        dict(stagnation_token_ids=(-1,)),
                        dict(stagnation_neighbors=0),
                        dict(ridge_stagnation_rounds=None)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes).validate()
        backend = Backend()
        with self.assertRaisesRegex(ValueError, 'outside model vocabulary'):
            recover(backend, stagnation_token_ids=(24,))
        self.assertEqual(backend.history, [])


if __name__ == '__main__':
    unittest.main()
