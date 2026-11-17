
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.engine import reconstruct
from kvrecon.search import SearchConfig, recover_position


class Backend:
    batch_capacity = 2
    dtype = np.dtype('float32')
    decoder_seconds = 0.

    def __init__(self, n=40, winner=None):
        self.vocab_size = n
        self.winner = winner
        self.history = []
        self.commits = []
        self.reset([])

    def reset(self, prefix):
        self.position = len(prefix)

    def probabilities(self):
        return np.arange(self.vocab_size)

    def query(self, ids):
        self.history.append(list(ids))
        return np.asarray([[0. if i == self.winner else 9.] for i in ids], dtype=np.float32)

    def values(self, codes):
        return codes.astype(np.float64)

    def commit(self, token):
        self.commits.append(token)
        raise AssertionError('An unresolved budget must not commit any token')


class Ridge:
    def __init__(self, n):
        self.n = n
        self.calls = 0
        self.neighbor_calls = 0

    def scores(self, *args):
        self.calls += 1
        return np.arange(self.n, dtype=float)

    def neighbors(self, center, count):
        self.neighbor_calls += 1
        return np.arange(self.n-count, self.n)


class Reference:
    def select(self, target, available, count, space):
        return np.flatnonzero(available)[:count]


def config(**changes):
    values = dict(method='guided', epsilon=.125, batch_size=2,
                  probability_seeds=2, random_seeds=0, proposals=2,
                  distance_space='original', max_ridge_rounds=15)
    values.update(changes)
    return SearchConfig(**values)


def run(backend=None, **changes):
    backend = backend or Backend()
    ridge = Ridge(backend.vocab_size)
    result = recover_position(backend, [0.], np.arange(backend.vocab_size),
                              0, config(**changes), ridge,
                              reference_initializer=Reference())
    return backend, ridge, result


class RidgeRoundBudgetTests(unittest.TestCase):
    def test_fifteen_rounds_do_not_accept_the_best_high_distance(self):
        backend, ridge, result = run()
        self.assertEqual(result['stop_reason'], 'ridge_round_budget')
        self.assertEqual(result['ridge_rounds'], 15)
        self.assertEqual(result['stagnant_ridge_rounds'], 15)
        self.assertEqual(ridge.calls, 15)
        self.assertEqual(result['queries'], 2 + 15*2)
        self.assertEqual(len(backend.history), 1 + 15)
        self.assertEqual(result['cache_restore_queries'], 0)
        self.assertEqual(result['neighborhood_checks'], [])
        self.assertIsNone(result['accepted_token_id'])
        self.assertIsNone(result['accepted_codes'])
        self.assertFalse(result['approximate'])
        self.assertFalse(result['threshold_passed'])

    def test_last_permitted_round_can_accept_and_is_counted(self):
        _, ridge, result = run(Backend(winner=31))
        self.assertEqual(result['accepted_token_id'], 31)
        self.assertEqual(result['stop_reason'], 'threshold')
        self.assertEqual(result['ridge_rounds'], 15)
        self.assertEqual(ridge.calls, 15)
        self.assertEqual(result['queries'], 32)
        self.assertFalse(result['approximate'])

    def test_initial_seed_acceptance_needs_no_ridge_round(self):
        for changes in ({}, dict(method='guided_kv', probability_seeds=0,
                                 reference_seeds=2)):
            with self.subTest(changes=changes):
                _, ridge, result = run(Backend(winner=1), **changes)
                self.assertEqual(result['accepted_token_id'], 1)
                self.assertEqual(result['ridge_rounds'], 0)
                self.assertEqual(ridge.calls, 0)
                self.assertEqual(result['queries'], 2)

    def test_split_physical_batches_count_as_one_ridge_round(self):
        backend, ridge, result = run(Backend(winner=6), proposals=5, max_ridge_rounds=1)
        self.assertEqual(backend.history, [[0, 1], [2, 3], [4, 5], [6]])
        self.assertEqual(result['accepted_token_id'], 6)
        self.assertEqual(result['ridge_rounds'], 1)
        self.assertEqual(ridge.calls, 1)

    def test_query_budget_can_expire_during_initial_seed_stages(self):
        _, ridge, result = run(method='guided_kv', probability_seeds=2,
                               reference_seeds=2, random_seeds=2, max_queries=6)
        self.assertEqual(result['stop_reason'], 'resource_budget')
        self.assertEqual(result['ridge_rounds'], 0)
        self.assertEqual(ridge.calls, 0)
        self.assertEqual(result['queries'], 6)
        self.assertIsNone(result['accepted_token_id'])

    def test_query_budget_can_limit_a_partial_ridge_round(self):
        backend, ridge, result = run(proposals=4, max_queries=7)
        self.assertEqual(result['stop_reason'], 'resource_budget')
        self.assertEqual(result['ridge_rounds'], 2)
        self.assertEqual(ridge.calls, 2)
        self.assertEqual(result['queries'], 7)
        self.assertEqual(backend.history[-1], [6])
        self.assertIsNone(result['accepted_token_id'])

    def test_threshold_still_wins_in_query_limited_last_round(self):
        _, _, result = run(Backend(winner=6), proposals=4, max_queries=7)
        self.assertEqual(result['stop_reason'], 'threshold')
        self.assertEqual(result['accepted_token_id'], 6)
        self.assertEqual(result['ridge_rounds'], 2)
        self.assertEqual(result['queries'], 7)

    def test_round_budget_precedes_a_simultaneous_query_budget(self):
        _, _, result = run(max_ridge_rounds=2, max_queries=6)
        self.assertEqual(result['stop_reason'], 'ridge_round_budget')
        self.assertEqual(result['queries'], 6)
        self.assertIsNone(result['accepted_token_id'])

    def test_round_budget_cannot_become_an_exhaustion_fallback(self):
        _, _, result = run(Backend(n=6), max_ridge_rounds=2, ridge_stagnation_rounds=99)
        self.assertEqual(result['stop_reason'], 'ridge_round_budget')
        self.assertEqual(result['queries'], 6)
        self.assertEqual(result['cache_restore_queries'], 0)
        self.assertFalse(result['approximate'])
        self.assertIsNone(result['accepted_token_id'])

    def test_round_budget_precedes_pending_stagnation_neighborhood(self):
        _, ridge, result = run(max_ridge_rounds=2, ridge_stagnation_rounds=2,
                               stagnation_neighbors=2)
        self.assertEqual(result['stop_reason'], 'ridge_round_budget')
        self.assertEqual(result['neighborhood_checks'], [])
        self.assertEqual(ridge.neighbor_calls, 0)
        self.assertIsNone(result['accepted_token_id'])

    def test_default_none_retains_legacy_local_stopping(self):
        self.assertIsNone(SearchConfig().max_ridge_rounds)
        _, ridge, result = run(max_ridge_rounds=None, ridge_stagnation_rounds=2,
                               stagnation_neighbors=2)
        self.assertEqual(result['stop_reason'], 'ridge_stagnation_neighborhood')
        self.assertEqual(ridge.calls, 2)
        self.assertEqual(ridge.neighbor_calls, 1)
        self.assertEqual(result['accepted_token_id'], 0)
        self.assertTrue(result['approximate'])

    def test_default_none_allows_more_than_fifteen_rounds(self):
        _, ridge, result = run(Backend(winner=39), max_ridge_rounds=None)
        self.assertEqual(result['stop_reason'], 'threshold')
        self.assertEqual(result['accepted_token_id'], 39)
        self.assertEqual(result['ridge_rounds'], 19)
        self.assertEqual(ridge.calls, 19)

    def test_json_asdict_exposes_the_optional_budget(self):
        original = config()
        restored = SearchConfig(**json.loads(json.dumps(asdict(original)))).validate()
        self.assertEqual(restored.max_ridge_rounds, 15)
        _, _, result = run(max_ridge_rounds=1)
        self.assertEqual(result['config']['max_ridge_rounds'], 1)
        self.assertIsNone(asdict(SearchConfig())['max_ridge_rounds'])

    def test_invalid_round_budgets_and_nonridge_methods_are_rejected(self):
        for value in (0, -1, True, False, 1.5, '15'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'max_ridge_rounds'):
                config(max_ridge_rounds=value).validate()
        for method in ('exhaustive', 'collision_plus', 'v_nearest', 'v_probability'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'max_ridge_rounds'):
                config(method=method).validate()

    def test_engine_leaves_position_unresolved_without_any_commit(self):
        backend = Backend()
        with tempfile.TemporaryDirectory() as folder:
            result = reconstruct(backend, np.zeros((2, 1)), [0, 1], [],
                                 config(), folder, surrogate=Ridge(backend.vocab_size))
            trace = [json.loads(x) for x in (Path(folder)/'trace.jsonl').read_text().splitlines()]
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['attempted_positions'], 1)
        self.assertEqual(result['recovered_token_ids'], [])
        self.assertEqual(result['recovered_token_positions'], [])
        self.assertEqual(result['covered_positions'], 0)
        self.assertEqual(backend.commits, [])
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]['stop_reason'], 'ridge_round_budget')
        self.assertEqual(trace[0]['ridge_rounds'], 15)


if __name__ == '__main__':
    unittest.main()
