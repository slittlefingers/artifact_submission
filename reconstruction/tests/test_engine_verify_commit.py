
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kvrecon.engine import reconstruct
from kvrecon.search import SearchConfig


class CommitBackend:
    

    vocab_size = 2
    batch_capacity = 2
    decoder_seconds = 0.

    def __init__(self, commit_value, capability=None):
        self.commit_value = commit_value
        self.committed = []
        self.retained = []
                                                                             
                                                                                
        if capability is not None:
            self.commit_revalidates_distance = capability

    def reset(self, prefix):
        self.position = len(prefix)

    def probabilities(self):
        return np.arange(self.vocab_size)

    def values(self, codes):
        return np.asarray(codes, dtype=np.float64)

    def commit(self, token):
        self.committed.append(token)
        self.position += 1
        return np.full((1, 2), self.commit_value, dtype=np.float32)

    def accept_query(self, token):
        self.retained.append(token)
        self.position += 1
        return np.full((1, 2), self.commit_value, dtype=np.float32)


class VerifyCommitTests(unittest.TestCase):
    epsilon = .125

    def run_recovery(self, backend, *, retained=False, approximate=True,
                     query_value=.25):
                                                                            
                                                                           
                                                                             
        def winner(_backend, _target, _order, index, *_args, **_kwargs):
            return dict(
                index=index,
                accepted_token_id=1,
                accepted_codes=[query_value, query_value],
                accepted_distance=query_value,
                accepted_l2=float(np.sqrt(2) * query_value),
                approximate=approximate,
                queries=2,
                search_seconds=0.,
                stop_reason='ridge_stagnation' if approximate else 'threshold',
                stages=[],
            )

        config = SearchConfig(
            method='guided', epsilon=self.epsilon, batch_size=2,
            probability_seeds=2, random_seeds=0,
            distance_space='original', retain_query_cache=retained,
        )
        with tempfile.TemporaryDirectory() as folder:
            with patch('kvrecon.engine.recover_position', side_effect=winner):
                result = reconstruct(
                    backend, np.zeros((2, 2)), [1, 2], [0], config, folder,
                )
            rows = [json.loads(line) for line in
                    (Path(folder) / 'trace.jsonl').read_text().splitlines()]
        return result, rows

    def test_verify_commit_within_epsilon_can_advance_despite_query_difference(self):
        backend = CommitBackend(.0625, capability=True)
        result, rows = self.run_recovery(backend)

        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['recovered_token_ids'], [1, 1])
        self.assertEqual(result['recovered_token_positions'], [1, 2])
        self.assertEqual(backend.committed, [1, 1])
        self.assertEqual(backend.retained, [])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertFalse(row['commit_matches_query'])
            self.assertEqual(row['commit_distance'], .0625)
            self.assertLess(row['commit_distance'], self.epsilon)

    def test_threshold_verify_commit_at_or_above_epsilon_stops(self):
                                                                             
                                                                           
        for commit_value in (self.epsilon, .25):
            with self.subTest(commit_value=commit_value):
                backend = CommitBackend(commit_value, capability=True)
                result, rows = self.run_recovery(
                    backend, approximate=False, query_value=.03125,
                )

                self.assertEqual(result['status'], 'commit_mismatch')
                self.assertEqual(result['recovered_token_ids'], [])
                self.assertEqual(result['attempted_positions'], 1)
                self.assertEqual(backend.committed, [1])
                self.assertEqual(len(rows), 1)
                self.assertGreaterEqual(rows[0]['commit_distance'], self.epsilon)
                self.assertFalse(rows[0]['commit_matches_query'])

    def test_approximate_verify_commit_above_epsilon_can_advance(self):
                                                                           
                                                                            
                                                                        
        backend = CommitBackend(.5, capability=True)
        result, rows = self.run_recovery(backend, query_value=.25)

        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['recovered_token_ids'], [1, 1])
        self.assertEqual(result['approximate_positions'], [1, 2])
        self.assertEqual(backend.committed, [1, 1])
        for row in rows:
            self.assertTrue(row['approximate'])
            self.assertTrue(row['commit_revalidated_by_distance'])
            self.assertFalse(row['commit_matches_query'])
            self.assertEqual(row['accepted_distance'], .25)
            self.assertEqual(row['commit_distance'], .5)

    def test_nonfinite_verify_commit_is_rejected_for_either_acceptance_policy(self):
        for approximate in (False, True):
            for commit_value in (np.nan, np.inf, -np.inf):
                with self.subTest(approximate=approximate, commit_value=commit_value):
                    backend = CommitBackend(commit_value, capability=True)
                    with self.assertRaisesRegex(ValueError, 'Nonfinite verify/decode commit K/V'):
                        self.run_recovery(backend, approximate=approximate, query_value=.03125)
                    self.assertEqual(backend.committed, [1])

    def test_legacy_approximate_commit_difference_still_stops(self):
        for capability in (None, False):
            with self.subTest(capability=capability):
                backend = CommitBackend(.0625, capability=capability)
                result, rows = self.run_recovery(backend)

                self.assertEqual(result['status'], 'commit_mismatch')
                self.assertEqual(result['recovered_token_ids'], [])
                self.assertEqual(backend.committed, [1])
                self.assertEqual(len(rows), 1)
                self.assertFalse(rows[0]['commit_matches_query'])
                self.assertLess(rows[0]['commit_distance'], self.epsilon)

    def test_retained_commit_remains_exact_despite_verify_capability(self):
                                                                        
                                                                             
        backend = CommitBackend(.0625, capability=True)
        result, rows = self.run_recovery(
            backend, retained=True, approximate=False, query_value=.03125,
        )

        self.assertEqual(result['status'], 'commit_mismatch')
        self.assertEqual(result['recovered_token_ids'], [])
        self.assertEqual(backend.committed, [])
        self.assertEqual(backend.retained, [1])
        self.assertFalse(rows[0]['commit_matches_query'])
        self.assertLess(rows[0]['commit_distance'], self.epsilon)


if __name__ == '__main__':
    unittest.main()
