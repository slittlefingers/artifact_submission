import tempfile,unittest
from pathlib import Path
import numpy as np
from kvrecon.search import SearchConfig,recover_position
from kvrecon.engine import reconstruct

class Backend:
    vocab_size=20;batch_capacity=2;dtype=np.dtype('float32');decoder_seconds=0.
    def __init__(self):self.reset([]);self.history=[]
    def reset(self,prefix):self.position=len(prefix)
    def begin_candidates(self):self.saved={}
    def query(self,ids):
        values=np.asarray([[{0:5,1:6,2:6,3:7,4:3}.get(i,9)] for i in ids],dtype=np.float32)
        self.history.extend(ids)
        for i,x in zip(ids,values):self.saved[i]=x.copy()
        return values
    def values(self,codes):return codes.astype(float)
    def accept_query(self,i):self.position+=1;return np.asarray([self.saved[i]])
    def probabilities(self):return np.arange(self.vocab_size)
class Ref:
    def select(self,target,available,count,space):return np.flatnonzero(available)[:count]
class Ridge:
    def scores(self,*args):return np.arange(20,dtype=float)
def config(**kw):
    return SearchConfig(method='guided_kv',batch_size=2,probability_seeds=0,random_seeds=0,
        reference_seeds=2,proposals=2,retain_query_cache=True,merge_initial_batch=True,
        distance_space='original',ridge_stagnation_rounds=2,**kw)
class Tests(unittest.TestCase):
    def test_gated_small_distance_still_requires_patience_and_neighbors(self):
        class Neighbors(Ridge):
            def neighbors(self,center,count):return np.array([18,19])
        r=recover_position(Backend(),[0],np.arange(20),0,
            config(epsilon=10,stagnation_neighbors=2,require_stagnation_for_acceptance=True),
            Neighbors(),reference_initializer=Ref())
        self.assertEqual(r['accepted_token_id'],4)
        self.assertEqual(r['stagnant_ridge_rounds'],2)
        self.assertFalse(r['neighborhood_checks'][-1]['improved'])
        self.assertTrue(r['acceptance_gate_passed'])
        self.assertTrue(r['distance_threshold_passed'])

    def test_gated_equal_or_high_distance_is_not_accepted_on_exhaustion(self):
        class Neighbors(Ridge):
            def neighbors(self,center,count):return np.array([0,1])
        for epsilon in (3,2):
            r=recover_position(Backend(),[0],np.arange(20),0,
                config(epsilon=epsilon,stagnation_neighbors=2,require_stagnation_for_acceptance=True),
                Neighbors(),reference_initializer=Ref())
            self.assertIsNone(r['accepted_token_id'])
            self.assertEqual(r['unique_candidate_queries'],20)
            self.assertTrue(r['neighborhood_checks'][0]['acceptance_blocked_by_distance'])
            self.assertFalse(r['acceptance_gate_passed'])

    def test_gated_high_distance_continues_to_later_better_candidate(self):
        class LaterBackend(Backend):
            def query(self,ids):
                values=super().query(ids)
                for j,i in enumerate(ids):
                    if i==12:values[j]=1;self.saved[i]=values[j].copy()
                return values
        class Neighbors(Ridge):
            def neighbors(self,center,count):return np.array([18,19])
        r=recover_position(LaterBackend(),[0],np.arange(20),0,
            config(epsilon=2,stagnation_neighbors=2,require_stagnation_for_acceptance=True),
            Neighbors(),reference_initializer=Ref())
        self.assertEqual(r['accepted_token_id'],12)
        self.assertTrue(r['neighborhood_checks'][0]['acceptance_blocked_by_distance'])
        self.assertTrue(r['acceptance_gate_passed'])
        self.assertEqual(len(r['queried_token_ids'][:-1]),len(set(r['queried_token_ids'][:-1])))

    def test_stalls_check_neighbors_and_resume_on_improvement(self):
        class Neighbors(Ridge):
            def neighbors(self,center,count):return np.array([4,19]) if center==0 else np.array([18,19])
                                                                            
                                                                       
        class DelayedRidge(Neighbors):
            def scores(self,*args):
                s=np.arange(20,dtype=float);s[4]=100;return s
        b=Backend();r=recover_position(b,[0],np.arange(20),0,config(stagnation_neighbors=2),DelayedRidge(),reference_initializer=Ref())
        self.assertEqual(r['accepted_token_id'],4)
        self.assertEqual([x['improved'] for x in r['neighborhood_checks']],[True,False])
        self.assertEqual(r['stop_reason'],'ridge_stagnation_neighborhood')
        self.assertEqual(len(b.history[:-1]),len(set(b.history[:-1])))
        self.assertFalse(r['threshold_passed'])

    def test_already_measured_neighbors_are_reused(self):
        class Neighbors(Ridge):
            def neighbors(self,center,count):return np.array([0,1])
        r=recover_position(Backend(),[0],np.arange(20),0,config(stagnation_neighbors=2),Neighbors(),reference_initializer=Ref())
        self.assertEqual(r['neighborhood_checks'][0]['new_queries'],0)
        self.assertEqual(r['neighborhood_checks'][0]['reused_queries'],2)

    def test_incomplete_neighborhood_does_not_claim_local_stop(self):
        class Neighbors(Ridge):
            def neighbors(self,center,count):return np.array([18,19])
        r=recover_position(Backend(),[0],np.arange(20),0,config(stagnation_neighbors=2,max_queries=11),Neighbors(),reference_initializer=Ref())
        self.assertEqual(r['stop_reason'],'resource_budget_before_neighborhood')
        self.assertIsNone(r['accepted_token_id'])

    def test_improvement_resets_patience_and_replays_old_winner(self):
        b=Backend();r=recover_position(b,[0],np.arange(20),0,config(),Ridge(),reference_initializer=Ref())
        self.assertEqual(r['accepted_token_id'],4);self.assertEqual(r['ridge_rounds'],4)
        self.assertEqual(r['stagnant_ridge_rounds'],2);self.assertEqual(r['queries'],11)
        self.assertEqual(b.history[-1],4);self.assertTrue(r['approximate']);self.assertFalse(r['threshold_passed'])
        np.testing.assert_array_equal(b.accept_query(4),[[3]])
    def test_threshold_wins_before_patience(self):
        r=recover_position(Backend(),[0],np.arange(20),0,config(epsilon=4),Ridge(),reference_initializer=Ref())
        self.assertFalse(r['approximate']);self.assertEqual(r['stop_reason'],'threshold');self.assertEqual(r['accepted_token_id'],4)
    def test_engine_continues_with_marked_approximate_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            r=reconstruct(Backend(),np.zeros((2,1)),[0,1],[],config(),Path(d),Ridge(),reference_initializer=Ref())
        self.assertEqual(r['recovered_token_ids'],[4,4]);self.assertEqual(r['approximate_positions'],[0,1])
    def test_reject_nonridge_setting(self):
        with self.assertRaises(ValueError):SearchConfig(method='exhaustive',ridge_stagnation_rounds=2).validate()
if __name__=='__main__':unittest.main()
