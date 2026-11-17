import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.kv_reference import VReferenceIndex
from kvrecon.search import SearchConfig,recover_position
from test_core import FakeBackend


class ReferenceTests(unittest.TestCase):
    def test_v_only_replays_fixed_order_without_probability_or_ridge(self):
        from kvrecon.engine import reconstruct
        class NoProbabilityBackend(FakeBackend):
            def probabilities(self):raise AssertionError('Ablation must not request probabilities')
        class Reference:
            def select(self,target,available,count,space):
                return np.flatnonzero(available)[::-1]
        b=NoProbabilityBackend([[9.],[0.],[8.],[7.],[6.]])
        c=SearchConfig(method='v_nearest',probability_seeds=0,random_seeds=0,batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            result=reconstruct(b,np.array([[0.]]),[1],[2],c,folder,reference_initializer=Reference())
            trace=json.loads((Path(folder)/'trace.jsonl').read_text())
        self.assertEqual(b.calls,[[4,3],[2,1]])
        self.assertEqual(result['recovered_token_ids'],[1])
        self.assertEqual(trace['accepted_reference_rank'],4)
        self.assertEqual({s['phase'] for s in trace['stages']},{'v_nearest'})

    def test_v_only_nearest_does_not_bypass_gate_or_budget(self):
        class Reference:
            def select(self,target,available,count,space):return np.flatnonzero(available)[::-1]
        b=FakeBackend(np.full((5,1),9.))
        c=SearchConfig(method='v_nearest',probability_seeds=0,random_seeds=0,batch_size=2,max_queries=3)
        r=recover_position(b,[0.],np.arange(5),0,c,reference_initializer=Reference())
        self.assertEqual(b.calls,[[4,3],[2]])
        self.assertIsNone(r['accepted_token_id'])
        self.assertEqual(r['stop_reason'],'resource_budget')
        with self.assertRaises(ValueError):
            SearchConfig(method='v_nearest',probability_seeds=1,random_seeds=0).validate()

    def test_reference_initialization_keeps_real_gate_and_ridge_fallback(self):
        b=FakeBackend(np.full((400,1),9.))
        class Reference:
            def select(self,target,available,count,space):
                return np.flatnonzero(available)[-count:]
        class Ridge:
            def scores(self,*args):return np.arange(400,dtype=float)
        c=SearchConfig(method='guided_kv',probability_seeds=50,random_seeds=0,
            reference_seeds=78,proposals=128,merge_initial_batch=True,max_queries=256)
        r=recover_position(b,[0.],np.arange(400),0,c,Ridge(),reference_initializer=Reference())
        self.assertEqual([len(x) for x in b.calls],[128,128])
        self.assertEqual(b.calls[0],list(range(50))+list(range(322,400)))
        self.assertEqual(len(set(r['queried_token_ids'])),256)
        self.assertIsNone(r['accepted_token_id'])
        self.assertEqual(r['stop_reason'],'resource_budget')
        self.assertEqual(r['effective_random_seeds'],0)

    def test_reference_cannot_reintroduce_excluded_candidates(self):
        class BadReference:
            def select(self,*args):return [0,0]
        c=SearchConfig(method='guided_kv',probability_seeds=1,random_seeds=0,reference_seeds=2)
        b=FakeBackend(np.zeros((6,1)))
        with self.assertRaisesRegex(ValueError,'duplicate, excluded'):
            recover_position(b,[0.],np.arange(6),0,c,reference_initializer=BadReference())
        self.assertEqual(b.calls,[])

    def test_index_units_masks_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            np.save(p/'v.npy',np.array([[0,0],[2,2],[4,4],[2,2]],dtype=np.int8))
            identity=dict(vocab_size=4,quantization=[dict(width=1),dict(width=2,scale=.5,zero_point=1)])
            metadata=dict(identity=identity,completed=4)
            (p/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
            index=VReferenceIndex(p,dict(identity,model_sha256='unused'),device='cpu')
            mask=np.array([True,False,True,True])
            self.assertEqual(index.select([100,2,2],mask,3,'lsb').tolist(),[3,0,2])
            self.assertEqual(index.select([100,.5,.5],mask,3,'original').tolist(),[3,0,2])
            self.assertEqual(VReferenceIndex(p,device='cpu').n,4)
            with self.assertRaisesRegex(ValueError,'match model'):
                VReferenceIndex(p,dict(identity,vocab_size=5),device='cpu')
            metadata['completed']=2
            (p/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                VReferenceIndex(p,identity,device='cpu')

if __name__=='__main__':unittest.main()
