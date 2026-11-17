import json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from kvrecon.engine import reconstruct
from test_ridge_stagnation import Backend,Ref,Ridge,config

class Tests(unittest.TestCase):
    def test_media_relaxation_does_not_leak_into_following_text(self):
        class Neighbors(Ridge):
            def neighbors(self,center,count):return np.array([17,18])
        base=config(epsilon=30,stagnation_neighbors=2,require_stagnation_for_acceptance=True)
        calls=[]
        def policy(position,previous_id,previous_position):
            calls.append((position,previous_id,previous_position))
            return replace(base,epsilon=2,require_stagnation_for_acceptance=False) if position==2 else base
        with tempfile.TemporaryDirectory() as folder:
            result=reconstruct(Backend(),np.array([[0.],[100.],[0.]]),[1,2,3],[2],base,folder,
                Neighbors(),reference_initializer=Ref(),position_config_provider=policy)
            rows=[json.loads(l) for l in (Path(folder)/'trace.jsonl').read_text().splitlines()]
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['covered_positions'],3)
        self.assertEqual(result['media_segments'],[])
        self.assertEqual(result['config']['epsilon'],30)
        self.assertEqual([r['config']['require_stagnation_for_acceptance'] for r in rows],[True,False,True])
        self.assertGreater(rows[1]['accepted_distance'],30)
        self.assertLess(rows[0]['accepted_distance'],30)
        self.assertLess(rows[2]['accepted_distance'],30)
        self.assertEqual(calls[0],(1,2,0))
        self.assertEqual(calls[2][1],rows[1]['accepted_token_id'])
        self.assertEqual(calls[2][2],2)

if __name__=='__main__':unittest.main()
