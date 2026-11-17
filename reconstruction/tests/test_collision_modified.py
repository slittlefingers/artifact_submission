import sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.search import SearchConfig,recover_position
from kvrecon.engine import reconstruct
from kvrecon.job_config import BoundaryRules
from test_core import FakeBackend

class ModifiedCollisionTests(unittest.TestCase):
 def cfg(self,**changes):
  return SearchConfig(**dict(dict(method='collision_plus_modified',boundary_token_ids=(8,9),batch_size=4,epsilon=.01),**changes))
 def test_low_rank_marker_is_in_first_batch_and_requires_real_match(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  r=recover_position(b,np.array([9.]),np.arange(10),0,self.cfg())
  self.assertEqual(b.calls,[[8,9,0,1]]);self.assertEqual(r['accepted_token_id'],9)
  self.assertEqual(r['queries'],4);self.assertEqual(r['stages'][0]['phase'],'boundary_priority')
 def test_failed_priority_falls_back_without_duplicate_queries(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  r=recover_position(b,np.array([5.]),np.arange(10),0,self.cfg())
  self.assertEqual(r['queried_token_ids'],[8,9,0,1,2,3,4,5]);self.assertEqual(r['accepted_token_id'],5)
 def test_priority_cannot_bypass_distance_or_candidate_pool(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  r=recover_position(b,np.array([9.]),np.arange(10),0,self.cfg(),candidate_ids=np.arange(5))
  self.assertIsNone(r['accepted_token_id']);self.assertEqual(r['queried_token_ids'],[0,1,2,3,4])
 def test_no_match_still_exhausts_all_candidates_once(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  r=recover_position(b,np.array([20.]),np.arange(10),0,self.cfg())
  self.assertIsNone(r['accepted_token_id']);self.assertEqual(sorted(r['queried_token_ids']),list(range(10)))
 def test_multitoken_delimiter_constituents_are_prioritised_each_position(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  with tempfile.TemporaryDirectory() as d:
   r=reconstruct(b,np.array([[8.],[9.],[2.]]),[1,2,3],[1],self.cfg(),d,boundaries=BoundaryRules({'assistant.start':[8,9]}))
  self.assertEqual(b.calls,[[8,9,0,1],[8,9,0,1]])
  self.assertEqual(r['status'],'boundary_reached');self.assertEqual(b.commits,[8])
 def test_unmodified_order_is_unchanged(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  r=recover_position(b,np.array([9.]),np.arange(10),0,self.cfg(method='collision_plus'))
  self.assertEqual(r['queried_token_ids'],list(range(10)))
 def test_missing_or_invalid_metadata_rejected(self):
  for ids in [(),(1,1),(-1,),('x',)]:
   with self.assertRaises(ValueError):self.cfg(boundary_token_ids=ids).validate()
if __name__=='__main__':unittest.main()
