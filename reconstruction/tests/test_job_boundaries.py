import sys,tempfile,unittest,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.engine import reconstruct
from kvrecon.job_config import BoundaryRules,load_job
from kvrecon.search import SearchConfig
from test_core import FakeBackend

class BoundaryTests(unittest.TestCase):
 def test_empty_stop_list_recovers_through_turn_markers(self):
  from unittest.mock import patch
  path=Path(__file__).resolve().parents[1]/'components/experiment5_gemma_layer4_multimodal/configs/text_prefix_layer4.json'
  config=json.loads(path.read_text(encoding='utf-8'));config['metadata']['stop_before']=[]
  original=Path.read_text
  def read(p,*a,**kw):
   return json.dumps(config) if p.resolve()==path.resolve() else original(p,*a,**kw)
  with patch.object(Path,'read_text',read):job=load_job(path)
  self.assertIsNone(job['boundaries'])
  b=FakeBackend(np.arange(10).reshape(-1,1))
  with tempfile.TemporaryDirectory() as d:
   r=reconstruct(b,np.array([[3.],[7.],[8.],[9.]]),[1,2,3,4],[1],SearchConfig(method='collision_plus',batch_size=2,epsilon=.01),d,boundaries=job['boundaries'])
  self.assertEqual(r['status'],'complete');self.assertEqual(b.commits,[3,7,8,9])
 def test_live_model_config_resolves_roles_and_same_gate(self):
  path=Path(__file__).resolve().parents[1]/'components/experiment5_gemma_layer4_multimodal/configs/text_prefix_layer4.json'
  job=load_job(path)
  self.assertEqual(job['public_prefix_ids'],[2,105,2364,107])
  self.assertEqual(job['resolved_markers']['assistant.start'],[105,4368,107])
  self.assertEqual(job['resolved_markers']['image.start'],[255999])
  a=job['search_configs']['guided'];b=job['search_configs']['collision_plus']
  self.assertEqual((a.epsilon,a.max_queries,a.batch_size),(b.epsilon,b.max_queries,b.batch_size))
 def test_multitoken_boundary_stops_before_body(self):
  b=FakeBackend(np.arange(10).reshape(-1,1));rules=BoundaryRules({'assistant.start':[7,8]})
  with tempfile.TemporaryDirectory() as d:
   r=reconstruct(b,np.array([[3.],[7.],[8.],[9.]]),[1,2,3,4],[1],SearchConfig(method='collision_plus',batch_size=2,epsilon=.01),d,boundaries=rules)
  self.assertEqual(r['status'],'boundary_reached');self.assertEqual(r['boundary']['start_position'],2)
  self.assertEqual(r['attempted_positions'],3);self.assertEqual(b.commits,[3,7])
 def test_unmatched_boundary_does_not_stop(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  with tempfile.TemporaryDirectory() as d:
   r=reconstruct(b,np.array([[7.],[3.]]),[1,2],[1],SearchConfig(method='collision_plus',batch_size=2,epsilon=.01),d,boundaries=BoundaryRules({'assistant.start':[7,8]}))
  self.assertEqual(r['status'],'complete');self.assertEqual(b.commits,[7,3])
 def test_media_marker_not_committed(self):
  b=FakeBackend(np.arange(10).reshape(-1,1))
  with tempfile.TemporaryDirectory() as d:
   r=reconstruct(b,np.array([[7.],[9.]]),[1,2],[1],SearchConfig(method='collision_plus',batch_size=2,epsilon=.01),d,boundaries=BoundaryRules({'image.start':[7]}))
  self.assertEqual(r['status'],'boundary_reached');self.assertEqual(b.commits,[])
if __name__=='__main__':unittest.main()
