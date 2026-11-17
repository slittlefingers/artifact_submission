import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from kvrecon.search import SearchConfig
from kvrecon.media import MediaUnavailable
from kvrecon.media.boundaries import VBoundaryLocator
from kvrecon.media.token_regions import TokenSurrogateRegionPolicy


def text_config():
    return SearchConfig(method='guided_kv', probability_seeds=50, reference_seeds=78,
        random_seeds=0, batch_size=128, merge_initial_batch=True, proposals=128,
        epsilon=30., distance_space='lsb', ridge_stagnation_rounds=2,
        stagnation_neighbors=127, require_stagnation_for_acceptance=True)


class Tests(unittest.TestCase):
    def policy(self, output=None):
        targets=np.zeros((10,2));targets[:,1]=50
        targets[3,1]=10                           
        targets[7,1]=20                           
        locator=VBoundaryLocator({'image':[[10]],'audio':[[20]]},1,
            {'image':(1,4),'audio':(1,4)})
        return TokenSurrogateRegionPolicy(text_config(),targets,np.arange(1,11),
            locator,output=output)

    def test_starts_only_after_an_accepted_marker_and_includes_end(self):
        p=self.policy()
        self.assertTrue(p(1,2,0).require_stagnation_for_acceptance)
        self.assertEqual(p.status['mode'],'text')
        self.assertFalse(p(2,255999,1).require_stagnation_for_acceptance)
        self.assertEqual(p.status['mode'],'image')
        self.assertEqual(p(3,17,2).epsilon,2.)
        self.assertFalse(p(4,18,3).require_stagnation_for_acceptance)
        self.assertTrue(p(5,19,4).require_stagnation_for_acceptance)
        self.assertEqual(p(5,19,4).epsilon,30.)
        self.assertEqual(p.regions[0]['start'],2)
        self.assertEqual(p.regions[0]['end'],4)
        self.assertFalse(p.regions[0]['verified'])

    def test_nested_marker_predictions_and_end_slot_do_not_open_regions(self):
        p=self.policy();p(1,2,0);p(2,255999,1)
        p(3,256000,2);p(4,255999,3)
        c=p(5,256000,4)
        self.assertEqual(len(p.regions),1)
        self.assertTrue(c.require_stagnation_for_acceptance)

    def test_audio_region_and_persisted_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'regions.json';p=self.policy(out)
            p(1,2,0);p(2,255999,1);p(3,17,2);p(4,18,3);p(5,19,4)
            p(6,256000,5);p(7,10,6);p(8,11,7)
            self.assertEqual(p.mode,'audio')
            self.assertTrue(p(9,12,8).require_stagnation_for_acceptance)
            report=json.loads(out.read_text())
            self.assertEqual([r['modality'] for r in report['regions']],['image','audio'])
            self.assertFalse(report['labels_used'])
            self.assertFalse(report['neural_media_models_used'])
            self.assertFalse(report['regions'][1]['verified'])

    def test_end_unavailable_stops_instead_of_relaxing_text(self):
        locator=VBoundaryLocator({'image':[[0]]},1,{'image':(5,10)})
        p=TokenSurrogateRegionPolicy(text_config(),np.zeros((3,2)),np.arange(1,4),locator)
        p(1,2,0)
        with self.assertRaisesRegex(MediaUnavailable,'outside_observation'):p(2,255999,1)
        self.assertEqual(p.status['mode'],'unresolved_boundary')
        self.assertFalse(p.regions)

    def test_invalid_sequence_and_shape_are_rejected(self):
        p=self.policy();p(1,2,0)
        with self.assertRaisesRegex(ValueError,'sequential'):p(3,10,2)
        with self.assertRaisesRegex(ValueError,'immediately'):self.policy()(1,2,9)
        with self.assertRaisesRegex(ValueError,'contiguous'):
            TokenSurrogateRegionPolicy(text_config(),np.zeros((2,2)),np.array([1,3]),p.locator)

    def test_same_position_configuration_is_idempotent(self):
        p=self.policy();p(1,2,0)
        first=p(2,255999,1);again=p(2,255999,1)
        self.assertIs(first,again)
        self.assertEqual(len(p.regions),1)


if __name__=='__main__':unittest.main()
