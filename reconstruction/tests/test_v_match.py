import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.v_match import compare_v


class VMatchTests(unittest.TestCase):
    def test_ties_across_chunks_and_unsorted_ids(self):
        rows=compare_v([[1,0],[10,0],[-1,0]],[9,2,3],[[0,0],[9,0]],chunk_size=1)
        self.assertEqual(rows[0]['nearest_token_ids'],[3,9])
        self.assertIsNone(rows[0]['unique_nearest_token_id'])
        self.assertEqual(rows[0]['l2_margin'],0)
        self.assertEqual(rows[1]['unique_nearest_token_id'],2)
        self.assertEqual(rows[1]['minimum_l2'],1)
        self.assertEqual(rows[1]['second_token_id'],9)

    def test_integer_overflow_and_single_candidate(self):
        r=compare_v(np.array([[-128,127]],dtype=np.int8),[7],np.array([127,-128],dtype=np.int8))[0]
        self.assertAlmostEqual(r['minimum_l2'],np.sqrt(2*255**2))
        self.assertIsNone(r['second_l2'])

    def test_invalid_inputs(self):
        for table,ids,target in [([[0],[1]],[2,2],[0]),([[0]],[1],[0,1]),([[np.nan]],[1],[0])]:
            with self.assertRaises(ValueError):compare_v(table,ids,target)


if __name__=='__main__':unittest.main()
