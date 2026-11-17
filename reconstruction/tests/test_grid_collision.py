import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from kvrecon.media.grid_collision import GridCollision,factor_grids,replay_dimensions,preprocessed_grid
from kvrecon.media.gemma_replay import GemmaMediaReplay
from kvrecon.media import MediaUnavailable


class Tests(unittest.TestCase):
    def test_every_feasible_layout_renders_to_its_grid(self):
        for n in range(1,281):
            for grid in factor_grids(n):
                self.assertEqual(grid[0]*grid[1],n)
                self.assertEqual(preprocessed_grid(*replay_dimensions(grid)),grid)
        self.assertIn((13,20),factor_grids(260))
        self.assertIn((20,13),factor_grids(260))
        self.assertEqual(factor_grids(1),[])

    def test_trials_share_prefix_and_only_winner_is_committed(self):
        from unittest.mock import patch
        class Backend:
            position=5
            def rpc(self,op,**args):
                if op=='save_media_prefix':self.saved=self.position
                elif op=='restore_media_prefix':self.position=self.saved
                elif op=='release_media_prefix':self.saved=None
        class Decoder:
            def decode(self,codes,out,grid):
                out.mkdir(parents=True,exist_ok=True)
                return dict(modality='image',artifact=str(out/'image.png'),grid_hw=grid)
        backend=Backend();starts=[]
        def replay(candidate,request):
            starts.append(backend.position);backend.position+=260
            return dict(status='complete',end_consumed=False,
                        grid_score_rmse=3. if candidate['grid_hw']==[13,20] else 9.)
        with tempfile.TemporaryDirectory() as folder,patch('kvrecon.media.grid_collision.factor_grids',return_value=[(13,20),(20,13)]):
            result=GridCollision().recover(backend,dict(output=Path(folder),targets=np.zeros((260,1024))),Decoder(),replay)
            self.assertEqual(result['grid_hw'],[13,20])
            self.assertEqual(starts,[5,5,5]);self.assertEqual(backend.position,265)
            self.assertIsNone(backend.saved)
            self.assertFalse(json.loads((Path(folder)/'grid_selection.json').read_text())['original_grid_used'])

    def test_audio_count_mismatch_never_advances_cache(self):
        class Backend:
            position=5
            def encode_audio_wav(self,path):return np.zeros((2,1536)),{}
            def rpc(self,*args,**kwargs):raise AssertionError('Must not append mismatched audio')
        b=Backend();replay=GemmaMediaReplay(b,{},'.');replay.end_embedding=lambda *args:None
        with self.assertRaisesRegex(MediaUnavailable,'token_count_mismatch'):
            replay(dict(modality='audio',artifact='recovered.wav'),
                   dict(output=Path('.'),targets=np.zeros((3,1024)),span=dict(start=5)))
        self.assertEqual(b.position,5)

    def test_audio_interior_stops_before_end(self):
        class Backend:
            position=5;batch_capacity=128;dtype=np.int8;width=1024
            def encode_audio_wav(self,path):return np.ones((3,1536)),dict(audio_rows=3)
            def rpc(self,op,**args):
                self.assert_op=op;self.position+=args['rows']
                return np.zeros((args['rows'],1024),dtype=np.int8).tobytes()
        b=Backend();replay=GemmaMediaReplay(b,{},'.');replay.end_embedding=lambda *args:None
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);wav=out/'recovered.wav';wav.write_bytes(b'test')
            result=replay(dict(modality='audio',artifact=str(wav)),
                dict(output=out,targets=np.zeros((3,1024)),span=dict(start=5)))
        self.assertEqual(b.position,8);self.assertFalse(result['end_consumed'])
        self.assertEqual(result['grid_score_rmse'],0)


if __name__=='__main__':unittest.main()
