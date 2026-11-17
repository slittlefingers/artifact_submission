
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from image_reconstruction.capture import conversation_image_start, gemma_visual_grid, recorded_grid, visual_kv
from image_reconstruction.metrics import VIEWS, score, ssim
from image_reconstruction.model import NativeGridReconstructor
from image_reconstruction.storage import Store


def capture(rows, width=4, start=0):
    return dict(k=np.arange(rows * width, dtype=np.int8).reshape(rows, width),
                v=-np.arange(rows * width, dtype=np.int8).reshape(rows, width),
                positions=np.arange(start, start + rows, dtype=np.int32))


def test_gemma_grid_matches_deployed_rule():
    assert gemma_visual_grid(480, 640) == (14, 19)
    assert gemma_visual_grid(10, 4000) == (1, 280)
    assert gemma_visual_grid(4000, 10) == (280, 1)


def test_recorded_grid_prefers_capture_then_metadata():
    assert recorded_grid({'grid': np.array([3, 4])}, {'grid': [9, 9]}) == (3, 4)
    assert recorded_grid({}, {'grid': [2, 5]}) == (2, 5)
    assert recorded_grid({}, {}) is None


def test_image_rows_from_recorded_positions():
    arrays = capture(10)
    keys, values = visual_kv(arrays, {'image_positions': [2, 3, 4, 5]}, 4)
    assert np.array_equal(keys, arrays['k'][2:6]) and np.array_equal(values, arrays['v'][2:6])


def test_image_rows_from_modality_sentinels():
    arrays = capture(8)
    arrays.update(modality_positions=np.array([1, 2, 3, 6]), modality_sentinel_ids=np.array([-1, -1, -1, -2]))
    keys, _ = visual_kv(arrays, {}, 3)
    assert np.array_equal(keys, arrays['k'][1:4])


def test_image_rows_from_conversation_turns():
    turns = [dict(user=dict(has_image=False), state=dict(step=20)), dict(user=dict(has_image=True))]
    assert conversation_image_start(turns) == 24
    assert conversation_image_start([dict(user=dict(has_image=True))]) == 5
    arrays = capture(40)
    keys, _ = visual_kv(arrays, {'turns': turns}, 6)
    assert np.array_equal(keys, arrays['k'][24:30])


def test_visual_arrays_and_image_only_captures():
    arrays = capture(6)
    arrays.update(visual_k=arrays['k'][:2], visual_v=arrays['v'][:2])
    assert len(visual_kv(arrays, {}, 2)[0]) == 2
    assert len(visual_kv(capture(6), {}, 6)[0]) == 6


def test_missing_positions_are_rejected():
    with pytest.raises(ValueError):
        visual_kv(capture(10, start=100), {'image_positions': [1, 2]}, 2)
    with pytest.raises(ValueError):
        visual_kv(capture(10), {}, 4)


def test_identical_images_score_perfectly():
    image = torch.rand(1, 3, 336, 336)
    result = score(image, image)
    assert result['mse'] == 0 and result['psnr'] == pytest.approx(150) and ssim(image, image) == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize('name', sorted(VIEWS))
def test_views_return_evaluation_size(name):
    pixels = np.random.default_rng(0).integers(0, 256, size=(96, 144, 3), dtype=np.uint8)
    view = VIEWS[name]
    assert view.target(pixels).shape == (1, 3, 336, 336)
    assert view.prediction(torch.rand(1, 3, 96, 144)).shape == (1, 3, 336, 336)


@pytest.mark.parametrize('merge', [2, 3])
def test_native_output_size_follows_merge(merge):
    model = NativeGridReconstructor(input_dim=8, merge=merge, checkpointing=False).eval()
    with torch.inference_mode():
        image = model(torch.zeros(1, 6, 8), (2, 3))
    assert image.shape == (1, 3, 2 * 16 * merge, 3 * 16 * merge)


def test_store_reads_directories_and_archives(tmp_path):
    (tmp_path / 'tree' / 'images').mkdir(parents=True)
    (tmp_path / 'tree' / 'images' / 'a.jpg').write_bytes(b'abc')
    with zipfile.ZipFile(tmp_path / 'bundle.zip', 'w') as archive:
        archive.writestr('data/images/a.jpg', b'abc')
    for store in (Store(tmp_path / 'tree'), Store(tmp_path / 'bundle.zip')):
        name = store.find('a.jpg')
        assert store.read(name) == b'abc' and 'a.jpg' in store and store.find('missing.jpg') is None


def test_candidate_grids_reproduce_their_own_grid():
    from image_reconstruction.grid_search import GemmaFamily, QwenFamily
    from image_reconstruction import qwen_vision
    qwen = QwenFamily.candidates(247)
    assert [13, 19] in [c['grid'] for c in qwen]
    assert all(qwen_vision.target_size(c['input_hw'][1], c['input_hw'][0]) == (c['input_hw'][1], c['input_hw'][0]) for c in qwen)
    grids = GemmaFamily.candidates(260)
    assert [13, 20] in [c['grid'] for c in grids] and [1, 260] in [c['grid'] for c in grids]
    for c in grids:
        assert gemma_visual_grid(*c['input_hw']) == tuple(c['grid'])


def test_rank_marks_the_unique_closest_grid():
    from image_reconstruction.grid_search import rank
    target = np.zeros((4, 4), dtype=np.int8)
    replayed = [([2, 2], target.copy()), ([1, 4], target + 3), ([4, 1], target + 1)]
    rankings, distances = rank(target, replayed, [2, 2])
    assert all(r['rank'] == 1 and r['unique_correct'] for r in rankings.values()) and len(distances) == 3
    rankings, _ = rank(target, replayed + [([2, 2], target)], [2, 2])
    assert not rankings['kv']['unique_correct']
