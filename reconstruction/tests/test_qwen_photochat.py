from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from kvrecon.adapters.qwen_verify.photochat import image_template,search_config,PhotoChatNpuBackend
from kvrecon.adapters.qwen_verify.image_span import locate_end,TOKEN_LIMITS
from kvrecon.media.qwen_vision import mrope_positions

def meta():return dict(public_prefix=[248045,846,198,248053],image_pad_id=248056,image_end_id=248054)

def test_image_mrope_offset_and_public_only_template():
    before,pad,end=image_template(meta());prefix=before+[pad]*64+[end]
    rope=mrope_positions(prefix,pad,(8,8))
    assert len(prefix)==69 and rope[:,-1].tolist()==[12,12,12]
    assert rope[0,-1]+1-len(prefix)==-56

def test_bad_template_rejected():
    m=meta();m['image_end_id']=m['image_pad_id']
    with pytest.raises(ValueError):image_template(m)

def test_end_marker_scan_recovers_image_length():
    rng=np.random.default_rng(0);observed=rng.normal(size=(300,8)).astype(np.float32);end=rng.normal(size=4)
    observed[4+100,4:]=end
    assert TOKEN_LIMITS==[64,256] and locate_end(observed,4,end[None])[0]==100

def test_existing_search_protocol():
    s=search_config(2,[1,2,1])
    assert s.batch_size==128 and s.probability_seeds==50 and s.reference_seeds==78
    assert s.ridge_stagnation_rounds==2 and s.stagnation_token_ids==(1,2)
    assert s.max_queries is None and not s.retain_query_cache

def test_restore_never_sends_future_rows():
    b=object.__new__(PhotoChatNpuBackend);b.position=3;b.observed=np.arange(20,dtype=np.float32).reshape(5,4)
    sent=[];b.write=lambda name,data:sent.append((name,data));b.rpc=lambda op,**kw:sent.append((op,kw))
    b._restore(2,1)
    assert np.frombuffer(sent[0][1],dtype='<f4').tolist()==[8,9,10,11]
    with pytest.raises(ValueError):b._restore(3,1)
    assert len(sent)==2

def test_search_reuses_validated_image_prefix_once():
    b=object.__new__(PhotoChatNpuBackend)
    b.public_prefix=[1,2,3];b.position=3;b.validated_prefix_ready=True
    b.reset([1,2,3])
    assert not b.validated_prefix_ready
    b.validated_prefix_ready=True;b.position=4
    with pytest.raises(RuntimeError):b.reset([1,2,3])

def test_commit_returns_measured_row_but_restores_accepted_cache(monkeypatch):
    from kvrecon.npu_standalone import StandaloneNpuBackend
    b=object.__new__(PhotoChatNpuBackend);b.position=3
    raw=np.array([[.1,.2]],dtype=np.float32)
    def commit(self,token):self.position+=1;return raw
    monkeypatch.setattr(StandaloneNpuBackend,'commit',commit)
    calls=[];b._restore=lambda start,count:calls.append((start,count))
    assert b.commit(17) is raw
    assert calls==[(3,1)]
