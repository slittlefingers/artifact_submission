import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.mark.parametrize('dataset,full,expected', [('lmsys',False,6),('base64',False,6),('all',True,12)])
def test_dataset_routing(monkeypatch,tmp_path,dataset,full,expected):
    root=Path(__file__).resolve().parents[2]
    path=root/'experiments/5.5 Prompt Recovery from Shallow Full-Attention Layers/run.py'
    spec=importlib.util.spec_from_file_location('experiment55',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    commands=[];scored=[]
    monkeypatch.setattr(module,'require_phone',lambda *args:None)
    monkeypatch.setattr(module,'output_root',lambda *args:tmp_path)
    monkeypatch.setattr(module,'run',lambda command,log:commands.append(command) or 0)
    monkeypatch.setattr(module,'score',lambda manifest,folder:scored.append(manifest) or (1,1,128))
    monkeypatch.setattr(sys,'argv',[str(path),'--dataset',dataset]+(['--full'] if full else []))
    assert module.main()==0
    assert len(commands)==len(scored)==expected
    for command,manifest in zip(commands,scored):
        assert command[command.index('--input-manifest')+1]==manifest
        cohort=manifest.parent.parent.name
        assert cohort.endswith(('_lmsys','_base64'))
        if dataset!='all':assert cohort.endswith('_'+dataset)
        assert command[command.index('--limit')+1]==(100 if full else 1)
        assert ('--positions' in command)==(not full)
    assert len({str(c[c.index('--output')+1]) for c in commands})==expected
