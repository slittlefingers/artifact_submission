from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kvrecon.search import lowest_scored_ids


class SelectionTests(unittest.TestCase):
    def test_partial_selection_exactly_matches_full_sort_including_ties(self):
        rng=np.random.default_rng(20261019)
        for size in (1,7,1000,262144):
            for dtype in (np.float32,np.float64):
                for ties in (False,True):
                    scores=(rng.integers(-3,4,size) if ties else rng.normal(size=size)).astype(dtype)
                    scores[0]=-0.
                    ids=np.flatnonzero(rng.random(size)>.1)
                    for k in (0,1,20,len(ids),len(ids)+1):
                        expected=ids[np.lexsort((ids,scores[ids]))[:k]]
                        np.testing.assert_array_equal(lowest_scored_ids(ids,scores,k),expected)

    def test_native_radix_matches_stable_numeric_sort(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'tests/float_ranking_test.cc'
        self.assertTrue(source.is_file(), 'Missing bundled C++ ranking test source')
        compiler=shutil.which(os.environ.get('CXX','')) or shutil.which('clang++') or shutil.which('g++') or shutil.which('cl')
        if not compiler:self.skipTest('C++17 compiler required: put clang++/g++ on PATH, or use a Visual Studio Developer Command Prompt; CXX may specify the compiler')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/('float_ranking_test.exe' if os.name=='nt' else 'float_ranking_test')
            if Path(compiler).stem.lower() in ('cl','clang-cl'):
                command=[compiler,'/nologo','/std:c++17','/O2','/EHsc','/W4','/I'+str(root/'native'),str(source),'/Fe:'+str(output)]
            else:
                command=[compiler,'-std=c++17','-O2','-Wall','-Wextra','-I',str(root/'native'),str(source),'-o',str(output)]
            built=subprocess.run(command,cwd=tmp,capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            result=subprocess.run([str(output)],check=True,capture_output=True,text=True)
            self.assertIn('PASS 100',result.stdout)


if __name__=='__main__':unittest.main()
