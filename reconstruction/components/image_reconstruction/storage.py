
import hashlib
import io
import os
import zipfile
from pathlib import Path, PurePosixPath


class Store:
    

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self._zip = None
            self._names = [(Path(folder)/name).relative_to(self.path).as_posix()
                           for folder, _, names in os.walk(self.path) for name in names]
        elif zipfile.is_zipfile(self.path):
            self._zip = zipfile.ZipFile(self.path)
            self._names = [i.filename for i in self._zip.infolist() if not i.is_dir()]
        else:
            raise FileNotFoundError(f'Expected a directory or zip archive: {self.path}')

    def __contains__(self, name):
        return self._resolve(name) is not None

    def names(self):
        return list(self._names)

    def find(self, filename):
        
        matches = [n for n in self._names if PurePosixPath(n).name == filename]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous file name {filename!r} in {self.path}')
        return matches[0] if matches else None

    def open(self, name):
        resolved = self._resolve(name)
        if resolved is None:
            raise FileNotFoundError(f'{name} not found in {self.path}')
        return self._zip.open(resolved) if self._zip else (self.path / resolved).open('rb')

    def read(self, name):
        with self.open(name) as stream:
            return stream.read()

    def buffer(self, name):
        return io.BytesIO(self.read(name))

    def sha256(self, name):
        with self.open(name) as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    def _resolve(self, name):
        if name in self._names:
            return name
        prefixed = [n for n in self._names if n.endswith('/' + name)]
        return prefixed[0] if len(prefixed) == 1 else None


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
