
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

from .storage import Store

IMAGE_NAME_FIELDS = ('image', 'image_name', 'source_image')
IMAGE_HASH_FIELDS = ('source_sha256', 'image_sha256')


@dataclass(frozen=True)
class Sample:
    id: str
    capture: Path
    metadata: dict


class Cohort:
    

    def __init__(self, root, images):
        self.root = Path(root)
        self.images = Store(images)
        self._by_hash = None
        nested = self.root / 'samples'
        self.sample_root = nested if nested.is_dir() else self.root
        if not any((f / 'sample.json').is_file() for f in self.sample_root.iterdir()):
            raise FileNotFoundError(f'No sample folders in {self.root}')

    def samples(self, split=None, limit=None):
        selected = []
        with os.scandir(self.sample_root) as entries:
            folders = sorted(Path(entry.path) for entry in entries if entry.is_dir())
        for folder in folders:
            metadata_path = folder / 'sample.json'
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if split is not None and metadata.get('split') != split:
                continue
            captures = sorted(folder.glob('layer*.npz'))
            if len(captures) != 1:
                raise ValueError(f'Expected one layer*.npz in {folder}, found {len(captures)}')
            selected.append(Sample(metadata.get('id', folder.name), captures[0], metadata))
            if limit is not None and len(selected) == limit:
                break
        if not selected:
            raise ValueError(f'No samples selected from {self.root} (split={split})')
        return selected

    def capture(self, sample_id):
        
        captures = sorted((self.sample_root / sample_id).glob('layer*.npz'))
        return captures[0] if len(captures) == 1 else None

    def source_image(self, sample):
        
        name = self._image_name(sample.metadata)
        expected = next((sample.metadata[f] for f in IMAGE_HASH_FIELDS if f in sample.metadata), None)
        if name is None and expected is None:
            raise ValueError(f'{sample.id}: no image name or hash recorded')
        stored = self.images.find(name) if name else self._find_by_hash(expected)
        if stored is None:
            raise FileNotFoundError(f'{sample.id}: original image not found in {self.images.path}')
        if expected is not None and self.images.sha256(stored) != expected:
            raise ValueError(f'{sample.id}: original image hash mismatch')
        with Image.open(self.images.buffer(stored)) as image:
            return ImageOps.exif_transpose(image).convert('RGB'), stored

    @staticmethod
    def _image_name(metadata):
        for field in IMAGE_NAME_FIELDS:
            if field in metadata:
                return PurePosixPath(metadata[field].replace('\\', '/')).name
        return None

    def _find_by_hash(self, digest):
        if self._by_hash is None:
            self._by_hash = {self.images.sha256(n): n for n in self.images.names() if n.lower().endswith(('.jpg', '.jpeg', '.png'))}
        return self._by_hash.get(digest)
