from pathlib import Path
import random
from collections import defaultdict

from PIL import Image, ImageOps
from torch.utils.data import Dataset
import numpy as np
import torch

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def select_images(root, samples, seed, generators='all'):
    root = Path(root)
    groups = defaultdict(list)
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        parts = path.relative_to(root).parts
        for i, part in enumerate(parts[:-1]):
            label = {'nature': 'real', 'ai': 'fake'}.get(part.lower())
            if label and i > 0 and 'val' in [p.lower() for p in parts[1:i]]:
                generator = parts[0]
                if generators == 'all' or generator in generators.split(','):
                    class_name = parts[i + 1] if i + 2 < len(parts) else ''
                    groups[(generator, label)].append((path, class_name))
                break
    if not groups:
        raise ValueError('No images found under generator/val/ai or nature directories: ' + str(root))
    rng = random.Random(seed)
    rows = []
    for generator in sorted({g for g, _ in groups}):
        for label in ('real', 'fake'):
            images = groups[(generator, label)]
            if len(images) < samples:
                raise ValueError(f'{generator}/{label}: found {len(images)}, need {samples}')
            for path, cls in rng.sample(images, samples):
                rows.append({'path': str(path.resolve()), 'generator': generator, 'label': label, 'class_name': cls})
    return rows


class Images(Dataset):
    def __init__(self, rows, size):
        self.rows, self.size = rows, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]['path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB').resize((self.size, self.size), Image.Resampling.BICUBIC)
            array = np.array(image, dtype=np.float32) / 255.0
        return index, torch.from_numpy(array).permute(2, 0, 1)
