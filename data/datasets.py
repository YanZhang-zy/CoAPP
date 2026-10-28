from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from utils.common import image_paths


def _read(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _augment(images, patch_size):
    width, height = images[0].size
    if min(width, height) < patch_size:
        scale = patch_size / min(width, height)
        size = (max(patch_size, round(width * scale)), max(patch_size, round(height * scale)))
        images = [im.resize(size, Image.Resampling.BICUBIC) for im in images]
        width, height = size
    left = random.randint(0, width - patch_size)
    top = random.randint(0, height - patch_size)
    box = (left, top, left + patch_size, top + patch_size)
    images = [im.crop(box) for im in images]
    if random.random() < 0.5:
        images = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in images]
    rotations = random.randrange(4)
    if rotations:
        operation = (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                     Image.Transpose.ROTATE_270)[rotations - 1]
        images = [im.transpose(operation) for im in images]
    return images


class PairedDataset(Dataset):

    def __init__(self, root, patch_size=256, training=True):
        self.root = Path(root)
        self.patch_size = int(patch_size)
        self.training = training
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        hazy = image_paths(self.root / "hazy")
        clear = image_paths(self.root / "clear")
        if not hazy or not clear:
            raise ValueError(f"Expected images in both {self.root / 'hazy'} and {self.root / 'clear'}.")
        lookup = {}
        for path in clear:
            stem = path.stem.lower()
            if stem in lookup:
                raise ValueError(f"Ambiguous clear image stem: {path.stem}")
            lookup[stem] = path
        self.pairs = []
        for path in hazy:
            target = lookup.get(path.stem.lower())
            if target is None:
                target = lookup.get(path.stem.split("_")[0].lower())
            if target is None:
                raise FileNotFoundError(f"No clear image matching {path.name} in {self.root / 'clear'}")
            self.pairs.append((path, target))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        hazy_path, clear_path = self.pairs[index]
        hazy, clear = _read(hazy_path), _read(clear_path)
        if hazy.size != clear.size:
            raise ValueError(f"Paired image sizes differ: {hazy_path.name} and {clear_path.name}")
        images = [hazy, clear]
        if self.training:
            images = _augment(images, self.patch_size)
        return tuple(to_tensor(im) for im in images)


class UnpairedDataset(Dataset):
    def __init__(self, root, patch_size=256, training=True):
        self.paths = image_paths(root, recursive=True)
        self.patch_size = int(patch_size)
        self.training = training
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if not self.paths:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = _read(self.paths[index])
        if self.training:
            image = _augment([image], self.patch_size)[0]
        return to_tensor(image)
