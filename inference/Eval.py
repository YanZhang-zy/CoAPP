import argparse
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

from utils.common import IMAGE_SUFFIXES, extract_state_dict, image_paths, load_checkpoint, normalize_input, resolve_device
from model import CoAPP, Student


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="An image or image directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Result directory; images and metrics are saved in separate subfolders.")
    parser.add_argument("--weights", type=Path, default=ROOT / "weight/CoAPP.pth")
    parser.add_argument("--model", choices=("deploy", "train"), default="deploy",
                        help="Use train for an unconverted student training checkpoint.")
    parser.add_argument("--checkpoint-key", help="Optional checkpoint entry, e.g. ema_model.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--musiq", action="store_true", help="Calculate MUSIQ for the saved output images.")
    return parser.parse_args(argv)


@torch.inference_mode()
def dehaze(model, image, device):
    tensor = normalize_input(to_tensor(image.convert("RGB"))).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    tensor = resize(tensor, [max(16, height // 16 * 16), max(16, width // 16 * 16)],
                    interpolation=InterpolationMode.BICUBIC, antialias=True)
    output = model(tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    output = output.clamp(0, 1)
    return resize(output, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    if args.input.is_file():
        if args.input.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {args.input}")
        images, input_root = [args.input], args.input.parent
    else:
        images, input_root = image_paths(args.input, args.recursive), args.input
        if args.recursive and args.output.resolve().is_relative_to(args.input.resolve()):
            raise ValueError("For recursive inference, the output directory must be outside the input directory.")
    if not images:
        raise ValueError(f"No images found in {args.input}")
    image_dir = args.output / "visualization"
    metrics_dir = args.output / "metrics"
    outputs = [image_dir / path.relative_to(input_root).with_suffix(".png") for path in images]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Input images with the same stem would overwrite the same PNG. Rename them first.")
    sources = {path.resolve() for path in images}
    if any(target.resolve() in sources for target in outputs):
        raise ValueError("The output directory must not overwrite input images.")
    if args.musiq and importlib.util.find_spec("pyiqa") is None:
        raise ImportError("MUSIQ requires pyiqa. Install it with: pip install pyiqa==0.1.11")
    model = CoAPP() if args.model == "deploy" else Student()
    state = extract_state_dict(load_checkpoint(args.weights), args.checkpoint_key)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for source, target in tqdm(list(zip(images, outputs)), desc="Dehazing"):
        with Image.open(source) as image:
            output = dehaze(model, image, device)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_image(output.squeeze(0), target)
    print(f"Saved {len(images)} image(s) to {image_dir.resolve()}")
    if args.musiq:
        from metrics.MUSIQ import load_musiq, evaluate_musiq
        metric, metric_version = load_musiq(device)
        evaluate_musiq(outputs, metrics_dir, metric, metric_version, device, image_dir=image_dir)


if __name__ == "__main__":
    main()
