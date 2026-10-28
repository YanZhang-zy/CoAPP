import csv
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm


def load_musiq(device):
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError("MUSIQ requires pyiqa. Install it with: pip install pyiqa==0.1.11") from exc
    metric = pyiqa.create_metric("musiq", device=device).eval()
    return metric, pyiqa.__version__


def evaluate_musiq(paths, output_dir, metric, version, device, image_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(image_dir) if image_dir is not None else output_dir
    scores_path = output_dir / "musiq_scores.csv"
    summary_path = output_dir / "musiq_summary.json"
    scores = []
    for path in tqdm(paths, desc="MUSIQ"):
        with torch.inference_mode():
            value = float(metric(str(path)).item())
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite MUSIQ score for {path}")
        scores.append((path.relative_to(image_dir).as_posix(), value))
    if not scores:
        raise ValueError("No output images to evaluate.")
    mean = sum(value for _, value in scores) / len(scores)
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "musiq"))
        writer.writerows(scores)
    summary = {"metric": "MUSIQ", "model": "pyiqa musiq default (KonIQ-10k)",
               "pyiqa_version": version, "device": str(device), "count": len(scores), "mean": mean}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"MUSIQ mean: {mean:.4f} ({len(scores)} images)")
    print(f"MUSIQ results: {scores_path.resolve()} and {summary_path.resolve()}")
    return mean

