import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def build_parser(stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if stage is None:
        parser.add_argument("stage", choices=("teacher", "kd", "adapt"))
    parser.add_argument("--train-dir", help="Paired root with hazy/clear, or unpaired image directory for adapt.")
    parser.add_argument("--val-dir", help="Optional paired root with hazy/clear; validated at original resolution.")
    parser.add_argument("--output", help="Experiment directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--iters-per-epoch", type=int, default=None)
    parser.add_argument("--max-steps", type=int, help="Stop at this total step, useful for a short smoke run.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--end-lr", type=float, default=None)
    parser.add_argument("--teacher-checkpoint", help="Frozen teacher checkpoint for KD.")
    parser.add_argument("--student-checkpoint", help="Training-form student checkpoint used to initialize adaptation.")
    parser.add_argument("--resume", help="Resume a full checkpoint saved by this script.")
    parser.add_argument("--save-every", type=int, default=1, help="Save and optionally validate every N complete epochs.")
    parser.add_argument("--ema-decay", type=float, default=0.95)
    parser.add_argument("--w-l1", type=float, default=0.8)
    parser.add_argument("--w-ssim", type=float, default=0.2)
    parser.add_argument("--w-contrast", type=float, default=0.05)
    parser.add_argument("--w-fa", type=float, default=1.0)
    parser.add_argument("--w-consistency", type=float, default=1.0)
    parser.add_argument("--w-direction", type=float, default=0.5)
    parser.add_argument("--w-positive", type=float, default=1.0)
    parser.add_argument("--w-negative", type=float, default=0.5)
    parser.add_argument("--vgg-checkpoint", help="Optional offline torchvision VGG19 state dict for contrastive loss.")
    return parser


def _image(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def _update_ema(student, teacher, decay):
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


def _rng_state(loader_generator, device):
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "generator": numpy_state[0], "keys": numpy_state[1].tolist(),
            "position": numpy_state[2], "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "loader_generator": loader_generator.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state, loader_generator, device):
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["generator"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    loader_generator.set_state(state["loader_generator"].cpu())
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@torch.no_grad()
def validate(model, loader, device, normalize_input, ssim_loss):
    model.eval()
    scores = []
    for hazy, clear in loader:
        hazy, clear = hazy.to(device), clear.to(device)
        height, width = hazy.shape[-2:]
        inputs = normalize_input(hazy)
        pad_height, pad_width = (-height) % 4, (-width) % 4
        if pad_height or pad_width:
            mode = "reflect" if pad_height < height and pad_width < width else "replicate"
            inputs = F.pad(inputs, (0, pad_width, 0, pad_height), mode=mode)
        prediction = _image(model(inputs))[..., :height, :width].clamp(0, 1)
        if prediction.shape != clear.shape:
            raise ValueError(f"Validation shapes differ: output={prediction.shape}, clear={clear.shape}.")
        mse = F.mse_loss(prediction, clear).item()
        score_psnr = -10.0 * math.log10(max(mse, 1e-12))
        score_ssim = ssim_loss(prediction, clear).item()
        scores.append((score_psnr, score_ssim))
    if not scores:
        raise ValueError("Validation dataset is empty.")
    return {"psnr": float(np.mean([value[0] for value in scores])),
            "ssim": float(np.mean([value[1] for value in scores])), "images": len(scores)}


def main(stage=None):
    parser = build_parser(stage)
    args = parser.parse_args()
    args.stage = stage or args.stage
    from utils.common import extract_state_dict, load_checkpoint, normalize_input, resolve_device, seed_everything
    from data import PairedDataset, UnpairedDataset
    from loss import ContrastLoss, DirectionalCLIPLoss, FA, SSIM
    from model import Student, Teacher

    checkpoint = load_checkpoint(args.resume, device="cpu") if args.resume else None
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or "optimizer" not in checkpoint or "model" not in checkpoint:
            parser.error("--resume requires a complete checkpoint from this training script.")
        previous_args = checkpoint.get("args", {})
        if previous_args.get("stage", args.stage) != args.stage:
            parser.error("Resume checkpoint stage does not match the selected stage.")
        supplied_options = {value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")}
        for key, value in previous_args.items():
            if key not in ("resume", "stage") and hasattr(args, key) and "--" + key.replace("_", "-") not in supplied_options:
                setattr(args, key, value)

    if args.iters_per_epoch is None:
        args.iters_per_epoch = 500 if args.stage == "adapt" else 5000
    if args.batch_size is None:
        args.batch_size = 16 if args.stage == "adapt" else 24
    args.lr = args.lr if args.lr is not None else (1e-7 if args.stage == "adapt" else 1e-4)
    args.end_lr = args.end_lr if args.end_lr is not None else (1e-8 if args.stage == "adapt" else 1e-6)
    args.output = args.output or f"experiment/{args.stage}"
    if not args.train_dir:
        parser.error("--train-dir is required.")
    if min(args.epochs, args.iters_per_epoch, args.batch_size, args.save_every) < 1:
        parser.error("Epoch, iteration, batch size, and save intervals must be positive.")
    if args.patch_size < 32 or args.patch_size % 4:
        parser.error("--patch-size must be at least 32 and divisible by four.")
    if args.num_workers < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("--num-workers must be nonnegative and --max-steps must be positive.")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be in [0, 1).")
    weight_names = ("w_l1", "w_ssim", "w_contrast", "w_fa", "w_consistency", "w_direction", "w_positive", "w_negative")
    if any(getattr(args, key) < 0 for key in weight_names) or args.lr <= 0 or args.end_lr < 0:
        parser.error("Loss weights and end learning rate must be nonnegative; learning rate must be positive.")
    if args.stage == "adapt" and args.w_consistency + args.w_direction == 0:
        parser.error("At least one adaptation loss must have a positive weight.")
    if args.stage != "adapt" and sum(getattr(args, key) for key in ("w_l1", "w_ssim", "w_contrast")) + (args.w_fa if args.stage == "kd" else 0) == 0:
        parser.error("At least one training loss must have a positive weight.")
    if args.stage == "kd" and not args.teacher_checkpoint and not (checkpoint and checkpoint.get("teacher_model")):
        parser.error("KD requires --teacher-checkpoint.")
    if args.stage == "adapt" and checkpoint is None and not args.student_checkpoint:
        parser.error("Adaptation requires --student-checkpoint in training form (before reparameterization).")

    device = resolve_device(args.device)
    seed_everything(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataset_type = UnpairedDataset if args.stage == "adapt" else PairedDataset
    dataset = dataset_type(args.train_dir, patch_size=args.patch_size, training=True)
    if len(dataset) == 0:
        parser.error("Training dataset is empty.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", drop_last=False, generator=loader_generator)
    validation_loader = None
    if args.val_dir:
        validation_set = PairedDataset(args.val_dir, patch_size=args.patch_size, training=False)
        validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = (Teacher() if args.stage == "teacher" else Student(return_features=args.stage == "kd")).to(device)
    reference = None
    if args.stage in ("kd", "adapt"):
        reference = (Teacher() if args.stage == "kd" else Student()).to(device)
        if checkpoint is not None:
            reference_key = "teacher_model" if args.stage == "kd" else "ema_model"
            if checkpoint.get(reference_key) is None:
                raise ValueError(f"Resume checkpoint is missing {reference_key}.")
            reference.load_state_dict(extract_state_dict(checkpoint[reference_key]), strict=True)
        else:
            initial_path = args.teacher_checkpoint if args.stage == "kd" else args.student_checkpoint
            initial = extract_state_dict(load_checkpoint(initial_path, device="cpu"))
            reference.load_state_dict(initial, strict=True)
            if args.stage == "adapt":
                model.load_state_dict(initial, strict=True)
        _freeze(reference)
    if checkpoint is not None:
        model.load_state_dict(extract_state_dict(checkpoint["model"]), strict=True)

    l1_loss = nn.L1Loss()
    ssim_loss = SSIM().to(device)
    feature_loss = FA().to(device) if args.stage == "kd" and args.w_fa > 0 else None
    contrast_loss = (ContrastLoss(vgg_checkpoint=args.vgg_checkpoint, device=device).to(device)
                     if args.stage != "adapt" and args.w_contrast > 0 else None)
    direction_loss = None
    if args.stage == "adapt" and args.w_direction > 0:
        direction_loss = DirectionalCLIPLoss(device=device,
                                            w_positive=args.w_positive, w_negative=args.w_negative).to(device)
        if checkpoint is not None:
            if checkpoint.get("fusion") is None:
                raise ValueError("Resume checkpoint is missing the semantic fusion parameters.")
            direction_loss.fusion.load_state_dict(checkpoint["fusion"], strict=True)
            if checkpoint.get("text_features") is not None:
                direction_loss.text_features.copy_(checkpoint["text_features"].to(device))

    parameters = list(model.parameters())
    if direction_loss is not None:
        parameters += list(direction_loss.fusion.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    step = 0
    best_psnr = float("-inf")
    last_validation = None
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        best_psnr = checkpoint.get("best_psnr", best_psnr)
        last_validation = checkpoint.get("validation")
        _restore_rng(checkpoint.get("rng_state"), loader_generator, device)
    total_steps = args.epochs * args.iters_per_epoch
    target_steps = min(total_steps, args.max_steps) if args.max_steps else total_steps
    if step >= target_steps:
        print(f"Checkpoint is already at step {step}; requested stop is {target_steps}.")
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(f"Stage: {args.stage} | device: {device} | images: {len(dataset)} | steps: {step} -> {target_steps}")

    def save(path, validation=None):
        state = {
            "model": model.state_dict(),
            "ema_model": reference.state_dict() if args.stage == "adapt" else None,
            "teacher_model": reference.state_dict() if args.stage == "kd" else None,
            "fusion": direction_loss.fusion.state_dict() if direction_loss is not None else None,
            "text_features": direction_loss.text_features.detach().cpu() if direction_loss is not None else None,
            "optimizer": optimizer.state_dict(), "step": step, "epoch": step // args.iters_per_epoch,
            "step_in_epoch": step % args.iters_per_epoch, "args": vars(args),
            "best_psnr": best_psnr, "validation": validation,
            "rng_state": _rng_state(loader_generator, device),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    iterator = iter(loader)
    while step < target_steps:
        model.train()
        if direction_loss is not None:
            direction_loss.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        progress = min(step + 1, total_steps) / total_steps
        learning_rate = args.end_lr + 0.5 * (args.lr - args.end_lr) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        losses = {}
        if args.stage == "adapt":
            hazy = batch.to(device)
            inputs = normalize_input(hazy)
            prediction = _image(model(inputs))
            if args.w_consistency > 0:
                with torch.no_grad():
                    reference_prediction = _image(reference(inputs))
                losses["consistency"] = args.w_consistency * l1_loss(prediction, reference_prediction)
            if direction_loss is not None:
                losses["direction"] = args.w_direction * direction_loss(prediction, hazy)
        else:
            hazy, clear = (value.to(device) for value in batch)
            inputs = normalize_input(hazy)
            result = model(inputs)
            prediction = _image(result)
            if args.w_l1 > 0:
                losses["l1"] = args.w_l1 * l1_loss(prediction, clear)
            if args.w_ssim > 0:
                losses["ssim"] = args.w_ssim * (1.0 - ssim_loss(prediction, clear))
            if contrast_loss is not None:
                losses["contrast"] = args.w_contrast * contrast_loss(prediction, clear, hazy)
            if feature_loss is not None:
                with torch.no_grad():
                    reference_result = reference(inputs)
                losses["fa"] = args.w_fa * feature_loss(result[1], reference_result[1])
        total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step + 1}.")
        total_loss.backward()
        optimizer.step()
        step += 1
        record = {"step": step, "epoch": step // args.iters_per_epoch, "lr": learning_rate,
                  "loss": total_loss.item(), **{key: value.item() for key, value in losses.items()}}
        with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step == 1 or step % 20 == 0 or step == target_steps:
            print(f"step {step}/{target_steps} | loss {record['loss']:.6f} | lr {learning_rate:.3g}")
        complete_epoch = step % args.iters_per_epoch == 0
        if complete_epoch and args.stage == "adapt":
            _update_ema(model, reference, args.ema_decay)
        if complete_epoch and (step // args.iters_per_epoch) % args.save_every == 0:
            metrics = validate(model, validation_loader, device, normalize_input, ssim_loss) if validation_loader else None
            last_validation = metrics
            if metrics is not None:
                print(f"Validation: PSNR {metrics['psnr']:.4f} | SSIM {metrics['ssim']:.4f} | images {metrics['images']}")
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    save(output / "best.pth", metrics)
            save(output / f"epoch_{step // args.iters_per_epoch:03d}.pth", metrics)
            save(output / "last.pth", metrics)
    save(output / "last.pth", last_validation)
    print(f"Saved full training checkpoint: {output / 'last.pth'}")


if __name__ == "__main__":
    main()
