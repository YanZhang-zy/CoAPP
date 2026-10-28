from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import OrderedDict
from collections.abc import Mapping
import torch

from model import CoAPP, Student
from model.modules.deconv import DEConv


def _fuse_deconv(state_dict, prefix):
    cd = state_dict[prefix + "conv1_1.conv.weight"].flatten(2)
    cd_kernel = cd.clone()
    cd_kernel[:, :, 4] = cd[:, :, 4] - cd.sum(dim=2)

    hd = state_dict[prefix + "conv1_2.conv.weight"]
    hd_kernel = hd.new_zeros(hd.shape[0], hd.shape[1], 9)
    hd_kernel[:, :, [0, 3, 6]] = hd
    hd_kernel[:, :, [2, 5, 8]] = -hd

    vd = state_dict[prefix + "conv1_3.conv.weight"]
    vd_kernel = vd.new_zeros(vd.shape[0], vd.shape[1], 9)
    vd_kernel[:, :, [0, 1, 2]] = vd
    vd_kernel[:, :, [6, 7, 8]] = -vd

    ad = state_dict[prefix + "conv1_4.conv.weight"].flatten(2)
    ad_kernel = ad - ad[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
    vanilla = state_dict[prefix + "conv1_5.weight"]
    shape = vanilla.shape
    kernel = (cd_kernel.reshape(shape) + hd_kernel.reshape(shape)
              + vd_kernel.reshape(shape) + ad_kernel.reshape(shape) + vanilla)
    bias = state_dict[prefix + "conv1_1.conv.bias"]
    for suffix in ("conv1_2.conv.bias", "conv1_3.conv.bias",
                   "conv1_4.conv.bias", "conv1_5.bias"):
        bias = bias + state_dict[prefix + suffix]
    return kernel, bias


@torch.no_grad()
def reparameterize_state_dict(state_dict: Mapping[str, torch.Tensor]):
    student = Student()
    student.load_state_dict(state_dict, strict=True)
    deployment = CoAPP()

    fused = {}
    for name, module in student.named_modules():
        if isinstance(module, DEConv):
            kernel, bias = _fuse_deconv(state_dict, name + ".")
            fused[name + ".weight"] = kernel
            fused[name + ".bias"] = bias

    converted = OrderedDict()
    for key in deployment.state_dict():
        value = fused[key] if key in fused else state_dict[key]
        converted[key] = value.detach().clone()
    deployment.load_state_dict(converted, strict=True)
    return converted


@torch.no_grad()
def reparameterize_model(student: Student) -> CoAPP:
    if not isinstance(student, Student):
        raise TypeError("Expected the training Student model.")
    parameter = next(student.parameters())
    deployment = CoAPP().to(device=parameter.device, dtype=parameter.dtype)
    deployment.load_state_dict(reparameterize_state_dict(student.state_dict()), strict=True)
    return deployment.eval()


def main(argv=None):
    from utils.common import extract_state_dict, load_checkpoint, resolve_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Training Student checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output deployment state dict.")
    parser.add_argument("--checkpoint-key", help="Optional full-checkpoint entry, e.g. ema_model.")
    parser.add_argument("--device", default="cpu", help="Conversion device: cpu, cuda, or cuda:N.")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input to preserve the training checkpoint.")

    device = resolve_device(args.device)
    state = extract_state_dict(load_checkpoint(args.input, device=device), args.checkpoint_key)
    converted = reparameterize_state_dict(state)
    portable = OrderedDict((key, value.cpu()) for key, value in converted.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable, args.output)
    print(f"Saved CoA++ deployment weights to {args.output} ({len(portable)} tensors).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
