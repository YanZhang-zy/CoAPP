from collections.abc import Mapping

import torch
from torch import nn


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, vgg_checkpoint=None):
        super().__init__()
        from torchvision import models
        from torchvision.models import VGG19_Weights

        if vgg_checkpoint is None:
            vgg = models.vgg19(weights=VGG19_Weights.DEFAULT)
        else:
            vgg = models.vgg19(weights=None)
            state = torch.load(vgg_checkpoint, map_location="cpu", weights_only=True)
            while isinstance(state, Mapping) and any(k in state for k in ("state_dict", "model_state_dict", "model")):
                state = next(state[k] for k in ("state_dict", "model_state_dict", "model") if k in state)
            if not isinstance(state, Mapping):
                raise ValueError("VGG checkpoint must contain a VGG19 state dictionary")
            cleaned = {}
            for key, value in state.items():
                while key.startswith("module."):
                    key = key[7:]
                cleaned[key] = value
            vgg.load_state_dict(cleaned, strict=True)

        features = vgg.features
        for name, start, end in (("slice1", 0, 2), ("slice2", 2, 7), ("slice3", 7, 12),
                                 ("slice4", 12, 21), ("slice5", 21, 30)):
            block = nn.Sequential()
            for index in range(start, end):
                block.add_module(str(index), features[index])
            setattr(self, name, block)
        self.requires_grad_(requires_grad)

    def forward(self, images):
        outputs = []
        for name in ("slice1", "slice2", "slice3", "slice4", "slice5"):
            images = getattr(self, name)(images)
            outputs.append(images)
        return outputs


class ContrastLoss(nn.Module):

    def __init__(self, ablation=False, vgg_checkpoint=None, device=None):
        super().__init__()
        self.vgg = Vgg19(vgg_checkpoint=vgg_checkpoint).eval()
        self.l1 = nn.L1Loss()
        self.weights = (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0)
        self.ab = ablation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        if device is not None:
            self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.vgg.eval()
        return self

    def forward(self, a, p, n):
        if a.ndim != 4 or a.shape[1] != 3 or a.shape != p.shape or a.shape != n.shape:
            raise ValueError("ContrastLoss expects equally shaped RGB BCHW anchor, positive, and negative tensors")
        a_vgg = self.vgg((a - self.mean) / self.std)
        with torch.no_grad():
            p_vgg = self.vgg((p - self.mean) / self.std)
            n_vgg = self.vgg((n - self.mean) / self.std) if not self.ab else None
        loss = a.new_zeros(())
        for i, weight in enumerate(self.weights):
            d_ap = self.l1(a_vgg[i], p_vgg[i])
            value = d_ap if self.ab else d_ap / (self.l1(a_vgg[i], n_vgg[i]) + 1e-7)
            loss = loss + weight * value
        return loss
