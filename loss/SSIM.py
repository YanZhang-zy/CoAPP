from math import exp

import torch
from torch import nn
from torch.nn import functional as F


def gaussian(window_size, sigma):
    values = torch.tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return values / values.sum()


def create_window(window_size, channel):
    kernel = gaussian(window_size, 1.5).unsqueeze(1)
    window = kernel.mm(kernel.t()).float().unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean() if size_average else ssim_map.mean((1, 2, 3))


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.register_buffer("window", create_window(window_size, 3), persistent=False)

    def forward(self, img1, img2):
        if img1.ndim != 4 or img1.shape != img2.shape:
            raise ValueError("SSIM expects equally shaped BCHW image tensors")
        channel = img1.shape[1]
        if self.window.shape[0] != channel:
            self.window = create_window(self.window_size, channel).to(img1)
        elif self.window.device != img1.device or self.window.dtype != img1.dtype:
            self.window = self.window.to(img1)
        return _ssim(img1, img2, self.window, self.window_size, channel, self.size_average)
