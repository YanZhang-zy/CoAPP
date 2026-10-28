from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttentionFusion(nn.Module):

    def __init__(self, in_dim=512, num_heads=8, seq_len=16):
        super().__init__()
        if in_dim < 1 or seq_len < 1 or in_dim % seq_len:
            raise ValueError("in_dim must be positive and divisible by seq_len")
        self.seq_len = seq_len
        self.head_dim = in_dim // seq_len
        if num_heads < 1 or self.head_dim % num_heads:
            raise ValueError("Feature group dimension must be divisible by num_heads")
        self.cross_attn = nn.MultiheadAttention(self.head_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(self.head_dim)
        self.norm2 = nn.LayerNorm(self.head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2), nn.ReLU(),
            nn.Linear(self.head_dim * 2, self.head_dim),
        )
        self.final_norm = nn.LayerNorm(in_dim)

    def forward(self, image_features, text_features):
        if image_features.ndim != 2 or image_features.shape != text_features.shape:
            raise ValueError("Fusion expects equal [batch, feature_dim] image and text tensors")
        batch, dim = image_features.shape
        if dim != self.seq_len * self.head_dim:
            raise ValueError(f"Expected {self.seq_len * self.head_dim} feature dimensions, got {dim}")
        image_seq = image_features.reshape(batch, self.seq_len, self.head_dim)
        text_seq = text_features.reshape(batch, self.seq_len, self.head_dim)
        attention, _ = self.cross_attn(query=image_seq, key=text_seq, value=text_seq)
        fused = self.norm1(image_seq + attention)
        fused = self.norm2(fused + self.ffn(fused))
        return self.final_norm(fused.reshape(batch, dim))


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.to(dtype=self.dtype) + self.positional_embedding.to(dtype=self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).to(dtype=self.dtype)
        positions = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), positions] @ self.text_projection


def _load_prompt_embeddings(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    while isinstance(state, Mapping) and any(k in state for k in ("state_dict", "model_state_dict", "model")):
        state = next(state[k] for k in ("state_dict", "model_state_dict", "model") if k in state)
    if torch.is_tensor(state):
        embeddings = state
    elif isinstance(state, Mapping):
        candidates = []
        for key, value in state.items():
            while key.startswith("module."):
                key = key[7:]
            if key == "embedding_prompt":
                candidates.append(value)
        if len(candidates) != 1:
            raise ValueError("Prompt checkpoint must contain exactly one embedding_prompt tensor")
        embeddings = candidates[0]
    else:
        raise ValueError("Prompt checkpoint must contain learned token embeddings")
    if not torch.is_tensor(embeddings) or tuple(embeddings.shape) != (2, 77, 512):
        raise ValueError("Expected learned ViT-B/32 prompt embeddings of shape [2, 77, 512]")
    return embeddings.detach()


class DirectionalCLIPLoss(nn.Module):

    def __init__(self, device="cpu", w_positive=1.0, w_negative=0.5):
        super().__init__()
        if w_positive < 0 or w_negative < 0:
            raise ValueError("Directional loss weights must be nonnegative")
        resource_dir = Path(__file__).resolve().parents[1] / "clip_model"
        prompt_checkpoint = resource_dir / "haze_prompt.pth"
        clip_checkpoint = resource_dir / "ViT-B-32.pt"
        missing = [str(path) for path in (prompt_checkpoint, clip_checkpoint) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing training resources: " + ", ".join(missing) + ". Download them as described in README.md.")
        embeddings = _load_prompt_embeddings(prompt_checkpoint)
        import clip

        self.clip_model, _ = clip.load(str(clip_checkpoint), device="cpu", jit=False)
        self.clip_model = self.clip_model.float().to(device)
        self.clip_model.requires_grad_(False)
        self.clip_model.eval()
        embeddings = embeddings.to(device)
        tokenized = clip.tokenize([" ".join(["X"] * 16)] * 2).to(device)
        with torch.no_grad():
            text_features = TextEncoder(self.clip_model)(embeddings, tokenized).float()
        if tuple(text_features.shape) != (2, 512):
            raise ValueError("Directional fusion requires a CLIP backbone with 512-dimensional image/text embeddings")
        self.register_buffer("text_features", text_features.detach())
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1))
        self.fusion = CrossAttentionFusion()
        self.w_positive = float(w_positive)
        self.w_negative = float(w_negative)
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        return self

    def preprocess(self, raw_images):
        resized = F.interpolate(raw_images, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        return (resized - self.clip_mean) / self.clip_std

    def forward(self, output_raw, input_raw):
        for name, value in (("output_raw", output_raw), ("input_raw", input_raw)):
            if value.ndim != 4 or value.shape[1] != 3 or not value.is_floating_point():
                raise ValueError(f"{name} must be an unnormalized floating point RGB BCHW tensor")
        if output_raw.shape[0] != input_raw.shape[0]:
            raise ValueError("Output and input batches must have equal batch sizes")
        with torch.no_grad():
            input_features = self.clip_model.encode_image(self.preprocess(input_raw)).float()
        output_features = self.clip_model.encode_image(self.preprocess(output_raw)).float()
        positive = self.text_features[1:2].expand(input_features.shape[0], -1)
        target = F.normalize(self.fusion(input_features, positive), dim=-1)
        output_features = F.normalize(output_features, dim=-1)
        negative = F.normalize(self.text_features[0:1], dim=-1)
        attract = 1 - (output_features * target).sum(dim=-1)
        repel = (output_features * negative).sum(dim=-1)
        return (self.w_positive * attract + self.w_negative * repel).mean()
