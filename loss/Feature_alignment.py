import torch
from torch import nn
from torch.nn import functional as F


def simplify_feature_map(feature_map, pool_size=2):
    return feature_map


def convert_to_distribution(feature_map):
    n, c, h, w = feature_map.shape
    distribution = F.softmax(feature_map.reshape(n, c, -1), dim=-1)
    return distribution.reshape(n, c, h, w)


def compute_emd_loss(teacher_dist, student_dist):
    return torch.mean((teacher_dist - student_dist) ** 2)


class GaussianLoss(nn.Module):
    def forward(self, teacher_feat, student_feat):
        teacher_mean = teacher_feat.mean(dim=(2, 3), keepdim=True)
        student_mean = student_feat.mean(dim=(2, 3), keepdim=True)
        teacher_var = teacher_feat.var(dim=(2, 3), keepdim=True, unbiased=False)
        student_var = student_feat.var(dim=(2, 3), keepdim=True, unbiased=False)
        return ((teacher_mean - student_mean) ** 2).mean() + ((teacher_var - student_var) ** 2).mean()


def compute_attention_weights(teacher_feats, student_feats):
    similarities = [
        F.cosine_similarity(t.reshape(t.shape[0], -1), s.reshape(s.shape[0], -1), dim=-1).mean()
        for t, s in zip(teacher_feats, student_feats)
    ]
    return F.softmax(torch.stack(similarities), dim=0)


def compute_local_emd_loss(teacher_feat, student_feat, patch_size=4):
    teacher_patches = teacher_feat.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    student_patches = student_feat.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return compute_emd_loss(teacher_patches, student_patches)


class FA(nn.Module):
    def __init__(self, patch_size=4, use_local_emd=True, pool_size=2):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.patch_size = patch_size
        self.use_local_emd = use_local_emd
        self.pool_size = pool_size
        self.gaussian_loss_fn = GaussianLoss()

    def forward(self, teacher_feats, student_feats):
        if not isinstance(teacher_feats, (list, tuple)) or not isinstance(student_feats, (list, tuple)):
            raise ValueError("FA expects lists or tuples of BCHW teacher and student features")
        if not teacher_feats or len(teacher_feats) != len(student_feats):
            raise ValueError("FA expects equal, nonzero numbers of teacher and student features")
        for i, (teacher, student) in enumerate(zip(teacher_feats, student_feats)):
            if teacher.ndim != 4 or teacher.shape != student.shape:
                raise ValueError(f"FA feature pair {i} must have equal BCHW shapes: {teacher.shape} vs {student.shape}")
            if self.use_local_emd and min(teacher.shape[-2:]) < self.patch_size:
                raise ValueError(f"FA feature pair {i} is smaller than patch_size={self.patch_size}")

        attention_weights = compute_attention_weights(teacher_feats, student_feats)
        total_loss = 0.0
        for i, (teacher, student) in enumerate(zip(teacher_feats, student_feats)):
            teacher_simple = simplify_feature_map(teacher, self.pool_size)
            student_simple = simplify_feature_map(student, self.pool_size)
            emd_loss = compute_emd_loss(
                convert_to_distribution(teacher_simple), convert_to_distribution(student_simple)
            )
            if self.use_local_emd:
                emd_loss = emd_loss + compute_local_emd_loss(teacher_simple, student_simple, self.patch_size)
            total_loss = total_loss + attention_weights[i] * (
                emd_loss + self.gaussian_loss_fn(teacher, student)
            )
        return total_loss
