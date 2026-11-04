"""
JointSelector with temporal consistency.
2-stage architecture following VGGT's alternating attention pattern.

Image Encoder: alternating frame(local) / global self-attention with register tokens.
Joint Encoder: cross-attn from image + global self-attn across frames.
Heads: alpha from image, weight from joint, translation from image+centroid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.attention import Block, CrossBlock, RotaryPositionEmbedding2D, PositionGetter


def weighted_procrustes(P, Q, w):
    w_ = w.unsqueeze(-1)
    cp = P[:, 0:1, :]
    cq = Q[:, 0:1, :]
    H = (w_ * (P - cp)).transpose(1, 2) @ (Q - cq)
    U, S, Vt = torch.linalg.svd(H)
    return Vt.transpose(-1, -2) @ U.transpose(-1, -2)


class JointSelectorTemporal(nn.Module):
    def __init__(self, d_model=384, n_joints=21, n_heads=6,
                 n_img_layers=4, n_joint_layers=2,
                 mlp_ratio=4.0, qk_norm=True,
                 init_values=0.01,
                 tau_start=1.0, tau_end=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_joints = n_joints
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau = tau_start

        # ── Image preprocessing ──
        self.mask_fusion = nn.Linear(d_model + 1, d_model)

        # 2D RoPE for spatial position in attention
        self.rope = RotaryPositionEmbedding2D(frequency=100.0)
        self.position_getter = PositionGetter()

        # ── Image Encoder: alternating frame/global (VGGT-style) ──
        self.img_frame_blocks = nn.ModuleList([
            Block(d_model, n_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                  init_values=init_values, rope=self.rope)
            for _ in range(n_img_layers)
        ])
        self.img_global_blocks = nn.ModuleList([
            Block(d_model, n_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                  init_values=init_values, rope=self.rope)
            for _ in range(n_img_layers)
        ])

        # ── Joint Encoder ──
        # Joint embedding
        self.joint_proj = nn.Linear(3, d_model)
        self.joint_pe = nn.Parameter(torch.randn(1, n_joints, d_model) * 0.02)
        freq_dim = d_model // 4
        self.register_buffer(
            'fourier_freqs',
            1.0 / (10000 ** (torch.arange(0, freq_dim).float() / freq_dim))
        )

        # Joint cross-attn (reads from image) + joint global self-attn
        self.joint_cross_blocks = nn.ModuleList([
            CrossBlock(d_model, n_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                       init_values=init_values)
            for _ in range(n_joint_layers)
        ])
        self.joint_global_blocks = nn.ModuleList([
            Block(d_model, n_heads, mlp_ratio=mlp_ratio, qk_norm=qk_norm,
                  init_values=init_values)
            for _ in range(n_joint_layers)
        ])

        # ── Heads ──
        # Alpha: image pair pool → scalar
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        # Weight: joint token → per-joint scalar
        self.rot_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        # Translation: per-frame image pool + centroid → delta
        self.centroid_enc = nn.Sequential(
            nn.Linear(3, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model // 4),
        )
        self.trans_head = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model), nn.GELU(),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 3),
        )

    def set_tau(self, progress):
        self.tau = self.tau_start + (self.tau_end - self.tau_start) * progress

    @staticmethod
    def _frame_sinusoidal_pe(n_frames, d_model, device):
        """Sinusoidal PE for frame index. Works for any N."""
        pos = torch.arange(n_frames, device=device).unsqueeze(1).float()  # (N, 1)
        dim = torch.arange(0, d_model, 2, device=device).float()          # (D/2,)
        freq = 1.0 / (10000 ** (dim / d_model))
        pe = torch.zeros(n_frames, d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return pe  # (N, D)

    def _fuse_mask(self, img_feat, obj_mask, H, W):
        B, T, D = img_feat.shape
        mask_ds = F.adaptive_avg_pool2d(obj_mask.unsqueeze(1).float(), (H, W))
        return self.mask_fusion(torch.cat([img_feat, mask_ds.view(B, T, 1)], dim=-1))

    def fourier_encode_2d(self, coords):
        x, y = coords[..., 0], coords[..., 1]
        freqs = self.fourier_freqs.view(1, 1, -1)
        return torch.cat([
            torch.sin(x.unsqueeze(-1) * freqs * math.pi),
            torch.cos(x.unsqueeze(-1) * freqs * math.pi),
            torch.sin(y.unsqueeze(-1) * freqs * math.pi),
            torch.cos(y.unsqueeze(-1) * freqs * math.pi),
        ], dim=-1)

    def forward(self, dino_feats, obj_masks, joints_2d, joints_3d, centroids,
                pairs=None):
        """
        Args:
            dino_feats: (B, N, H, W, D)
            obj_masks: (B, N, Himg, Wimg)
            joints_2d: (B, N, 21, 2)
            joints_3d: (B, N, 21, 3)
            centroids: (B, N, 3)
            pairs: list of (i, j), default [(0,1),(1,2),(0,2)]
        """
        B, N, H, W, D = dino_feats.shape
        T = H * W

        if pairs is None:
            pairs = [(0, 1), (1, 2), (0, 2)] if N == 3 else [(i, i+1) for i in range(N-1)]

        # ══════════════════════════════════════
        # Stage 1: Image Encoder
        # ══════════════════════════════════════

        # Image tokens: (B*N, T, D)
        img_tokens = dino_feats.view(B * N, T, D)
        masks_flat = obj_masks.view(B * N, *obj_masks.shape[2:])
        img_tokens = self._fuse_mask(img_tokens, masks_flat, H, W)

        # Frame PE (sinusoidal, any N)
        fpe = self._frame_sinusoidal_pe(N, D, img_tokens.device)  # (N, D)
        fpe = fpe.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 1, D)
        img_tokens = img_tokens + fpe

        # 2D RoPE positions for patches
        pos_local = self.position_getter(B * N, H, W, img_tokens.device)  # (B*N, T, 2)
        pos_global = pos_local.view(B, N, T, 2).view(B, N * T, 2)          # (B, N*T, 2)

        # Alternating frame/global attention (VGGT-style)
        for i in range(len(self.img_frame_blocks)):
            # Frame (local): (B*N, T, D) with local positions
            img_tokens = self.img_frame_blocks[i](img_tokens, pos=pos_local)

            # Global: (B, N*T, D) with global positions
            img_tokens = img_tokens.view(B, N * T, D)
            img_tokens = self.img_global_blocks[i](img_tokens, pos=pos_global)
            img_tokens = img_tokens.view(B * N, T, D)

        img_patch = img_tokens  # (B*N, T, D)

        # Per-frame pool (from patches only)
        img_per_frame = img_patch.view(B, N, T, D)
        frame_pools = img_per_frame.mean(dim=2)  # (B, N, D)

        # ══════════════════════════════════════
        # Stage 2: Joint Encoder
        # ══════════════════════════════════════

        # Joint tokens: (B*N, 21, D)
        j2d_flat = joints_2d.view(B * N, 21, 2)
        j3d_flat = joints_3d.view(B * N, 21, 3)
        joint_tokens = (self.joint_proj(j3d_flat) + self.joint_pe
                        + self.fourier_encode_2d(j2d_flat))

        # Cross-attn (joint ← image) + global self-attn (joint across frames)
        for i in range(len(self.joint_cross_blocks)):
            # Joint reads from image patches (per-frame)
            joint_tokens = self.joint_cross_blocks[i](joint_tokens, img_patch)

            # Joint global: (B, N*21, D)
            joint_tokens = joint_tokens.view(B, N * 21, D)
            joint_tokens = self.joint_global_blocks[i](joint_tokens)
            joint_tokens = joint_tokens.view(B * N, 21, D)

        joint_per_frame = joint_tokens.view(B, N, 21, D)

        # ══════════════════════════════════════
        # Heads
        # ══════════════════════════════════════

        # Translation: per-frame (from image)
        cent_feat = self.centroid_enc(centroids)
        trans_deltas = self.trans_head(torch.cat([frame_pools, cent_feat], dim=-1))

        # Per-pair predictions
        results = []
        for (fi, fj) in pairs:
            # Weight: from joint tokens (avg pair)
            joint_pair = (joint_per_frame[:, fi] + joint_per_frame[:, fj]) / 2
            logits = self.rot_head(joint_pair).squeeze(-1)  # (B, 21)
            w = F.softmax(logits / self.tau, dim=-1)

            # Alpha: from image pool (no joint info)
            pair_feat = torch.cat([frame_pools[:, fi], frame_pools[:, fj]], dim=-1)
            alpha = torch.sigmoid(self.alpha_mlp(pair_feat).squeeze(-1))

            # Procrustes
            R_hand = weighted_procrustes(joints_3d[:, fi], joints_3d[:, fj], w)

            results.append({'w': w, 'alpha': alpha, 'R_hand': R_hand,
                           'fi': fi, 'fj': fj})

        return results, trans_deltas
