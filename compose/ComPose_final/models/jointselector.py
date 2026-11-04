import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def hand_frame_from_points(w, m, i):
    """Build rotation matrix from 3 points. Fully differentiable."""
    z = m - w
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    x = i - w
    x = x - z * (x * z).sum(dim=-1, keepdim=True)
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)  # (B, 3, 3)


def weighted_procrustes(P, Q, w):
    """Weighted Procrustes: find R that best aligns P to Q with weights w."""
    w_ = w.unsqueeze(-1)
    cp = P[:, 0:1, :]
    cq = Q[:, 0:1, :]
    Pc = P - cp
    Qc = Q - cq
    H = (w_ * Pc).transpose(1, 2) @ Qc
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.transpose(-1, -2) @ U.transpose(-1, -2)
    return R


class JointSelectorBlock(nn.Module):
    """Image self-attn → cross-attn (joints→image) + FFN."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Image self-attention
        self.img_self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.img_self_norm = nn.LayerNorm(d_model)

        # Cross-attention: joints read from image
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, img_tokens):
        # Image self-attn first
        h, _ = self.img_self_attn(img_tokens, img_tokens, img_tokens)
        img_tokens = self.img_self_norm(img_tokens + h)
        # Then joints read from refined image
        h, _ = self.cross_attn(x, img_tokens, img_tokens)
        x = self.cross_norm(x + h)
        x = self.ffn_norm(x + self.ffn(x))
        return x, img_tokens


class JointSelector(nn.Module):
    def __init__(self, d_model=384, n_joints=21, n_heads=6, n_layers=4,
                 mode='procrustes', tau_start=1.0, tau_end=0.1):
        super().__init__()
        self.n_joints = n_joints
        self.d_model = d_model
        self.mode = mode
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau = tau_start

        # Joint queries
        self.joint_queries = nn.Parameter(torch.randn(1, n_joints, d_model) * 0.02)

        # Mask fusion: cat(img_feat, mask_val) → proj back to d_model
        self.mask_fusion = nn.Linear(d_model + 1, d_model)

        # Frame embedding (frame i vs frame j)
        self.frame_embed = nn.Parameter(torch.randn(2, 1, d_model) * 0.02)

        # Fourier PE + learnable PE
        freq_dim = d_model // 4
        self.register_buffer(
            'fourier_freqs',
            1.0 / (10000 ** (torch.arange(0, freq_dim).float() / freq_dim))
        )
        self.joint_pe = nn.Parameter(torch.randn(1, n_joints, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            JointSelectorBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Rotation head (4 layers)
        self.head_w = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model // 8), nn.GELU(),
            nn.Linear(d_model // 8, 1),
        )

        # Alpha gate: from image global pool (4 layers)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model // 8), nn.GELU(),
            nn.Linear(d_model // 8, 1),
        )

        # ── Translation branch (per-frame) ──
        # Centroid encoder: 3D → D/4
        self.centroid_enc = nn.Sequential(
            nn.Linear(3, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model // 4),
        )
        # Translation head: img pool (D) + centroid feat (D/4) → delta (3)
        self.trans_head = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model), nn.GELU(),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, d_model // 8), nn.GELU(),
            nn.Linear(d_model // 8, 3),
        )

    def set_tau(self, progress):
        self.tau = self.tau_start + (self.tau_end - self.tau_start) * progress

    def fourier_encode_2d(self, coords):
        B, N, _ = coords.shape
        x, y = coords[..., 0], coords[..., 1]
        freqs = self.fourier_freqs.view(1, 1, -1)
        pe = torch.cat([
            torch.sin(x.unsqueeze(-1) * freqs * math.pi),
            torch.cos(x.unsqueeze(-1) * freqs * math.pi),
            torch.sin(y.unsqueeze(-1) * freqs * math.pi),
            torch.cos(y.unsqueeze(-1) * freqs * math.pi),
        ], dim=-1)
        return pe

    def _fuse_mask(self, img_feat, obj_mask, H, W):
        """Cat mask value to each image token, project back to D."""
        B, T, D = img_feat.shape
        mask_ds = F.adaptive_avg_pool2d(obj_mask.unsqueeze(1).float(), (H, W))  # (B, 1, H, W)
        mask_flat = mask_ds.view(B, T, 1)  # (B, T, 1)
        return self.mask_fusion(torch.cat([img_feat, mask_flat], dim=-1))  # (B, T, D)

    def _encode(self, dino_feat, joints_2d, obj_mask=None, dino_feat_j=None, obj_mask_j=None):
        B, H, W, D = dino_feat.shape
        img_tokens = dino_feat.view(B, H * W, D)

        if obj_mask is not None:
            img_tokens = self._fuse_mask(img_tokens, obj_mask, H, W)

        img_tokens = img_tokens + self.frame_embed[0]

        if dino_feat_j is not None:
            img_tokens_j = dino_feat_j.view(B, H * W, D)
            if obj_mask_j is not None:
                img_tokens_j = self._fuse_mask(img_tokens_j, obj_mask_j, H, W)
            img_tokens_j = img_tokens_j + self.frame_embed[1]
            img_tokens = torch.cat([img_tokens, img_tokens_j], dim=1)

        fourier_pe = self.fourier_encode_2d(joints_2d)
        x = self.joint_queries.expand(B, -1, -1) + self.joint_pe + fourier_pe
        for block in self.blocks:
            x, img_tokens = block(x, img_tokens)
        return x, img_tokens

    def _predict_trans(self, img_tokens, centroid):
        """Predict centroid-to-origin delta for one frame."""
        pool = img_tokens.mean(dim=1)  # (B, D)
        cent_feat = self.centroid_enc(centroid)  # (B, D//4)
        trans_in = torch.cat([pool, cent_feat], dim=-1)  # (B, D + D//4)
        return self.trans_head(trans_in)  # (B, 3)

    def forward(self, dino_feat, joints_2d, joints_3d,
                dino_feat_j=None, obj_mask=None, obj_mask_j=None,
                icp_centroid_i=None, icp_centroid_j=None):
        joint_out, img_out = self._encode(dino_feat, joints_2d, obj_mask=obj_mask,
                                            dino_feat_j=dino_feat_j, obj_mask_j=obj_mask_j)
        B = joint_out.shape[0]
        H, W = dino_feat.shape[1], dino_feat.shape[2]
        T = H * W

        # ── Alpha gate (from image mean pool) ──
        alpha = torch.sigmoid(self.alpha_mlp(img_out.mean(dim=1)).squeeze(-1))

        # ── Rotation (from joint tokens) ──
        logits = self.head_w(joint_out).squeeze(-1)
        w = F.softmax(logits / self.tau, dim=-1)

        # ── Translation (per-frame centroid→origin delta) ──
        trans_delta_i, trans_delta_j = None, None
        if icp_centroid_i is not None:
            trans_delta_i = self._predict_trans(img_out[:, :T], icp_centroid_i)
        if icp_centroid_j is not None and dino_feat_j is not None:
            trans_delta_j = self._predict_trans(img_out[:, T:2*T], icp_centroid_j)

        return w, alpha, (trans_delta_i, trans_delta_j), {'w': w}

    def compute_procrustes_R(self, joints_3d_i, joints_3d_j, w):
        return weighted_procrustes(joints_3d_i, joints_3d_j, w)
