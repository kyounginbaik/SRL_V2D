"""
Attention layers for JointSelectorTemporal.
Based on VGGT (vggt/layers/attention.py, block.py, mlp.py).
Includes 2D RoPE from VGGT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional


# ═══════════════════════════════════════════
# 2D RoPE (from VGGT)
# ═══════════════════════════════════════════

class PositionGetter:
    """Generates and caches 2D spatial positions for patches."""
    def __init__(self):
        self.cache: Dict[Tuple[int, int], Tensor] = {}

    def __call__(self, batch_size, height, width, device):
        if (height, width) not in self.cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self.cache[height, width] = torch.cartesian_prod(y, x)
        pos = self.cache[height, width]
        return pos.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D RoPE from VGGT."""
    def __init__(self, frequency=100.0):
        super().__init__()
        self.base_frequency = frequency
        self.freq_cache: Dict[Tuple, Tuple[Tensor, Tensor]] = {}

    def _get_freq(self, dim, seq_len, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self.freq_cache:
            exp = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exp)
            pos = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", pos, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.freq_cache[key] = (angles.cos(), angles.sin())
        return self.freq_cache[key]

    @staticmethod
    def _rotate(x):
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d(self, tokens, positions, cos, sin):
        c = F.embedding(positions, cos)[:, None, :, :]
        s = F.embedding(positions, sin)[:, None, :, :]
        return tokens * c + self._rotate(tokens) * s

    def forward(self, tokens, positions):
        """
        tokens: (B, n_heads, N, head_dim)
        positions: (B, N, 2) — y, x coords
        """
        feat_dim = tokens.size(-1) // 2
        max_pos = int(positions.max()) + 1
        cos, sin = self._get_freq(feat_dim, max_pos, tokens.device, tokens.dtype)
        vert, horiz = tokens.chunk(2, dim=-1)
        vert = self._apply_1d(vert, positions[..., 0], cos, sin)
        horiz = self._apply_1d(horiz, positions[..., 1], cos, sin)
        return torch.cat((vert, horiz), dim=-1)


# ═══════════════════════════════════════════
# Attention layers
# ═══════════════════════════════════════════

class Attention(nn.Module):
    """Multi-head self-attention with optional 2D RoPE."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, qk_norm=False, rope=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        x = F.scaled_dot_product_attention(q, k, v,
                                            dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q from x, KV from ctx."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True,
                 attn_drop=0.0, proj_drop=0.0, qk_norm=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        B, N, C = x.shape
        _, M, _ = ctx.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(ctx).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v,
                                            dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm self-attention + FFN with optional RoPE and layer scale."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 proj_bias=True, ffn_bias=True, qk_norm=False,
                 init_values=None, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              proj_bias=proj_bias, qk_norm=qk_norm, rope=rope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       bias=ffn_bias)

        if init_values is not None:
            self.ls1 = nn.Parameter(init_values * torch.ones(dim))
            self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.ls1 = self.ls2 = None

    def forward(self, x: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        h = self.attn(self.norm1(x), pos=pos)
        x = x + (self.ls1 * h if self.ls1 is not None else h)
        h = self.mlp(self.norm2(x))
        x = x + (self.ls2 * h if self.ls2 is not None else h)
        return x


class CrossBlock(nn.Module):
    """Pre-norm cross-attention + FFN with layer scale."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 proj_bias=True, ffn_bias=True, qk_norm=False,
                 init_values=None):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    proj_bias=proj_bias, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       bias=ffn_bias)

        if init_values is not None:
            self.ls1 = nn.Parameter(init_values * torch.ones(dim))
            self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.ls1 = self.ls2 = None

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        h = self.attn(self.norm_q(x), self.norm_kv(ctx))
        x = x + (self.ls1 * h if self.ls1 is not None else h)
        h = self.mlp(self.norm2(x))
        x = x + (self.ls2 * h if self.ls2 is not None else h)
        return x
