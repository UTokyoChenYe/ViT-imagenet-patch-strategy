import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


# -------------------------------
# 位置编码 (pos + size)
# -------------------------------
class BVHPosEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, pos, size):
        return self.mlp(torch.cat([pos, size], dim=-1))  # (B,N,D)


# -------------------------------
# Patch 嵌入 (像素块 -> 向量)
# -------------------------------
class BVHPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=(8, 8)):
        super().__init__()
        H, W = patch_size[:2]
        self.linear = nn.Linear(in_chans * H * W, embed_dim)

    def forward(self, x):
        B, N, C, H, W = x.shape
        return self.linear(x.view(B, N, -1))  # (B,N,D)


# -------------------------------
# DropPath 用于正则
# -------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# -------------------------------
# 支持邻接掩码的 Attention 模块
# -------------------------------
class MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,L,Dh)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,L,L)

        if attn_mask is not None:
            big_neg = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(~attn_mask.unsqueeze(1), big_neg)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -------------------------------
# Transformer Block（带邻接支持）
# -------------------------------
class MaskedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskedAttention(dim, num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -------------------------------
# 主模型：BVHViT
# -------------------------------
class BVHViT(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, num_classes=1000, mlp_ratio=4.0):
        super().__init__()
        self.embed = BVHPatchEmbed()
        self.pos_embed = BVHPosEmbed(embed_dim)

        base_vit = VisionTransformer(img_size=224, patch_size=16,
                                     embed_dim=embed_dim, depth=depth,
                                     num_heads=num_heads, num_classes=num_classes)
        self.head = base_vit.head
        self.cls_token = base_vit.cls_token
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 自定义 blocks
        self.blocks = nn.ModuleList([
            MaskedBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        del base_vit  # 删除 timm 内部结构防止 DDP 报错

    def _build_attn_mask(self, adj, B, N, device):
        """根据邻接矩阵构造注意力掩码 (B,1+N,1+N)"""
        if adj is None:
            return None
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        adj_bool = (adj.to(torch.bool) | eye)
        top = torch.ones(B, 1, N, device=device, dtype=torch.bool)
        left = torch.ones(B, N, 1, device=device, dtype=torch.bool)
        cls_self = torch.ones(B, 1, 1, device=device, dtype=torch.bool)
        upper = torch.cat([cls_self, top], dim=2)
        lower = torch.cat([left, adj_bool], dim=2)
        return torch.cat([upper, lower], dim=1)

    def forward(self, batch):
        patches, pos, size = batch["patches"], batch["positions"], batch["sizes"]
        adj = batch.get("adj", None)
        B, N = patches.shape[:2]
        device = patches.device

        x = self.embed(patches) + self.pos_embed(pos, size)
        cls_token = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls_token, x], dim=1)

        attn_mask = self._build_attn_mask(adj, B, N, device)
        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = self.norm(x)
        return self.head(x[:, 0])

