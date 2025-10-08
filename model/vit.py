import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class BVHPosEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, pos, size):
        return self.mlp(torch.cat([pos, size], dim=-1))

class BVHPatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=(8,8)):
        super().__init__()
        H, W = patch_size[:2]
        self.linear = nn.Linear(in_chans*H*W, embed_dim)

    def forward(self, x):
        B, N, C, H, W = x.shape
        return self.linear(x.view(B, N, -1))

class BVHViT(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.embed = BVHPatchEmbed()
        self.pos_embed = BVHPosEmbed(embed_dim)
        self.vit = VisionTransformer(img_size=224, patch_size=16,
                                     embed_dim=embed_dim, depth=depth,
                                     num_heads=num_heads, num_classes=num_classes)

    def forward(self, batch):
        patches, pos, size = batch['patches'], batch['positions'], batch['sizes']
        B = patches.shape[0]
        x = self.embed(patches) + self.pos_embed(pos, size)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed[:, :x.size(1), :]
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        return self.vit.head(x[:, 0])
