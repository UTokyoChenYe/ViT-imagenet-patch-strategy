import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from patching.patchify import bvh_patchify
import logging


class ImageNetBVHDataset(Dataset):
    def __init__(self, root, transform=None, max_nodes=512, patch_size=(8, 8, 3)):
        self.root = root
        self.transform = transform
        self.samples = []
        self.max_nodes = max_nodes
        self.patch_size = patch_size
        self.bad_images_count = 0

        classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in classes:
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except (Image.UnidentifiedImageError, OSError) as e:
            # 只在 rank 0 打印警告（避免多卡重复输出）
            import torch.distributed as dist
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logging.warning(f"[ImageNet] Skipping corrupted image: {path}")
            # 创建一张黑图占位，保证后续不会出错
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            self.bad_images_count += 1
        
        
        img_np = np.array(img)

        # ✅ bvh_patchify 已返回 (seq_patch, seq_pos, seq_size, adj)
        seq_patch, seq_pos, seq_size, adj = bvh_patchify(
            img_np, patch_size=self.patch_size, max_nodes=self.max_nodes
        )

        if seq_patch.shape[0] > self.max_nodes:
            seq_patch = seq_patch[:self.max_nodes]
            seq_pos = seq_pos[:self.max_nodes]
            seq_size = seq_size[:self.max_nodes]
            adj = adj[:self.max_nodes, :self.max_nodes]

        # ✅ 转换为 Tensor（bvh_patchify 已经返回 torch.Tensor 就不用重复转）
        if not isinstance(seq_patch, torch.Tensor):
            seq_patch = torch.tensor(seq_patch, dtype=torch.float32)
        if not isinstance(seq_pos, torch.Tensor):
            seq_pos = torch.tensor(seq_pos, dtype=torch.float32)
        if not isinstance(seq_size, torch.Tensor):
            seq_size = torch.tensor(seq_size, dtype=torch.float32)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.bool)

        # ✅ 返回完整 batch
        return {
            "patches": seq_patch,   # [N, C, H, W]
            "positions": seq_pos,   # [N, 2]
            "sizes": seq_size,      # [N, 1]
            "adj": adj,             # [N, N] 真实 BVH 邻接矩阵
            "label": torch.tensor(label, dtype=torch.long)
        }


