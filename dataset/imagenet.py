import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from patching.patchify import bvh_patchify

class ImageNetBVHDataset(Dataset):
    def __init__(self, root, transform=None, max_nodes=512, patch_size=(8,8,3)):
        self.root = root
        self.transform = transform
        self.samples = []
        self.max_nodes = max_nodes
        self.patch_size = patch_size

        classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        for cls in classes:
            for f in os.listdir(os.path.join(root, cls)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(root, cls, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        seq_patch, seq_pos, seq_size = bvh_patchify(img_np, patch_size=self.patch_size, max_nodes=self.max_nodes)
        return {
            "patches": seq_patch,
            "positions": seq_pos,
            "sizes": seq_size,
            "label": torch.tensor(label, dtype=torch.long)
        }
