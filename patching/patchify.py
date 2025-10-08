# patching/patchify.py
import numpy as np
import torch
import cv2
from .bvh_tree import BVH2D, AABB2D, BuildParams2D, otsu_threshold, connected_components

def bvh_patchify(img_np, patch_size=(8,8,3), max_nodes=512, max_leaf_prims=6, bins=16):
    """输入图像 → BVH patch 序列"""
    h, w = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    th = otsu_threshold(gray)
    binary = (gray < th).astype(np.uint8)
    boxes_px = connected_components(binary, min_area=max(50, (w*h)//5000))
    if not boxes_px:
        boxes_px = [(w*0.45,h*0.45,w*0.55,h*0.55)]
    aabbs = [AABB2D(x0,y0,x1,y1) for (x0,y0,x1,y1) in boxes_px]
    bvh = BVH2D(aabbs, BuildParams2D(max_leaf_prims=max_leaf_prims,bins=bins), max_nodes=max_nodes)
    seq_patch, seq_size, seq_pos, adj = bvh.serialize(img_np, size=patch_size)
    seq_patch = torch.from_numpy(np.stack(seq_patch)).permute(0,3,1,2).float()/255.0
    seq_pos = torch.tensor(seq_pos, dtype=torch.float32)
    seq_size = torch.tensor(seq_size, dtype=torch.float32).unsqueeze(-1)
    adj = torch.tensor(adj, dtype=torch.bool)  
    return seq_patch, seq_pos, seq_size, adj
