import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
from collections import deque
import cv2 as cv


from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

# ---------------- Utilities ----------------
def ensure_demo_image(path: str, size=(512, 512)):
    """Create a synthetic demo image with a few rectangles and blobs if the given path does not exist."""
    if os.path.exists(path):
        return path
    w, h = size
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # draw some rectangles
    rects = [ (50, 60, 160, 150),
              (220, 40, 320, 180),
              (350, 300, 470, 430),
              (80, 300, 200, 420) ]
    for x0,y0,x1,y1 in rects:
        draw.rectangle([x0,y0,x1,y1], fill=(80,80,80))
    # draw some ellipses
    ellipses = [ (300, 220, 360, 280), (120, 200, 200, 260) ]
    for x0,y0,x1,y1 in ellipses:
        draw.ellipse([x0,y0,x1,y1], fill=(60,60,60))
    img.save(path)
    return path

def otsu_threshold(gray: np.ndarray) -> int:
    """Compute Otsu threshold for a uint8 grayscale image."""
    # histogram
    hist, _ = np.histogram(gray, bins=256, range=(0,256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold

def connected_components(binary: np.ndarray, min_area: int = 50):
    """Simple 8-connected component labeling. Returns list of bounding boxes (xmin,ymin,xmax,ymax)."""
    H, W = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    boxes = []
    # neighbor offsets (8-connectivity)
    neigh = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for y in range(H):
        for x in range(W):
            if binary[y, x] and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                minx = maxx = x
                miny = maxy = y
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    if cx < minx: minx = cx
                    if cx > maxx: maxx = cx
                    if cy < miny: miny = cy
                    if cy > maxy: maxy = cy
                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if area >= min_area:
                    boxes.append((minx, miny, maxx, maxy))
    return boxes

# ---------------- AABB 2D ----------------

class AABB2D:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        """
        表示一个二维轴对齐包围盒 (Axis-Aligned Bounding Box in 2D)
        
        参数:
            xmin, ymin: 左下角坐标
            xmax, ymax: 右上角坐标
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def expand(self, other: 'AABB2D'):
        """将当前AABB扩展，使其包含另一个AABB"""
        self.xmin = min(self.xmin, other.xmin)
        self.ymin = min(self.ymin, other.ymin)
        self.xmax = max(self.xmax, other.xmax)
        self.ymax = max(self.ymax, other.ymax)

    @staticmethod
    def union(a: 'AABB2D', b: 'AABB2D') -> 'AABB2D':
        """返回两个AABB的并集AABB（新对象）"""
        return AABB2D(
            min(a.xmin, b.xmin), min(a.ymin, b.ymin),
            max(a.xmax, b.xmax), max(a.ymax, b.ymax)
        )

    def centroid(self) -> Tuple[float, float]:
        """返回AABB中心点"""
        return ((self.xmin + self.xmax) * 0.5,
                (self.ymin + self.ymax) * 0.5)

    def width(self) -> float:
        """返回宽度"""
        return max(0.0, self.xmax - self.xmin)

    def height(self) -> float:
        """返回高度"""
        return max(0.0, self.ymax - self.ymin)

    def area(self) -> float:
        """返回面积"""
        return self.width() * self.height()

    def perimeter(self) -> float:
        """返回周长"""
        w, h = self.width(), self.height()
        return 2.0 * (w + h)

    def intersects(self, other: 'AABB2D') -> bool:
        """判断是否与另一个AABB相交"""
        return not (self.xmax < other.xmin or self.xmin > other.xmax or
                    self.ymax < other.ymin or self.ymin > other.ymax)

    def __repr__(self):
        return f"AABB2D(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"


# ---------------- BVH Node ----------------
class BVHNode2D:
    def __init__(
        self,
        bbox: 'AABB2D',
        left: Optional['BVHNode2D'] = None,
        right: Optional['BVHNode2D'] = None,
        indices: Optional[List[int]] = None
    ):
        self.bbox = bbox
        self.left = left
        self.right = right
        self.indices = indices

    @property
    def is_leaf(self) -> bool:
        return self.indices is not None

# ---------------- 构建参数 ----------------
class BuildParams2D:
    def __init__(
        self,
        max_leaf_prims: int = 4,
        max_depth: int = 32,
        bins: int = 16,
        Ct: float = 1.0,
        Ci: float = 1.0,
        measure: Callable[['AABB2D'], float] = None
    ):
        self.max_leaf_prims = max_leaf_prims
        self.max_depth = max_depth
        self.bins = bins
        self.Ct = Ct
        self.Ci = Ci
        # 默认使用 AABB2D.perimeter
        self.measure = measure if measure is not None else AABB2D.perimeter

# ---------------- 主构建器 ----------------
class BVH2D:
    def __init__(self, boxes: List[AABB2D], params: BuildParams2D = BuildParams2D(), max_nodes: int = 1024):
        assert len(boxes) > 0
        self.boxes = boxes
        self.params = params
        self.max_nodes = max_nodes         # 存下最大节点数
        self.node_count = 0                # 节点计数器
        idx = list(range(len(boxes)))
        self.root = self._build(idx, 0)

    def _node_bbox(self, indices: List[int])->AABB2D:
        b = self.boxes[indices[0]]
        xmin,ymin,xmax,ymax = b.xmin,b.ymin,b.xmax,b.ymax
        for i in indices[1:]:
            bb = self.boxes[i]
            xmin = min(xmin, bb.xmin); ymin = min(ymin, bb.ymin)
            xmax = max(xmax, bb.xmax); ymax = max(ymax, bb.ymax)
        return AABB2D(xmin,ymin,xmax,ymax)

    def _leaf_cost(self, n:int)->float:
        return n * self.params.Ci

    def _build(self, indices: List[int], depth:int)->BVHNode2D:
        if self.node_count >= self.max_nodes:
            node_bbox = self._node_bbox(indices)
            return BVHNode2D(node_bbox, indices=indices)  # 超出限制直接建叶子
        
        self.node_count += 1  # 每建一个节点加1
        node_bbox = self._node_bbox(indices)

        # 叶子早停
        if len(indices) <= self.params.max_leaf_prims or depth >= self.params.max_depth:
            return BVHNode2D(node_bbox, indices=indices)

        best = {'cost': self._leaf_cost(len(indices)), 'axis': -1, 'threshold': None}
        parent_measure = max(1e-12, self.params.measure(node_bbox))

        # 尝试 x(0)/y(1) 两轴 + binned SAH
        for axis in (0,1):
            cand = self._binned_sah_best_split(indices, node_bbox, axis, parent_measure)
            if cand and cand['cost'] < best['cost']:
                best = cand

        # 没有更优切分 → 建叶子
        if best['axis'] == -1:
            return BVHNode2D(node_bbox, indices=indices)

        # 按阈值划分
        left,right = [],[]
        axis = best['axis']; th = best['threshold']
        for i in indices:
            cx,cy = self.boxes[i].centroid()
            key = cx if axis==0 else cy
            (left if key<=th else right).append(i)

        # 退化保护
        if len(left)==0 or len(right)==0:
            indices.sort(key=lambda i: self.boxes[i].centroid()[axis])
            mid = len(indices)//2
            left, right = indices[:mid], indices[mid:]

        lnode = self._build(left,  depth+1)
        rnode = self._build(right, depth+1)
        return BVHNode2D(node_bbox, left=lnode, right=rnode)

    def _binned_sah_best_split(self, indices: List[int], node_bbox:AABB2D, axis:int, parent_measure:float):
        B = self.params.bins
        cs = [ (self.boxes[i].centroid()[axis], i) for i in indices ]
        minC = min(cs)[0]; maxC = max(cs)[0]
        if maxC - minC < 1e-12:  # 所有质心重合
            return None

        # 分箱
        bins_count = [0]*B
        bins_bbox: List[Optional[AABB2D]] = [None]*B
        scale = B / (maxC - minC)
        for c,i in cs:
            b = int((c - minC) * scale)
            if b==B: b = B-1
            bins_count[b] += 1
            bins_bbox[b] = self.boxes[i] if bins_bbox[b] is None else AABB2D.union(bins_bbox[b], self.boxes[i])

        # 前后缀聚合
        pre_cnt = [0]*B; pre_box=[None]*B
        agg_cnt=0; agg_box=None
        for k in range(B):
            if bins_count[k]>0:
                agg_box = bins_bbox[k] if agg_box is None else AABB2D.union(agg_box, bins_bbox[k])
            agg_cnt += bins_count[k]
            pre_cnt[k] = agg_cnt
            pre_box[k] = agg_box

        suf_cnt = [0]*B; suf_box=[None]*B
        agg_cnt=0; agg_box=None
        for k in reversed(range(B)):
            if bins_count[k]>0:
                agg_box = bins_bbox[k] if agg_box is None else AABB2D.union(agg_box, bins_bbox[k])
            agg_cnt += bins_count[k]
            suf_cnt[k] = agg_cnt
            suf_box[k] = agg_box

        # 评估切点（bin 边界）
        best = None
        Ct, Ci = self.params.Ct, self.params.Ci
        measure = self.params.measure
        for cut in range(B-1):
            Lc, Rc = pre_cnt[cut], suf_cnt[cut+1]
            if Lc==0 or Rc==0: 
                continue
            Lbox, Rbox = pre_box[cut], suf_box[cut+1]
            cost = Ct + (measure(Lbox)/parent_measure)*Lc*Ci + (measure(Rbox)/parent_measure)*Rc*Ci
            if (best is None) or (cost < best['cost']):
                threshold = minC + (cut+1) * (maxC - minC)/B
                best = {'cost': cost, 'axis': axis, 'threshold': threshold}
        return best

    def serialize(self, img: np.ndarray, size=(8,8,3)) -> Tuple[List[np.ndarray], List[int], List[Tuple[float, float]], np.ndarray]:
        """
        将 BVH 中的节点序列化为 patch 序列，同时输出 patch 大小、位置、邻接矩阵。

        Args:
            img (np.ndarray): 输入图像 (H, W, C)
            size (tuple): 输出 patch 尺寸 (h, w, c)

        Returns:
            seq_patch: 所有节点对应的 patch（缩放至指定 size）
            seq_size: 每个 patch 对应的原始 bbox 宽度（可用作 scale encoding）
            seq_pos: 每个 patch 的 bbox 中心位置（可用于位置编码）
            adj_matrix: 邻接矩阵 (N, N)，包含父子连接
        """
        h2, w2, c2 = size
        seq_patch, seq_size, seq_pos = [], [], []
        node_list = []
        parent_map = {}

        # 遍历整棵树，记录节点编号和父子关系
        def dfs(node, parent_idx=None):
            idx = len(node_list)
            node_list.append(node)
            if parent_idx is not None:
                parent_map[idx] = parent_idx
            if not node.is_leaf:
                dfs(node.left, idx)
                dfs(node.right, idx)
        dfs(self.root)
        N = len(node_list)

        # 提取 patch 内容、大小、中心位置
        for node in node_list:
            bbox = node.bbox
            patch = img[int(bbox.ymin):int(bbox.ymax), int(bbox.xmin):int(bbox.xmax)]
            h1, w1 = patch.shape[:2]
            if h1 == 0 or w1 == 0:
                patch = np.zeros((h2, w2, c2), dtype=np.uint8)
            else:
                patch = cv.resize(patch, (w2, h2), interpolation=cv.INTER_NEAREST)
            if patch.ndim == 2:
                patch = np.expand_dims(patch, axis=-1)
            seq_patch.append(patch)
            seq_size.append(int(bbox.width()))
            seq_pos.append(bbox.centroid())

        # 邻接矩阵 (父子连接)
        adj = np.zeros((N, N), dtype=np.uint8)
        for child, parent in parent_map.items():
            adj[parent, child] = 1
            adj[child, parent] = 1  # 可选：若是无向图

        return seq_patch, seq_size, seq_pos, adj



    # ----------- 可视化：画 BVH 框 -----------
    def draw_bvh(self, ax, depth_color=False, linewidth=1.2):
        q = deque([(self.root,0)])
        while q:
            node, d = q.popleft()
            w,h = node.bbox.width(), node.bbox.height()
            rect = patches.Rectangle((node.bbox.xmin, node.bbox.ymin), w, h,
                                     fill=False, linewidth=linewidth, edgecolor="green")
            if depth_color:
                rect.set_alpha(max(0.1, 0.8/(1+d)))
            ax.add_patch(rect)
            if not node.is_leaf:
                q.append((node.left,d+1)); q.append((node.right,d+1))

# ---------------- Main Demo ----------------
# if __name__ == "__main__":
#     # 1) Load or create image
#     img_path = "/storage/chenye/data/ImageNet/val/ILSVRC2010_val_00000136.JPEG"   # <- 如果你上传图片到此路径，将会用你的图片
#     ensure_demo_image(img_path)             # 如果没有，会自动生成一张演示图

#     # 2) Read and preprocess
#     img = Image.open(img_path).convert("RGB")
#     # Optionally downscale very large images for speed
#     MAX_SIDE = 768
#     w, h = img.size
#     scale = 1.0
#     if max(w,h) > MAX_SIDE:
#         scale = MAX_SIDE / max(w,h)
#         img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
#         w, h = img.size

#     gray = np.array(img.convert("L"))
#     th = otsu_threshold(gray)
#     binary = (gray < th).astype(np.uint8)  # objects assumed darker than background

#     # 3) Extract bounding boxes via connected components
#     boxes_px = connected_components(binary, min_area=max(50, (w*h)//5000))  # adaptive min area
#     if not boxes_px:
#         # fallback: try inverse if background assumption was wrong
#         binary_inv = 1 - binary
#         boxes_px = connected_components(binary_inv, min_area=max(50, (w*h)//5000))

#     # 4) Build BVH over pixel-space AABBs
#     aabbs = [AABB2D(x0, y0, x1, y1) for (x0,y0,x1,y1) in boxes_px]
#     if not aabbs:
#         # Ensure at least one box to avoid errors; add a tiny box in the center
#         aabbs = [AABB2D(w*0.45, h*0.45, w*0.55, h*0.55)]
#     bvh = BVH2D(aabbs, BuildParams2D(max_leaf_prims=6, bins=16, measure=AABB2D.perimeter), max_nodes=512)

#     # 5) Plot: image + original boxes (red) + BVH boxes (gray)
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(img)
#     for b in aabbs:
#         ax.add_patch(patches.Rectangle((b.xmin, b.ymin), b.width(), b.height(),
#                                     fill=False, linewidth=1.0, edgecolor="red"))
#     bvh.draw_bvh(ax, depth_color=False, linewidth=1.2)
#     ax.set_title("Objects (red) + BVH bounding boxes (gray)")
#     ax.set_axis_off()

#     out_path = "./bvh_on_image.png"
#     fig.savefig(out_path, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import networkx as nx
    import cv2
    from PIL import Image
    import numpy as np

    # 1) Load or create image
    img_path = "/storage/chenye/data/ImageNet/val/ILSVRC2010_val_00000136.JPEG"
    ensure_demo_image(img_path)

    # 2) Read and preprocess
    img = Image.open(img_path).convert("RGB")
    MAX_SIDE = 768
    w, h = img.size
    scale = 1.0
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        w, h = img.size

    gray = np.array(img.convert("L"))
    th = otsu_threshold(gray)
    binary = (gray < th).astype(np.uint8)

    # 3) Extract bounding boxes via connected components
    boxes_px = connected_components(binary, min_area=max(50, (w * h) // 5000))
    if not boxes_px:
        binary_inv = 1 - binary
        boxes_px = connected_components(binary_inv, min_area=max(50, (w * h) // 5000))

    # 4) Build BVH over pixel-space AABBs
    aabbs = [AABB2D(x0, y0, x1, y1) for (x0, y0, x1, y1) in boxes_px]
    if not aabbs:
        aabbs = [AABB2D(w * 0.45, h * 0.45, w * 0.55, h * 0.55)]
    bvh = BVH2D(aabbs, BuildParams2D(max_leaf_prims=6, bins=16, measure=AABB2D.perimeter), max_nodes=512)

    # 5) Plot: image + original boxes (red) + BVH boxes (gray)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    for b in aabbs:
        ax.add_patch(patches.Rectangle((b.xmin, b.ymin), b.width(), b.height(),
                                       fill=False, linewidth=1.0, edgecolor="red"))
    bvh.draw_bvh(ax, depth_color=False, linewidth=1.2)
    ax.set_title("Objects (red) + BVH bounding boxes (gray)")
    ax.set_axis_off()
    fig.savefig("./bvh_on_image.png", dpi=200, bbox_inches="tight")

    # 6) Serialization to patch sequence
    print("📦 Serializing BVH...")
    patch_size = (8, 8, 3)
    seq_patch, seq_size, seq_pos, adj = bvh.serialize(np.array(img), size=patch_size)

    print(f"✅ Patch count: {len(seq_patch)}")
    print(f"📏 Size list: {seq_size[:5]} ...")
    print(f"📍 Position list: {seq_pos[:5]} ...")
    print(f"🧱 Adjacency shape: {adj.shape}, nonzeros: {np.count_nonzero(adj)}")

    # 7) Patch grid image (前 64 个 patch 可视化)
    patch_h, patch_w, patch_c = patch_size
    grid_side = int(np.ceil(np.sqrt(min(len(seq_patch), 64))))
    vis_grid = np.zeros((grid_side * patch_h, grid_side * patch_w, patch_c), dtype=np.uint8)
    for idx, patch in enumerate(seq_patch[:64]):
        row, col = divmod(idx, grid_side)
        vis_grid[row * patch_h:(row + 1) * patch_h, col * patch_w:(col + 1) * patch_w] = patch
    cv2.imwrite("./debug_patch_grid.png", vis_grid)

    # 8) 中心点标注图
    img_np = np.array(img).copy()
    for (x, y) in seq_pos:
        if x < 0 or y < 0: continue
        cv2.circle(img_np, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=-1)
    cv2.imwrite("./debug_patch_center.png", img_np)

    # 9) 邻接矩阵结构图可视化
    G = nx.Graph()
    for i in range(len(seq_patch)):
        G.add_node(i)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j]:
                G.add_edge(i, j)
    plt.figure(figsize=(12, 12))
    nx.draw(G, with_labels=True, node_size=300, font_size=8)
    plt.title("Adjacency Matrix Graph (BVH Structure)")
    plt.savefig("./debug_adjacency_graph.png")
