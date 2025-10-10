#!/bin/bash
# ======================================
# 🔧 环境变量设置（防显存碎片 + 调试稳定）
# ======================================

# 防止显存碎片化（最重要）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 让 CUDA 报错能定位到具体行（调试时有用）
export CUDA_LAUNCH_BLOCKING=1

# 限制每个进程 CPU 线程数，避免 dataloader 占太多资源
export OMP_NUM_THREADS=4

# 避免 NCCL P2P bug（多机或多 GPU 通信时偶尔卡死）
export NCCL_P2P_DISABLE=1

# ======================================
# 🚀 启动训练
# ======================================
python main.py --config_path configs/bvh_vit_imagenet.yaml