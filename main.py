import os
import sys
import yaml
import argparse
import shutil
import subprocess
import random
from datetime import datetime
from easydict import EasyDict

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import cv2
cv2.setNumThreads(0)

# 加入项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.vit_imagenet import bvh_vit_imagenet_ddp_train, bvh_vit_imagenet_local_train


def main():
    parser = argparse.ArgumentParser(description="Training ViT in BVH patching")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    cli_args = parser.parse_args()

    # === 1. 读取配置文件 ===
    with open(cli_args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = EasyDict(config)

    # === 2. 输出目录创建 ===
    expertiment_name = args.get("expertiment_name", "experiment")
    output_base_path = args.get("output_base_path", "./output")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_path = os.path.join(output_base_path, expertiment_name, timestamp)
    os.makedirs(args.output_path, exist_ok=True)

    # === 3. 拷贝当前配置文件到输出目录 ===
    config_copy_path = os.path.join(args.output_path, "config_used.yaml")
    shutil.copy(cli_args.config_path, config_copy_path)
    print(f"📝 Saved config copy to: {config_copy_path}")

    # === 4. 判断是否分布式训练 ===
    is_ddp = args.get("is_ddp", False)
    runtime_cfg = args.get("runtime", {})
    num_gpus = runtime_cfg.get("num_gpus", 1)
    master_port = runtime_cfg.get("master_port", random.randint(10000, 20000))

    if expertiment_name != "BVH_ViT_imagenet":
        raise ValueError(f"Invalid experiment name: {expertiment_name}")

    # === 5. 启动训练 ===
    if is_ddp and num_gpus > 1:
        print(f"🚀 Launching DDP training with {num_gpus} GPUs (port {master_port}) ...")
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={master_port}",
            "train/vit_imagenet.py",
            "--config",
            config_copy_path  # 用复制后的配置文件
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("🚀 Launching single GPU / CPU training ...")
        bvh_vit_imagenet_local_train(args)

    print("✅ Training job finished successfully.")


if __name__ == "__main__":
    main()
