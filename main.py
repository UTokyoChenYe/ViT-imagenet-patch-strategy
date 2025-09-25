import os
import sys
import yaml
import argparse

from easydict import EasyDict
from datetime import datetime

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import cv2
cv2.setNumThreads(0)

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.vit_imagenet import bvh_vit_imagenet_ddp_train, bvh_vit_imagenet_local_train


def main():
    parser = argparse.ArgumentParser(description="Training ViT in BVH patching")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    
    args = parser.parse_args()

    with open(args.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    args = EasyDict(config)

    expertiment_name = args.get("expertiment_name", "experiment")
    output_base_path = args.get("output_base_path", "./output")
    args.output_path = os.path.join(output_base_path, expertiment_name)
    args.output_path = os.path.join(args.output_path, datetime.now().strftime("%Y%m%d_%H%M%S")) # 加入timestamp为子文件夹
    is_ddp = args.get("is_ddp", False)

    os.makedirs(args.output_path, exist_ok=True)

    if expertiment_name == "BVH_ViT_imagenet":
        if is_ddp:
            print("=======DDP Training for BVH_ViT_imagenet=======")
            bvh_vit_imagenet_ddp_train(args)
        else:
            print("=======Local Training for BVH_ViT_imagenet=======")
            bvh_vit_imagenet_local_train(args)
    else:
        raise ValueError(f"Invalid expertiment name: {expertiment_name}")

if __name__ == "__main__":
    main()