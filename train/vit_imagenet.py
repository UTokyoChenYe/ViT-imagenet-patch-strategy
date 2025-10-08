import os
import sys
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import logging
import argparse
import contextlib
import time
from typing import Dict, Any
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from train.utils import bvh_collate_fn

sys.path.append("./")

# ==== 数据与模型（按我们的工程） ====
from dataset.imagenet import ImageNetBVHDataset
from dataset.utils import param_groups_lrd
from model.vit import BVHViT

# ---------------- Logging ----------------
def setup_logging(output_path: str):
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("BVH-ViT")



# ---------------- Eval (dict batch) ----------------
@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, is_ddp: bool = False) -> float:
    model.eval()
    correct, total = 0, 0
    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for batch in val_loader:
            # 将 batch 字典移动到 device
            for k in ("patches", "positions", "sizes", "label"):
                batch[k] = batch[k].to(device, non_blocking=True)
            logits = model(batch)
            pred = logits.argmax(dim=1)
            total += batch["label"].size(0)
            correct += (pred == batch["label"]).sum().item()

    if is_ddp:
        t = torch.tensor([total, correct], dtype=torch.long, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, correct = t[0].item(), t[1].item()

    return (100.0 * correct / total) if total > 0 else 0.0


# ---------------- Train ----------------
def check_memory(threshold_gb=10):
    """监控显存剩余情况并在不足时清理缓存"""
    if not torch.cuda.is_available():
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return  # 只让 rank 0 执行监控
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024 ** 3
    if free_gb < threshold_gb:
        logging.warning(f"[WARN] Free GPU memory only {free_gb:.2f} GB — waiting & clearing cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(3)

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    args: Dict[str, Any],
    is_ddp: bool,
    logger: logging.Logger,
):
    model.train()
    scaler = GradScaler(enabled=True)
    accum_steps = args.get("train", {}).get("gradient_accumulation_steps", 1)
    log_interval = args.get("train", {}).get("log_interval", 10)

    running_loss, running_corrects, running_total = 0.0, 0, 0

    if is_ddp and hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
    iterable = train_loader
    if is_main:
        iterable = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.get('train', {}).get('epochs', 100)}", ncols=100)

    for it, batch in enumerate(iterable):
        is_accum = ((it + 1) % accum_steps) != 0

        # move to device
        for k in ("patches", "positions", "sizes", "label"):
            batch[k] = batch[k].to(device, non_blocking=True)

        # 采用 label smoothing（更适配 dict 输入；mixup 默认关闭）
        labels = batch["label"]

        sync_ctx = model.no_sync() if (is_ddp and is_accum) else contextlib.nullcontext()
        with sync_ctx:
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch)
                loss = criterion(logits, labels) / accum_steps

            scaler.scale(loss).backward()

        if not is_accum:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        # 统计
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            running_total += labels.size(0)
            running_corrects += (pred == labels).sum().item()
            running_loss += loss.item() * accum_steps

        if (it + 1) % log_interval == 0 and (not is_ddp or dist.get_rank() == 0):
            train_acc = 100.0 * running_corrects / max(1, running_total)
            avg_loss = running_loss / log_interval
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"[Epoch {epoch+1} | Iter {it+1}/{len(train_loader)}] "
                        f"loss={avg_loss:.4f} acc={train_acc:.2f}% lr={lr:.6e}")
            if "tensorboard_writer" in args and (not is_ddp or dist.get_rank() == 0):
                writer = args["tensorboard_writer"]
                step = epoch * len(train_loader) + it
                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/acc", train_acc, step)
                writer.add_scalar("train/lr", lr, step)
            running_loss, running_corrects, running_total = 0.0, 0, 0
        
        if (it + 1) % 100 == 0 and (not is_ddp or dist.get_rank() == 0):
            torch.cuda.empty_cache()


# ---------------- Build loaders ----------------
def build_loaders(cfg: Dict[str, Any], is_ddp: bool):
    ds_cfg = cfg["dataset"]
    train_root = ds_cfg["root"]
    val_root = ds_cfg.get("val_root", None)
    num_workers = ds_cfg.get("num_workers", 8)
    batch_size = ds_cfg.get("batch_size", 64)
    max_nodes = ds_cfg.get("max_nodes", 512)
    patch_size = tuple(ds_cfg.get("patch_size", [8, 8, 3]))

    train_set = ImageNetBVHDataset(train_root, transform=None, max_nodes=max_nodes, patch_size=patch_size)
    if val_root is None:
        raise ValueError("Please set dataset.val_root in yaml for validation set.")
    val_set = ImageNetBVHDataset(val_root, transform=None, max_nodes=max_nodes, patch_size=patch_size)

    if is_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=bvh_collate_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(2, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=bvh_collate_fn,
        persistent_workers=True,
    )
    return train_loader, val_loader


# ---------------- Schedulers ----------------
def build_scheduler(optimizer, cfg: Dict[str, Any], steps_per_epoch: int):
    tr_cfg = cfg.get("train", {})
    num_epochs = tr_cfg.get("epochs", 100)
    warmup_epochs = tr_cfg.get("warmup_epochs", 0)
    eta_min = tr_cfg.get("eta_min", 1e-6)

    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)

    if warmup_steps > 0:
        warm = LinearLR(optimizer, start_factor=tr_cfg.get("warmup_start_factor", 0.01), total_iters=warmup_steps)
        main = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup_steps])
    else:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)


# ---------------- DDP init ----------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank, world_size = 0, 1

    if world_size > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl")
        torch.distributed.barrier()
    return world_size > 1


# ---------------- Main train loop ----------------
def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    out_dir = cfg.get("train", {}).get("output_dir", "./output/bvh_vit_imagenet")
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))
    cfg["tensorboard_writer"] = writer

    is_ddp = setup_ddp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders
    train_loader, val_loader = build_loaders(cfg, is_ddp=is_ddp)

    # model
    m_cfg = cfg["model"]
    num_classes = m_cfg.get("num_classes", cfg["dataset"].get("num_classes", 1000))
    model = BVHViT(
        embed_dim=m_cfg.get("embed_dim", 768),
        depth=m_cfg.get("depth", 12),
        num_heads=m_cfg.get("num_heads", 12),
        num_classes=num_classes,
    ).to(device)

    # optional compile
    if cfg.get("train", {}).get("use_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)  # py>=2.0

    # optimizer param groups with layer-wise decay (compatible fallback)
    if hasattr(model, "no_weight_decay"):
        nwd = model.no_weight_decay()
    else:
        nwd = set()
    wd = cfg.get("train", {}).get("weight_decay", 0.05)
    groups = param_groups_lrd(model, weight_decay=wd, no_weight_decay_list=nwd, layer_decay=cfg.get("train", {}).get("layer_decay", 0.75))

    lr = cfg.get("train", {}).get("lr", 3e-4)
    betas = tuple(cfg.get("train", {}).get("betas", (0.9, 0.95)))
    use_fused = bool(cfg.get("train", {}).get("use_fused_optimizer", False))
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=betas, weight_decay=wd, fused=use_fused)

    # scheduler (per-iteration)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # loss（label smoothing 推荐值 0.1）
    ls = cfg.get("train", {}).get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls).to(device)

    # DDP wrap
    if is_ddp:
        model = DDP(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=False,      # 不允许存在未参与反向的参数
            broadcast_buffers=False,          # 小优化，通常不需要同步 buffers
            gradient_as_bucket_view=True      # 更快的 bucket 视图
        )

    # resume
    start_epoch = 0
    best = 0.0
    ckpt_dir = out_dir
    best_pth = os.path.join(ckpt_dir, "best_model.pth")
    latest_pth = os.path.join(ckpt_dir, "latest_checkpoint.pth")
    if cfg.get("train", {}).get("reload", False) and os.path.exists(best_pth):
        map_location = "cpu"
        state = torch.load(best_pth, map_location=map_location)
        (model.module if is_ddp else model).load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best = float(state.get("best_val_acc", 0.0))
        if (not is_ddp) or dist.get_rank() == 0:
            logger.info(f"Resumed from {best_pth}, start_epoch={start_epoch}, best={best:.3f}")

    # train loop
    epochs = cfg.get("train", {}).get("epochs", 100)
    for epoch in range(start_epoch, epochs):
        check_memory()
        if (not is_ddp) or dist.get_rank() == 0:
            free, total = torch.cuda.mem_get_info()
            logger.info(f"[GPU] Free {free / 1e9:.2f} GB / Total {total / 1e9:.2f} GB before epoch {epoch+1}")
            logger.info(f"=== Epoch {epoch+1}/{epochs} ===")

        train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            args=cfg,
            is_ddp=is_ddp,
            logger=logger,
        )

        # eval
        val_acc = evaluate(model, val_loader, device, is_ddp=is_ddp)
        if (not is_ddp) or dist.get_rank() == 0:
            logger.info(f"Val Acc: {val_acc:.3f} | LR: {optimizer.param_groups[0]['lr']:.6e}")
            if "tensorboard_writer" in cfg and (not is_ddp or dist.get_rank() == 0):
                writer = cfg["tensorboard_writer"]
                step = epoch * len(val_loader)
                writer.add_scalar("val/loss", val_acc, step)
                writer.add_scalar("val/acc", val_acc, step)
                writer.add_scalar("val/lr", optimizer.param_groups[0]['lr'], step)

            # save latest
            to_save = {
                "epoch": epoch,
                "model_state_dict": (model.module if is_ddp else model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best,
            }
            torch.save(to_save, latest_pth)

            # save best
            if val_acc > best:
                best = val_acc
                torch.save(to_save, best_pth)
                logger.info(f"New best: {best:.3f} (saved to best_model.pth)")
            
            # 打印跳过坏图像统计
            if hasattr(train_loader.dataset, "bad_images_count") and train_loader.dataset.bad_images_count > 0:
                logger.warning(f"[ImageNet] {train_loader.dataset.bad_images_count} corrupted images skipped in epoch {epoch+1}.")
                train_loader.dataset.bad_images_count = 0

    is_main_process = (not is_ddp) or (dist.is_initialized() and dist.get_rank() == 0) 

    if is_main_process:
        logger.info(f"Finished. Best Val Acc: {best:.3f}")
        if "tensorboard_writer" in cfg:
            cfg["tensorboard_writer"].close()


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bvh_vit_imagenet.yaml")
    args = parser.parse_args()
    main(args.config)
