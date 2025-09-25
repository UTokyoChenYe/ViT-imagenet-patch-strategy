import os
import sys
from torch.distributed.distributed_c10d import _World
from torch.optim import optimizer
import yaml
import logging
import argparse
import contextlib

import torch
from torch import nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast 

sys.path.append("./")
from dataset.imagenet import imagenet_distribute, imagenet_subloaders
from model.vit import create_timm_vit as create_vit_model
from dataset.utlis import param_groups_lrd

def setup_logging(args):
    log_file = os.path.join(args.output_path, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    return logger

def evaluate_model_compatible(model, val_loader, device, is_ddp=False):
    """评估函数。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    if is_ddp:
        total_tensor = torch.tensor(total, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        dist.all_reduce(total_tensor)
        dist.all_reduce(correct_tensor)
        total, correct = total_tensor.item(), correct_tensor.item()
        
    return 100 * correct / total if total > 0 else 0


def train_vit_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args, device_id, start_epoch, best_val_acc=0.0, is_ddp=False):
    # 1. 初始化 GradScaler
    scaler = GradScaler(enabled=True)
    num_epochs = args.get("train.num_epochs", 100)
    accumulation_steps = args.get("train.gradient_accumulation_steps", 1)
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        label_smoothing=0.1,
        prob=1.0,              # 总使用概率
        switch_prob=0.5        # mixup vs cutmix 切换概率
    )
    
    # checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    use_compile = args.get("train.use_compile", False)
    use_fused_optimizer = args.get("train.use_fused_optimizer", False)
    use_checkpointing = args.get("train.use_checkpointing", False)
    if is_main_process:
        logging.info("Starting ViT training for %d epochs with Automatic Mixed Precision (AMP)...", num_epochs)   
        logging.info("开始BF16优化训练...")
        logging.info(f"torch.compile: {use_compile}, Fused Optimizer: {use_fused_optimizer}, Activation Checkpointing: {use_checkpointing}")
        logging.info(f"将从 Epoch {start_epoch + 1} 开始训练...")
        
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            is_accumulation_step = (i + 1) % accumulation_steps != 0
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            original_labels = labels.clone()
            # *** 修改: 应用Mixup/CutMix ***
            if args.get("train.use_mixup", False):
                images, soft_labels = mixup_fn(images, labels)
            else:
                soft_labels = nn.functional.one_hot(labels, args.get("model.num_classes", 1000)).float()
                
            sync_context = model.no_sync() if (is_ddp and is_accumulation_step) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, soft_labels)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()

            if not is_accumulation_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler and not args.get("train.use_postrain", False): scheduler.step()
                
            # --- 之后的评估逻辑保持不变 ---
            _, predicted = torch.max(outputs.data, 1)
            running_total += original_labels.size(0)
            running_corrects += (predicted == original_labels).sum().item()
            # 注意：loss.item() 会自动返回未缩放的、float32类型的损失值
            running_loss += loss.item() * accumulation_steps

            if (i + 1) % 10 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%, current_lr: {current_lr:.6f}')
                running_loss, running_corrects, running_total = 0.0, 0, 0

        # 调用兼容性更强的评估函数
        val_acc = evaluate_model_compatible(
            model, 
            val_loader, 
            device_id, 
            is_ddp=is_ddp
        )
                
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f} | current_lr: {current_lr:.6f}")
            
            # *** 修改: 完整的检查点保存逻辑 ***
            checkpoint_dir = os.path.join(args.output, args.savefile)
            # checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
            
            # 总是保存最新的检查点
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, latest_checkpoint_path)

            # 如果是最佳模型，则另外保存一份
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"新的最佳验证精度: {best_val_acc:.4f}. 保存最佳模型...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, best_checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Training. Best Validation Accuracy: {best_val_acc:.4f}')



def setup_ddp(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def bvh_vit_imagenet_ddp_train(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup_ddp(rank, world_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    if rank == 0:
        logger = setup_logging(args)
    logger.info(f"DDP training with {world_size} GPUs")
    logger.info(f"Rank: {rank}, Local Rank: {local_rank}, Device ID: {device_id}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Arguments: {args}")
    
    img_size = args.get("model.img_size", 224)
    batch_size = args.get("train.batch_size", 128)
    num_workers = args.get("train.num_workers", 8)
    data_path = args.get("data.data_path", "./data/imagenet")

    dataloaders = imagenet_distribute(
        img_size=img_size,
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=args.num_workers)

    model = create_vit_model(args)

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss()
    model_without_ddp = model.module
    weight_decay = args.get("train.weight_decay", 0.05)
    param_groups = param_groups_lrd(
        model_without_ddp,
        weight_decay=weight_decay, 
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=0.75)
    
    learning_rate = args.get("train.learning_rate", 1e-3)
    use_fuse_adafactor = args.get("train.use_fuse_adafactor", False)
    betas = args.get("train.betas", (0.9, 0.95))
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=learning_rate, 
        weight_decay=weight_decay, 
        betas=betas,
        fused = use_fuse_adafactor)
    
    # *** 修改: 创建包含线性预热和余弦退火的组合调度器 ***
    training_config = args.get("train", {})
    num_epochs = training_config.get('num_epochs', 100)
    warmup_epochs = training_config.get('warmup_epochs', 0)
    
    # 计算总的训练步数和预热步数
    steps_per_epoch = len(dataloaders['train'])
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch
    
    if num_warmup_steps > 0:
        # 预热调度器：从一个很小的值线性增长到1
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=num_warmup_steps)
        # 主调度器：在预热结束后，进行余弦退火
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=1e-6)
        # 使用SequentialLR将两者串联起来
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        # 如果不使用预热，则只使用余弦退火
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
        
    # *** 新增: 完整的检查点加载逻辑 ***
    start_epoch = 0
    best_val_acc = 0.0
    output_path = args.get("output_path", "./output")
    checkpoint_path = os.path.join(output_path, "best_model.pth")

    reload = args.get("train.reload", False)
    if reload and os.path.exists(checkpoint_path):
        if dist.get_rank() == 0:
            logging.info(f"从检查点恢复训练: {checkpoint_path}")
        
        # 加载到CPU以避免GPU内存冲突，并确保所有进程加载相同的权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型权重 (注意要加载到 model.module)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        
        if dist.get_rank() == 0:
            logging.info(f"成功恢复，将从 Epoch {start_epoch + 1} 开始。")
            
    train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, args, device_id, start_epoch=start_epoch, best_val_acc=best_val_acc, is_ddp=(world_size > 1))
    dist.destroy_process_group()





def bvh_vit_imagenet_local_train(args):
    pass