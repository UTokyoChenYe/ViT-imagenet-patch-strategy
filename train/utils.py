import torch

def bvh_collate_fn(batch):
    # batch 是列表，每个元素是 dataset 返回的一个样本(dict)
    max_n = max(item['patches'].shape[0] for item in batch)

    patches_list, pos_list, size_list, labels = [], [], [], []

    for item in batch:
        patches = item['patches']
        pos = item['positions']
        size = item['sizes']
        label = item['label']

        n = patches.shape[0]
        pad = max_n - n

        # padding 补零
        if pad > 0:
            patches = torch.cat([patches, torch.zeros(pad, *patches.shape[1:], device=patches.device)], dim=0)
            pos = torch.cat([pos, torch.zeros(pad, *pos.shape[1:], device=pos.device)], dim=0)
            size = torch.cat([size, torch.zeros(pad, *size.shape[1:], device=size.device)], dim=0)

        patches_list.append(patches)
        pos_list.append(pos)
        size_list.append(size)
        labels.append(label)

    batch_out = {
        'patches': torch.stack(patches_list, dim=0),
        'positions': torch.stack(pos_list, dim=0),
        'sizes': torch.stack(size_list, dim=0),
        'label': torch.tensor(labels)
    }
    return batch_out
