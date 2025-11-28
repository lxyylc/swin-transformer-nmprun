import torch
import torch.nn as nn
from collections import defaultdict
import os
import time
from swin import swin_t


def n_m_prune_linear_layer(layer, n, m):
    #传进来的layer参数一定是nn.Linear
    weight = layer.weight.data
    #因为是线性层，所以只需要考虑输入维度和输出维度
    out_features, in_features = weight.shape

    mask = torch.ones_like(weight)

    group_size = m
    num_groups = in_features // group_size
    remaining = in_features % group_size

    if num_groups > 0:
        # 将权重的前 num_groups*group_size 列重塑为 (out_features, 分组数, 每组大小),原来的形状为（out_features,in_features)
        weight_groups = weight[:, :num_groups * group_size].view(out_features, num_groups, group_size)
        # 计算每个分组内的绝对值排序索引
        sorted_indices = torch.argsort(torch.abs(weight_groups), dim=2)
        # 保留每个分组中绝对值最大的n个元素，其余位置掩码置0
        prune_idx = sorted_indices[:, :, :-n] if n > 0 else sorted_indices
        # 构造掩码的分组部分
        mask_groups = torch.ones_like(weight_groups)
        mask_groups = mask_groups.scatter_(2, prune_idx, 0)
        mask[:, :num_groups * group_size] = mask_groups.view(out_features, -1)

    if remaining > 0 and remaining <= n:
        mask[:, num_groups * group_size:] = 1.0

    with torch.no_grad():
        layer.weight.data *= mask

    return mask


def apply_n_m_pruning(model, n, m):
    masks = defaultdict(torch.Tensor)

    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)]
    total_layers = len(linear_layers)
    print(f"发现 {total_layers} 个Linear层需要剪枝...")

    for i, (name, module) in enumerate(linear_layers, 1):
        start_time = time.time()
        new_mask = n_m_prune_linear_layer(
            layer=module,
            n=n,
            m=m
        )
        masks[name] = new_mask
        # 打印剪枝进度
        layer_type = f"{module.in_features}→{module.out_features}"
        elapsed = time.time() - start_time
        print(f"[{i}/{total_layers}] 剪枝完成: {name} ({layer_type}) | 耗时: {elapsed:.2f}s")

    return masks

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = swin_t(
        patch_size=4,
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        channels=3,
        num_classes=10,
        head_dim=32,
        window_size=7,
        downscaling_factors=(1, 2, 2, 2),
        relative_pos_embedding=True
    ).to(device)

    N = 2
    M = 4
    print(f"开始N:M剪枝 (N={N}, M={M})...")

    # 应用剪枝并获取掩码
    start_total = time.time()
    pruning_masks = apply_n_m_pruning(model, N, M)
    total_time = time.time() - start_total
    print(f"所有层剪枝完成 | 总耗时: {total_time:.2f}s")

    # 保存掩码
    os.makedirs('models', exist_ok=True)
    torch.save(pruning_masks, 'models/swin_pruning_masks.pt')
    print(f"掩码已保存至 models/swin_pruning_masks.pt")

    # 验证剪枝效果
    first_linear = next((m for m in model.modules() if isinstance(m, nn.Linear)), None)
    if first_linear is not None:
        weight = first_linear.weight.data
        total_params = weight.numel()
        pruned_params = (weight == 0).sum().item()
        print(f"\n剪枝效果验证:")
        print(f"实际剪枝比例: {pruned_params / total_params:.2%}")
        print(f"理论剪枝比例: {(M - N) / M:.2%}")


def load_pruning_masks(model, mask_path):
    masks = torch.load(mask_path,weights_only=False)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in masks:
            mask = masks[name].to(module.weight.device)  # 确保掩码与设备一致

            with torch.no_grad():
                module.weight.data *= mask

            def make_hook(mask):
                #grad参数是PyTorch自动传入的
                def hook(grad):
                    return grad * mask

                return hook

            module.weight.register_hook(make_hook(mask))

    return masks
