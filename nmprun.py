import torch
import torch.nn as nn
from collections import defaultdict
import os
import time  # 增加时间模块用于计时


def n_m_prune_linear_layer(layer, n, m, mask=None):
    """对nn.Linear层进行N:M半结构化剪枝"""
    weight = layer.weight.data
    out_features, in_features = weight.shape

    if mask is None:
        mask = torch.ones_like(weight)

        # 优化：使用向量化操作减少循环次数（原代码逐输出通道+逐分组循环效率低）
        # 重塑为 (out_features, -1, m) 便于批量处理分组
        group_size = m
        num_groups = in_features // group_size
        remaining = in_features % group_size

        # 处理完整分组
        if num_groups > 0:
            weight_groups = weight[:, :num_groups * group_size].view(out_features, num_groups, group_size)
            # 计算每个分组内的绝对值排序索引
            sorted_indices = torch.argsort(torch.abs(weight_groups), dim=2)
            # 保留每个分组中绝对值最大的n个元素，其余位置掩码置0
            prune_idx = sorted_indices[:, :, :-n] if n > 0 else sorted_indices
            # 构造掩码的分组部分
            mask_groups = torch.ones_like(weight_groups)
            mask_groups = mask_groups.scatter_(2, prune_idx, 0)
            mask[:, :num_groups * group_size] = mask_groups.view(out_features, -1)

        # 处理剩余不足m的部分（不剪枝）
        if remaining > 0 and remaining <= n:
            mask[:, num_groups * group_size:] = 1.0

    # 应用掩码
    with torch.no_grad():
        layer.weight.data *= mask

    return mask


def apply_n_m_pruning(model, n, m, masks=None):
    """对模型中所有nn.Linear层应用N:M半结构化剪枝，增加进度打印"""
    if masks is None:
        masks = defaultdict(torch.Tensor)

    # 先收集所有Linear层信息，方便统计进度
    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)]
    total_layers = len(linear_layers)
    print(f"发现 {total_layers} 个Linear层需要剪枝...")

    for i, (name, module) in enumerate(linear_layers, 1):
        start_time = time.time()
        layer_mask = masks.get(name, None)
        new_mask = n_m_prune_linear_layer(
            layer=module,
            n=n,
            m=m,
            mask=layer_mask
        )
        masks[name] = new_mask
        # 打印当前剪枝进度
        layer_type = f"{module.in_features}→{module.out_features}"
        elapsed = time.time() - start_time
        print(f"[{i}/{total_layers}] 剪枝完成: {name} ({layer_type}) | 耗时: {elapsed:.2f}s")

    return masks


# 使用示例
if __name__ == '__main__':
    # 新增：解决swin_t未导入的问题
    from swin import swin_t  # 从train.py中可知模型定义在swin.py

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
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

    # 定义剪枝参数
    N = 2
    M = 4
    print(f"开始N:M剪枝 (N={N}, M={M})...")

    # 应用剪枝并获取掩码
    start_total = time.time()
    pruning_masks = apply_n_m_pruning(model, N, M)
    total_time = time.time() - start_total
    print(f"所有层剪枝完成 | 总耗时: {total_time:.2f}s")

    # 保存掩码
    os.makedirs('models', exist_ok=True)  # 确保目录存在
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
    """加载掩码并应用到模型，确保剪枝后的参数不被更新"""
    masks = torch.load(mask_path,weights_only=False)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in masks:
            mask = masks[name].to(module.weight.device)  # 确保掩码与设备一致

            with torch.no_grad():
                module.weight.data *= mask

            # 注册梯度钩子
            def make_hook(mask):
                def hook(grad):
                    return grad * mask

                return hook

            module.weight.register_hook(make_hook(mask))

    return masks