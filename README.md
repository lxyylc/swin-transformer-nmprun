version 1：在cifar10（图片大小32*32）上用swin transformer进行分类，acc 0.83，模型配置
```
model = swin_t(
    patch_size=2,
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=10,  # CIFAR-10有10个类别
    head_dim=32,
    window_size=4,
    downscaling_factors=(2, 2, 1, 1),
    relative_pos_embedding=True
).to(device)
```

version 2: 对cifar10数据集插值到224*224，模型配置
```
model = swin_t(
    patch_size=4,
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=10,  # CIFAR-10有10个类别
    head_dim=32,
    window_size=7,
    downscaling_factors=(1, 2, 2, 2),#注意下采样的因子，否则可能出现维度无法整除的问题
    relative_pos_embedding=True
).to(device)
```
训练swin transformer：

```python train.py```

剪枝：

```python nmprun.py```

微调：

```python retrain.py```

训练集acc 0.96，测试集acc 0.889，剪枝后微调acc 0.885

参考https://github.com/Cydia2018/Vision-Transformer-CIFAR10
