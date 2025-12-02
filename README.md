对swin transoformer进行N:M半结构化剪枝，这是一种介于结构化剪枝和非结构化剪枝之间的方法，核心是在模型权重中按 “每 M 个连续参数保留 N 个非零值” 的规则进行剪枝，数据集cifar10

<img width="2165" height="624" alt="image" src="https://github.com/user-attachments/assets/1ba2ccc9-1260-4b15-a375-7f705e5bec1f" />


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

```
python train.py
```

剪枝：

```
python nmprun.py
```

微调：

```
python retrain.py
```
|模型|剪枝方法|剪枝前acc|剪枝后acc|
|-|-|-|-|
|swin transformer|2:4剪枝|0.889|0.885|

训练epochs=200，微调epochs=20

参考https://github.com/Cydia2018/Vision-Transformer-CIFAR10
