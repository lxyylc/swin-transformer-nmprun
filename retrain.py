import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from swin import swin_t  # 导入Swin Transformer模型
from nmprun import load_pruning_masks  # 导入加载掩码的函数

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据预处理（与train.py保持一致）
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10测试集
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

# 数据加载器
batch_size = 32
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)

# 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 初始化模型（与train.py保持一致的参数）
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

# 加载剪枝前的最佳模型权重
model.load_state_dict(torch.load('models/swin_cifar10_best.pth'))

# 加载剪枝掩码并应用到模型
mask_path = 'models/swin_pruning_masks.pt'  # 剪枝生成的掩码路径
load_pruning_masks(model, mask_path)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 测试剪枝后模型的函数
def test_pruned_model():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 打印批次信息
            if batch_idx % 50 == 0:
                print(f'[Batch {batch_idx}] Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}%')

    test_acc = 100. * correct / total
    test_avg_loss = test_loss / len(test_loader)
    print(f'\n剪枝后模型测试结果 | 平均损失: {test_avg_loss:.3f} | 准确率: {test_acc:.3f}%')
    return test_avg_loss, test_acc

# 执行测试
if __name__ == '__main__':
    test_pruned_model()