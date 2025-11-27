import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from swin import swin_t

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),        # 插值到224×224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),        # 测试集也插值到224×224
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if not os.path.exists('models'):
    os.makedirs('models')


# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 打印批次信息
        if batch_idx % 50 == 0:
            print(
                f'[{epoch}, {batch_idx}] Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}%')

    epoch_time = time.time() - start_time
    train_acc = 100. * correct / total
    train_avg_loss = train_loss / len(train_loader)

    print(f'Epoch {epoch} Training | Loss: {train_avg_loss:.3f} | Acc: {train_acc:.3f}% | Time: {epoch_time:.2f}s')
    return train_avg_loss, train_acc

def test(epoch):
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

    test_acc = 100. * correct / total
    test_avg_loss = test_loss / len(test_loader)
    print(f'Epoch {epoch} Testing  | Loss: {test_avg_loss:.3f} | Acc: {test_acc:.3f}%')
    return test_avg_loss, test_acc

best_acc = 0
num_epochs = 200

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()
    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f'models/swin_cifar10_best.pth')
        print(f'Saved best model with accuracy: {best_acc:.3f}%')

print(f'Training complete. Best test accuracy: {best_acc:.3f}%')
