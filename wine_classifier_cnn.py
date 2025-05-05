import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置 matplotlib 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

# # 设置随机种子以确保结果可重现
random_state = 0
# torch.manual_seed(random_state)
# np.random.seed(random_state)

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 8, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class WineClassifierCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(WineClassifierCNN, self).__init__()

        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
        )

        # 残差块
        self.res_block1 = ResidualBlock(32, 64)
        self.attention1 = AttentionBlock(64)
        self.pool1 = nn.MaxPool1d(2)

        self.res_block2 = ResidualBlock(64, 128)
        self.attention2 = AttentionBlock(128)
        self.pool2 = nn.MaxPool1d(2)

        self.res_block3 = ResidualBlock(128, 256)
        self.attention3 = AttentionBlock(256)
        self.pool3 = nn.MaxPool1d(2)

        # 计算全连接层的输入大小
        self.fc_input_size = 256 * (input_size // 8)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # 添加通道维度
        x = x.unsqueeze(1)

        # 初始卷积
        x = self.initial_conv(x)

        # 第一个残差块
        x = self.res_block1(x)
        x = self.attention1(x)
        x = self.pool1(x)

        # 第二个残差块
        x = self.res_block2(x)
        x = self.attention2(x)
        x = self.pool2(x)

        # 第三个残差块
        x = self.res_block3(x)
        x = self.attention3(x)
        x = self.pool3(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)

        return x


# 数据预处理函数
def prepare_data():
    # 读取数据
    df = pd.read_csv("频率数据宽表.csv")

    # 提取20-120Hz的阻抗值作为特征
    frequency_columns = [str(i) for i in range(20, 121)]
    X = df[frequency_columns].values

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 获取类别标签
    y = df["类别"].values

    # 编码目标变量
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # 转换为 PyTorch 张量并移动到设备
    X = torch.FloatTensor(X).to(device)
    y = torch.LongTensor(y).to(device)

    return X, y, label_encoder, df["wine_name"].values


# 训练函数
def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    criterion,
    optimizer,
    scheduler,
    num_epochs=2000,
    use_early_stopping=False,
):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience = 20
    counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())

            # 更新学习率
            scheduler.step(val_loss)

            # 早停检查
            if use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        model.load_state_dict(best_model_state)
                        break

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return train_losses, val_losses


def main():
    # 准备数据
    X, y, label_encoder, wine_names = prepare_data()

    # 使用 K 折交叉验证
    n_splits = 3
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 设置模型参数
        input_size = X.shape[1]
        num_classes = len(label_encoder.classes_)

        # 创建模型实例
        model = WineClassifierCNN(input_size, num_classes).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # 训练模型，默认不启用早停
        train_losses, val_losses = train_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            criterion,
            optimizer,
            scheduler,
            use_early_stopping=False,  # 默认不启用早停
            num_epochs=300,
        )

        # 评估模型
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            fold_accuracies.append(accuracy)
            print(f"Fold {fold + 1} Test Accuracy: {accuracy:.4f}")

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="训练损失")
        plt.plot(val_losses, label="验证损失")
        plt.title(f"Fold {fold + 1} 训练和验证损失曲线")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loss_curve_fold_{fold + 1}.png")
        plt.close()

    # 打印平均准确率
    print(
        f"\nAverage Test Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}"
    )

    # 打印类别映射
    print("\n类别编码映射：")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {i}")


if __name__ == "__main__":
    main()
