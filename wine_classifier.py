import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# # 设置随机种子以确保结果可重现
# torch.manual_seed(42)
# np.random.seed(42)


# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义卷积神经网络模型
class WineCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(WineCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        # 第二个卷积块
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # 第三个卷积块
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # 计算全连接层的输入维度
        # 原始序列长度是 101（20-120Hz），经过 3 次池化后长度为 101//(2^3) = 12
        self.fc_input_size = 128 * 12

        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 数据预处理函数
def prepare_data():
    # 读取数据
    df = pd.read_csv("频率数据宽表.csv")

    # 提取 20-120Hz 的阻抗值作为特征
    frequency_columns = [str(i) for i in range(20, 121)]
    X = df[frequency_columns].values

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # # 使用 PCA 进行降维
    # pca = PCA(n_components=20)  # 降维到 20 个主成分
    # X = pca.fit_transform(X)

    # # 打印解释方差比
    # print("PCA 解释方差比：", pca.explained_variance_ratio_)
    # print("累计解释方差比：", np.cumsum(pca.explained_variance_ratio_))

    # 获取类别标签
    y = df["类别"].values

    # 编码目标变量
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # 转换为 PyTorch 张量并重塑为 CNN 输入格式 (batch_size, channels, sequence_length)
    X = torch.FloatTensor(X).unsqueeze(1).to(device)  # 添加通道维度
    y = torch.LongTensor(y).to(device)

    return X, y, label_encoder, df["wine_name"].values


# 训练函数
def train_model(
    model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs=5000
):
    train_losses = []
    val_losses = []

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

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
            )

    return train_losses, val_losses


def train_multiple_models(
    X_train,
    y_train,
    X_val,
    y_val,
    input_channels,
    num_classes,
    num_trials=5,
    num_epochs=1000,
):
    best_accuracy = 0
    best_model = None
    best_train_losses = None
    best_val_losses = None

    for trial in range(num_trials):
        print(f"\n开始第 {trial + 1} 次训练...")

        # 创建新的模型实例
        model = WineCNN(input_channels, num_classes).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        train_losses, val_losses = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            criterion,
            optimizer,
            num_epochs=num_epochs,
        )

        # 评估模型
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_val).sum().item() / y_val.size(0)
            print(f"第 {trial + 1} 次训练的验证集准确率：{accuracy:.4f}")

        # 如果这是目前最好的模型，保存它
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_train_losses = train_losses
            best_val_losses = val_losses
            # 保存最佳模型
            torch.save(model, "best_wine_classifier.pth")
            print(f"新的最佳模型已保存，准确率：{accuracy:.4f}")

    return best_model, best_train_losses, best_val_losses, best_accuracy


def main():
    # 准备数据
    X, y, label_encoder, wine_names = prepare_data()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 设置模型参数
    input_channels = X.shape[1]  # 输入通道数
    num_classes = len(label_encoder.classes_)

    # 多次训练并获取最佳模型
    best_model, train_losses, val_losses, best_accuracy = train_multiple_models(
        X_train,
        y_train,
        X_test,
        y_test,
        input_channels,
        num_classes,
        num_trials=5,
        num_epochs=1000,
    )

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    # 评估最佳模型
    best_model.eval()
    with torch.no_grad():
        outputs = best_model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"最佳模型在测试集上的准确率：{accuracy:.4f}")

    # 打印类别映射
    print("\n类别编码映射：")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {i}")


if __name__ == "__main__":
    main()
