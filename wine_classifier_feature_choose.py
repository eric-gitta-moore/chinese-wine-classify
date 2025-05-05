import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)


# 定义神经网络模型
class WineClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(WineClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


# 数据预处理函数
def prepare_data():
    # 读取数据
    df = pd.read_csv("频率数据宽表.csv")

    # 选择关键频率点，每隔5个点取一个
    frequency_columns = [str(i) for i in range(20, 121, 5)]
    print(f"选择的频率点: {frequency_columns}")
    print(f"特征数量: {len(frequency_columns)}")
    
    X = df[frequency_columns].values

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 获取类别标签
    y = df["类别"].values

    # 编码目标变量
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # 转换为PyTorch张量
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

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


def main():
    # 准备数据
    X, y, label_encoder, wine_names = prepare_data()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 设置模型参数
    input_size = X.shape[1]  # 使用选择后的频率点作为输入
    hidden_size = 64  # 由于输入维度减小，可以适当减小隐藏层大小
    num_classes = len(label_encoder.classes_)

    # 创建模型实例
    model = WineClassifier(input_size, hidden_size, num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test, criterion, optimizer
    )

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss_curve_feature_choose.png")
    plt.close()

    # 评估模型
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f"Test Accuracy: {accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "wine_classifier_feature_choose.pth")

    # 打印类别映射
    print("\n类别编码映射：")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {i}")


if __name__ == "__main__":
    main() 