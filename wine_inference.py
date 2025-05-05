import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class WineClassifier:
    def __init__(self, model_path="best_wine_classifier.pth"):
        # 检查 CUDA 是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()  # 设置为评估模式
        
        # 初始化标准化器
        self.scaler = StandardScaler()
        
    def _load_model(self, model_path):
        # 定义模型结构
        class WineCNN(torch.nn.Module):
            def __init__(self, input_channels=1, num_classes=3):
                super(WineCNN, self).__init__()
                self.conv1 = torch.nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
                self.bn1 = torch.nn.BatchNorm1d(32)
                self.relu = torch.nn.ReLU()
                self.pool1 = torch.nn.MaxPool1d(2)

                self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.bn2 = torch.nn.BatchNorm1d(64)
                self.pool2 = torch.nn.MaxPool1d(2)

                self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.bn3 = torch.nn.BatchNorm1d(128)
                self.pool3 = torch.nn.MaxPool1d(2)

                self.fc_input_size = 128 * 12
                self.fc1 = torch.nn.Linear(self.fc_input_size, 256)
                self.fc2 = torch.nn.Linear(256, num_classes)
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
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

                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        # 创建模型实例并加载权重
        model = WineCNN()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def preprocess_data(self, data):
        """预处理输入数据"""
        # 确保数据是 numpy 数组
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # 标准化数据
        data = self.scaler.fit_transform(data)
        
        # 转换为 PyTorch 张量并添加通道维度
        data = torch.FloatTensor(data).unsqueeze(1).to(self.device)
        return data

    def predict(self, data):
        """进行预测"""
        # 预处理数据
        processed_data = self.preprocess_data(data)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(processed_data)
            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        return predicted.cpu().numpy(), probabilities.cpu().numpy()

def main():
    # 创建分类器实例
    classifier = WineClassifier()
    
    # 示例：读取测试数据
    try:
        df = pd.read_csv("频率数据宽表.csv")
        frequency_columns = [str(i) for i in range(20, 121)]
        test_data = df[frequency_columns].values
        
        # 进行预测
        predictions, probabilities = classifier.predict(test_data)
        
        # 打印结果
        print("预测结果：", predictions)
        print("预测概率：", probabilities)
        
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    main() 