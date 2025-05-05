import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from wine_classifier import WineCNN

class WineClassifier:
    def __init__(self, model_path="models/best_wine_classifier.pth"):
        # 检查 CUDA 是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载整个模型，设置 weights_only=False
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()  # 设置为评估模式
        
        # 初始化标准化器
        self.scaler = StandardScaler()
        
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
        
        # 获取预测类别的概率
        predicted_probabilities = [prob[pred] for pred, prob in zip(predictions, probabilities)]
        
        # 将预测结果和概率添加到原始数据框中
        df['预测结果'] = predictions
        df['预测概率'] = predicted_probabilities
        
        # 保存结果到新的 CSV 文件
        df.to_csv('result.csv', index=False, encoding='utf-8-sig')
        print("预测结果已保存到 result.csv")
        
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    main() 