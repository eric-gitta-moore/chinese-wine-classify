# 酒类分类模型说明文档

## 模型文件
- `best_wine_classifier.pth`: 预训练的酒类分类模型

## 模型性能
```sh
wine-groupby-hz on  main [!?⇡] is 󰏗 v0.1.0 via  v3.11.12 (wine-groupby-hz) via  py311 at 16:47:32
❯ python .\wine_classifier.py
第 5 次训练的验证集准确率：0.7073
最佳模型在测试集上的准确率：0.8171

类别编码映射：
啤酒: 0
果酒: 1
白酒: 2
鸡尾酒: 3
```

## 推理方法

### 1. 环境要求
- Python 3.11.12
- PyTorch
- pandas
- numpy
- scikit-learn

### 2. 推理步骤
1. 准备输入数据
   - 数据格式：CSV 文件（频率数据宽表.csv）
   - 特征：20-120Hz 的阻抗值
   - 数据列名：20, 21, ..., 120

2. 运行推理
   ```python
   from wine_inference import WineClassifier
   
   # 创建分类器实例
   classifier = WineClassifier()
   
   # 读取测试数据
   df = pd.read_csv("频率数据宽表.csv")
   frequency_columns = [str(i) for i in range(20, 121)]
   test_data = df[frequency_columns].values
   
   # 进行预测
   predictions, probabilities = classifier.predict(test_data)
   ```

3. 输出结果
   - 预测结果保存在 `result.csv` 文件中
   - 包含原始数据和预测结果
   - 每行数据包含：
     - 原始频率数据（20-120Hz）
     - 预测的酒类（0-3）
     - 预测概率

### 3. 类别编码
- 0: 啤酒
- 1: 果酒
- 2: 白酒
- 3: 鸡尾酒

### 4. 注意事项
- 确保模型文件 `best_wine_classifier.pth` 存在于 models 目录下
- 输入数据需要进行标准化处理（已在推理代码中实现）
- 推理结果会自动保存到 `result.csv` 文件中
- `预测结果.xlsx` 是做了公式高亮的 `result.csv`