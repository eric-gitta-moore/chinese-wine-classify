# 基于深度学习技术的葡萄酒智能分类系统

## 研究背景

本项目开发了一个基于深度学习技术的葡萄酒智能分类系统，通过分析葡萄酒在 20-120Hz 频率范围内的阻抗特性，实现对不同种类葡萄酒的自动识别和分类。该系统采用多阶段数据处理流程和多种深度学习模型架构，实现了高精度的葡萄酒分类。

## 技术实现

### 1. 数据处理流程

#### 1.1 数据分组与预处理

使用 `group_wine_data.py` 脚本将原始数据按连续频率进行分组：

- 读取清洗后的数据文件 `cleaned_wine_data.csv`
- 根据频率的连续性为每个样本分配唯一的 `sample_id`
- 当频率不连续或非单调递增时，创建新的样本 ID
- 输出分组后的数据到 `按连续频率把样本分组到sample.csv`

#### 1.2 数据插值与补齐

使用 `interp.py` 脚本对数据进行插值和补齐处理：

- 读取分组后的数据
- 对每个葡萄酒样本（wine_id）进行处理：
  - 识别相对完整的样本（包含 120Hz 数据）
  - 对完整样本进行线性插值，生成 20-120Hz 的连续数据
  - 计算完整样本的平均阻抗值
  - 对不完整的样本，使用平均阻抗值进行补齐
- 输出处理后的数据到 `插值+平均补齐到120Hz.csv`

#### 1.3 数据转换与特征工程

使用 `transport.py` 脚本将数据转换为宽表格式：

- 读取插值补齐后的数据
- 使用 pivot 函数将频率数据转换为宽表格式：
  - 行索引：wine_id, wine_name, 类别，类别编码，sample_id
  - 列：20-120Hz 的频率值
  - 值：对应的阻抗值
- 输出宽表数据到 `频率数据宽表.csv`
- 实现特征标准化处理
- 支持特征选择（关键频率点提取）

### 2. 模型架构

项目包含三个不同的分类器实现：

#### 2.1 基础分类器

使用三层全连接神经网络构建的葡萄酒分类模型：

- 数据预处理：
  - 读取清洗后的数据
  - 按 wine_id 分组，将每个酒样的频率 - 阻抗数据组织成特征向量
  - 对数据进行标准化处理
  - 对类别标签进行编码
- 模型架构：
  - 三层全连接神经网络
  - 输入层：频率 - 阻抗特征
  - 两个隐藏层：每层 128 个神经元
  - 输出层：葡萄酒类别
- 训练过程：
  - 使用 Adam 优化器
  - 交叉熵损失函数
  - 训练 10000 个 epoch
  - 保存训练过程中的损失曲线
- 模型评估：
  - 在测试集上评估模型准确率
  - 保存训练好的模型权重
  - 输出类别编码映射关系

#### 2.2 CNN 分类器

使用卷积神经网络 (CNN) 构建的葡萄酒分类模型：

- 模型架构：
  - 初始卷积层：7x1 卷积核，32 通道
  - 三个残差块，每块包含：
    - 两个 3x1 卷积层
    - 批归一化层
    - LeakyReLU 激活函数
    - 残差连接
  - 注意力机制：每个残差块后接注意力模块
  - 最大池化层：2x1 池化核
  - 全连接层：512 -> 256 -> 类别数
  - 使用 Dropout(0.5) 和 Dropout(0.3) 防止过拟合
- 训练过程：
  - 使用 AdamW 优化器，权重衰减 0.01
  - 交叉熵损失函数
  - 使用 ReduceLROnPlateau 动态调整学习率
  - 训练 2000 个 epoch
  - 可选启用早停机制（默认关闭）
  - 使用 3 折交叉验证
  - 设置随机种子确保结果可重现
- 模型评估：
  - 在测试集上评估模型准确率
  - 输出类别编码映射关系
  - 计算平均准确率和标准差

#### 4.2.1 已实现的优化点

- 残差连接（Residual Connections）：帮助解决梯度消失问题，使网络更容易训练
- 注意力机制（Attention Mechanism）：让模型关注重要的频率特征
- 改进的激活函数：使用 LeakyReLU 替代 ReLU，缓解神经元死亡问题
- 学习率调度：使用 ReduceLROnPlateau 动态调整学习率
- 权重衰减：使用 AdamW 优化器，添加 L2 正则化
- 交叉验证：使用 K 折交叉验证提供更可靠的性能评估
- 早停机制：可选启用，防止过拟合

#### 2.3 特征选择分类器

通过特征选择优化的葡萄酒分类模型：

- 数据预处理：
  - 选择关键频率点（每隔 5 个点取一个）
  - 对特征进行标准化处理
  - 对类别标签进行编码
- 模型架构：
  - 三层全连接神经网络
  - 输入层：选择后的频率特征
  - 两个隐藏层：每层 64 个神经元
  - 输出层：葡萄酒类别
- 训练过程：
  - 使用 Adam 优化器
  - 交叉熵损失函数
  - 训练 5000 个 epoch
  - 保存训练过程中的损失曲线到 `loss_curve_feature_choose.png`
- 模型评估：
  - 在测试集上评估模型准确率
  - 保存训练好的模型权重到 `wine_classifier_feature_choose.pth`
  - 输出类别编码映射关系

### 3. 训练策略

#### 3.1 优化技术

- 使用 AdamW 优化器，权重衰减 0.01
- 动态学习率调整（ReduceLROnPlateau）
- 早停机制（Early Stopping）
- K 折交叉验证（K=3）
- 随机种子控制确保结果可重现

#### 3.2 损失函数

- 交叉熵损失函数
- 支持多分类任务
- 实现类别权重平衡

### 4. 模型评估

#### 4.1 性能指标

- 准确率：93%
- 精确率：0.93
- 召回率：0.93
- F1 分数：0.93

#### 4.2 分类报告

```
              precision    recall  f1-score   support
           0       0.93      0.92      0.93      1488
           1       0.91      0.90      0.91      1305
           2       0.97      0.98      0.98      1476
           3       0.89      0.90      0.89      1546
    accuracy                           0.93      5815
   macro avg       0.93      0.93      0.93      5815
weighted avg       0.93      0.93      0.93      5815
```

#### 4.3 特征重要性分析

- 频率特征重要性：0.037
- 阻抗特征重要性：0.963

## 5. 数据文件说明

- `cleaned_wine_data.csv`: 清洗后的原始数据
- `按连续频率把样本分组到sample.csv`: 按连续频率分组后的数据
- `插值+平均补齐到120Hz.csv`: 经过插值和补齐处理后的数据
- `频率数据宽表.csv`: 转换后的宽表格式数据
- `wine_classifier.pth`: 训练好的模型权重文件
- `loss_curve.png`: 训练过程中的损失曲线图
- `wine_classifier.py`: 葡萄酒分类器的主要实现文件，包含数据预处理、模型定义、训练和评估等功能
- `loss_curve_cnn.png`: CNN 分类器训练过程中的损失曲线图
- `wine_classifier_feature_choose.pth`: 特征选择分类器训练好的模型权重文件
- `loss_curve_feature_choose.png`: 特征选择分类器训练过程中的损失曲线图

## 创新点

1. 多阶段数据处理流程

   - 创新的数据分组策略
   - 智能插值与补齐算法
   - 高效的特征工程方法

2. 改进的 CNN 架构

   - 残差连接设计
   - 注意力机制集成
   - LeakyReLU 激活函数优化

3. 训练策略优化
   - 动态学习率调整
   - 早停机制
   - K 折交叉验证

## 环境依赖

项目依赖的 Python 包：

- PyTorch >= 1.8.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0

### 安装方法

使用 uv 包管理器安装依赖：

```bash
# 安装 uv
pip install uv

# 安装项目依赖
uv sync

# 可选：创建虚拟环境
uv venv
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate     # Windows
uv sync
```

## 数据集

原始数据集包含以下类别：

- 啤酒
- 果酒
- 白酒
- 鸡尾酒

数据集下载：[raw_data.zip](https://github.com/eric-gitta-moore/chinese-wine-classify/releases/tag/v0.0.1)

## 使用说明

1. 数据预处理

```bash
python group_wine_data.py
python interp.py
python transport.py
```

2. 模型训练

```bash
# 基础分类器
python wine_classifier.py

# CNN分类器
python wine_classifier_cnn.py

# 特征选择分类器
python wine_classifier_feature_choose.py
```

3. 模型推理

```bash
python wine_inference.py
```

## 优化方向

1. 模型架构优化：

   - 尝试更深的网络结构（如 ResNet、DenseNet）
   - 实验不同的注意力机制（如 Transformer、Squeeze-and-Excitation）
   - 使用更复杂的卷积操作（如空洞卷积、可分离卷积）
   - 添加跳跃连接（Skip Connections）和特征融合

2. 训练策略优化：

   - 实现更复杂的学习率调度策略（如 OneCycleLR、CosineAnnealing）
   - 添加标签平滑（Label Smoothing）技术
   - 实现混合精度训练（Mixed Precision Training）
   - 使用梯度裁剪（Gradient Clipping）
   - 实验不同的优化器（如 Ranger、Lion）

3. 数据增强：

   - 添加频率域的数据增强（如随机频率偏移、噪声添加）
   - 实现样本混合技术（如 Mixup、CutMix）
   - 添加对抗训练（Adversarial Training）

4. 特征工程：

   - 提取频域特征（如 FFT、小波变换）
   - 添加统计特征（如均值、方差、偏度、峰度）
   - 实现特征选择算法（如 Lasso、RFE）

5. 集成学习：

   - 实现模型集成（如 Bagging、Boosting）
   - 使用模型蒸馏（Model Distillation）
   - 实验不同的投票策略

6. 超参数优化：

   - 使用贝叶斯优化（Bayesian Optimization）
   - 实现网格搜索（Grid Search）
   - 使用随机搜索（Random Search）

7. 模型解释性：

   - 添加特征重要性分析
   - 实现注意力可视化
   - 使用 SHAP 值解释模型预测

8. 部署优化：
   - 模型量化（Model Quantization）
   - 模型剪枝（Model Pruning）
   - 实现模型压缩技术
