# 基于阻抗频率的葡萄酒智能分类系统

本项目是一个基于葡萄酒阻抗频率数据的智能分类系统，通过深度学习技术实现对不同种类葡萄酒的自动识别和分类。系统采用多阶段数据处理流程，结合多种深度学习模型，实现了高精度的葡萄酒分类。

## 1. 数据分组

使用 `group_wine_data.py` 脚本将原始数据按连续频率进行分组：
- 读取清洗后的数据文件 `cleaned_wine_data.csv`
- 根据频率的连续性为每个样本分配唯一的 `sample_id`
- 当频率不连续或非单调递增时，创建新的样本 ID
- 输出分组后的数据到 `按连续频率把样本分组到sample.csv`

## 2. 数据插值和补齐

使用 `interp.py` 脚本对数据进行插值和补齐处理：
- 读取分组后的数据
- 对每个葡萄酒样本（wine_id）进行处理：
  - 识别相对完整的样本（包含 120Hz 数据）
  - 对完整样本进行线性插值，生成 20-120Hz 的连续数据
  - 计算完整样本的平均阻抗值
  - 对不完整的样本，使用平均阻抗值进行补齐
- 输出处理后的数据到 `插值+平均补齐到120Hz.csv`

## 3. 数据转宽表

使用 `transport.py` 脚本将数据转换为宽表格式：
- 读取插值补齐后的数据
- 使用 pivot 函数将频率数据转换为宽表格式：
  - 行索引：wine_id, wine_name, 类别，类别编码，sample_id
  - 列：20-120Hz 的频率值
  - 值：对应的阻抗值
- 输出宽表数据到 `频率数据宽表.csv`

## 4. 葡萄酒分类器

项目包含三个不同的分类器实现：

### 4.1 基础分类器 (wine_classifier.py)
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

### 4.2 CNN 分类器 (wine_classifier_cnn.py)
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

#### 4.2.2 未来优化方向
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

### 4.3 特征选择分类器 (wine_classifier_feature_choose.py)
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
  - 保存训练过程中的损失曲线到`loss_curve_feature_choose.png`
- 模型评估：
  - 在测试集上评估模型准确率
  - 保存训练好的模型权重到`wine_classifier_feature_choose.pth`
  - 输出类别编码映射关系

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

## 6. 环境依赖

项目依赖的 Python 包在 `requirements.txt` 中列出，主要包括：
- pandas
- numpy
- scipy
- torch
- sklearn
- matplotlib

### 使用 uv 安装依赖

[uv](https://github.com/astral-sh/uv) 是一个快速的 Python 包安装器和解析器，比 pip 更快。以下是使用 uv 安装项目依赖的步骤：

1. 安装 uv（如果尚未安装）：
   ```bash
   pip install uv
   ```

2. 使用 uv 安装项目依赖：
   ```bash
   uv sync
   ```

3. 可选：创建虚拟环境并安装依赖：
   ```bash
   # 创建虚拟环境
   uv venv
   
   # 激活虚拟环境
   # Windows:
   .venv\Scripts\activate
   # Linux/MacOS:
   source .venv/bin/activate
   
   # 安装依赖
   uv sync
   ```

uv 的优势：
- 比 pip 快 10-100 倍
- 支持并行安装
- 更好的依赖解析
- 支持锁定文件
- 与 pip 完全兼容

## 数据集

- [raw_data.zip](https://github.com/eric-gitta-moore/chinese-wine-classify/releases/tag/v0.0.1)