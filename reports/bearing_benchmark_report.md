# XJTU-SY 滚动轴承数据分析与多模型对比报告

- **生成时间**：2026-11-24 10:46:53
- **数据目录**：`C:\Users\bigci\Desktop\DEMO\AGENT\data\xjtu_sy`
- **轴承总数**：12 个（有效样本 244 条，54 维时频特征）
- **深度学习配置**：epochs=35，batch_size=32，seed=42

## 一、特征相关性分析与 PCA 降维

- 原始特征维度：**54 维**，发现高相关冗余特征对（|r| ≥ 0.90）：**62 对**。
- PCA 降维（95% 累计方差阈值）：由 **54 维** 压缩至 **6 维**（累计解释方差 `95.12%`）。
- 空间划分下随机森林对比：原始 54 维 Accuracy=`95.08%` / Macro-F1=`94.89%` vs PCA 降维后 Accuracy=`91.80%` / Macro-F1=`91.75%`。

## 二、故障分类多模型对比（按轴承单元 7:3 空间不重叠划分）

- 训练轴承（7 个）：`35Hz12kN/Bearing1_4, 40Hz10kN/Bearing3_4, 40Hz10kN/Bearing3_3, 40Hz10kN/Bearing3_2, 35Hz12kN/Bearing1_2, 37.5Hz11kN/Bearing2_2, 40Hz10kN/Bearing3_1`
- 测试轴承（5 个）：`37.5Hz11kN/Bearing2_3, 37.5Hz11kN/Bearing2_1, 35Hz12kN/Bearing1_5, 37.5Hz11kN/Bearing2_4, 35Hz12kN/Bearing1_1`

| 模型 | Accuracy | Macro-F1 | 训练耗时 (s) | 单样本推理耗时 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 随机森林 (RF) | 95.08% | 94.89% | 1.105 | 1.0280 |
| 1D-CNN (PyTorch) | 85.25% | 85.88% | 3.862 | 0.2005 |
| LSTM (PyTorch) | 83.61% | 84.79% | 1.000 | 0.1274 |

## 三、时间窗划分对比（训练集 20%~55% → 测试集 55%~70%）

- 训练样本数：`86` | 测试样本数：`39`

| 模型 | Accuracy | Macro-F1 | 训练耗时 (s) | 单样本推理耗时 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 随机森林 (RF) | 100.00% | 100.00% | 0.507 | 1.8349 |
| 1D-CNN (PyTorch) | 66.67% | 74.04% | 0.925 | 0.1996 |
| LSTM (PyTorch) | 100.00% | 100.00% | 1.041 | 0.1747 |

## 四、RUL 剩余寿命预测归一化策略对比

| 归一化策略 | 相对尺度 RMSE | 相对尺度 MAE | 相对尺度 R² | 分钟尺度 RMSE | 分钟尺度 MAE | 分钟尺度 R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max-based 相对归一化 [0,1] | 0.0473 | 0.0338 | 0.9731 | 0.93 | 0.67 | 0.9739 |
| 工况分组归一化 | 0.0796 | 0.0624 | 0.9237 | 1.62 | 1.25 | 0.9211 |
| 绝对剩余寿命 (分钟) | 0.0754 | 0.0574 | 0.9316 | 1.46 | 1.12 | 0.9352 |

## 五、生成的对比图表

- `reports/figures/bearing_model_comparison.png`：RF / 1D-CNN / LSTM 准确率、Macro-F1 与计算耗时对比
- `reports/figures/bearing_confusion_matrices.png`：三模型混淆矩阵并排对比图
- `reports/figures/bearing_accuracy_over_time.png`：三模型随退化时间推移准确率与 Macro-F1 曲线
- `reports/figures/bearing_rul_degradation.png`：Max-based vs 分组归一化 vs 绝对 RUL 对比及退化曲线
- `reports/figures/bearing_feature_correlation_pca.png`：Top-12 特征相关性热力图与 PCA 方差解释率
