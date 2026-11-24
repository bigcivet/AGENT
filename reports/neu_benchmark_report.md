# NEU-DET 金属表面瑕疵多模型分类对比报告

- **生成时间**：2026-11-24 10:48:21
- **数据目录**：`C:\Users\bigci\Desktop\DEMO\AGENT\data\neu_det`
- **数据集划分（70% / 15% / 15% 分层）**：总计 `360` 张（训练集 `252` / 验证集 `54` / 测试集 `54`）
- **训练配置**：epochs=5，batch_size=16，image_size=224，freeze_backbone=True

## 一、总体性能与效率对比（独立测试集）

| 模型 | 预训练权重 | 参数量 (M) | Test Accuracy | Test Macro-F1 | 训练耗时 (s) | 单图推理耗时 (ms) |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 是 | 11.18 | 100.00% | 100.00% | 56.61 | 36.66 |
| EfficientNet-B0 | 是 | 4.02 | 100.00% | 100.00% | 41.73 | 19.44 |

## 二、6 类金属表面缺陷逐类 Precision / Recall / F1 明细

| 缺陷类别 | ResNet18 P | ResNet18 R | ResNet18 F1 | EfficientNet-B0 P | EfficientNet-B0 R | EfficientNet-B0 F1 | 测试样本数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 裂纹 (`crazing`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 |
| 夹杂 (`inclusion`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 |
| 斑块 (`patches`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 |
| 麻点 (`pitted_surface`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 |
| 压入氧化皮 (`rolled-in_scale`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 |
| 划痕 (`scratches`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 9 |

## 三、生成的对比图表

- `reports/figures/neu_model_comparison.png`：ResNet18 vs EfficientNet-B0 准确率、Macro-F1、每类 F1 及推理耗时对比
- `reports/figures/neu_confusion_matrices.png`：ResNet18 vs EfficientNet-B0 测试集 6×6 混淆矩阵对比
