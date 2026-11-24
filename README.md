# 工业数据分析与挖掘综合实训 — 工业智能体与多模型分析综合工作台

本项目是「工业数据分析与挖掘综合实训」的完整交付成果，在一套模块化工程架构与三页式 Streamlit 工作台下，整合了**西交大 XJTU-SY 滚动轴承数据分析与设备运维诊断智能体**、**东北大学 NEU-DET 金属表面缺陷多模型分类与视觉质检智能体**，以及**统一的数据分析与多模型基准对比中心**：

1. **🔍 视觉质检智能体（`src/vision_agent`）**：基于 NEU-DET 6 类金属表面缺陷，支持 **ResNet18 / EfficientNet-B0** 迁移学习识别（离线自动回退 `weights=None`，含可选 YOLO 检测适配）→ **SQLite 质量数字孪生记忆库** → **自然语言查库（Tool Calling + 受限只读 SQL）** → **连续缺陷主动预警与工艺文档 RAG 溯源**；
2. **🛠️ 设备运维诊断智能体（`src/equipment_agent`）**：基于 XJTU-SY 滚动轴承振动信号，打通 **54 维双通道时频特征提取（含相关性分析与可选 PCA 降维）** → **Isolation Forest 异常检测** → **可切换故障分类模型（RF / 1D-CNN / LSTM）** → **Max-based 相对 RUL 预测** → **大模型结构化诊断报告与多轮对话**；
3. **📊 数据分析与模型对比中心（`app/benchmark_ui.py` + `scripts/run_*_benchmark.py`）**：一键运行轴承（空间 7:3 划分、时间窗 20%~55%→55%~70% 划分、随时间推移趋势、RUL 归一化策略对比）与 NEU-DET（70%/15%/15% 分层划分、ResNet18 vs EfficientNet-B0）对比实验，自动输出 JSON/Markdown 报告与高清 PNG 图表并在工作台集中展示。

两个智能体共用同一套 OpenAI 兼容大模型配置，且内置模拟数据生成与本地规则降级机制，**无真实数据集、无 GPU、无 API Key** 也可一键完整复现全部实验与界面功能。

---

## 一、项目目录结构

```text
AGENT/
├── app/
│   ├── streamlit_app.py                  # 统一 Streamlit 3 页工作台入口（侧边栏切换三大页面）
│   ├── vision_ui.py                      # 页面 1：视觉质检智能体（支持切换 ResNet18 / EfficientNet-B0）
│   └── benchmark_ui.py                   # 页面 3：数据分析与多模型对比看板（读取 reports/ 指标与图表）
├── docs/                                 # 实训考核交付文档
│   ├── 实训报告.md                        # 完整综合实训报告（含架构、真实实验数据表、图表引用与测试总结）
│   └── 实训手册每日记录.md                # 按 10 天（两周）实训节奏整理的每日记录详稿
├── reports/                              # 一键基准对比脚本生成的评估报告与高清 PNG 图表
│   ├── bearing_benchmark_report.json     # XJTU-SY 轴承多模型与 RUL 归一化对比 JSON 报告
│   ├── bearing_benchmark_report.md       # XJTU-SY 轴承多模型对比 Markdown 报告
│   ├── neu_benchmark_report.json         # NEU-DET 多模型分类对比 JSON 报告
│   ├── neu_benchmark_report.md           # NEU-DET 多模型分类对比 Markdown 报告
│   └── figures/                          # 全部对比实验高清 PNG 图表
├── data/
│   └── process_docs/                     # 工艺知识库（版本管理，供视觉质检 RAG 溯源引用）
│       ├── 01_热轧带钢轧制工艺规程.md
│       ├── 02_金属表面缺陷成因与处置手册.md
│       ├── 03_轧辊维护与导卫检修作业指导书.md
│       └── 04_产线质量异常分级预警与应急预案.md
├── scripts/
│   ├── generate_mock_data.py             # 生成 XJTU-SY 模拟振动数据（25.6kHz / 32768点）
│   ├── train_pipeline.py                 # 设备运维智能体一键训练与端到端诊断验证
│   ├── run_bearing_benchmark.py          # 【基准对比】XJTU-SY 轴承 RF / 1D-CNN / LSTM + RUL 归一化一键对比
│   ├── generate_mock_neu_det.py          # 生成 NEU-DET 200×200 灰度模拟缺陷图与 PASCAL VOC XML 标注
│   ├── simulate_inspection_stream.py     # 模拟多台机台、多天质检流水写入 SQLite 并触发主动预警
│   ├── train_vision_pipeline.py          # 视觉质检一键训练与端到端联调脚本
│   └── run_neu_benchmark.py              # 【基准对比】NEU-DET 70/15/15 ResNet18 vs EfficientNet-B0 一键对比
├── src/
│   ├── equipment_agent/                  # 【模块一】设备运维诊断智能体与轴承分析
│   │   ├── config.py                     # 全局配置与 .env 加载
│   │   ├── data/loader.py                # XJTU-SY 加载器、按轴承 7:3 划分、时间窗划分与时间分箱评估
│   │   ├── features/
│   │   │   ├── extractor.py              # 双通道 54 维时域 + FFT 频域特征提取
│   │   │   └── analysis.py               # 特征相关性分析与可选 PCA 降维工具
│   │   ├── models/
│   │   │   ├── base.py                   # 异常检测 / 故障分类 / RUL 预测统一抽象基类
│   │   │   ├── anomaly.py                # Isolation Forest 无监督异常检测
│   │   │   ├── classifier.py             # 随机森林故障分类器（RF）
│   │   │   ├── deep_classifiers.py       # PyTorch 1D-CNN 与 LSTM 故障分类器及工厂加载器
│   │   │   ├── rul.py                    # 随机森林 RUL 预测 + Max-based/分组/绝对归一化对比
│   │   │   └── trainer.py                # ModelBundle（支持 RF/1D-CNN/LSTM 切换）与训练流水线
│   │   ├── report/generator.py           # OpenAI 兼容结构化诊断报告 + 规则模板降级
│   │   └── agent/                        # 设备运维诊断工具集与编排器
│   └── vision_agent/                     # 【模块二】视觉质检智能体（NEU-DET）
│       ├── config.py                     # 视觉配置与 6 类缺陷中英文映射
│       ├── llm.py                        # 复用的 OpenAI 兼容大模型客户端工厂
│       ├── vision/
│       │   ├── dataset.py                # NEU-DET 加载器、VOC XML 解析、70/15/15 划分与 YOLO 导出
│       │   ├── classifier.py             # ResNet18 & EfficientNet-B0 分类器（离线自动回退）与切换工厂
│       │   └── yolo_detector.py          # 可选 Ultralytics YOLO 目标检测适配器
│       ├── memory/store.py               # SQLite 产线质量数字孪生记忆库 + 受限只读 SQL 执行器
│       ├── decision/                     # TF-IDF 工艺文档 RAG + 连续缺陷主动预警引擎
│       └── agent/                        # 视觉质检工具集与自然语言推理 Agent
├── tests/                                # pytest 单元测试、端到端测试与 Streamlit 三页 AppTest
├── .env.example
├── requirements.txt
└── README.md
```

---

## 二、环境安装

建议使用 Python 3.10 及以上版本（已在 Python 3.12 + CPU 环境验证通过）。

```bash
# 1. 创建并激活虚拟环境（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS 激活方式：
# source .venv/bin/activate

# 2. 若仅使用 CPU 运行 PyTorch，建议先安装轻量 CPU 版 torch/torchvision（可选）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装项目其余依赖（使用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. （可选）如需使用 YOLO 目标检测适配器：
# pip install ultralytics
```

---

## 三、大模型 API 配置（可选，两个智能体共用）

不配置 API Key 也可完整运行（系统自动降级为本地规则路由与机理模板引擎）。如需接入 DeepSeek、通义千问等 OpenAI 兼容大模型：

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux / macOS
```

在 `.env` 中填入：
```ini
OPENAI_API_KEY=sk-你的真实密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

---

## 四、一键多模型基准对比实验（生成报告与高清图表）

### 1. 西交大 XJTU-SY 轴承多模型对比（RF / 1D-CNN / LSTM + RUL 归一化 + PCA）

```bash
# 使用模拟数据快速跑通（若已有数据则直接读取 data/xjtu_sy/），同时保存 RF/1D-CNN/LSTM 部署权重
python scripts/run_bearing_benchmark.py --generate-mock --epochs 35 --batch-size 32
```

自动输出：
- 对比报告：`reports/bearing_benchmark_report.json`、`reports/bearing_benchmark_report.md`
- 5 张高清图表（`reports/figures/`）：
  - `bearing_model_comparison.png`（RF / 1D-CNN / LSTM 空间划分、时间窗划分与耗时柱状对比）
  - `bearing_confusion_matrices.png`（三模型混淆矩阵并排热力图）
  - `bearing_accuracy_over_time.png`（随退化时间推移准确率与 Macro-F1 曲线）
  - `bearing_rul_degradation.png`（Max-based vs 工况分组 vs 绝对 RUL 误差对比与退化预测曲线）
  - `bearing_feature_correlation_pca.png`（Top-12 退化敏感特征相关性热力图与 PCA 累计方差曲线）
- 部署模型权重（`artifacts/models/`）：`isolation_forest.joblib`、`rf_fault_classifier.joblib`、`cnn1d_fault_classifier.pt`、`lstm_fault_classifier.pt`、`rf_rul_predictor.joblib`。

### 2. NEU-DET 表面缺陷多模型对比（70/15/15 划分：ResNet18 vs EfficientNet-B0）

```bash
# CPU 友好配置（若离线无法下载 ImageNet 权重，将自动优雅回退 weights=None）
python scripts/run_neu_benchmark.py --generate-mock --epochs 5 --batch-size 16 --freeze-backbone
```

自动输出：
- 对比报告：`reports/neu_benchmark_report.json`、`reports/neu_benchmark_report.md`
- 2 张高清图表（`reports/figures/`）：
  - `neu_model_comparison.png`（ResNet18 vs EfficientNet-B0 准确率、Macro-F1、每类 F1 与推理耗时对比）
  - `neu_confusion_matrices.png`（ResNet18 vs EfficientNet-B0 测试集 6×6 混淆矩阵对比）
- 双模型权重（`artifacts/vision_models/`）：`resnet18_neu_det.pt`、`efficientnet_b0_neu_det.pt` 及各自指标 JSON。

---

## 五、替换为真实工业数据集与一键重训指南

拿到真实官方数据集后，**无需修改任何业务代码**，按如下路径放置并重新运行命令即可得到真实结果：

### 1. 放置真实 XJTU-SY 数据集
将官方解压后的 3 个工况文件夹放入 `data/xjtu_sy/`：
```text
data/xjtu_sy/
├── 35Hz12kN/Bearing1_1/1.csv ...
├── 37.5Hz11kN/Bearing2_1/1.csv ...
└── 40Hz10kN/Bearing3_1/1.csv ...
```
一键重训与重新生成对比报告（真实数据集样本较多，可先用 `--step 2` 或 `--step 5` 加速特征提取，全量实验设为 `--step 1`）：
```bash
python scripts/run_bearing_benchmark.py --data-dir data/xjtu_sy --epochs 50 --batch-size 64 --step 1
```

### 2. 放置真实 NEU-DET 数据集
将官方 NEU-DET 压缩包解压至 `data/neu_det/`（自动兼容 `train/validation` 检测版、`IMAGES/ANNOTATIONS` 扁平版或 6 类别子目录版）：
```bash
# GPU 环境可去掉 --freeze-backbone 并调大 --epochs 15~20 获得更优迁移学习精度
python scripts/run_neu_benchmark.py --data-dir data/neu_det --epochs 10 --batch-size 32
# 同步重置并模拟多机台产线流水（可选）
python scripts/simulate_inspection_stream.py --days 7 --threshold 3
```

---

## 六、启动统一 Streamlit 三页工作台

```bash
streamlit run app/streamlit_app.py
```

访问 `http://localhost:8501`，侧边栏可自由切换三大核心页面：

1. **🔍 视觉质检智能体（NEU-DET）**：侧边栏支持切换 **ResNet18 / EfficientNet-B0** 分类模型，内置「图片上传检测」「质量数字孪生看板」「自然语言对话查询」「主动预警与工艺知识库」4 个标签页；
2. **🛠️ 设备运维诊断智能体（XJTU-SY）**：侧边栏支持切换 **随机森林 (RF) / 1D-CNN / LSTM** 故障分类器，内置「AI 老师傅对话诊断」「实时监测与时频分析」「结构化报告与模型评估」3 个标签页；
3. **📊 数据分析与模型对比**：集中加载展示 XJTU-SY（RF / 1D-CNN / LSTM + PCA + RUL 归一化）与 NEU-DET（ResNet18 vs EfficientNet-B0）的全部基准指标表与高清对比图表。

---

## 七、实训考核文档与自动化测试

- **综合实训报告**：[docs/实训报告.md](docs/实训报告.md)（含需求分析、架构设计、轴承与 NEU-DET 真实运行对比表与图表、双智能体实现、测试汇总与改进反思）
- **10 天实训手册每日记录**：[docs/实训手册每日记录.md](docs/实训手册每日记录.md)
- **运行全部自动化测试**（涵盖数据加载、特征/PCA、RF/1D-CNN/LSTM、Max-based RUL、ResNet18/EfficientNet-B0、SQLite/RAG/预警、智能体对话及 Streamlit 三页 AppTest）：
  ```bash
  pytest -v
  ```
