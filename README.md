# 工业数据分析与挖掘综合实训 — 工业智能体综合工作台

本项目是「工业数据分析与挖掘综合实训」的智能体开发完整成果，在同一套统一架构与 Streamlit 工作台下集成了两大核心工业智能体：
1. **设备运维诊断智能体（`src/equipment_agent`）**：基于西交大 **XJTU-SY 滚动轴承加速寿命试验数据集**，打通「振动信号加载 → 54 维时频特征提取 → Isolation Forest 异常检测 → 随机森林故障分类与 RUL 预测 → 大模型结构化诊断报告 → 对话交互」闭环；
2. **视觉质检智能体（`src/vision_agent`）**：基于东北大学 **NEU-DET 金属表面缺陷数据集**（6 类表面缺陷），打通「PyTorch ResNet18 迁移学习识别（含可选 YOLO 检测适配）→ SQLite 质量数字孪生记忆库 → 自然语言查库（Tool Calling + 受限只读 SQL）→ 连续缺陷主动预警与本地工艺文档 RAG 溯源 → 交互式质检看板」全流程。

两个智能体共用同一套大模型客户端配置（OpenAI 兼容接口）与 Streamlit 交互工作台，并提供完整的模拟数据与模拟产线流水生成脚本，保证**无真实数据集、无 GPU、无 API Key** 也能完整跑通全流程。

---

## 一、项目目录结构

```text
AGENT/
├── app/
│   ├── streamlit_app.py                  # 统一 Streamlit 工业智能体工作台入口（侧边栏切换两大智能体）
│   └── vision_ui.py                      # 视觉质检智能体交互页面（上传检测、看板、对话查询、预警与 RAG）
├── data/
│   └── process_docs/                     # 工艺知识库（版本管理，供视觉质检 RAG 溯源引用）
│       ├── 01_热轧带钢轧制工艺规程.md
│       ├── 02_金属表面缺陷成因与处置手册.md
│       ├── 03_轧辊维护与导卫检修作业指导书.md
│       └── 04_产线质量异常分级预警与应急预案.md
├── scripts/
│   ├── generate_mock_data.py             # 生成 XJTU-SY 模拟振动数据（25.6kHz / 32768点）
│   ├── train_pipeline.py                 # 设备运维智能体一键训练与端到端诊断验证
│   ├── generate_mock_neu_det.py          # 生成 NEU-DET 200x200 灰度模拟缺陷图与 PASCAL VOC XML 标注
│   ├── simulate_inspection_stream.py     # 模拟多台机台、多天质检流水写入 SQLite 并触发主动预警
│   └── train_vision_pipeline.py          # 视觉质检 ResNet18 一键训练、评估与端到端联调脚本
├── src/
│   ├── equipment_agent/                  # 【模块一】设备运维诊断智能体（AI 老师傅）
│   │   ├── config.py                     # 全局配置与 .env 加载（供两个智能体复用）
│   │   ├── data/loader.py                # XJTU-SY 轴承数据加载器
│   │   ├── features/extractor.py         # 双通道 54 维时域 + FFT 频域特征提取
│   │   ├── models/                       # 异常检测(IF)、故障分类(RF)、RUL预测(RF)及 CNN/LSTM 扩展基类
│   │   ├── report/generator.py           # OpenAI 兼容结构化诊断报告 + 规则模板降级
│   │   └── agent/                        # 运维诊断工具集与编排器
│   └── vision_agent/                     # 【模块二】视觉质检智能体（NEU-DET）
│       ├── config.py                     # 视觉配置（复用大模型配置，定义 6 类缺陷映射与阈值）
│       ├── llm.py                        # 复用的 OpenAI 兼容大模型客户端工厂
│       ├── vision/
│       │   ├── dataset.py                # NEU-DET 目录加载器、VOC XML 解析、数据增强与 YOLO 导出
│       │   ├── classifier.py             # PyTorch ResNet18 迁移学习分类器与评估（准确率/F1/混淆矩阵）
│       │   └── yolo_detector.py          # 可选 Ultralytics YOLO 目标检测适配器
│       ├── memory/
│       │   └── store.py                  # SQLite 产线质量“数字孪生”记忆库 + 受限只读 SQL 执行器
│       ├── decision/
│       │   ├── rag.py                    # 轻量级工艺文档 RAG 检索引擎（TF-IDF 字符 n-gram，零重型依赖）
│       │   └── alert_engine.py           # 连续同类缺陷主动预警引擎（阈值可配置 + 自动关联工艺原因）
│       └── agent/
│           ├── tools.py                  # 视觉质检工具集（统计、机台趋势、对比、预警、RAG、只读 SQL）
│           └── orchestrator.py           # 自然语言质检推理 Agent（LLM Function Calling + 本地规则回退）
├── tests/                                # pytest 单元测试与端到端集成测试
├── .env.example                          # 大模型 API 与路径环境变量模板
├── requirements.txt                      # Python 依赖清单
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

不配置 API Key 也可完整运行（系统会自动降级为本地规则路由器与模板引擎）。  
如需接入 DeepSeek、通义千问等 OpenAI 兼容大模型：

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux / macOS
```

在 `.env` 中配置密钥与模型名，例如：
```ini
OPENAI_API_KEY=sk-你的真实密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

---

## 四、视觉质检智能体运行指南（NEU-DET）

### 1. 数据准备（二选一）

- **方式 A：快速体验 — 生成模拟 200×200 灰度缺陷图与 VOC XML 标注**
  ```bash
  # 默认每类 60 张（共 360 张），如需对齐官方规模可设 --per-class 300
  python scripts/generate_mock_neu_det.py --output data/neu_det --per-class 60
  ```
- **方式 B：放入真实 NEU-DET 数据集**  
  将真实 NEU-DET 数据解压至 `data/neu_det/`，加载器自动兼容以下任意目录布局：
  - 标准检测版：`data/neu_det/train/images/<类别>/*.jpg` + `train/annotations/*.xml`（以及 `validation/`）
  - 扁平检测版：`data/neu_det/IMAGES/*.jpg` + `data/neu_det/ANNOTATIONS/*.xml`
  - 分类目录版：`data/neu_det/<类别>/*.jpg`（6 类：`crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled-in_scale`、`scratches`）

### 2. 一键训练 ResNet18 分类模型 + 模拟产线流水 + 联调验证

```bash
# CPU 友好配置：可自由调整 --epochs / --batch-size / --freeze-backbone
python scripts/train_vision_pipeline.py --generate-mock --simulate-stream --epochs 5 --freeze-backbone
```

产物说明：
- `artifacts/vision_models/resnet18_neu_det.pt`：PyTorch ResNet18 缺陷分类权重
- `artifacts/vision_models/vision_metrics.json`：验证集准确率、Macro-F1、每类指标与 6×6 混淆矩阵
- `data/vision_memory.db`：SQLite 产线质量数字孪生数据库（含检测流水表 `inspections` 与预警表 `alerts`）

若需单独重新模拟多机台（`M01`~`M04`）、多天检测流水并触发连续缺陷预警：
```bash
python scripts/simulate_inspection_stream.py --days 7 --threshold 3
```

---

## 五、设备运维诊断智能体运行指南（XJTU-SY）

```bash
# 生成 XJTU-SY 模拟振动数据（或将真实数据集放入 data/xjtu_sy/）并一键训练评估
python scripts/train_pipeline.py --generate-mock
```

模型工件保存在 `artifacts/models/`（`isolation_forest.joblib`、`rf_fault_classifier.joblib`、`rf_rul_predictor.joblib`、`metrics.json`）。

---

## 六、启动统一 Streamlit 工作台

```bash
streamlit run app/streamlit_app.py
```

打开浏览器访问 `http://localhost:8501`，在左侧边栏可自由切换两大智能体：

### 1. 🔍 视觉质检智能体（NEU-DET）
- **📷 图片上传检测**：上传本地金属表面图片或抽取 NEU-DET 样本，实时输出 6 类缺陷概率分布、自动写入 SQLite，若连续同类缺陷达到阈值立即触发主动预警并展示工艺依据。
- **📊 质量数字孪生看板**：展示缺陷总量 KPI、各类缺陷分布柱状图、各机台按天缺陷趋势曲线、机台横向对比、流水明细表，以及 ResNet18 验证指标与 6×6 混淆矩阵热力图。
- **💬 质量对话查询**：支持自然语言提问（如“最近哪种缺陷最多？”、“3 号机台有没有异常趋势？”、“压入氧化皮的工艺原因和处置措施是什么？”），支持展开查看工具调用与工艺文档引用轨迹，也支持直接执行受限只读 `SELECT` 查询。
- **🚨 主动预警与工艺知识库**：支持动态调整连续缺陷预警阈值、一键重扫、预警状态流转（确认/闭环解决），以及直接对 `data/process_docs/` 进行 RAG 关键词检索。

### 2. 🛠️ 设备运维诊断智能体（XJTU-SY）
- **💬 AI 老师傅对话诊断** / **📈 实时监测与时频分析** / **📋 结构化报告与模型评估**。

---

## 七、运行自动化测试

```bash
pytest -v
```
