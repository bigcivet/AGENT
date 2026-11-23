# 工业数据分析与挖掘综合实训 — 设备运维诊断智能体（AI 老师傅）

本项目是「工业数据分析与挖掘综合实训」智能体开发部分的第一阶段成果：**设备运维诊断智能体**。  
参考工业界“AI 老师傅”思路，基于西交大 **XJTU-SY 滚动轴承加速寿命试验数据集**，打通了从**振动信号加载 → 时频域特征提取 → 无监督异常检测 → 故障分类与剩余寿命（RUL）预测 → 大模型结构化诊断报告 → 交互式对话界面**的完整闭环，并提供模拟振动数据生成器，保证无真实数据、无 API Key 也能完整跑通全流程。

---

## 一、核心功能与架构

```text
AGENT/
├── app/
│   └── streamlit_app.py              # Streamlit 交互式运维工作台（对话诊断 + 波形监测 + 报告导出）
├── scripts/
│   ├── generate_mock_data.py         # 符合 XJTU-SY 目录与格式的模拟振动数据生成脚本
│   └── train_pipeline.py             # 一键特征提取、模型训练、离线评估与端到端诊断验证脚本
├── src/
│   └── equipment_agent/              # 设备运维诊断智能体核心包（预留 src/vision_agent/ 扩展位）
│       ├── config.py                 # 全局配置与环境变量（.env）加载
│       ├── data/
│       │   └── loader.py             # XJTU-SY 数据加载器（支持逐文件流式读取、按轴承单元切分）
│       ├── features/
│       │   └── extractor.py          # 双通道时域 + FFT 频域特征提取（54 维物理特征）
│       ├── models/
│       │   ├── base.py               # 抽象基类（统一接口，方便后续无缝替换为 CNN / LSTM）
│       │   ├── anomaly.py            # 孤立森林（Isolation Forest）无监督异常检测
│       │   ├── classifier.py         # 随机森林故障分类基线模型
│       │   ├── rul.py                # 随机森林 RUL 剩余寿命预测基线模型（Max-based 归一化）
│       │   └── trainer.py            # 离线训练、评估与 joblib 模型持久化流水线
│       ├── report/
│       │   └── generator.py          # 结构化诊断报告生成器（OpenAI 兼容接口 + 规则模板自动降级）
│       └── agent/
│           ├── tools.py              # 智能体诊断工具集（OpenAI Function Calling Schema）
│           └── orchestrator.py       # AI 老师傅编排器（大模型工具调用 + 本地意图路由回退）
├── tests/                            # pytest 单元测试与端到端集成测试
├── .env.example                      # 大模型 API 环境变量配置模板
├── requirements.txt                  # Python 依赖清单
└── README.md
```

### 核心技术亮点
1. **严格对齐 XJTU-SY 数据规范**：支持 25.6 kHz、每文件 32,768 点、双列（`Horizontal_vibration_signals`, `Vertical_vibration_signals`）标准 CSV，内置西交大 15 个轴承工况与失效类型映射，且按“单个轴承单元”切分训练/测试集避免数据泄露。
2. **完整时频域特征工程**：水平/垂直双通道各提取 11 维时域特征（RMS、峭度、偏度、峰值因子、波形因子、脉冲因子、裕度因子等）与 15 维频域特征（FFT 主频、频谱重心、均方频率、频谱熵及 0-1k / 1-3k / 3-6k / 6-12.8kHz 四频带能量占比）+ 通道间 RMS 比值与相关系数，共 54 维。
3. **可扩展模型基类**：异常检测采用无监督 `IsolationForest`；故障分类与 RUL 预测默认使用随机森林基线，并通过 `BaseFaultClassifier` / `BaseRULPredictor` 抽象基类预留接口，后续可直接替换为 PyTorch `CNN` / `LSTM` 而无需修改智能体和前端代码。
4. **双模式智能体与报告生成**：配置 OpenAI 兼容 API（如 DeepSeek、通义千问）时自动启用大模型 Function Calling 与 JSON 结构化报告；未配置 Key 或断网时无缝降级为本地规则路由器与工业机理模板报告，保证实训演示 100% 可用。

---

## 二、环境安装

建议使用 Python 3.10 及以上版本（已在 Python 3.12 环境验证通过）。

```bash
# 1. 创建并激活虚拟环境（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS 激活方式：
# source .venv/bin/activate

# 2. 安装项目依赖（使用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 三、数据准备（二选一）

### 方式 A：快速体验 — 使用模拟振动数据（推荐首次运行）
没有真实数据集时，可直接运行模拟数据脚本生成与 XJTU-SY 目录、采样率（25.6kHz）及单文件点数（32768）完全一致的轴承全生命周期振动数据：

```bash
python scripts/generate_mock_data.py --output data/xjtu_sy --files-per-bearing 24 --samples-per-file 32768
```

### 方式 B：使用真实西交大 XJTU-SY 轴承数据集
将下载并解压后的真实数据放到 `data/xjtu_sy/` 目录下，保持标准二级目录结构即可自动识别：

```text
data/xjtu_sy/
├── 35Hz12kN/
│   ├── Bearing1_1/
│   │   ├── 1.csv
│   │   ├── 2.csv
│   │   └── ...
│   └── ...
├── 37.5Hz11kN/
│   └── Bearing2_1/ ...
└── 40Hz10kN/
    └── Bearing3_1/ ...
```

> 真实数据集参考下载地址见实训指导书（百度网盘），每个文件默认 1 分钟采样、两列水平/垂直振动信号。

---

## 四、大模型 API 配置（可选）

不配置 API Key 也可完整运行（系统会自动使用本地规则路由器与规则模板生成结构化报告）。  
如需接入 DeepSeek、通义千问等 OpenAI 兼容大模型：

1. 复制 `.env.example` 为 `.env`：
   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # Linux / macOS
   ```
2. 在 `.env` 中填入密钥与模型名称，例如 DeepSeek：
   ```ini
   OPENAI_API_KEY=sk-你的真实密钥
   OPENAI_BASE_URL=https://api.deepseek.com/v1
   OPENAI_MODEL=deepseek-chat
   ```
   或阿里云通义千问（DashScope 兼容模式）：
   ```ini
   OPENAI_API_KEY=sk-你的真实密钥
   OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   OPENAI_MODEL=qwen-plus
   ```

---

## 五、运行全流程

### 1. 一键训练模型并执行命令行端到端验证
```bash
# 若 data/xjtu_sy 为空，附加 --generate-mock 会自动生成模拟数据并完成训练与诊断演示
python scripts/train_pipeline.py --generate-mock
```
训练完成后，模型文件将保存在 `artifacts/models/`：
- `isolation_forest.joblib`（无监督异常检测器）
- `rf_fault_classifier.joblib`（随机森林故障分类器）
- `rf_rul_predictor.joblib`（随机森林 RUL 预测器）
- `metrics.json`（离线评估指标报告）

### 2. 启动 Streamlit 交互式运维工作台
```bash
streamlit run app/streamlit_app.py
```
启动后浏览器访问 `http://localhost:8501`，包含三个核心功能页：
- **💬 AI 老师傅对话诊断**：模拟“工人提问 → Agent 自主调用诊断工具 → 解释结论与维修建议”，支持展开查看每一步工具调用入参与出参。
- **📈 实时监测与时频分析**：逐文件流式回放轴承 RMS 退化、Isolation Forest 异常分数、RUL 预测曲线，查看当前分钟时域波形与 FFT 频谱，支持上传单个自定义 CSV 文件即时诊断。
- **📋 结构化报告与模型评估**：一键生成并导出 Markdown / JSON 格式诊断报告，查看模型离线准确率、Macro-F1、MAE/RMSE/R² 与特征重要性排行。

### 3. 运行自动化测试
```bash
pytest -v
```

---

## 六、后续扩展说明

1. **升级深度学习模型（CNN / LSTM）**：
   在 `src/equipment_agent/models/` 下新建 `cnn_classifier.py` 继承 `BaseFaultClassifier`，新建 `lstm_rul.py` 继承 `BaseRULPredictor`，实现 `fit / predict / evaluate / save / load` 后在 `trainer.py` 中替换即可，无需改动 Agent 和前端代码。
2. **下一阶段：视觉质检智能体**：
   可在 `src/vision_agent/` 下平行构建 NEU-DET 表面缺陷检测工具（YOLO/ResNet）、SQLite 质量数字孪生记忆库、自然语言统计查询（Text-to-SQL / Tool Calling）以及基于本地工艺文档的 RAG 根因预警模块，并与当前运维工作台统一集成。
