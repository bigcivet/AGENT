"""一键训练与端到端诊断验证脚本。

功能：
1. 检查 XJTU-SY 数据目录；若无数据且指定 `--generate-mock`，自动生成模拟振动数据；
2. 提取全量轴承的时域 + 频域特征，按轴承单元切分训练集与测试集；
3. 训练并持久化：孤立森林异常检测模型、随机森林故障分类模型、随机森林 RUL 预测模型；
4. 打印离线评估指标，并使用设备运维诊断智能体执行一次端到端示例诊断。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保直接运行脚本时可以导入 src 下的包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from equipment_agent.agent.orchestrator import EquipmentDiagnosisAgent
from equipment_agent.agent.tools import EquipmentToolkit
from equipment_agent.config import load_config
from equipment_agent.data.loader import XJTUSYDataLoader
from equipment_agent.models.trainer import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="设备运维诊断智能体一键训练与端到端验证脚本")
    parser.add_argument("--data-dir", type=str, default=None, help="XJTU-SY 数据目录（默认读取配置）")
    parser.add_argument("--artifact-dir", type=str, default=None, help="模型保存目录（默认读取配置）")
    parser.add_argument(
        "--generate-mock",
        action="store_true",
        help="当数据目录为空时，自动生成模拟 XJTU-SY 振动数据",
    )
    parser.add_argument("--files-per-bearing", type=int, default=24, help="模拟数据单轴承基准文件数")
    parser.add_argument("--samples-per-file", type=int, default=32768, help="模拟数据单文件采样点数")
    parser.add_argument("--step", type=int, default=1, help="特征提取采样步长（真实大数据集可适当增大加速）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else cfg.artifact_dir

    loader = XJTUSYDataLoader(data_dir=data_dir, sampling_rate=cfg.sampling_rate)
    if not loader.is_available:
        if args.generate_mock:
            from generate_mock_data import generate_mock_xjtu_dataset

            print(f"[提示] 目录 {data_dir} 下未发现数据，正在生成模拟 XJTU-SY 数据集...")
            generate_mock_xjtu_dataset(
                output_dir=data_dir,
                files_per_bearing=args.files_per_bearing,
                samples_per_file=args.samples_per_file,
                sampling_rate=cfg.sampling_rate,
            )
            loader.refresh()
        else:
            print(
                f"[错误] 未在 `{data_dir}` 找到任何 XJTU-SY 轴承 CSV 数据。\n"
                "请放入真实 XJTU-SY 数据，或附加 `--generate-mock` 参数自动生成模拟数据。"
            )
            sys.exit(1)

    print(f"[1/3] 扫描到 {len(loader.list_bearings())} 个轴承，开始提取时频特征并训练诊断模型...")
    pipeline = TrainingPipeline(
        loader=loader,
        artifact_dir=artifact_dir,
        step=args.step,
    )
    bundle = pipeline.run()
    m = bundle.metrics

    print("\n[2/3] 模型训练与离线评估完成，模型已保存至：", artifact_dir.resolve())
    print(f"  - 训练轴承 ({len(m['train_bearings'])})：{m['train_bearings']}")
    print(f"  - 测试轴承 ({len(m['test_bearings'])})：{m['test_bearings']}")
    print(
        f"  - 孤立森林异常检测：早期健康段误报率={m['anomaly_detection']['early_false_alarm_rate']*100:.1f}%，"
        f"晚期退化段检出率={m['anomaly_detection']['late_stage_detection_rate']*100:.1f}%"
    )
    print(
        f"  - 随机森林故障分类：准确率={m['fault_classification']['accuracy']*100:.2f}%，"
        f"Macro-F1={m['fault_classification']['macro_f1']*100:.2f}%"
    )
    print(
        f"  - 随机森林 RUL 预测（相对寿命）：MAE={m['rul_prediction']['mae']:.4f}，"
        f"RMSE={m['rul_prediction']['rmse']:.4f}，R²={m['rul_prediction']['r2']:.4f}"
    )

    print("\n[3/3] 启动设备运维诊断智能体进行端到端对话验证...")
    toolkit = EquipmentToolkit(loader=loader, bundle=bundle, config=cfg)
    agent = EquipmentDiagnosisAgent(toolkit=toolkit, config=cfg)

    sample_bearing = loader.list_bearings()[0]
    sample_minute = max(1, int(round(sample_bearing.total_life_minutes * 0.85)))
    question = f"请帮我全面诊断 {sample_bearing.bearing_id} 第 {sample_minute} 分钟的运行状态并给出处置建议"
    print(f"\n>>> 工人提问：{question}")
    response = agent.chat(question)
    print(f">>> 智能体模式：{response.mode} | 调用工具：{[t.tool_name for t in response.tool_calls]}\n")
    print(response.reply)


if __name__ == "__main__":
    main()
