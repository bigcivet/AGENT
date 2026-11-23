"""模拟多台机台、多天产线视觉质检流水并写入 SQLite 数字孪生库。

模拟场景设计（便于实训演示自然语言查询与主动预警）：
- 4 台机台：M01、M02、M03、M04，默认覆盖最近 7 天；
- M01 / M04：总体平稳运行，各类缺陷偶发；
- M02（2 号机台）：最近出现连续 `rolled-in_scale`（压入氧化皮），模拟除鳞水压偏低工况；
- M03（3 号机台）：近 2 天 `scratches`（划痕）数量显著激增且末尾连续出现 5 次，
  模拟导卫磨损/死辊刮擦导致的明显异常上升趋势。
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vision_agent.config import DEFECT_CLASSES, DEFECT_CN_MAP, DEFECT_SEVERITY_MAP, load_vision_config
from vision_agent.decision.alert_engine import AlertEngine
from vision_agent.decision.rag import ProcessDocRAG
from vision_agent.memory.store import InspectionRecord, QualityMemoryStore
from vision_agent.vision.dataset import NEUDetCatalog


def simulate_inspection_stream(
    db_path: str | Path,
    process_docs_dir: str | Path,
    neu_det_dir: str | Path | None = None,
    days: int = 7,
    alert_threshold: int = 3,
    reset_db: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """生成多机台、多天检测流水写入 SQLite，并执行连续缺陷预警扫描。"""
    rng = random.Random(seed)
    store = QualityMemoryStore(db_path)
    if reset_db:
        store.clear_all()

    # 若本地已有 NEU-DET 图像，关联真实相对路径以便前端回溯查看
    sample_images_by_cls: dict[str, list[str]] = {c: [] for c in DEFECT_CLASSES}
    if neu_det_dir is not None and Path(neu_det_dir).exists():
        catalog = NEUDetCatalog(neu_det_dir)
        for s in catalog.samples:
            sample_images_by_cls[s.label].append(str(s.image_path))

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    machines = ["M01", "M02", "M03", "M04"]

    # 各机台基础缺陷权重（模拟不同工位偏好）
    base_weights: dict[str, dict[str, float]] = {
        "M01": {
            "crazing": 0.14,
            "inclusion": 0.20,
            "patches": 0.24,
            "pitted_surface": 0.16,
            "rolled-in_scale": 0.14,
            "scratches": 0.12,
        },
        "M02": {
            "crazing": 0.10,
            "inclusion": 0.12,
            "patches": 0.16,
            "pitted_surface": 0.14,
            "rolled-in_scale": 0.36,
            "scratches": 0.12,
        },
        "M03": {
            "crazing": 0.10,
            "inclusion": 0.10,
            "patches": 0.14,
            "pitted_surface": 0.12,
            "rolled-in_scale": 0.14,
            "scratches": 0.40,
        },
        "M04": {
            "crazing": 0.18,
            "inclusion": 0.16,
            "patches": 0.22,
            "pitted_surface": 0.22,
            "rolled-in_scale": 0.12,
            "scratches": 0.10,
        },
    }

    records: list[InspectionRecord] = []

    for day_offset in range(days - 1, -1, -1):
        day_dt = now - timedelta(days=day_offset)
        day_str = day_dt.strftime("%Y%m%d")

        for machine in machines:
            # M03 近 2 天缺陷量明显飙升，模拟导卫异常劣化趋势
            if machine == "M03" and day_offset == 1:
                daily_n = 11
            elif machine == "M03" and day_offset == 0:
                daily_n = 15
            elif machine == "M02" and day_offset == 0:
                daily_n = 10
            else:
                daily_n = rng.randint(5, 7)

            weights_dict = dict(base_weights[machine])
            if machine == "M03" and day_offset <= 1:
                weights_dict["scratches"] = 0.65

            classes = list(weights_dict.keys())
            probs = list(weights_dict.values())

            for idx in range(daily_n):
                ts = day_dt.replace(hour=8, minute=0, second=0) + timedelta(
                    minutes=int((idx + 1) * (600 / max(1, daily_n))) + rng.randint(0, 12)
                )
                shift = "A" if ts.hour < 14 else "B"
                batch_id = f"B{day_str}-{machine}-{shift}"

                # 刻意在最近一天末尾注入连续同类缺陷，验证主动预警模块
                if machine == "M03" and day_offset == 0 and idx >= daily_n - 5:
                    defect = "scratches"
                    conf = round(rng.uniform(0.88, 0.98), 4)
                elif machine == "M02" and day_offset == 0 and idx >= daily_n - 4:
                    defect = "rolled-in_scale"
                    conf = round(rng.uniform(0.84, 0.96), 4)
                elif machine == "M04" and day_offset == 1 and 2 <= idx <= 4:
                    defect = "pitted_surface"
                    conf = round(rng.uniform(0.82, 0.94), 4)
                else:
                    defect = rng.choices(classes, weights=probs, k=1)[0]
                    conf = round(rng.uniform(0.76, 0.97), 4)

                img_pool = sample_images_by_cls.get(defect, [])
                img_path = rng.choice(img_pool) if img_pool else f"mock://{defect}_{rng.randint(1, 60)}.jpg"

                records.append(
                    InspectionRecord(
                        inspected_at=ts.strftime("%Y-%m-%d %H:%M:%S"),
                        machine_id=machine,
                        batch_id=batch_id,
                        defect_type=defect,
                        defect_type_cn=DEFECT_CN_MAP[defect],
                        confidence=conf,
                        image_path=img_path,
                        model_name="resnet18",
                        severity=DEFECT_SEVERITY_MAP[defect],
                        notes="在线视觉巡检自动入库",
                    )
                )

    records.sort(key=lambda r: r.inspected_at)
    inserted = store.record_batch(records)

    rag = ProcessDocRAG(process_docs_dir)
    engine = AlertEngine(store=store, rag=rag, consecutive_threshold=alert_threshold)
    alerts = engine.scan_all_machines(recent_only=False)

    return {
        "inserted_records": inserted,
        "machines": machines,
        "days": days,
        "alerts_triggered": len(alerts),
        "db_path": str(db_path),
    }


def main() -> None:
    cfg = load_vision_config()
    parser = argparse.ArgumentParser(description="模拟多机台多天视觉质检流水写入 SQLite 并触发主动预警")
    parser.add_argument("--db", default=str(cfg.db_path), help="SQLite 数据库路径")
    parser.add_argument("--docs", default=str(cfg.process_docs_dir), help="工艺知识文档目录")
    parser.add_argument("--neu-det", default=str(cfg.neu_det_dir), help="NEU-DET 图像目录（可选）")
    parser.add_argument("--days", type=int, default=7, help="模拟天数（默认 7 天）")
    parser.add_argument("--threshold", type=int, default=cfg.alert_consecutive_threshold, help="连续缺陷预警阈值")
    parser.add_argument("--no-reset", action="store_true", help="不清空已有记录，追加写入")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    res = simulate_inspection_stream(
        db_path=args.db,
        process_docs_dir=args.docs,
        neu_det_dir=args.neu_det,
        days=args.days,
        alert_threshold=args.threshold,
        reset_db=not args.no_reset,
        seed=args.seed,
    )
    print(
        f"[OK] 已写入 {res['inserted_records']} 条多机台检测流水（{res['days']} 天，机台：{', '.join(res['machines'])}），"
        f"触发 {res['alerts_triggered']} 条连续缺陷主动预警 -> {res['db_path']}"
    )


if __name__ == "__main__":
    main()
