"""连续缺陷主动预警与工艺溯源决策引擎。

核心规则：
- 监控每台机台的检测流水；当同一机台连续出现同一类缺陷达到阈值（默认 3 次，可配置）时，
  自动分级触发预警（警告 / 严重 / 紧急）；
- 自动通过 ProcessDocRAG 检索 `data/process_docs/` 中的工艺规程与缺陷处置手册，
  关联可能工艺原因、现场处置建议及文档引用出处，并持久化写入 SQLite 预警表。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vision_agent.config import defect_to_cn
from vision_agent.decision.rag import ProcessDocRAG
from vision_agent.memory.store import QualityMemoryStore


@dataclass
class QualityAlert:
    """结构化主动质量预警。"""

    alert_id: int
    machine_id: str
    defect_type: str
    defect_type_cn: str
    consecutive_count: int
    threshold: int
    severity: str
    summary: str
    possible_causes: str
    recommended_actions: str
    doc_references: list[dict[str, Any]]
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertEngine:
    """连续缺陷主动预警引擎。"""

    def __init__(
        self,
        store: QualityMemoryStore,
        rag: ProcessDocRAG,
        consecutive_threshold: int = 3,
    ) -> None:
        self.store = store
        self.rag = rag
        self.consecutive_threshold = max(2, int(consecutive_threshold))

    def classify_alert_severity(self, consecutive_count: int) -> str:
        """按连续出现次数与配置阈值评定预警等级（对齐 PROC-EM-004）。"""
        if consecutive_count >= self.consecutive_threshold + 4:
            return "紧急"
        if consecutive_count >= self.consecutive_threshold + 2:
            return "严重"
        return "警告"

    def _create_alert_from_run(
        self,
        machine_id: str,
        defect_type: str,
        consecutive_count: int,
        last_inspection_id: int,
        created_at: str | None = None,
    ) -> QualityAlert:
        cn_name = defect_to_cn(defect_type)
        severity = self.classify_alert_severity(consecutive_count)
        guidance = self.rag.lookup_defect_guidance(defect_type, top_k=3)

        citations = [ref["citation"] for ref in guidance["references"]]
        cite_str = "、".join(citations[:2]) if citations else "工艺知识库"
        summary = (
            f"【{severity}预警】机台 {machine_id} 连续检出 {consecutive_count} 次「{cn_name}（{defect_type}）」"
            f"（阈值={self.consecutive_threshold}），请立即排查工艺异常（依据：{cite_str}）。"
        )

        alert_id = self.store.save_alert(
            machine_id=machine_id,
            defect_type=defect_type,
            consecutive_count=consecutive_count,
            threshold=self.consecutive_threshold,
            severity=severity,
            summary=summary,
            possible_causes=guidance["possible_causes"],
            recommended_actions=guidance["recommended_actions"],
            doc_references=guidance["references"],
            last_inspection_id=last_inspection_id,
            created_at=created_at,
        )
        return QualityAlert(
            alert_id=alert_id,
            machine_id=machine_id,
            defect_type=defect_type,
            defect_type_cn=cn_name,
            consecutive_count=consecutive_count,
            threshold=self.consecutive_threshold,
            severity=severity,
            summary=summary,
            possible_causes=guidance["possible_causes"],
            recommended_actions=guidance["recommended_actions"],
            doc_references=guidance["references"],
            created_at=created_at or "",
        )

    def evaluate_machine(self, machine_id: str) -> QualityAlert | None:
        """实时评估某机台最新一次检测后是否触发连续缺陷预警。"""
        streak = self.store.get_recent_streak(machine_id)
        if not streak:
            return None
        if int(streak["consecutive_count"]) < self.consecutive_threshold:
            return None

        return self._create_alert_from_run(
            machine_id=machine_id,
            defect_type=str(streak["defect_type"]),
            consecutive_count=int(streak["consecutive_count"]),
            last_inspection_id=int(streak["latest_inspection_id"]),
            created_at=str(streak["latest_inspected_at"]),
        )

    def scan_all_machines(self, recent_only: bool = False) -> list[QualityAlert]:
        """扫描全库所有机台的检测流水，发现并生成所有达到阈值的连续缺陷预警。

        Args:
            recent_only: True 时仅检查每台机台当前末尾连续段；False 时扫描历史全部连续段。
        """
        triggered: list[QualityAlert] = []
        for machine_id in self.store.list_machines():
            if recent_only:
                alert = self.evaluate_machine(machine_id)
                if alert is not None:
                    triggered.append(alert)
                continue

            seq = self.store.get_machine_sequence(machine_id)
            if not seq:
                continue

            run_defect = str(seq[0]["defect_type"])
            run_count = 1
            run_last = seq[0]

            for item in seq[1:]:
                if str(item["defect_type"]) == run_defect:
                    run_count += 1
                    run_last = item
                else:
                    if run_count >= self.consecutive_threshold:
                        triggered.append(
                            self._create_alert_from_run(
                                machine_id=machine_id,
                                defect_type=run_defect,
                                consecutive_count=run_count,
                                last_inspection_id=int(run_last["id"]),
                                created_at=str(run_last["inspected_at"]),
                            )
                        )
                    run_defect = str(item["defect_type"])
                    run_count = 1
                    run_last = item

            if run_count >= self.consecutive_threshold:
                triggered.append(
                    self._create_alert_from_run(
                        machine_id=machine_id,
                        defect_type=run_defect,
                        consecutive_count=run_count,
                        last_inspection_id=int(run_last["id"]),
                        created_at=str(run_last["inspected_at"]),
                    )
                )

        return triggered
