"""视觉质检智能体工具集（Tool Registry + OpenAI Function Calling Schemas）。

整合视觉识别、SQLite 质量记忆库统计、受限只读 SQL 查询、主动预警查询与工艺文档 RAG 检索。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_agent.config import VisionConfig, load_vision_config
from vision_agent.decision.alert_engine import AlertEngine
from vision_agent.decision.rag import ProcessDocRAG
from vision_agent.memory.store import InspectionRecord, QualityMemoryStore
from vision_agent.vision.classifier import BaseVisionDefectClassifier


class VisionToolkit:
    """视觉质检智能体可调用工具集。"""

    def __init__(
        self,
        store: QualityMemoryStore,
        rag: ProcessDocRAG,
        alert_engine: AlertEngine,
        classifier: BaseVisionDefectClassifier | None = None,
        config: VisionConfig | None = None,
    ) -> None:
        self.store = store
        self.rag = rag
        self.alert_engine = alert_engine
        self.classifier = classifier
        self.config = config or load_vision_config()

    def set_classifier(self, classifier: BaseVisionDefectClassifier | None) -> None:
        """动态切换视觉质检智能体使用的缺陷分类模型（ResNet18 / EfficientNet-B0）。"""
        self.classifier = classifier

    @staticmethod
    def openai_tool_schemas() -> list[dict[str, Any]]:
        """返回供 OpenAI 兼容大模型使用的 tools schema 列表。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_defect_summary",
                    "description": "统计近 N 天全产线或指定机台各类金属表面缺陷的数量、占比与平均置信度，回答‘最近哪种缺陷最多’等问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "统计近多少天，默认 7 天。"},
                            "machine_id": {
                                "type": "string",
                                "description": "可选机台号，如 M01、M02、M03、M04；不传则统计全产线。",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_machine_trend",
                    "description": "分析指定机台（如 M03 / 3号机台）近 N 天的缺陷变化趋势、环比增幅、主导缺陷和是否存在异常上升。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "machine_id": {"type": "string", "description": "机台号，如 M03"},
                            "days": {"type": "integer", "description": "分析天数，默认 7 天。"},
                        },
                        "required": ["machine_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_machine_comparison",
                    "description": "横向对比所有机台近 N 天的缺陷总量、高危缺陷数和主要缺陷类型。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "对比天数，默认 7 天。"}
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_quality_alerts",
                    "description": "查询当前产线连续缺陷主动预警列表（包含关联的工艺原因、处置建议和参考工艺文件）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "machine_id": {"type": "string", "description": "可选机台过滤，如 M03"},
                            "status": {
                                "type": "string",
                                "description": "预警状态过滤：open / acknowledged / resolved / all，默认 open。",
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_process_docs",
                    "description": "在本地工艺规程与缺陷处置手册（data/process_docs/）中进行 RAG 检索，查找缺陷成因、轧制工艺参数、导卫/轧辊检修措施并引用出处。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "检索关键词或工艺问题"},
                            "top_k": {"type": "integer", "description": "返回最相关片段数，默认 3"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_readonly_sql",
                    "description": (
                        "对质量数字孪生 SQLite 数据库执行受限的只读 SELECT 查询。"
                        "表 inspections 字段：id, inspected_at, machine_id, batch_id, defect_type, defect_type_cn, confidence, image_path, model_name, severity, notes；"
                        "表 alerts 字段：id, created_at, machine_id, defect_type, defect_type_cn, consecutive_count, threshold, severity, status, summary。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "合法的单条 SQLite SELECT 语句"}
                        },
                        "required": ["sql"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # 工具实现
    # ------------------------------------------------------------------
    def get_defect_summary(self, days: int = 7, machine_id: str | None = None) -> dict[str, Any]:
        return self.store.get_defect_summary(days=days, machine_id=machine_id)

    def analyze_machine_trend(self, machine_id: str, days: int = 7) -> dict[str, Any]:
        trend = self.store.analyze_machine_trend(machine_id=machine_id, days=days)
        dominant = trend.get("dominant_defect")
        if dominant and self.rag.is_ready:
            guidance = self.rag.lookup_defect_guidance(str(dominant["defect_type"]), top_k=2)
            trend["process_guidance"] = guidance
        return trend

    def get_machine_comparison(self, days: int = 7) -> dict[str, Any]:
        items = self.store.get_machine_comparison(days=days)
        return {
            "days": days,
            "machine_count": len(items),
            "machines": items,
        }

    def list_quality_alerts(
        self, machine_id: str | None = None, status: str = "open"
    ) -> dict[str, Any]:
        alerts = self.store.list_alerts(machine_id=machine_id, status=status)
        return {
            "total_alerts": len(alerts),
            "status_filter": status,
            "machine_id": machine_id or "ALL",
            "alerts": alerts,
        }

    def search_process_docs(self, query: str, top_k: int = 3) -> dict[str, Any]:
        hits = self.rag.search(query=query, top_k=top_k)
        return {
            "query": query,
            "hit_count": len(hits),
            "results": [h.to_dict() for h in hits],
        }

    def run_readonly_sql(self, sql: str) -> dict[str, Any]:
        return self.store.execute_readonly_sql(sql)

    def inspect_image_and_record(
        self,
        image_source: Any,
        machine_id: str = "M01",
        batch_id: str = "B-LIVE",
        image_path: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        """执行图像缺陷识别 → 自动写入 SQLite 记忆库 → 检查连续缺陷预警 → 返回工艺参考。"""
        if self.classifier is None:
            raise RuntimeError("视觉分类模型尚未加载，请先训练 ResNet18 模型。")

        pred = self.classifier.predict_image(image_source)
        record = InspectionRecord(
            machine_id=machine_id,
            batch_id=batch_id,
            defect_type=pred.defect_type,
            defect_type_cn=pred.defect_type_cn,
            confidence=pred.confidence,
            image_path=str(image_path),
            model_name=pred.model_name,
            severity=pred.severity,
            notes=notes,
        )
        record_id = self.store.record_inspection(record)
        alert = self.alert_engine.evaluate_machine(machine_id)
        guidance = self.rag.lookup_defect_guidance(pred.defect_type, top_k=2)

        return {
            "record_id": record_id,
            "record": record.to_dict(),
            "prediction": pred.to_dict(),
            "triggered_alert": alert.to_dict() if alert else None,
            "process_guidance": guidance,
        }

    def execute(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """按工具名统一调度执行。"""
        args = arguments or {}
        dispatch = {
            "get_defect_summary": self.get_defect_summary,
            "analyze_machine_trend": self.analyze_machine_trend,
            "get_machine_comparison": self.get_machine_comparison,
            "list_quality_alerts": self.list_quality_alerts,
            "search_process_docs": self.search_process_docs,
            "run_readonly_sql": self.run_readonly_sql,
        }
        if tool_name not in dispatch:
            return {"error": f"未知视觉质检工具：{tool_name}"}
        try:
            return dispatch[tool_name](**args)
        except Exception as exc:
            return {"error": f"工具 {tool_name} 执行失败：{exc}"}
