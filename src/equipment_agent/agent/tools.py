"""设备运维诊断智能体可调用的工具集（Tool Registry + OpenAI Function Schemas）。

将底层数据读取、时频特征提取、异常检测、故障分类、RUL 预测与报告生成
封装为标准工具，供大模型 Function Calling 或本地规则路由器统一调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equipment_agent.config import AppConfig, load_config
from equipment_agent.data.loader import XJTUSYDataLoader, read_vibration_csv
from equipment_agent.features.extractor import (
    compute_fft_spectrum,
    extract_dual_channel_features,
    extract_sample_features,
)
from equipment_agent.models.trainer import ModelBundle
from equipment_agent.report.generator import DiagnosticReport, DiagnosticReportGenerator


class EquipmentToolkit:
    """设备运维诊断工具集。"""

    def __init__(
        self,
        loader: XJTUSYDataLoader,
        bundle: ModelBundle,
        report_generator: DiagnosticReportGenerator | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.config = config or load_config()
        self.loader = loader
        self.bundle = bundle
        self.report_generator = report_generator or DiagnosticReportGenerator(self.config)

    # ------------------------------------------------------------------
    # OpenAI Function Calling 工具元数据定义
    # ------------------------------------------------------------------
    @staticmethod
    def openai_tool_schemas() -> list[dict[str, Any]]:
        """返回供 OpenAI 兼容接口使用的 tools schema 列表。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_available_bearings",
                    "description": "查看当前数据目录下所有可用的工况、轴承编号及总采样分钟数。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_anomaly",
                    "description": "使用孤立森林（Isolation Forest）对指定轴承的某一时间点（分钟）进行无监督异常检测。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bearing_id": {"type": "string", "description": "轴承编号，如 Bearing1_1"},
                            "file_index": {
                                "type": "integer",
                                "description": "采样文件序号（1-based 分钟数），不传则默认检测最后一个时间点。",
                            },
                        },
                        "required": ["bearing_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "classify_fault",
                    "description": "调用故障分类模型识别指定轴承当前时间点的故障类型及置信度。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bearing_id": {"type": "string", "description": "轴承编号，如 Bearing1_1"},
                            "file_index": {"type": "integer", "description": "采样文件序号（分钟数），可选。"},
                        },
                        "required": ["bearing_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "predict_rul",
                    "description": "预测指定轴承当前时间点的剩余使用寿命（RUL 比例与估算剩余分钟数）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bearing_id": {"type": "string", "description": "轴承编号，如 Bearing1_1"},
                            "file_index": {"type": "integer", "description": "采样文件序号（分钟数），可选。"},
                        },
                        "required": ["bearing_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_diagnostic_report",
                    "description": "综合异常检测、故障分类、RUL 预测与时频特征，生成结构化运维诊断报告与处置建议。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bearing_id": {"type": "string", "description": "轴承编号，如 Bearing1_1"},
                            "file_index": {"type": "integer", "description": "采样文件序号（分钟数），可选。"},
                        },
                        "required": ["bearing_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scan_all_bearings_health",
                    "description": "对当前目录下所有轴承的最新时间点执行巡检扫描，快速定位存在异常或高风险的轴承。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    def _resolve_file_index(self, bearing_id: str, file_index: int | None) -> int:
        info = self.loader.get_bearing(bearing_id)
        if file_index is None:
            return info.total_life_minutes
        return max(1, min(int(file_index), info.total_life_minutes))

    def _load_and_extract(
        self, bearing_id: str, file_index: int | None = None
    ) -> tuple[Any, dict[str, float]]:
        idx = self._resolve_file_index(bearing_id, file_index)
        sample = self.loader.load_file(bearing_id, idx)
        feats = extract_dual_channel_features(
            sample.horizontal, sample.vertical, sampling_rate=sample.sampling_rate
        )
        return sample, feats

    # ------------------------------------------------------------------
    # 具体工具实现
    # ------------------------------------------------------------------
    def list_available_bearings(self) -> dict[str, Any]:
        """列出全部可用轴承及其工况与采样分钟数。"""
        bearings = self.loader.list_bearings()
        return {
            "total_bearings": len(bearings),
            "conditions": self.loader.list_conditions(),
            "bearings": [
                {
                    "bearing_id": b.bearing_id,
                    "condition": b.condition,
                    "total_minutes": b.total_life_minutes,
                    "reference_fault_label": b.fault_type,
                }
                for b in bearings
            ],
        }

    def detect_anomaly(self, bearing_id: str, file_index: int | None = None) -> dict[str, Any]:
        """对指定轴承的时间点执行无监督异常检测。"""
        sample, feats = self._load_and_extract(bearing_id, file_index)
        res = self.bundle.anomaly_detector.detect_one(feats)
        return {
            "bearing_id": sample.bearing_id,
            "condition": sample.condition,
            "file_index": sample.file_index,
            "total_files": sample.total_files,
            **res.to_dict(),
            "key_features": {
                "h_rms": round(feats["h_rms"], 4),
                "v_rms": round(feats["v_rms"], 4),
                "h_kurtosis": round(feats["h_kurtosis"], 3),
                "v_kurtosis": round(feats["v_kurtosis"], 3),
                "h_crest_factor": round(feats["h_crest_factor"], 3),
                "h_dominant_freq": round(feats["h_dominant_freq"], 1),
            },
        }

    def classify_fault(self, bearing_id: str, file_index: int | None = None) -> dict[str, Any]:
        """对指定轴承的时间点执行故障类型分类。"""
        sample, feats = self._load_and_extract(bearing_id, file_index)
        res = self.bundle.fault_classifier.classify_one(feats)
        return {
            "bearing_id": sample.bearing_id,
            "condition": sample.condition,
            "file_index": sample.file_index,
            **res.to_dict(),
        }

    def predict_rul(self, bearing_id: str, file_index: int | None = None) -> dict[str, Any]:
        """对指定轴承的时间点预测剩余使用寿命（RUL）。"""
        sample, feats = self._load_and_extract(bearing_id, file_index)
        res = self.bundle.rul_predictor.predict_one(
            feats,
            condition=sample.condition,
            reference_total_life=float(sample.total_files),
        )
        return {
            "bearing_id": sample.bearing_id,
            "condition": sample.condition,
            "file_index": sample.file_index,
            "total_files": sample.total_files,
            **res.to_dict(),
        }

    def generate_diagnostic_report(
        self,
        bearing_id: str,
        file_index: int | None = None,
        prefer_llm: bool = True,
    ) -> dict[str, Any]:
        """一键串联检测、分类、RUL 与结构化报告生成。"""
        sample, feats = self._load_and_extract(bearing_id, file_index)
        anom = self.bundle.anomaly_detector.detect_one(feats).to_dict()
        clf = self.bundle.fault_classifier.classify_one(feats).to_dict()
        rul = self.bundle.rul_predictor.predict_one(
            feats,
            condition=sample.condition,
            reference_total_life=float(sample.total_files),
        ).to_dict()

        report: DiagnosticReport = self.report_generator.generate(
            bearing_id=sample.bearing_id,
            condition=sample.condition,
            file_index=sample.file_index,
            anomaly_result=anom,
            classification_result=clf,
            rul_result=rul,
            features=feats,
            prefer_llm=prefer_llm,
        )
        return {
            "report": report.to_dict(),
            "markdown": report.to_markdown(),
            "anomaly": anom,
            "classification": clf,
            "rul": rul,
        }

    def diagnose_csv_file(
        self,
        csv_path: str | Path,
        bearing_label: str = "上传设备",
        condition: str = "自定义工况",
        prefer_llm: bool = True,
    ) -> dict[str, Any]:
        """对用户上传的单个双通道振动 CSV 文件直接执行完整诊断。"""
        h_sig, v_sig = read_vibration_csv(csv_path)
        feats = extract_dual_channel_features(
            h_sig, v_sig, sampling_rate=self.config.sampling_rate
        )
        anom = self.bundle.anomaly_detector.detect_one(feats).to_dict()
        clf = self.bundle.fault_classifier.classify_one(feats).to_dict()
        rul = self.bundle.rul_predictor.predict_one(feats, condition=condition).to_dict()
        report = self.report_generator.generate(
            bearing_id=bearing_label,
            condition=condition,
            file_index=1,
            anomaly_result=anom,
            classification_result=clf,
            rul_result=rul,
            features=feats,
            prefer_llm=prefer_llm,
        )
        return {
            "report": report.to_dict(),
            "markdown": report.to_markdown(),
            "anomaly": anom,
            "classification": clf,
            "rul": rul,
            "features": feats,
            "horizontal": h_sig,
            "vertical": v_sig,
        }

    def scan_all_bearings_health(self) -> dict[str, Any]:
        """巡检全部轴承的最近一个采样点（默认 80% 寿命点以展示运行中预警状态）。"""
        rows: list[dict[str, Any]] = []
        for info in self.loader.list_bearings():
            # 取倒数第 2 个或最后一个采样点模拟当前在线状态
            idx = max(1, int(round(info.total_life_minutes * 0.85)))
            sample, feats = self._load_and_extract(info.bearing_id, idx)
            anom = self.bundle.anomaly_detector.detect_one(feats)
            clf = self.bundle.fault_classifier.classify_one(feats)
            rul = self.bundle.rul_predictor.predict_one(
                feats, condition=info.condition, reference_total_life=float(info.total_life_minutes)
            )
            rows.append(
                {
                    "bearing_id": info.bearing_id,
                    "condition": info.condition,
                    "inspected_minute": sample.file_index,
                    "total_minutes": info.total_life_minutes,
                    "is_anomaly": anom.is_anomaly,
                    "health_index": anom.health_index,
                    "severity_hint": anom.severity_hint,
                    "fault_type": clf.fault_type,
                    "fault_confidence": clf.confidence,
                    "rul_ratio": rul.rul_ratio,
                    "estimated_rul_minutes": rul.estimated_rul_minutes,
                }
            )
        rows.sort(key=lambda r: r["health_index"])
        anomaly_count = sum(1 for r in rows if r["is_anomaly"])
        return {
            "total_checked": len(rows),
            "anomaly_count": anomaly_count,
            "items": rows,
        }

    def monitor_bearing_trend(
        self, bearing_id: str, step: int = 1
    ) -> pd.DataFrame:
        """逐文件流式回放指定轴承全生命周期的异常分数、RMS、峭度与 RUL 曲线。"""
        records: list[dict[str, Any]] = []
        for sample in self.loader.iter_bearing_stream(bearing_id, step=step):
            rec = extract_sample_features(sample)
            anom = self.bundle.anomaly_detector.detect_one(rec)
            clf = self.bundle.fault_classifier.classify_one(rec)
            rul = self.bundle.rul_predictor.predict_one(
                rec,
                condition=sample.condition,
                reference_total_life=float(sample.total_files),
            )
            records.append(
                {
                    "file_index": sample.file_index,
                    "life_ratio": round(sample.life_ratio, 4),
                    "h_rms": round(float(rec["h_rms"]), 4),
                    "v_rms": round(float(rec["v_rms"]), 4),
                    "h_kurtosis": round(float(rec["h_kurtosis"]), 3),
                    "v_kurtosis": round(float(rec["v_kurtosis"]), 3),
                    "is_anomaly": anom.is_anomaly,
                    "anomaly_score": anom.anomaly_score,
                    "health_index": anom.health_index,
                    "pred_fault_type": clf.fault_type,
                    "pred_rul_ratio": rul.rul_ratio,
                    "true_rul_ratio": round(sample.rul_normalized, 4),
                    "pred_rul_minutes": rul.estimated_rul_minutes,
                    "true_rul_minutes": round(sample.rul_minutes, 1),
                }
            )
        return pd.DataFrame(records)

    def get_waveform_and_spectrum(
        self, bearing_id: str, file_index: int
    ) -> dict[str, Any]:
        """获取单文件的时域波形与 FFT 频谱，用于可视化展示。"""
        sample = self.loader.load_file(bearing_id, file_index)
        h_freqs, h_amps = compute_fft_spectrum(sample.horizontal, sample.sampling_rate)
        v_freqs, v_amps = compute_fft_spectrum(sample.vertical, sample.sampling_rate)
        t = np.arange(len(sample.horizontal)) / float(sample.sampling_rate)
        return {
            "sample": sample,
            "time_axis": t,
            "h_freqs": h_freqs,
            "h_amps": h_amps,
            "v_freqs": v_freqs,
            "v_amps": v_amps,
        }

    def execute(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """按工具名动态执行工具调用。"""
        args = arguments or {}
        dispatch = {
            "list_available_bearings": self.list_available_bearings,
            "detect_anomaly": self.detect_anomaly,
            "classify_fault": self.classify_fault,
            "predict_rul": self.predict_rul,
            "generate_diagnostic_report": self.generate_diagnostic_report,
            "scan_all_bearings_health": self.scan_all_bearings_health,
        }
        if tool_name not in dispatch:
            return {"error": f"未知工具名称：{tool_name}"}
        try:
            return dispatch[tool_name](**args)
        except Exception as exc:
            return {"error": f"工具 {tool_name} 执行失败：{exc}"}
