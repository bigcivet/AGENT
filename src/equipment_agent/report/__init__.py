"""结构化诊断报告生成子包（支持大模型 + 规则模板自动降级）。"""

from .generator import DiagnosticReport, DiagnosticReportGenerator

__all__ = ["DiagnosticReport", "DiagnosticReportGenerator"]
