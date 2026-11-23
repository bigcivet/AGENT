"""大模型结构化诊断报告生成模块。

核心特性：
1. 优先通过 OpenAI 兼容接口（可对接 DeepSeek、通义千问、OpenAI 等）生成结构化 JSON 诊断报告；
2. 未配置 API Key 或网络/接口调用异常时，自动回退为基于工业机理规则的模板报告引擎，
   保证离线实训环境或演示环境下始终输出高质量、结构化的运维诊断报告。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from equipment_agent.config import AppConfig, load_config


# 不同故障类型的工业运维先验知识（用于规则模板生成及辅助 Prompt 约束）
FAULT_KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "正常": {
        "mechanism": "当前水平与垂直通道振动幅值及峭度均处于健康基线范围内，未观察到明显周期性冲击或高频共振抬升。",
        "actions": [
            "保持常规点检与在线监测频率，记录基线振动趋势。",
            "按润滑保养周期检查油脂状态与轴承座温升。",
        ],
        "checks": [
            "核对轴承座表面温度是否在正常范围（一般 < 65℃）。",
            "确认采样传感器磁座紧固无松动。",
        ],
    },
    "外圈故障": {
        "mechanism": "轴承外圈滚道出现点蚀、剥落或微裂纹，滚动体周期性经过承载区缺陷时诱发规律性高频冲击，表现为峭度与 3~6kHz 共振带能量抬升。",
        "actions": [
            "缩短点检间隔并开启连续振动监测，密切跟踪 RMS 与峭度增长斜率。",
            "检查润滑油膜及油脂污染情况，必要时补充或更换指定牌号润滑脂。",
            "提前申领同型号备件轴承，结合计划检修窗口安排拆检更换。",
        ],
        "checks": [
            "检查轴承座安装同轴度与外圈配合间隙，排除偏载引起的单侧剥落。",
            "用听诊器/红外测温仪复核外圈承载侧是否存在局部过热或异响。",
        ],
    },
    "内圈故障": {
        "mechanism": "轴承内圈随主轴高速旋转，缺陷周期性进出承载区，导致振动信号出现明显的幅值调制（边频带）与峰值因子增大。",
        "actions": [
            "适当降低负载或转速以减缓内圈裂纹扩展速率。",
            "建立备件预案，当剩余寿命比例低于 20% 或 RMS 骤增时立即安排计划停机更换。",
            "检查主轴轴颈配合与锁紧螺母，防止内圈跑圈加剧磨损。",
        ],
        "checks": [
            "测量主轴径向跳动与联轴器对中偏差。",
            "抽检润滑脂中是否含有金属微粒或变黑乳化现象。",
        ],
    },
    "保持架故障": {
        "mechanism": "保持架磨损、变形或铆钉松动会造成滚动体间距失衡及滑动摩擦，低频保持架通过频率附近能量上升并伴随宽带高频摩擦噪声。",
        "actions": [
            "保持架失效往往具有突发性（易引发卡死抱轴），建议尽快安排停机检查。",
            "检查润滑是否充分、油路是否堵塞，避免干摩擦导致保持架高温断裂。",
            "更换轴承时同步清理轴承室碎屑，防止二次损伤。",
        ],
        "checks": [
            "监听设备是否存在不规则的金属刮擦或咔哒异响。",
            "检查润滑油/脂中是否有铜屑或保持架材料碎末。",
        ],
    },
    "复合故障": {
        "mechanism": "轴承内圈、外圈、滚动体或保持架中的多个部位同时存在退化，振动能量在全频带显著抬升，冲击脉冲剧烈且频谱结构复杂。",
        "actions": [
            "复合故障处于加速失效阶段，建议最高优先级安排停机检修并更换轴承总成。",
            "停机前严禁超载或提速运行，加强温升与剧烈振动联锁保护。",
            "拆解后对失效件做根因分析（润滑失效、过载、安装偏心或电流腐蚀）。",
        ],
        "checks": [
            "复核设备联锁报警阈值是否生效，防止发生抱轴或断轴事故。",
            "同步检查关联齿轮、轴颈及密封件是否受到波及。",
        ],
    },
}


@dataclass
class DiagnosticReport:
    """结构化设备运维诊断报告。"""

    bearing_id: str
    condition: str
    file_index: int
    generated_at: str
    is_anomaly: bool
    anomaly_score: float
    health_index: float
    fault_type: str
    fault_confidence: float
    severity: str                    # 正常 / 轻度 / 中度 / 严重 / 危急
    rul_ratio: float                 # 0 ~ 1
    estimated_rul_minutes: float
    summary: str                     # 一句话核心结论
    signal_analysis: str             # 振动特征物理解读
    recommended_actions: list[str]   # 处置建议列表
    inspection_checklist: list[str]  # 现场排查清单
    source: str                      # "llm" 或 "rule_template"
    llm_model: str = ""
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """将结构化报告渲染为易读的 Markdown 文本。"""
        status_label = "⚠️ 检测到异常" if self.is_anomaly else "✅ 运行正常"
        source_label = (
            f"大模型生成（{self.llm_model}）"
            if self.source == "llm"
            else "规则模板引擎（本地离线模式）"
        )
        actions_md = "\n".join(f"{i}. {item}" for i, item in enumerate(self.recommended_actions, 1))
        checks_md = "\n".join(f"- [ ] {item}" for item in self.inspection_checklist)

        extra_note = f"\n> 说明：{self.fallback_reason}\n" if self.fallback_reason else ""

        return f"""## 设备运维诊断报告 — {self.bearing_id}（工况：{self.condition}）

- **诊断时间**：{self.generated_at}
- **采样时间点**：第 `{self.file_index}` 分钟
- **报告来源**：{source_label}{extra_note}

### 一、核心诊断结论
| 指标 | 诊断结果 |
| --- | --- |
| 异常状态 | {status_label}（异常分 `{self.anomaly_score:.4f}`，健康度 `{self.health_index:.1f}/100`） |
| 故障类型 | **{self.fault_type}**（置信度 `{self.fault_confidence * 100:.1f}%`） |
| 严重程度 | **{self.severity}** |
| 预测剩余寿命 (RUL) | 剩余比例 **`{self.rul_ratio * 100:.1f}%`**（约 `{self.estimated_rul_minutes:.1f}` 分钟） |

**综合判断**：{self.summary}

### 二、振动信号特征解读
{self.signal_analysis}

### 三、运维处置建议
{actions_md}

### 四、现场排查清单
{checks_md}
"""


class DiagnosticReportGenerator:
    """结构化诊断报告生成器（OpenAI 兼容接口 + 本地规则模板降级）。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

    @staticmethod
    def assess_severity(
        is_anomaly: bool,
        health_index: float,
        rul_ratio: float,
        fault_type: str,
    ) -> str:
        """综合异常状态、健康指数、RUL 与故障类型判定严重程度等级。"""
        if not is_anomaly and health_index >= 75.0 and rul_ratio >= 0.60:
            return "正常"
        if rul_ratio < 0.12 or health_index < 20.0 or (fault_type == "复合故障" and health_index < 40.0):
            return "危急"
        if rul_ratio < 0.28 or health_index < 40.0:
            return "严重"
        if rul_ratio < 0.50 or health_index < 65.0 or is_anomaly:
            return "中度"
        return "轻度"

    @staticmethod
    def _build_signal_highlights(features: dict[str, float]) -> str:
        h_rms = features.get("h_rms", 0.0)
        v_rms = features.get("v_rms", 0.0)
        h_kurt = features.get("h_kurtosis", 3.0)
        v_kurt = features.get("v_kurtosis", 3.0)
        h_crest = features.get("h_crest_factor", 0.0)
        h_dom = features.get("h_dominant_freq", 0.0)
        v_dom = features.get("v_dominant_freq", 0.0)
        h_high_ratio = (
            features.get("h_band_3_6k_ratio", 0.0) + features.get("h_band_6_12k_ratio", 0.0)
        ) * 100.0

        kurt_note = (
            "峭度显著高于高斯正常值(≈3)，存在明显冲击脉冲特征"
            if max(h_kurt, v_kurt) >= 4.5
            else "峭度接近常规水平，冲击脉冲尚不剧烈"
        )
        return (
            f"水平通道 RMS={h_rms:.4f}、峭度={h_kurt:.2f}、峰值因子={h_crest:.2f}、主频={h_dom:.1f}Hz；"
            f"垂直通道 RMS={v_rms:.4f}、峭度={v_kurt:.2f}、主频={v_dom:.1f}Hz；"
            f"水平方向 3kHz 以上高频能量占比约 {h_high_ratio:.1f}%。{kurt_note}。"
        )

    def _generate_with_rules(
        self,
        bearing_id: str,
        condition: str,
        file_index: int,
        anomaly_result: dict[str, Any],
        classification_result: dict[str, Any],
        rul_result: dict[str, Any],
        features: dict[str, float],
        fallback_reason: str = "",
    ) -> DiagnosticReport:
        is_anomaly = bool(anomaly_result["is_anomaly"])
        anomaly_score = float(anomaly_result["anomaly_score"])
        health_index = float(anomaly_result["health_index"])

        raw_fault = str(classification_result["fault_type"])
        fault_conf = float(classification_result["confidence"])
        # 若无监督异常检测正常且健康度很高，展示为正常并备注潜在趋势
        effective_fault = "正常" if (not is_anomaly and health_index >= 78.0 and raw_fault == "正常") else raw_fault

        rul_ratio = float(rul_result["rul_ratio"])
        est_minutes = float(rul_result["estimated_rul_minutes"])
        severity = self.assess_severity(is_anomaly, health_index, rul_ratio, effective_fault)

        kb = FAULT_KNOWLEDGE_BASE.get(effective_fault, FAULT_KNOWLEDGE_BASE["外圈故障"])
        signal_text = f"{self._build_signal_highlights(features)} {kb['mechanism']}"

        actions = list(kb["actions"])
        if severity == "危急":
            actions.insert(0, "【立即停机预警】设备剩余寿命已进入危急区间，建议立即安排停机检查并更换轴承，避免抱轴损坏主轴。")
        elif severity == "严重":
            actions.insert(0, "【紧急排检】建议在最近一个班次内安排停机复检，同时准备同型号备件。")

        if severity == "正常":
            summary = (
                f"轴承 {bearing_id} 在第 {file_index} 分钟运行平稳，健康度 {health_index:.1f}/100，"
                f"预计剩余寿命比例 {rul_ratio * 100:.1f}%（约 {est_minutes:.1f} 分钟），可继续正常运行。"
            )
        else:
            summary = (
                f"轴承 {bearing_id} 在第 {file_index} 分钟呈现【{severity}】级退化，"
                f"疑似【{effective_fault}】（置信度 {fault_conf * 100:.1f}%），"
                f"当前健康度 {health_index:.1f}/100，预计剩余寿命比例 {rul_ratio * 100:.1f}%（约 {est_minutes:.1f} 分钟）。"
            )

        return DiagnosticReport(
            bearing_id=bearing_id,
            condition=condition,
            file_index=file_index,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            health_index=health_index,
            fault_type=effective_fault,
            fault_confidence=fault_conf,
            severity=severity,
            rul_ratio=rul_ratio,
            estimated_rul_minutes=est_minutes,
            summary=summary,
            signal_analysis=signal_text,
            recommended_actions=actions,
            inspection_checklist=list(kb["checks"]),
            source="rule_template",
            fallback_reason=fallback_reason,
        )

    def _generate_with_llm(
        self,
        bearing_id: str,
        condition: str,
        file_index: int,
        anomaly_result: dict[str, Any],
        classification_result: dict[str, Any],
        rul_result: dict[str, Any],
        features: dict[str, float],
    ) -> DiagnosticReport:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            timeout=self.config.llm_timeout_seconds,
        )

        # 仅挑选最具代表性的时频特征传给大模型，避免上下文冗长
        key_feature_names = [
            "h_rms",
            "v_rms",
            "h_kurtosis",
            "v_kurtosis",
            "h_crest_factor",
            "v_crest_factor",
            "h_dominant_freq",
            "v_dominant_freq",
            "h_band_0_1k_ratio",
            "h_band_3_6k_ratio",
            "h_band_6_12k_ratio",
            "hv_rms_ratio",
        ]
        selected_features = {
            k: round(float(features[k]), 4) for k in key_feature_names if k in features
        }

        payload = {
            "bearing_id": bearing_id,
            "condition": condition,
            "file_index_minute": file_index,
            "anomaly_detection": anomaly_result,
            "fault_classification": classification_result,
            "rul_prediction": rul_result,
            "key_vibration_features": selected_features,
        }

        system_prompt = (
            "你是一名拥有 20 年旋转机械故障诊断经验的工业设备运维专家（AI老师傅）。"
            "请根据给定的轴承振动特征、孤立森林异常检测结果、故障分类结果和剩余寿命（RUL）预测，"
            "输出严格合法的 JSON 对象（不要输出额外 Markdown 代码块标记以外的文字），必须包含以下字段：\n"
            '- "fault_type": 字符串（如 正常/外圈故障/内圈故障/保持架故障/复合故障）\n'
            '- "severity": 字符串（仅限 正常/轻度/中度/严重/危急 之一）\n'
            '- "summary": 一句话运维核心结论（中文）\n'
            '- "signal_analysis": 结合 RMS、峭度、主频、高频带能量的物理解读（中文，2~3 句）\n'
            '- "recommended_actions": 3~4 条具体可落地的运维处置建议（中文字符串数组）\n'
            '- "inspection_checklist": 2~3 条现场点检排查清单（中文字符串数组）'
        )

        response = client.chat.completions.create(
            model=self.config.openai_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"请基于以下监测数据生成结构化运维诊断报告 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
                },
            ],
        )

        raw_content = (response.choices[0].message.content or "").strip()
        parsed = self._extract_json(raw_content)

        default_severity = self.assess_severity(
            bool(anomaly_result["is_anomaly"]),
            float(anomaly_result["health_index"]),
            float(rul_result["rul_ratio"]),
            str(classification_result["fault_type"]),
        )
        severity = str(parsed.get("severity", default_severity))
        if severity not in {"正常", "轻度", "中度", "严重", "危急"}:
            severity = default_severity

        return DiagnosticReport(
            bearing_id=bearing_id,
            condition=condition,
            file_index=file_index,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_anomaly=bool(anomaly_result["is_anomaly"]),
            anomaly_score=float(anomaly_result["anomaly_score"]),
            health_index=float(anomaly_result["health_index"]),
            fault_type=str(parsed.get("fault_type", classification_result["fault_type"])),
            fault_confidence=float(classification_result["confidence"]),
            severity=severity,
            rul_ratio=float(rul_result["rul_ratio"]),
            estimated_rul_minutes=float(rul_result["estimated_rul_minutes"]),
            summary=str(parsed.get("summary", "")),
            signal_analysis=str(parsed.get("signal_analysis", "")),
            recommended_actions=[str(x) for x in parsed.get("recommended_actions", [])],
            inspection_checklist=[str(x) for x in parsed.get("inspection_checklist", [])],
            source="llm",
            llm_model=self.config.openai_model,
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """兼容提取纯 JSON 或 ```json ... ``` 包裹的内容。"""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            candidate = candidate[start : end + 1]
        return json.loads(candidate)

    def generate(
        self,
        bearing_id: str,
        condition: str,
        file_index: int,
        anomaly_result: dict[str, Any],
        classification_result: dict[str, Any],
        rul_result: dict[str, Any],
        features: dict[str, float],
        prefer_llm: bool = True,
    ) -> DiagnosticReport:
        """生成结构化诊断报告：优先大模型，未配 Key 或失败时自动降级为规则模板。"""
        if prefer_llm and self.config.has_llm_key:
            try:
                return self._generate_with_llm(
                    bearing_id=bearing_id,
                    condition=condition,
                    file_index=file_index,
                    anomaly_result=anomaly_result,
                    classification_result=classification_result,
                    rul_result=rul_result,
                    features=features,
                )
            except Exception as exc:  # pragma: no cover - 网络调用异常兜底
                return self._generate_with_rules(
                    bearing_id=bearing_id,
                    condition=condition,
                    file_index=file_index,
                    anomaly_result=anomaly_result,
                    classification_result=classification_result,
                    rul_result=rul_result,
                    features=features,
                    fallback_reason=f"大模型接口调用失败（{exc.__class__.__name__}），已自动切换为规则模板报告。",
                )

        reason = (
            "未配置 OPENAI_API_KEY，已自动使用规则模板引擎生成结构化报告。"
            if prefer_llm
            else ""
        )
        return self._generate_with_rules(
            bearing_id=bearing_id,
            condition=condition,
            file_index=file_index,
            anomaly_result=anomaly_result,
            classification_result=classification_result,
            rul_result=rul_result,
            features=features,
            fallback_reason=reason,
        )
