"""设备运维诊断智能体编排器（AI 老师傅）。

支持两种运行模式，统一对外暴露 `chat(user_message)` 接口：
1. **大模型 Function Calling 模式**（配置 `OPENAI_API_KEY`）：
   由大模型自主规划并多轮调用异常检测、故障分类、RUL 预测、报告生成等工具，
   综合工具输出用自然语言回复现场工人；
2. **本地规则工具调度模式**（未配置 API Key 或接口异常时自动切换）：
   解析工人提问中的意图、轴承编号与时间点，自动调用对应工具并组织清晰回答，
   支持多轮上下文记忆（记住上一轮讨论的轴承编号与时间点）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from equipment_agent.agent.tools import EquipmentToolkit
from equipment_agent.config import AppConfig, load_config


@dataclass
class ToolCallRecord:
    """记录智能体一次工具调用的入参和返回结果，便于前端可解释性展示。"""

    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentResponse:
    """智能体单轮对话响应。"""

    reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    report: dict[str, Any] | None = None
    mode: str = "rule_router"  # "llm_agent" 或 "rule_router"


class EquipmentDiagnosisAgent:
    """设备运维诊断智能体（模拟“工人提问 → Agent 调用工具诊断 → 输出结论与建议”）。"""

    SYSTEM_PROMPT = (
        "你是工厂旋转机械运维诊断智能体（AI老师傅），专门负责滚动轴承的健康监测与故障诊断。\n"
        "你配备了以下专业诊断工具：\n"
        "- list_available_bearings：查询当前可用轴承列表\n"
        "- scan_all_bearings_health：全厂轴承巡检，找出异常和高风险轴承\n"
        "- detect_anomaly：基于孤立森林对指定轴承执行无监督异常检测\n"
        "- classify_fault：基于故障分类模型识别故障类型\n"
        "- predict_rul：预测轴承剩余使用寿命（RUL）\n"
        "- generate_diagnostic_report：生成完整结构化诊断报告与处置建议\n\n"
        "工作原则：\n"
        "1. 回答设备状态问题前，必须先调用相应工具获取真实检测数据，严禁凭空编造数值。\n"
        "2. 当工人请求全面诊断、维修建议或报告时，调用 generate_diagnostic_report。\n"
        "3. 用清晰、专业、接地气的中文向现场运维工人解释结论，突出严重程度、剩余寿命和可执行处置建议。"
    )

    def __init__(
        self,
        toolkit: EquipmentToolkit,
        config: AppConfig | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.toolkit = toolkit
        self.config = config or load_config()
        self.max_tool_rounds = max_tool_rounds
        self.history: list[dict[str, str]] = []
        self.last_bearing_id: str | None = None
        self.last_file_index: int | None = None

    def reset(self) -> None:
        """清空多轮对话历史与上下文记忆。"""
        self.history.clear()
        self.last_bearing_id = None
        self.last_file_index = None

    def chat(self, user_message: str, prefer_llm: bool = True) -> AgentResponse:
        """接收现场工人自然语言提问，调度工具并返回诊断回复。"""
        text = user_message.strip()
        if not text:
            return AgentResponse(
                reply="请输入您想诊断的设备问题，例如：“帮我全面诊断 Bearing1_1 第 25 分钟的状态”或“巡检一下所有轴承”。"
            )

        if prefer_llm and self.config.has_llm_key:
            try:
                resp = self._chat_with_llm(text)
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": resp.reply})
                return resp
            except Exception as exc:  # pragma: no cover - 网络调用失败自动回退
                fallback = self._chat_with_rules(text)
                fallback.reply = (
                    f"> 注：大模型对话接口暂不可用（{exc.__class__.__name__}），已自动切换至本地工具调度模式。\n\n"
                    + fallback.reply
                )
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": fallback.reply})
                return fallback

        resp = self._chat_with_rules(text)
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": resp.reply})
        return resp

    # ------------------------------------------------------------------
    # 模式一：大模型 Function Calling 多轮工具调度
    # ------------------------------------------------------------------
    def _chat_with_llm(self, user_message: str) -> AgentResponse:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
            timeout=self.config.llm_timeout_seconds,
        )

        context_hint = ""
        if self.last_bearing_id:
            context_hint = (
                f"\n[上下文记忆：上一轮关注的轴承为 {self.last_bearing_id}"
                + (f"，时间点为第 {self.last_file_index} 分钟" if self.last_file_index else "")
                + "]"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT + context_hint},
            *self.history[-8:],
            {"role": "user", "content": user_message},
        ]

        tool_records: list[ToolCallRecord] = []
        latest_report: dict[str, Any] | None = None
        tools_schema = self.toolkit.openai_tool_schemas()

        for _ in range(self.max_tool_rounds):
            completion = client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = completion.choices[0].message

            if not msg.tool_calls:
                return AgentResponse(
                    reply=(msg.content or "").strip(),
                    tool_calls=tool_records,
                    report=latest_report,
                    mode="llm_agent",
                )

            # 记录 assistant 的工具调用请求
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            # 执行每个工具并回传给大模型
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if "bearing_id" in args and args["bearing_id"]:
                    self.last_bearing_id = str(args["bearing_id"])
                if "file_index" in args and args["file_index"] is not None:
                    self.last_file_index = int(args["file_index"])

                result = self.toolkit.execute(tool_name, args)
                tool_records.append(ToolCallRecord(tool_name=tool_name, arguments=args, result=result))
                if tool_name == "generate_diagnostic_report" and "report" in result:
                    latest_report = result["report"]

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        # 达到最大轮数兜底总结
        final = client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=0.2,
        )
        return AgentResponse(
            reply=(final.choices[0].message.content or "已完成工具调用与诊断。").strip(),
            tool_calls=tool_records,
            report=latest_report,
            mode="llm_agent",
        )

    # ------------------------------------------------------------------
    # 模式二：本地意图识别 + 工具调度回退模式（零 API Key 也能完整演示）
    # ------------------------------------------------------------------
    def _extract_bearing_and_minute(self, text: str) -> tuple[str | None, int | None]:
        # 匹配类似 Bearing1_1、bearing2_3、1_1
        bearing_id: str | None = None
        m_bearing = re.search(r"(bearing\s*\d+[_\-]\d+)", text, flags=re.IGNORECASE)
        if m_bearing:
            normalized = re.sub(r"\s+", "", m_bearing.group(1)).replace("-", "_")
            # 规范首字母大写：BearingX_Y
            bearing_id = "Bearing" + normalized[len("bearing") :]
        else:
            # 尝试匹配仓库中真实存在的任意轴承名称
            for info in self.toolkit.loader.list_bearings():
                if info.bearing_id.lower() in text.lower():
                    bearing_id = info.bearing_id
                    break

        # 匹配采样时间点（如 “第 20 分钟”、“第25个文件”、“file 18”、“20min”）
        file_index: int | None = None
        for pattern in (
            r"第\s*(\d+)\s*(?:分钟|分|个文件|号文件|点)",
            r"(?:file|minute|min|index)\s*[=:：]?\s*(\d+)",
            r"(\d+)\s*(?:分钟|min)",
        ):
            m_idx = re.search(pattern, text, flags=re.IGNORECASE)
            if m_idx:
                file_index = int(m_idx.group(1))
                break

        return bearing_id, file_index

    def _chat_with_rules(self, text: str) -> AgentResponse:
        lower = text.lower()
        tool_records: list[ToolCallRecord] = []
        latest_report: dict[str, Any] | None = None

        bearing_id, file_index = self._extract_bearing_and_minute(text)

        # 1) 询问有哪些设备/轴承
        if any(k in lower for k in ("列表", "哪些轴承", "有哪些", "多少个轴承", "可用设备", "list")):
            res = self.toolkit.execute("list_available_bearings")
            tool_records.append(ToolCallRecord("list_available_bearings", {}, res))
            lines = [f"当前数据集共有 **{res['total_bearings']}** 个轴承（涵盖工况：{', '.join(res['conditions'])}）：\n"]
            for b in res["bearings"]:
                lines.append(
                    f"- `{b['bearing_id']}`（工况 `{b['condition']}`，共 {b['total_minutes']} 分钟采样）"
                )
            lines.append("\n您可以直接向我提问，例如：**“全面诊断 Bearing1_1 第 20 分钟的状态”**。")
            return AgentResponse(reply="\n".join(lines), tool_calls=tool_records, mode="rule_router")

        # 2) 全厂/所有轴承巡检
        if any(k in lower for k in ("巡检", "所有轴承", "全部轴承", "哪些有异常", "哪些异常", "最危险", "scan")):
            res = self.toolkit.execute("scan_all_bearings_health")
            tool_records.append(ToolCallRecord("scan_all_bearings_health", {}, res))
            lines = [
                f"已完成全量轴承巡检，共检查 **{res['total_checked']}** 个轴承，其中 **{res['anomaly_count']}** 个存在异常预警（按健康度从低到高排序）：\n",
                "| 轴承 | 工况 | 抽检时间 | 状态 | 健康度 | 疑似故障 | 预测剩余寿命 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for item in res["items"]:
                flag = "⚠️ 异常" if item["is_anomaly"] else "✅ 正常"
                lines.append(
                    f"| `{item['bearing_id']}` | {item['condition']} | 第 {item['inspected_minute']} 分 | "
                    f"{flag} | {item['health_index']:.1f} | {item['fault_type']} | "
                    f"{item['rul_ratio'] * 100:.1f}%（约 {item['estimated_rul_minutes']:.1f} 分） |"
                )
            if res["items"]:
                top_risk = res["items"][0]["bearing_id"]
                self.last_bearing_id = top_risk
                lines.append(
                    f"\n建议优先关注健康度最低的 **`{top_risk}`**，您可以继续说：“帮我生成 {top_risk} 的诊断报告”。"
                )
            return AgentResponse(reply="\n".join(lines), tool_calls=tool_records, mode="rule_router")

        # 解析目标轴承（若本轮未提及则复用上下文记忆，否则默认选第一个可用轴承）
        if bearing_id:
            self.last_bearing_id = bearing_id
        elif self.last_bearing_id:
            bearing_id = self.last_bearing_id
        else:
            available = self.toolkit.loader.list_bearings()
            if not available:
                return AgentResponse(
                    reply="当前数据目录下尚未加载到任何轴承数据，请先运行模拟数据生成脚本或放入 XJTU-SY 数据集。"
                )
            bearing_id = available[0].bearing_id
            self.last_bearing_id = bearing_id

        if file_index is not None:
            self.last_file_index = file_index

        args: dict[str, Any] = {"bearing_id": bearing_id}
        if file_index is not None:
            args["file_index"] = file_index

        # 3) 判断具体诊断意图
        wants_report = any(
            k in lower
            for k in ("报告", "建议", "全面", "完整", "怎么处理", "处置", "维修", "诊断", "状态", "report")
        )
        wants_rul = any(k in lower for k in ("寿命", "rul", "还能撑多久", "还能运行多久", "还能用多久", "剩余"))
        wants_fault = any(k in lower for k in ("故障", "什么毛病", "分类", "坏在哪", "外圈", "内圈", "保持架"))
        wants_anomaly = any(k in lower for k in ("异常", "健康", "正不正常", "报警", "孤立森林"))

        # 仅询问 RUL
        if wants_rul and not (wants_report or wants_fault or wants_anomaly):
            rul_res = self.toolkit.execute("predict_rul", args)
            tool_records.append(ToolCallRecord("predict_rul", args, rul_res))
            if "error" in rul_res:
                return AgentResponse(reply=f"查询失败：{rul_res['error']}", tool_calls=tool_records)
            reply = (
                f"对 **`{rul_res['bearing_id']}`**（第 {rul_res['file_index']}/{rul_res['total_files']} 分钟）的剩余寿命（RUL）预测结果如下：\n\n"
                f"- **剩余寿命比例**：`{rul_res['rul_ratio'] * 100:.1f}%`\n"
                f"- **估算剩余运行时间**：约 **`{rul_res['estimated_rul_minutes']:.1f}` 分钟**（参考总寿命 {rul_res['reference_total_life']:.0f} 分钟）\n"
                f"- **紧急程度**：**{rul_res['urgency_level']}**\n\n"
                "如需结合异常检测与故障类型出具完整处置方案，请回复“生成诊断报告”。"
            )
            return AgentResponse(reply=reply, tool_calls=tool_records, mode="rule_router")

        # 仅询问故障类型
        if wants_fault and not (wants_report or wants_rul or wants_anomaly):
            clf_res = self.toolkit.execute("classify_fault", args)
            tool_records.append(ToolCallRecord("classify_fault", args, clf_res))
            if "error" in clf_res:
                return AgentResponse(reply=f"查询失败：{clf_res['error']}", tool_calls=tool_records)
            prob_lines = "\n".join(
                f"  - {k}：{v * 100:.1f}%"
                for k, v in sorted(clf_res["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
            )
            reply = (
                f"对 **`{clf_res['bearing_id']}`**（第 {clf_res['file_index']} 分钟）的故障分类结果：\n\n"
                f"- **识别故障类型**：**{clf_res['fault_type']}**（置信度 `{clf_res['confidence'] * 100:.1f}%`）\n"
                f"- **各类别概率分布**：\n{prob_lines}"
            )
            return AgentResponse(reply=reply, tool_calls=tool_records, mode="rule_router")

        # 仅询问异常检测
        if wants_anomaly and not (wants_report or wants_rul or wants_fault):
            anom_res = self.toolkit.execute("detect_anomaly", args)
            tool_records.append(ToolCallRecord("detect_anomaly", args, anom_res))
            if "error" in anom_res:
                return AgentResponse(reply=f"查询失败：{anom_res['error']}", tool_calls=tool_records)
            flag = "⚠️ 检测到异常" if anom_res["is_anomaly"] else "✅ 未检测到异常"
            kf = anom_res["key_features"]
            reply = (
                f"对 **`{anom_res['bearing_id']}`**（第 {anom_res['file_index']} 分钟）的孤立森林异常检测结果：\n\n"
                f"- **检测结论**：{flag}（初评：**{anom_res['severity_hint']}**）\n"
                f"- **异常分数**：`{anom_res['anomaly_score']:.4f}`（> 0 表示偏离健康基线）\n"
                f"- **健康指数**：`{anom_res['health_index']:.1f} / 100`\n"
                f"- **关键振动特征**：水平 RMS=`{kf['h_rms']}`，水平峭度=`{kf['h_kurtosis']}`，"
                f"垂直 RMS=`{kf['v_rms']}`，主频=`{kf['h_dominant_freq']}Hz`"
            )
            return AgentResponse(reply=reply, tool_calls=tool_records, mode="rule_router")

        # 默认：执行全流程工具链（异常检测 → 故障分类 → RUL 预测 → 结构化诊断报告）
        anom_res = self.toolkit.execute("detect_anomaly", args)
        tool_records.append(ToolCallRecord("detect_anomaly", args, anom_res))
        if "error" in anom_res:
            return AgentResponse(reply=f"诊断失败：{anom_res['error']}", tool_calls=tool_records)

        clf_res = self.toolkit.execute("classify_fault", args)
        tool_records.append(ToolCallRecord("classify_fault", args, clf_res))

        rul_res = self.toolkit.execute("predict_rul", args)
        tool_records.append(ToolCallRecord("predict_rul", args, rul_res))

        rep_res = self.toolkit.execute("generate_diagnostic_report", {**args, "prefer_llm": False})
        tool_records.append(ToolCallRecord("generate_diagnostic_report", args, rep_res))

        if "error" in rep_res:
            return AgentResponse(reply=f"生成报告失败：{rep_res['error']}", tool_calls=tool_records)

        latest_report = rep_res["report"]
        return AgentResponse(
            reply=rep_res["markdown"],
            tool_calls=tool_records,
            report=latest_report,
            mode="rule_router",
        )
