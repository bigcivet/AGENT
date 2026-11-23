"""视觉质检智能体对话推理编排器。

复用 `equipment_agent.agent.orchestrator` 的 `ToolCallRecord` / `AgentResponse` 数据结构，
以及 `vision_agent.llm.create_openai_client`（统一读取 `.env` 中的 OpenAI 兼容配置）：
1. 配置 `OPENAI_API_KEY` 时：启用大模型多轮 Function Calling（统计工具 + 受限只读 SQL + RAG 工艺检索）；
2. 未配置 API Key 或接口异常时：无缝回退到本地规则路由模式，准确解析中文机台号、天数、缺陷别名，
   调用工具并引用工艺文件作答。
"""

from __future__ import annotations

import json
import re
from typing import Any

from equipment_agent.agent.orchestrator import AgentResponse, ToolCallRecord
from vision_agent.agent.tools import VisionToolkit
from vision_agent.config import DEFECT_ALIAS_MAP, DEFECT_CN_MAP, VisionConfig, load_vision_config
from vision_agent.llm import create_openai_client


class VisionQualityAgent:
    """金属表面视觉质检智能体（支持自然语言查库、趋势分析、预警溯源与工艺问答）。"""

    SYSTEM_PROMPT = (
        "你是钢铁冷/热轧产线的视觉质检智能体（AI 质量工程师），管理着基于 NEU-DET 视觉模型构建的"
        "产线质量数字孪生数据库（SQLite）和本地工艺规程知识库。\n"
        "你可以调用以下工具：\n"
        "- get_defect_summary：统计近 N 天各类表面缺陷数量与占比\n"
        "- analyze_machine_trend：分析指定机台（如 M01/M02/M03/M04）近期缺陷趋势与异常漂移\n"
        "- get_machine_comparison：横向对比各机台缺陷情况\n"
        "- list_quality_alerts：查询连续缺陷主动预警列表\n"
        "- search_process_docs：从工艺文件中检索缺陷成因、轧制参数和处置规范\n"
        "- run_readonly_sql：对数据库执行受限只读 SELECT 语句处理复杂统计问题\n\n"
        "回答原则：\n"
        "1. 任何数量、占比、趋势判断都必须先调用工具查询数据库，严禁编造数据。\n"
        "2. 当发现异常趋势、某类缺陷高发或被问及工艺原因时，务必调用 search_process_docs，并在回答中明确引用工艺文档名称与章节。\n"
        "3. 用清晰、结构化的中文给出数据结论、可能工艺原因及可执行现场处置建议。"
    )

    def __init__(
        self,
        toolkit: VisionToolkit,
        config: VisionConfig | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.toolkit = toolkit
        self.config = config or load_vision_config()
        self.max_tool_rounds = max_tool_rounds
        self.history: list[dict[str, str]] = []
        self.last_machine_id: str | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_machine_id = None

    def chat(self, user_message: str, prefer_llm: bool = True) -> AgentResponse:
        """处理质检员或工艺工程师的自然语言查询。"""
        text = user_message.strip()
        if not text:
            return AgentResponse(
                reply="请输入您想查询的质量问题，例如：“最近哪种缺陷最多？”或“3 号机台有没有异常趋势？”。"
            )

        if prefer_llm and self.config.has_llm_key:
            try:
                resp = self._chat_with_llm(text)
                self.history.append({"role": "user", "content": text})
                self.history.append({"role": "assistant", "content": resp.reply})
                return resp
            except Exception as exc:  # pragma: no cover - 网络异常自动降级
                fallback = self._chat_with_rules(text)
                fallback.reply = (
                    f"> 注：大模型接口暂不可用（{exc.__class__.__name__}），已自动切换至本地规则查询模式。\n\n"
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
    # 模式一：大模型 Function Calling（含统计工具 + 只读 SQL + RAG）
    # ------------------------------------------------------------------
    def _chat_with_llm(self, user_message: str) -> AgentResponse:
        client = create_openai_client(self.config.app_config)

        context_hint = ""
        if self.last_machine_id:
            context_hint = f"\n[上下文记忆：上一轮关注的机台为 {self.last_machine_id}]"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT + context_hint},
            *self.history[-8:],
            {"role": "user", "content": user_message},
        ]

        tool_records: list[ToolCallRecord] = []
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
                    mode="llm_agent",
                )

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

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if args.get("machine_id"):
                    self.last_machine_id = str(args["machine_id"])

                result = self.toolkit.execute(tool_name, args)
                tool_records.append(ToolCallRecord(tool_name=tool_name, arguments=args, result=result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        final = client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=0.2,
        )
        return AgentResponse(
            reply=(final.choices[0].message.content or "已完成查询分析。").strip(),
            tool_calls=tool_records,
            mode="llm_agent",
        )

    # ------------------------------------------------------------------
    # 模式二：无 API Key 时的本地规则意图解析与工具调度
    # ------------------------------------------------------------------
    def _extract_machine_id(self, text: str) -> str | None:
        # 匹配 M03 / m3 / 3号机台 / 3#机台 / 三号机台
        m_code = re.search(r"\bM0?(\d+)\b", text, flags=re.IGNORECASE)
        if m_code:
            return f"M{int(m_code.group(1)):02d}"

        cn_digit_map = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6}
        m_cn = re.search(r"([1-9一二两三四五六])\s*[号#]\s*机台?", text)
        if m_cn:
            raw = m_cn.group(1)
            num = cn_digit_map.get(raw, int(raw) if raw.isdigit() else 1)
            return f"M{num:02d}"

        for known in self.toolkit.store.list_machines():
            if known.lower() in text.lower():
                return known
        return None

    @staticmethod
    def _extract_days(text: str, default: int = 7) -> int:
        if "今天" in text or "当日" in text or "24小时" in text:
            return 1
        if "昨天" in text or "近两天" in text or "最近两天" in text:
            return 2
        if "本周" in text or "一周" in text or "近一周" in text:
            return 7
        if "本月" in text or "一个月" in text or "30天" in text:
            return 30
        m = re.search(r"(?:最近|近)\s*(\d+)\s*天", text)
        if m:
            return max(1, int(m.group(1)))
        return default

    @staticmethod
    def _extract_defect(text: str) -> str | None:
        lower = text.lower()
        for alias, canonical in DEFECT_ALIAS_MAP.items():
            if alias in lower:
                return canonical
        return None

    def _chat_with_rules(self, text: str) -> AgentResponse:
        lower = text.lower()
        tool_records: list[ToolCallRecord] = []

        machine_id = self._extract_machine_id(text)
        days = self._extract_days(text, default=7)
        defect = self._extract_defect(text)

        if machine_id:
            self.last_machine_id = machine_id

        # 0) 如果用户直接输入了 SELECT 语句，走受限只读 SQL 工具
        if re.match(r"^(select|with)\b", lower):
            sql_res = self.toolkit.execute("run_readonly_sql", {"sql": text})
            tool_records.append(ToolCallRecord("run_readonly_sql", {"sql": text}, sql_res))
            if "error" in sql_res:
                return AgentResponse(reply=f"SQL 查询被拒绝或失败：{sql_res['error']}", tool_calls=tool_records)
            reply = (
                f"只读 SQL 查询完成，共返回 **{sql_res['row_count']}** 条记录：\n\n"
                f"```json\n{json.dumps(sql_res['rows'][:10], ensure_ascii=False, indent=2)}\n```"
            )
            return AgentResponse(reply=reply, tool_calls=tool_records, mode="rule_router")

        # 1) 询问预警列表 / 连续报警
        if any(k in lower for k in ("预警", "报警", "告警", "alert", "连续出现", "连续缺陷")):
            args: dict[str, Any] = {"status": "open"}
            if machine_id:
                args["machine_id"] = machine_id
            res = self.toolkit.execute("list_quality_alerts", args)
            tool_records.append(ToolCallRecord("list_quality_alerts", args, res))
            alerts = res.get("alerts", [])
            if not alerts:
                return AgentResponse(
                    reply=f"当前未发现处于活跃状态的质量预警（筛选机台：`{res['machine_id']}`）。",
                    tool_calls=tool_records,
                    mode="rule_router",
                )
            lines = [f"当前共有 **{len(alerts)}** 条活跃连续缺陷预警：\n"]
            for a in alerts[:6]:
                cites = "、".join(r.get("citation", "") for r in a.get("doc_references", [])[:2])
                lines.append(
                    f"- **#{a['id']} [{a['severity']}] 机台 `{a['machine_id']}`**："
                    f"连续 **{a['consecutive_count']}** 次出现 **{a['defect_type_cn']}（{a['defect_type']}）**\n"
                    f"  - 可能工艺原因：{a['possible_causes']}\n"
                    f"  - 处置建议：{a['recommended_actions']}\n"
                    f"  - 工艺依据：{cites or '工艺知识库'}"
                )
            return AgentResponse(reply="\n".join(lines), tool_calls=tool_records, mode="rule_router")

        # 2) 询问具体某类缺陷的工艺原因/处置方案（RAG 知识库问答）
        asks_process = any(
            k in lower for k in ("原因", "为什么", "怎么处理", "处置", "工艺", "排查", "规程", "怎么办", "标准")
        )
        if asks_process and (defect or not machine_id):
            query = f"{DEFECT_CN_MAP.get(defect, '')} {defect or ''} {text}".strip()
            rag_res = self.toolkit.execute("search_process_docs", {"query": query, "top_k": 3})
            tool_records.append(ToolCallRecord("search_process_docs", {"query": query, "top_k": 3}, rag_res))
            hits = rag_res.get("results", [])
            if not hits:
                return AgentResponse(
                    reply="工艺知识库中暂未检索到直接匹配条目，请确认 `data/process_docs/` 目录中已放置工艺文件。",
                    tool_calls=tool_records,
                    mode="rule_router",
                )
            lines = ["已为您检索本地工艺知识库，相关成因与处置规范如下：\n"]
            for idx, hit in enumerate(hits, 1):
                lines.append(f"#### 参考 {idx}：{hit['citation']}（相关度 `{hit['score']:.2f}`）\n{hit['content']}\n")
            return AgentResponse(reply="\n".join(lines), tool_calls=tool_records, mode="rule_router")

        # 3) 询问某机台趋势 / 有没有异常趋势（如“3 号机台有没有异常趋势？”）
        wants_trend = any(k in lower for k in ("趋势", "异常", "飙升", "上升", "变化", "怎么样", "情况", "波动"))
        if machine_id or (wants_trend and self.last_machine_id and "对比" not in lower and "各机台" not in lower):
            target_m = machine_id or self.last_machine_id
            assert target_m is not None
            args = {"machine_id": target_m, "days": days}
            trend = self.toolkit.execute("analyze_machine_trend", args)
            tool_records.append(ToolCallRecord("analyze_machine_trend", args, trend))

            alerts_res = self.toolkit.execute("list_quality_alerts", {"machine_id": target_m, "status": "open"})
            tool_records.append(
                ToolCallRecord("list_quality_alerts", {"machine_id": target_m, "status": "open"}, alerts_res)
            )

            if trend["total_defects"] == 0:
                return AgentResponse(
                    reply=f"机台 **`{target_m}`** 在近 {days} 天内暂无缺陷记录。",
                    tool_calls=tool_records,
                    mode="rule_router",
                )

            dom = trend["dominant_defect"]
            daily_str = " → ".join(f"{d['date'][5:]}({d['count']}次)" for d in trend["daily_counts"])
            status_badge = "⚠️ **存在异常上升趋势**" if trend["has_anomaly_trend"] else "✅ **运行总体平稳**"
            lines = [
                f"### 机台 `{target_m}` 近 {days} 天质量趋势分析\n",
                f"- **趋势判定**：{status_badge}（等级：**{trend['trend_level']}**）",
                f"- **缺陷总量**：`{trend['total_defects']}` 条（前期日均 `{trend['earlier_daily_avg']}` → 近两日日均 `{trend['recent_daily_avg']}`，环比变化 **`{trend['growth_rate']*100:+.1f}%`**）",
                f"- **主导缺陷**：**{dom['defect_type_cn']}（{dom['defect_type']}）**，共 `{dom['count']}` 次",
                f"- **每日趋势**：{daily_str}",
            ]

            streak = trend.get("latest_streak")
            if streak and streak["consecutive_count"] >= 2:
                lines.append(
                    f"- **最近连续缺陷**：连续 `{streak['consecutive_count']}` 次检出 **{streak['defect_type_cn']}**（最新批次 `{streak['latest_batch_id']}`）"
                )

            if alerts_res.get("total_alerts", 0) > 0:
                lines.append(f"- **关联活跃预警**：当前有 **{alerts_res['total_alerts']}** 条未闭环预警。")

            guidance = trend.get("process_guidance")
            if guidance:
                cites = "、".join(r["citation"] for r in guidance.get("references", [])[:2])
                lines.extend(
                    [
                        "\n#### 关联工艺分析与建议",
                        f"- **可能工艺原因**：{guidance['possible_causes']}",
                        f"- **建议处置措施**：{guidance['recommended_actions']}",
                        f"- **引用工艺文档**：{cites}",
                    ]
                )
            return AgentResponse(reply="\n".join(lines), tool_calls=tool_records, mode="rule_router")

        # 4) 各机台对比
        if any(k in lower for k in ("机台对比", "哪台机台", "哪个机台", "各机台", "最差", "排名")):
            res = self.toolkit.execute("get_machine_comparison", {"days": days})
            tool_records.append(ToolCallRecord("get_machine_comparison", {"days": days}, res))
            machines = res.get("machines", [])
            if not machines:
                return AgentResponse(reply="当前质量记忆库中暂无检测记录。", tool_calls=tool_records, mode="rule_router")
            lines = [
                f"近 **{days}** 天各机台质量横向对比（按缺陷总数降序）：\n",
                "| 机台号 | 缺陷总数 | 高危缺陷数 | 平均置信度 | 主要缺陷（次数） |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
            for m in machines:
                lines.append(
                    f"| `{m['machine_id']}` | {m['total_defects']} | {m['high_severity_count']} | "
                    f"{m['avg_confidence']*100:.1f}% | {m['top_defect_cn']}（{m['top_defect_count']}） |"
                )
            top_m = machines[0]["machine_id"]
            self.last_machine_id = top_m
            lines.append(f"\n建议重点排查缺陷最多的机台 **`{top_m}`**，您可以继续问：“{top_m} 最近有没有异常趋势？”。")
            return AgentResponse(reply="\n".join(lines), tool_calls=tool_records, mode="rule_router")

        # 5) 默认：缺陷总量统计与排行（如“最近哪种缺陷最多？”）
        sum_args: dict[str, Any] = {"days": days}
        if machine_id:
            sum_args["machine_id"] = machine_id
        summary = self.toolkit.execute("get_defect_summary", sum_args)
        tool_records.append(ToolCallRecord("get_defect_summary", sum_args, summary))

        if summary["total_records"] == 0:
            return AgentResponse(
                reply="当前质量记忆库尚无检测流水，请先上传图片检测或点击侧边栏“生成模拟多机台流水”。",
                tool_calls=tool_records,
                mode="rule_router",
            )

        top = summary["top_defect"]
        rag_res = self.toolkit.execute(
            "search_process_docs",
            {"query": f"{top['defect_type_cn']} {top['defect_type']} 工艺原因 处置措施", "top_k": 2},
        )
        tool_records.append(
            ToolCallRecord(
                "search_process_docs",
                {"query": f"{top['defect_type_cn']} {top['defect_type']} 工艺原因 处置措施", "top_k": 2},
                rag_res,
            )
        )

        lines = [
            f"近 **{days}** 天（范围：`{summary['machine_id']}`）共记录 **{summary['total_records']}** 条表面缺陷检测，"
            f"其中出现最多的是 **{top['defect_type_cn']}（`{top['defect_type']}`）**，"
            f"共 **{top['count']}** 次（占比 **`{top['ratio']*100:.1f}%`**）：\n",
            "| 缺陷类别 | 英文编码 | 检出次数 | 占比 | 平均置信度 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for row in summary["breakdown"]:
            lines.append(
                f"| {row['defect_type_cn']} | `{row['defect_type']}` | {row['count']} | "
                f"{row['ratio']*100:.1f}% | {row['avg_confidence']*100:.1f}% |"
            )

        hits = rag_res.get("results", [])
        if hits:
            cites = "、".join(h["citation"] for h in hits)
            lines.append(f"\n#### 针对高发缺陷「{top['defect_type_cn']}」的工艺参考")
            lines.append(f"- 引用工艺文档：{cites}")
            lines.append(f"- 核心片段摘要：\n{hits[0]['content'][:320].strip()}...")

        return AgentResponse(reply="\n".join(lines), tool_calls=tool_records, mode="rule_router")
