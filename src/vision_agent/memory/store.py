"""产线质量“数字孪生”记忆模块（基于 SQLite）。

职责：
1. 每次视觉检测结果自动入库（时间、机台号、批次、缺陷类型、置信度、图片路径等）；
2. 持久化主动预警记录与处理状态；
3. 提供结构化统计分析接口（缺陷排行、按天/机台趋势、机台异常检测、连续同类缺陷计算）；
4. 提供严格受限的只读 SQL 执行器，支持 LLM 按需生成安全的 SELECT 查询。
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from vision_agent.config import (
    DEFECT_CN_MAP,
    DEFECT_SEVERITY_MAP,
    defect_to_cn,
    normalize_defect_name,
)

# 严禁在只读 SQL 中出现的危险关键字（防写入与防注入）
_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    flags=re.IGNORECASE,
)


@dataclass
class InspectionRecord:
    """单条视觉质检流水记录。"""

    machine_id: str
    batch_id: str
    defect_type: str
    confidence: float
    image_path: str = ""
    inspected_at: str = ""
    defect_type_cn: str = ""
    model_name: str = "resnet18"
    severity: str = ""
    bboxes: list[dict[str, Any]] | None = None
    notes: str = ""
    record_id: int | None = None

    def __post_init__(self) -> None:
        self.defect_type = normalize_defect_name(self.defect_type)
        if not self.defect_type_cn:
            self.defect_type_cn = defect_to_cn(self.defect_type)
        if not self.inspected_at:
            self.inspected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.severity:
            self.severity = DEFECT_SEVERITY_MAP.get(self.defect_type, "中")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QualityMemoryStore:
    """SQLite 产线质量记忆库。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspected_at TEXT NOT NULL,
                    machine_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    defect_type_cn TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT DEFAULT '',
                    model_name TEXT DEFAULT 'resnet18',
                    severity TEXT DEFAULT '中',
                    bbox_json TEXT DEFAULT '[]',
                    notes TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_inspections_time ON inspections(inspected_at);
                CREATE INDEX IF NOT EXISTS idx_inspections_machine ON inspections(machine_id, inspected_at);
                CREATE INDEX IF NOT EXISTS idx_inspections_defect ON inspections(defect_type);

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    machine_id TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    defect_type_cn TEXT NOT NULL,
                    consecutive_count INTEGER NOT NULL,
                    threshold INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    summary TEXT NOT NULL,
                    possible_causes TEXT NOT NULL,
                    recommended_actions TEXT NOT NULL,
                    doc_references_json TEXT NOT NULL DEFAULT '[]',
                    last_inspection_id INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_alerts_machine ON alerts(machine_id, defect_type);
                """
            )

    def clear_all(self) -> None:
        """清空所有检测流水与预警记录（用于重新模拟演示）。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM alerts")
            conn.execute("DELETE FROM inspections")

    # ------------------------------------------------------------------
    # 写入接口
    # ------------------------------------------------------------------
    def record_inspection(self, record: InspectionRecord) -> int:
        """写入一条视觉检测结果，返回自增主键 ID。"""
        bbox_json = json.dumps(record.bboxes or [], ensure_ascii=False)
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO inspections (
                    inspected_at, machine_id, batch_id, defect_type, defect_type_cn,
                    confidence, image_path, model_name, severity, bbox_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.inspected_at,
                    record.machine_id,
                    record.batch_id,
                    record.defect_type,
                    record.defect_type_cn,
                    float(record.confidence),
                    record.image_path,
                    record.model_name,
                    record.severity,
                    bbox_json,
                    record.notes,
                ),
            )
            record_id = int(cur.lastrowid)
            record.record_id = record_id
            return record_id

    def record_batch(self, records: list[InspectionRecord]) -> int:
        """批量写入检测流水（用于模拟多机台多天数据）。"""
        if not records:
            return 0
        rows = [
            (
                r.inspected_at,
                r.machine_id,
                r.batch_id,
                r.defect_type,
                r.defect_type_cn,
                float(r.confidence),
                r.image_path,
                r.model_name,
                r.severity,
                json.dumps(r.bboxes or [], ensure_ascii=False),
                r.notes,
            )
            for r in records
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO inspections (
                    inspected_at, machine_id, batch_id, defect_type, defect_type_cn,
                    confidence, image_path, model_name, severity, bbox_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    # ------------------------------------------------------------------
    # 基础查询与统计
    # ------------------------------------------------------------------
    def count_inspections(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM inspections").fetchone()
            return int(row["cnt"]) if row else 0

    def list_machines(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT machine_id FROM inspections ORDER BY machine_id"
            ).fetchall()
            return [str(r["machine_id"]) for r in rows]

    def get_latest_timestamp(self) -> datetime:
        """获取库内最新一条记录的时间戳；若库为空则返回当前时间。"""
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(inspected_at) AS max_ts FROM inspections").fetchone()
        if row and row["max_ts"]:
            try:
                return datetime.strptime(str(row["max_ts"])[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        return datetime.now()

    def _window_start(self, days: int) -> str:
        ref_dt = self.get_latest_timestamp()
        start_dt = ref_dt - timedelta(days=max(1, int(days)))
        return start_dt.strftime("%Y-%m-%d %H:%M:%S")

    def query_recent_inspections(
        self,
        limit: int = 100,
        machine_id: str | None = None,
        defect_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """按时间倒序查询最近的质检流水记录。"""
        clauses: list[str] = []
        params: list[Any] = []
        if machine_id:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        if defect_type:
            clauses.append("defect_type = ?")
            params.append(normalize_defect_name(defect_type))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT id, inspected_at, machine_id, batch_id, defect_type, defect_type_cn,
                   confidence, image_path, model_name, severity, notes
            FROM inspections
            {where_sql}
            ORDER BY inspected_at DESC, id DESC
            LIMIT ?
        """
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_defect_summary(
        self,
        days: int = 7,
        machine_id: str | None = None,
    ) -> dict[str, Any]:
        """统计近 N 天各类缺陷检出次数、占比与平均置信度。"""
        since = self._window_start(days)
        clauses = ["inspected_at >= ?"]
        params: list[Any] = [since]
        if machine_id:
            clauses.append("machine_id = ?")
            params.append(machine_id)

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT defect_type,
                   defect_type_cn,
                   COUNT(*) AS count,
                   ROUND(AVG(confidence), 4) AS avg_confidence
            FROM inspections
            WHERE {where_sql}
            GROUP BY defect_type, defect_type_cn
            ORDER BY count DESC, defect_type ASC
        """
        with self._connect() as conn:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

        total = sum(int(r["count"]) for r in rows)
        for r in rows:
            r["ratio"] = round(int(r["count"]) / total, 4) if total else 0.0

        top_defect = rows[0] if rows else None
        return {
            "days": days,
            "since": since,
            "machine_id": machine_id or "ALL",
            "total_records": total,
            "top_defect": top_defect,
            "breakdown": rows,
        }

    def get_daily_trend(
        self,
        days: int = 7,
        machine_id: str | None = None,
        defect_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """按日期统计缺陷检出趋势（可选过滤机台与缺陷类型）。"""
        since = self._window_start(days)
        clauses = ["inspected_at >= ?"]
        params: list[Any] = [since]
        if machine_id:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        if defect_type:
            clauses.append("defect_type = ?")
            params.append(normalize_defect_name(defect_type))

        sql = f"""
            SELECT substr(inspected_at, 1, 10) AS date,
                   machine_id,
                   defect_type,
                   defect_type_cn,
                   COUNT(*) AS count
            FROM inspections
            WHERE {' AND '.join(clauses)}
            GROUP BY date, machine_id, defect_type, defect_type_cn
            ORDER BY date ASC, machine_id ASC
        """
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_machine_comparison(self, days: int = 7) -> list[dict[str, Any]]:
        """横向对比各机台近 N 天的缺陷总量、高危缺陷数与首要缺陷。"""
        since = self._window_start(days)
        with self._connect() as conn:
            machines = conn.execute(
                """
                SELECT machine_id,
                       COUNT(*) AS total_defects,
                       SUM(CASE WHEN severity = '高' THEN 1 ELSE 0 END) AS high_severity_count,
                       ROUND(AVG(confidence), 4) AS avg_confidence
                FROM inspections
                WHERE inspected_at >= ?
                GROUP BY machine_id
                ORDER BY total_defects DESC, machine_id ASC
                """,
                (since,),
            ).fetchall()

            results: list[dict[str, Any]] = []
            for m in machines:
                mid = str(m["machine_id"])
                top_row = conn.execute(
                    """
                    SELECT defect_type, defect_type_cn, COUNT(*) AS cnt
                    FROM inspections
                    WHERE inspected_at >= ? AND machine_id = ?
                    GROUP BY defect_type, defect_type_cn
                    ORDER BY cnt DESC
                    LIMIT 1
                    """,
                    (since, mid),
                ).fetchone()
                item = dict(m)
                item["top_defect_type"] = str(top_row["defect_type"]) if top_row else ""
                item["top_defect_cn"] = str(top_row["defect_type_cn"]) if top_row else ""
                item["top_defect_count"] = int(top_row["cnt"]) if top_row else 0
                results.append(item)
            return results

    def analyze_machine_trend(self, machine_id: str, days: int = 7) -> dict[str, Any]:
        """分析指定机台近 N 天的缺陷数量变化、环比增幅及异常趋势判定。"""
        since = self._window_start(days)
        with self._connect() as conn:
            daily_rows = conn.execute(
                """
                SELECT substr(inspected_at, 1, 10) AS date, COUNT(*) AS count
                FROM inspections
                WHERE inspected_at >= ? AND machine_id = ?
                GROUP BY date
                ORDER BY date ASC
                """,
                (since, machine_id),
            ).fetchall()

            defect_rows = conn.execute(
                """
                SELECT defect_type, defect_type_cn, COUNT(*) AS count
                FROM inspections
                WHERE inspected_at >= ? AND machine_id = ?
                GROUP BY defect_type, defect_type_cn
                ORDER BY count DESC
                """,
                (since, machine_id),
            ).fetchall()

        daily = [{"date": str(r["date"]), "count": int(r["count"])} for r in daily_rows]
        defects = [dict(r) for r in defect_rows]
        total = sum(d["count"] for d in daily)

        if len(daily) >= 3:
            recent_window = daily[-2:]
            earlier_window = daily[:-2]
            recent_avg = sum(d["count"] for d in recent_window) / len(recent_window)
            earlier_avg = sum(d["count"] for d in earlier_window) / max(1, len(earlier_window))
            growth_rate = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0.0
            monotonic_up = (
                len(daily) >= 3 and daily[-1]["count"] > daily[-2]["count"] > daily[-3]["count"]
            )
        elif len(daily) == 2:
            earlier_avg = float(daily[0]["count"])
            recent_avg = float(daily[1]["count"])
            growth_rate = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0.0
            monotonic_up = recent_avg > earlier_avg
        else:
            earlier_avg = float(total)
            recent_avg = float(total)
            growth_rate = 0.0
            monotonic_up = False

        streak = self.get_recent_streak(machine_id)
        has_anomaly_trend = (
            growth_rate >= 0.50
            or monotonic_up
            or (streak is not None and streak["consecutive_count"] >= 3)
        )

        if growth_rate >= 0.8 or (streak and streak["consecutive_count"] >= 5):
            trend_level = "显著异常上升"
        elif has_anomaly_trend:
            trend_level = "异常上升预警"
        elif growth_rate <= -0.25:
            trend_level = "持续改善下降"
        else:
            trend_level = "总体平稳"

        return {
            "machine_id": machine_id,
            "days": days,
            "total_defects": total,
            "earlier_daily_avg": round(earlier_avg, 2),
            "recent_daily_avg": round(recent_avg, 2),
            "growth_rate": round(growth_rate, 4),
            "monotonic_increasing": monotonic_up,
            "has_anomaly_trend": has_anomaly_trend,
            "trend_level": trend_level,
            "dominant_defect": defects[0] if defects else None,
            "defect_breakdown": defects,
            "daily_counts": daily,
            "latest_streak": streak,
        }

    def get_recent_streak(self, machine_id: str, limit: int = 20) -> dict[str, Any] | None:
        """获取指定机台最近一次连续相同缺陷的出现次数及起止记录 ID。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, inspected_at, batch_id, defect_type, defect_type_cn, confidence
                FROM inspections
                WHERE machine_id = ?
                ORDER BY inspected_at DESC, id DESC
                LIMIT ?
                """,
                (machine_id, int(limit)),
            ).fetchall()

        if not rows:
            return None

        target_defect = str(rows[0]["defect_type"])
        target_cn = str(rows[0]["defect_type_cn"])
        streak_rows = []
        for r in rows:
            if str(r["defect_type"]) == target_defect:
                streak_rows.append(dict(r))
            else:
                break

        return {
            "machine_id": machine_id,
            "defect_type": target_defect,
            "defect_type_cn": target_cn,
            "consecutive_count": len(streak_rows),
            "latest_inspection_id": int(streak_rows[0]["id"]),
            "latest_inspected_at": str(streak_rows[0]["inspected_at"]),
            "latest_batch_id": str(streak_rows[0]["batch_id"]),
        }

    def get_machine_sequence(self, machine_id: str) -> list[dict[str, Any]]:
        """按时间升序获取某机台的全部检测流水（供全库历史连续缺陷扫描使用）。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, inspected_at, machine_id, batch_id, defect_type, defect_type_cn, confidence
                FROM inspections
                WHERE machine_id = ?
                ORDER BY inspected_at ASC, id ASC
                """,
                (machine_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # 预警表 CRUD
    # ------------------------------------------------------------------
    def save_alert(
        self,
        *,
        machine_id: str,
        defect_type: str,
        consecutive_count: int,
        threshold: int,
        severity: str,
        summary: str,
        possible_causes: str,
        recommended_actions: str,
        doc_references: list[dict[str, Any]],
        last_inspection_id: int = 0,
        created_at: str | None = None,
    ) -> int:
        """保存或更新同一机台同类缺陷的活跃预警（避免重复刷屏）。"""
        ts = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        canonical = normalize_defect_name(defect_type)
        cn_name = DEFECT_CN_MAP.get(canonical, defect_to_cn(canonical))
        refs_json = json.dumps(doc_references, ensure_ascii=False)

        with self._connect() as conn:
            # 若同一机台、同类缺陷已有未关闭预警且覆盖同一连续段，则就地升级更新
            existing = conn.execute(
                """
                SELECT id, consecutive_count, last_inspection_id
                FROM alerts
                WHERE machine_id = ? AND defect_type = ? AND status = 'open'
                ORDER BY id DESC
                LIMIT 1
                """,
                (machine_id, canonical),
            ).fetchone()

            if existing and int(existing["last_inspection_id"]) == int(last_inspection_id):
                return int(existing["id"])

            if existing and int(consecutive_count) >= int(existing["consecutive_count"]):
                conn.execute(
                    """
                    UPDATE alerts
                    SET created_at = ?, consecutive_count = ?, threshold = ?, severity = ?,
                        summary = ?, possible_causes = ?, recommended_actions = ?,
                        doc_references_json = ?, last_inspection_id = ?
                    WHERE id = ?
                    """,
                    (
                        ts,
                        int(consecutive_count),
                        int(threshold),
                        severity,
                        summary,
                        possible_causes,
                        recommended_actions,
                        refs_json,
                        int(last_inspection_id),
                        int(existing["id"]),
                    ),
                )
                return int(existing["id"])

            cur = conn.execute(
                """
                INSERT INTO alerts (
                    created_at, machine_id, defect_type, defect_type_cn,
                    consecutive_count, threshold, severity, status,
                    summary, possible_causes, recommended_actions,
                    doc_references_json, last_inspection_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    machine_id,
                    canonical,
                    cn_name,
                    int(consecutive_count),
                    int(threshold),
                    severity,
                    summary,
                    possible_causes,
                    recommended_actions,
                    refs_json,
                    int(last_inspection_id),
                ),
            )
            return int(cur.lastrowid)

    def list_alerts(
        self,
        machine_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """查询预警列表（自动反序列化工艺文档引用列表）。"""
        clauses: list[str] = []
        params: list[Any] = []
        if machine_id:
            clauses.append("machine_id = ?")
            params.append(machine_id)
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT *
            FROM alerts
            {where_sql}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            try:
                item["doc_references"] = json.loads(item.pop("doc_references_json", "[]"))
            except json.JSONDecodeError:
                item["doc_references"] = []
            results.append(item)
        return results

    def update_alert_status(self, alert_id: int, status: str) -> bool:
        """更新预警处理状态（open / acknowledged / resolved）。"""
        if status not in {"open", "acknowledged", "resolved"}:
            raise ValueError(f"非法的预警状态：{status}")
        with self._connect() as conn:
            cur = conn.execute("UPDATE alerts SET status = ? WHERE id = ?", (status, int(alert_id)))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # 受限只读 SQL 执行器（供 LLM Text-to-SQL 工具调用）
    # ------------------------------------------------------------------
    def execute_readonly_sql(self, sql: str, max_rows: int = 50) -> dict[str, Any]:
        """严格校验并执行只读 SELECT SQL，防止任何数据修改或注入风险。"""
        cleaned = sql.strip().rstrip(";").strip()
        if not cleaned:
            raise ValueError("SQL 语句不能为空。")
        if ";" in cleaned:
            raise ValueError("仅允许执行单条只读 SELECT 语句，禁止多语句拼接。")
        if not re.match(r"^(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
            raise ValueError("出于数据安全限制，仅允许以 SELECT 或 WITH 开头的只读查询。")
        if _FORBIDDEN_SQL_PATTERN.search(cleaned):
            raise ValueError("检测到被禁止的非只读 SQL 关键字，已拒绝执行。")

        # 强制限制返回行数
        if not re.search(r"\bLIMIT\b", cleaned, flags=re.IGNORECASE):
            cleaned = f"{cleaned} LIMIT {int(max_rows)}"

        # 使用 SQLite 只读 URI 模式作为第二道底层防线
        uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute(cleaned).fetchall()[:max_rows]]
        finally:
            conn.close()

        return {
            "sql": cleaned,
            "row_count": len(rows),
            "rows": rows,
        }
