"""轻量级本地工艺文件 RAG 检索引擎（基于 TF-IDF 字符 n-gram + 关键词融合）。

设计理念：
- 零额外重型向量数据库或分词库依赖（仅依赖已安装的 scikit-learn + numpy）；
- 字符级 n-gram `(1, 3)` 天然适配中文工艺术语及英文缺陷编码（如 `rolled-in_scale`、`scratches`）；
- 按 Markdown 标题（`##` / `###`）切分知识块，保留源文件名与章节标题，确保预警和回答均可溯源引用。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from vision_agent.config import DEFECT_ALIAS_MAP, DEFECT_CN_MAP


@dataclass
class DocChunk:
    """切分后的工艺文档知识块。"""

    chunk_id: str
    doc_name: str
    doc_path: str
    section: str
    content: str


@dataclass
class RetrievalResult:
    """RAG 单条检索命中结果。"""

    doc_name: str
    section: str
    content: str
    score: float
    citation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_markdown_by_headings(text: str, doc_name: str, doc_path: str) -> list[DocChunk]:
    """按 Markdown 二级/三级标题切分为语义完整的工艺片段。"""
    lines = text.splitlines()
    doc_title = doc_name
    current_section = "总览"
    buffer: list[str] = []
    chunks: list[DocChunk] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body and len(body) >= 15:
            chunks.append(
                DocChunk(
                    chunk_id=f"{doc_name}#{len(chunks) + 1}",
                    doc_name=doc_name,
                    doc_path=doc_path,
                    section=current_section,
                    content=f"【{doc_title} - {current_section}】\n{body}",
                )
            )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            doc_title = stripped[2:].strip()
            continue
        if stripped.startswith("## ") or stripped.startswith("### "):
            flush()
            buffer = []
            current_section = re.sub(r"^#{2,3}\s*", "", stripped).strip()
            continue
        buffer.append(line)

    flush()
    return chunks


class ProcessDocRAG:
    """基于 `data/process_docs/` Markdown 工艺文件的轻量 RAG 检索器。"""

    def __init__(self, docs_dir: str | Path) -> None:
        self.docs_dir = Path(docs_dir)
        self.chunks: list[DocChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None
        self.reload()

    @property
    def is_ready(self) -> bool:
        return len(self.chunks) > 0 and self._matrix is not None

    def list_documents(self) -> list[str]:
        if not self.docs_dir.exists():
            return []
        return sorted(p.name for p in self.docs_dir.glob("*.md"))

    def reload(self) -> int:
        """重新扫描工艺文档目录并构建 TF-IDF 检索索引。"""
        self.chunks = []
        if not self.docs_dir.exists():
            self._vectorizer = None
            self._matrix = None
            return 0

        for md_file in sorted(self.docs_dir.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            self.chunks.extend(_split_markdown_by_headings(text, md_file.name, str(md_file)))

        if not self.chunks:
            self._vectorizer = None
            self._matrix = None
            return 0

        corpus = [f"{c.doc_name} {c.section}\n{c.content}" for c in self.chunks]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)
        return len(self.chunks)

    def _expand_query(self, query: str) -> tuple[str, list[str]]:
        """对查询进行中英文缺陷同义词扩展，提升跨语种召回精度。"""
        expanded_terms = [query]
        boost_keywords: list[str] = []
        lower = query.lower()

        for alias, canonical in DEFECT_ALIAS_MAP.items():
            if alias in lower:
                cn = DEFECT_CN_MAP.get(canonical, canonical)
                expanded_terms.extend([canonical, cn])
                boost_keywords.extend([canonical.lower(), cn])

        # 提取额外中英文实体词（长度 >= 2）
        for token in re.findall(r"[\u4e00-\u9fa5]{2,6}|[a-zA-Z0-9_\-]{3,}", query):
            if token not in ("什么", "怎么", "原因", "如何", "有没有", "可能"):
                boost_keywords.append(token.lower())

        return " ".join(expanded_terms), list(dict.fromkeys(boost_keywords))

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """检索与查询最相关的前 K 个工艺文档片段。"""
        if not self.is_ready or not query.strip():
            return []

        expanded_q, boost_keywords = self._expand_query(query)
        q_vec = self._vectorizer.transform([expanded_q])
        sims = cosine_similarity(q_vec, self._matrix).ravel()

        scores = sims.astype(float).copy()
        for idx, chunk in enumerate(self.chunks):
            haystack = f"{chunk.section} {chunk.content}".lower()
            title_haystack = chunk.section.lower()
            bonus = 0.0
            for kw in boost_keywords:
                if kw in title_haystack:
                    bonus += 0.25
                elif kw in haystack:
                    bonus += 0.08
            scores[idx] += min(bonus, 0.8)

        order = np.argsort(scores)[::-1][: max(1, int(top_k))]
        results: list[RetrievalResult] = []
        for idx in order:
            sc = float(scores[idx])
            if sc <= 0.01:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    doc_name=chunk.doc_name,
                    section=chunk.section,
                    content=chunk.content,
                    score=round(sc, 4),
                    citation=f"《{chunk.doc_name}》— {chunk.section}",
                )
            )
        return results

    def lookup_defect_guidance(self, defect_type: str, top_k: int = 3) -> dict[str, Any]:
        """针对某一类金属缺陷定向检索工艺成因、排查与处置建议。"""
        cn = DEFECT_CN_MAP.get(defect_type, defect_type)
        hits = self.search(f"{cn} {defect_type} 工艺原因 处置措施 检修", top_k=top_k)

        causes_lines: list[str] = []
        action_lines: list[str] = []
        for hit in hits:
            mode = ""
            for raw_line in hit.content.splitlines():
                line = raw_line.strip()
                if "工艺原因" in line or "原因" in line and line.startswith("-"):
                    mode = "cause"
                    continue
                if "处置措施" in line or "检修" in line and line.startswith("-"):
                    mode = "action"
                    continue
                if re.match(r"^\d+\.\s+", line):
                    cleaned = re.sub(r"^\d+\.\s+", "", line).rstrip("；;。")
                    if mode == "cause" and len(causes_lines) < 3:
                        causes_lines.append(cleaned)
                    elif mode == "action" and len(action_lines) < 3:
                        action_lines.append(cleaned)

        if not causes_lines and hits:
            causes_lines.append(hits[0].content.splitlines()[1] if len(hits[0].content.splitlines()) > 1 else hits[0].content[:160])
        if not action_lines:
            action_lines.append(f"请参照《{hits[0].doc_name}》对应章节执行现场复检与工艺参数核查。" if hits else "建议现场复核工艺参数并抽检当前批次。")

        return {
            "defect_type": defect_type,
            "defect_type_cn": cn,
            "possible_causes": "；".join(causes_lines),
            "recommended_actions": "；".join(action_lines),
            "references": [h.to_dict() for h in hits],
        }
