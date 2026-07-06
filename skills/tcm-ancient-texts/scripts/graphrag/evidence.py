"""证据对象与证据卡片 —— 系统的产出单位。

每条结果不是"答案",而是可追溯、可复核的 Evidence Card:绑定原文 span、
出处、模型理由、排除检查与专家状态。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class Candidate:
    """召回阶段的候选条文(含来源路由与子分)。"""
    passage_id: int
    book: str
    seq: int
    path: str
    text: str
    routes: List[str] = field(default_factory=list)   # lexical/synonym/semantic/graph
    matched_terms: List[str] = field(default_factory=list)
    graph_trace: List[str] = field(default_factory=list)
    # 每个召回列表内的名次(键 "route:term" → 1 起的最好名次),供 RRF 融合。
    # RRF 只看名次不看分值,天然免疫 BM25/余弦/密度代理三种分值尺度不可比的问题。
    route_ranks: Dict[str, int] = field(default_factory=dict)
    subscores: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    def cite(self) -> str:
        return f"《{self.book}》· {self.path or '(卷首)'} · 段{self.seq}"


@dataclass
class EvidenceCard:
    passage_id: int
    citation: str
    book: str
    seq: int
    path: str
    original_text: str
    evidence_span: str = ""
    modern_translation: str = ""
    # 结构化抽取(Extractor)
    ancient_disease_terms: List[str] = field(default_factory=list)
    manifestations: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    therapeutic_principles: List[str] = field(default_factory=list)
    formulas_or_herbs: List[str] = field(default_factory=list)
    population: List[str] = field(default_factory=list)
    # 归一化(Normalizer)
    modern_phenotypes: List[str] = field(default_factory=list)
    mapping_type: str = ""            # exact/broad/narrow/related/negative Match
    # 方剂/治法极性(Negation/ConText):处方 vs 禁忌 vs 辨证使用
    polarity: str = "neutral"         # affirm/negate/conditional/neutral
    polarity_note: str = ""           # 命中的否定/禁忌线索说明
    # 裁判(Evidence Judge)
    relevance_score: float = 0.0
    grade: str = ""                   # A/B/C/D/E
    inclusion_reason: str = ""
    exclusion_flags: List[str] = field(default_factory=list)
    # 核验(Verifier)
    grounded: Optional[bool] = None
    verifier_note: str = ""
    # 元数据
    dynasty: str = ""
    author: str = ""
    edition: str = ""
    quality: str = ""
    routes: List[str] = field(default_factory=list)
    expert_status: str = "未审"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        g = self.grade or "?"
        badge = {"A": "A级·核心证据", "B": "B级·支持证据", "C": "C级·背景证据",
                 "D": "D级·不建议纳入", "E": "E级·已排除"}.get(g, f"{g}级")
        ground = {True: "✅ 已核验(判断有原文支撑)",
                  False: "⚠️ 未通过核验(判断超出原文)",
                  None: "—"}[self.grounded]
        lines = [
            f"### {badge}　相关性 {self.relevance_score:.2f}",
            f"> {self.evidence_span or self.original_text[:120]}",
            "",
            f"- **出处**：{self.citation}（{self.dynasty}·{self.author}"
            + (f"，底本:{self.edition}" if self.edition else "") + "）",
        ]
        if self.modern_translation:
            lines.append(f"- **现代释义**：{self.modern_translation}")
        if self.ancient_disease_terms:
            lines.append(f"- **古病名/术语**：{'、'.join(self.ancient_disease_terms)}")
        if self.manifestations:
            lines.append(f"- **症状表型**：{'、'.join(self.manifestations)}")
        if self.patterns:
            lines.append(f"- **证候/病机**：{'、'.join(self.patterns)}")
        if self.therapeutic_principles:
            lines.append(f"- **治法**：{'、'.join(self.therapeutic_principles)}")
        if self.formulas_or_herbs:
            lines.append(f"- **方药**：{'、'.join(self.formulas_or_herbs)}")
        if self.population:
            lines.append(f"- **人群**：{'、'.join(self.population)}")
        if self.modern_phenotypes:
            lines.append(f"- **映射现代表型**：{'、'.join(self.modern_phenotypes)}"
                         + (f"（{self.mapping_type}）" if self.mapping_type else ""))
        if self.polarity and self.polarity != "neutral":
            pbadge = {"affirm": "✅ 处方（该证可用）",
                      "negate": "⛔ 禁忌（此证禁用）",
                      "conditional": "⚖️ 辨证使用（随证可用可禁）"}.get(
                          self.polarity, self.polarity)
            lines.append(f"- **方证极性**：{pbadge}"
                         + (f"（线索：{self.polarity_note}）" if self.polarity_note else ""))
        if self.inclusion_reason:
            lines.append(f"- **纳入理由**：{self.inclusion_reason}")
        lines.append(f"- **排除检查**："
                     + ("命中排除项 " + "、".join(self.exclusion_flags) if self.exclusion_flags
                        else "未见外伤/温病/危候等排除项"))
        lines.append(f"- **事实核验**：{ground}"
                     + (f"（{self.verifier_note}）" if self.verifier_note else ""))
        if self.quality:
            lines.append(f"- **录文品质**：{self.quality}")
        lines.append(f"- **专家状态**：{self.expert_status}　|　召回路由：{'+'.join(self.routes)}")
        return "\n".join(lines)


def cards_to_json(query_analysis: dict, cards: List[EvidenceCard]) -> str:
    return json.dumps({
        "query_analysis": query_analysis,
        "n_cards": len(cards),
        "cards": [c.to_dict() for c in cards],
    }, ensure_ascii=False, indent=2)
