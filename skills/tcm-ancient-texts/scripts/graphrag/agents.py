"""五重模型角色 + 查询解析。

每个角色都有两条实现:
  - LLM 路径:client 非空,走 complete_json(结构化 schema);
  - 规则路径:client 为 None(provider=rule),用本体做确定性推断。

这样有 API key 时用大模型提升质量,无 key 时仍能端到端产出可用证据卡片。
"""
from __future__ import annotations

import re
from typing import List, Optional

from .llm import LLMClient
from .ontology import Ontology, to_traditional

_SENT_SPLIT = re.compile(r"[。！？；\n]")


# ---------------------------------------------------------------------------
# 0. 查询理解与拆解
# ---------------------------------------------------------------------------

def analyze_query(client: Optional[LLMClient], query: str, onto: Ontology) -> dict:
    if client is None:
        return _rule_analyze(query, onto)
    sys = ("你是中医古籍检索的查询解析器。把用户查询拆解为结构化检索意图,"
           "用于古籍条文召回。ancient_terms 必须是繁体古籍用词(如 骨痿/腰痛/腎虛),"
           "不要现代病名。")
    schema = ('{"modern_disease": [..], "modern_phenotypes": [..], '
              '"tcm_patterns": [繁体证候], "ancient_terms": [繁体古籍检索词], '
              '"seed_patterns": [繁体证候,用于图谱扩展]}')
    hint = "、".join(onto.all_synonyms()[:40])
    user = (f"查询:{query}\n\n本病种({onto.label})常见古籍词参考:{hint}\n\n"
            f"输出 JSON,字段:{schema}")
    try:
        data = client.complete_json(sys, user)
        # 合并规则结果兜底,避免 LLM 漏词
        rule = _rule_analyze(query, onto)
        for k in ("ancient_terms", "seed_patterns", "tcm_patterns"):
            merged = list(dict.fromkeys((data.get(k) or []) + rule.get(k, [])))
            data[k] = merged
        # 用 or 而非 setdefault:模型给 null 时也回退规则结果
        data["modern_disease"] = data.get("modern_disease") or rule["modern_disease"]
        data["modern_phenotypes"] = data.get("modern_phenotypes") or rule["modern_phenotypes"]
        data["has_domain_signal"] = bool(
            data.get("ancient_terms") or data.get("seed_patterns")
            or rule["has_domain_signal"])
        return data
    except Exception:
        return _rule_analyze(query, onto)


def _rule_analyze(query: str, onto: Ontology) -> dict:
    query = to_traditional(query)   # 简体查询归一到繁体,避免离线路径丢词
    graph_srcs = onto.graph_sources()
    ancient, seeds, patterns = [], [], []
    # 直接命中的古籍同义词。种子取"命中词 ∩ 图谱源节点",不再只限 L2,
    # 使 骨痿/腰痛 等病名/表型类查询也能触发图谱路径召回。
    for t in onto.terms:
        for s in t.synonyms:
            if s in query:
                ancient.append(t.term)
                if t.term in graph_srcs:
                    seeds.append(t.term)
                if t.layer == "L2_pattern":
                    patterns.append(t.term)
                break
    # 经桥接矩阵:现代表型词 → 古籍词
    modern_pheno, modern_dis = [], []
    for b in onto.bridges:
        for mp in b.get("modern_phenotype", []):
            if mp in query:
                modern_pheno.append(mp)
                ancient.extend(b.get("ancient", []))
                patterns.extend(b.get("tcm", []))
                seeds.extend(a for a in b.get("ancient", []) if a in graph_srcs)
    for d in onto.modern_layers.get("M1_disease", []):
        if any(part in query for part in re.split(r"[（(]", d)):
            modern_dis.append(d)
    dedup = lambda xs: list(dict.fromkeys(xs))
    # 是否检出与本病种相关的任何信号(古籍词/证候/现代表型/现代病名)。
    # 无信号时不回退默认词 —— 否则会给无关查询伪造高置信证据。
    has_signal = bool(ancient or seeds or modern_pheno or modern_dis)
    return {
        "modern_disease": dedup(modern_dis),
        "modern_phenotypes": dedup(modern_pheno),
        "tcm_patterns": dedup(patterns),
        "ancient_terms": dedup(ancient),
        "seed_patterns": dedup(seeds),
        "has_domain_signal": has_signal,
    }


# ---------------------------------------------------------------------------
# 1. Extractor
# ---------------------------------------------------------------------------

def extract(client: Optional[LLMClient], text: str, onto: Ontology) -> dict:
    if client is None:
        return _rule_extract(text, onto)
    sys = ("你是中医古籍实体抽取器。从给定条文原文中抽取要素,evidence_span 必须"
           "是原文中真正支撑判断的连续片段(逐字照抄,不得改写)。")
    user = (f"条文原文:\n{text}\n\n输出 JSON:"
            '{"ancient_disease_terms":[],"manifestations":[],"patterns":[],'
            '"therapeutic_principles":[],"formulas_or_herbs":[],"population":[],'
            '"evidence_span":"原文片段"}')
    try:
        d = client.complete_json(sys, user)
        if not isinstance(d, dict):
            return _rule_extract(text, onto)
        if not d.get("evidence_span"):
            d["evidence_span"] = _best_span(text, _rule_extract(text, onto))
        return d
    except Exception:
        return _rule_extract(text, onto)


def _rule_extract(text: str, onto: Ontology) -> dict:
    buckets = {"L1_disease": [], "L2_pattern": [], "L3_manifestation": [],
               "L5_therapy": [], "L6_formula_herb": [], "L4_population": []}
    for t in onto.terms:
        if t.layer in buckets and any(s in text for s in t.synonyms):
            buckets[t.layer].append(t.term)
    d = {
        "ancient_disease_terms": buckets["L1_disease"],
        "manifestations": buckets["L3_manifestation"],
        "patterns": buckets["L2_pattern"],
        "therapeutic_principles": buckets["L5_therapy"],
        "formulas_or_herbs": buckets["L6_formula_herb"],
        "population": buckets["L4_population"],
    }
    d["evidence_span"] = _best_span(text, d)
    return d


def _best_span(text: str, entities: dict) -> str:
    """选包含最多命中术语的句子作为 evidence span。"""
    terms = []
    for v in entities.values():
        if isinstance(v, list):
            terms.extend(v)
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return text[:120]
    best = max(sents, key=lambda s: sum(1 for t in terms if t and t in s))
    return best[:160]


# ---------------------------------------------------------------------------
# 2. Normalizer
# ---------------------------------------------------------------------------

_MATCH_RANK = {"exactMatch": 4, "broadMatch": 3, "narrowMatch": 2,
               "relatedMatch": 1, "negativeMatch": 0}


def normalize(client: Optional[LLMClient], entities: dict, onto: Ontology,
              text: str) -> dict:
    if client is None:
        return _rule_normalize(entities, onto, text)
    sys = ("你是古今术语归一器。把古籍术语映射到现代医学表型,保留原词,"
           "并给出映射类型(exactMatch/broadMatch/narrowMatch/relatedMatch/negativeMatch)。")
    user = (f"古籍要素:{entities}\n病种:{onto.label}\n"
            f"现代表型候选:{onto.bridge_phenotypes()}\n\n"
            '输出 JSON:{"modern_phenotypes":[],"mapping_type":"","confidence":0.0}')
    try:
        d = client.complete_json(sys, user)
        if not isinstance(d, dict):
            return _rule_normalize(entities, onto, text)
        # 净化字段类型,避免下游 join/迭代崩溃
        if not isinstance(d.get("modern_phenotypes"), list):
            d["modern_phenotypes"] = _rule_normalize(entities, onto, text)["modern_phenotypes"]
        d["mapping_type"] = str(d.get("mapping_type") or "")
        return d
    except Exception:
        return _rule_normalize(entities, onto, text)


def _rule_normalize(entities: dict, onto: Ontology, text: str) -> dict:
    phenos, best_rank, best_type = [], -1, "relatedMatch"
    excl = onto.exclusions_in_text(text)
    for b in onto.bridges:
        if any(a in text for a in b.get("ancient", [])):
            phenos.extend(b.get("modern_phenotype", []))
            r = _MATCH_RANK.get(b.get("match", "relatedMatch"), 1)
            if r > best_rank:
                best_rank, best_type = r, b.get("match", "relatedMatch")
    if excl:
        best_type = "negativeMatch"
    return {
        "modern_phenotypes": list(dict.fromkeys(phenos)),
        "mapping_type": best_type,
        "confidence": round(min(0.95, 0.4 + 0.15 * len(set(phenos))), 2),
    }


# ---------------------------------------------------------------------------
# 3. Evidence Judge
# ---------------------------------------------------------------------------

def judge(client: Optional[LLMClient], query_analysis: dict, card_data: dict,
          onto: Ontology) -> dict:
    if client is None:
        return _rule_judge(card_data, onto)
    sys = ("你是中医古籍证据裁判。判断该条文能否作为现代疾病研究证据,给出等级:"
           "A 病名/表型/病机/治法高度一致;B 表型病机一致但病名不全;"
           "C 仅单一症状或宽泛相关;D 牵强易误配;E 明确排除(命中排除项)。")
    user = (f"研究意图:{query_analysis}\n条文抽取:{card_data}\n\n"
            '输出 JSON:{"grade":"A/B/C/D/E","relevance_score":0.0,"reason":""}')
    try:
        d = client.complete_json(sys, user)
        if not isinstance(d, dict):
            return _rule_judge(card_data, onto)
        grade = str(d.get("grade") or "C").strip().upper()[:1]
        d["grade"] = grade if grade in "ABCDE" else "C"
        # 净化相关性分:模型可能给 null / 文字 / 超界
        try:
            d["relevance_score"] = max(0.0, min(1.0, float(d.get("relevance_score"))))
        except (TypeError, ValueError):
            d["relevance_score"] = _rule_judge(card_data, onto)["relevance_score"]
        return d
    except Exception:
        return _rule_judge(card_data, onto)


def _rule_judge(card_data: dict, onto: Ontology) -> dict:
    excl = card_data.get("exclusion_flags") or []
    has_symptom = bool(card_data.get("ancient_disease_terms")
                       or card_data.get("manifestations"))
    has_pattern = bool(card_data.get("patterns"))
    has_treat = bool(card_data.get("therapeutic_principles")
                     or card_data.get("formulas_or_herbs"))
    n_pheno = len(card_data.get("modern_phenotypes") or [])
    mtype = card_data.get("mapping_type", "")

    if excl and not (has_pattern and has_treat):
        grade, reason = "E", f"命中排除项({'、'.join(excl)}),语境不支持作为骨质疏松证据。"
    elif mtype == "negativeMatch":
        grade, reason = "D", "映射为负向(排除)语境,易误配,不建议纳入。"
    elif has_symptom and has_pattern and has_treat and n_pheno >= 1:
        grade, reason = "A", "病名/表型、病机、治法方药形成完整证据链,且可映射现代表型。"
    elif (has_symptom or n_pheno >= 1) and has_pattern:
        grade, reason = "B", "表型与病机一致,链条部分完整,可作支持证据。"
    elif has_symptom or n_pheno >= 1:
        grade, reason = "C", "仅见单一表型或宽泛相关,作背景证据。"
    else:
        grade, reason = "D", "与本病种关联牵强。"
    base = {"A": 0.88, "B": 0.72, "C": 0.55, "D": 0.35, "E": 0.15}[grade]
    score = round(base + 0.03 * min(3, n_pheno) - 0.05 * len(excl), 3)
    return {"grade": grade, "relevance_score": max(0.0, min(1.0, score)), "reason": reason}


# ---------------------------------------------------------------------------
# 4. Verifier(幻觉/断言核验)
# ---------------------------------------------------------------------------

def verify(client: Optional[LLMClient], evidence_span: str, card_data: dict,
           full_text: str, onto: Ontology = None) -> dict:
    if client is None:
        return _rule_verify(card_data, full_text, onto)
    sys = ("你是事实核验器。检查抽取出的每个术语/表型是否真的能在条文原文中找到"
           "依据(直接出现或明确语义支撑)。不得放行原文未支撑的断言。")
    user = (f"原文:{full_text}\nevidence_span:{evidence_span}\n"
            f"待核验要素:{card_data}\n\n"
            '输出 JSON:{"grounded":true/false,"note":"不成立的断言或说明"}')
    try:
        d = client.complete_json(sys, user)
        if not isinstance(d, dict):
            return _rule_verify(card_data, full_text, onto)
        d["grounded"] = bool(d.get("grounded"))
        d["note"] = str(d.get("note") or "")
        return d
    except Exception:
        return _rule_verify(card_data, full_text, onto)


def _rule_verify(card_data: dict, full_text: str, onto: Ontology = None) -> dict:
    # 构建 术语→同义词 索引,使核验按"原词或其任一同义词出现"判定,
    # 避免抽取经同义词命中、核验却只查规范词导致的假阴性。
    syn_index = {}
    if onto is not None:
        for t in onto.terms:
            syn_index[t.term] = t.synonyms
    missing = []
    for key in ("ancient_disease_terms", "manifestations", "formulas_or_herbs"):
        val = card_data.get(key) or []
        if not isinstance(val, list):   # 防 LLM 把数组给成字符串导致逐字迭代
            val = [val]
        for term in val:
            term = str(term)
            variants = syn_index.get(term, [term])
            if not any(v in full_text for v in variants):
                missing.append(term)
    grounded = not missing
    note = "全部命中术语均见于原文(含同义写法)。" if grounded else \
        f"以下术语在原文无直接依据:{'、'.join(missing)}"
    return {"grounded": grounded, "note": note}
