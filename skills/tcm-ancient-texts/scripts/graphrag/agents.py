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
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 0. 查询理解与拆解
# ---------------------------------------------------------------------------

def analyze_query(client: Optional[LLMClient], query: str, onto: Ontology) -> dict:
    if client is None:
        return _rule_analyze(query, onto)
    sys = ("你是中医古籍检索的查询解析器。用户查询可能是简体或口语,请先在心中转为"
           "繁体古籍语汇,再拆解为结构化检索意图,用于古籍条文召回。"
           "ancient_terms 必须是繁体古籍用词(如 骨痿/腰痛/腎虛),不要现代病名;"
           "简体词一律转繁体(汤→湯、证→證、肾→腎、酸软→痠軟)。")
    schema = ('{"modern_disease": [..], "modern_phenotypes": [..], '
              '"tcm_patterns": [繁体证候], "ancient_terms": [繁体古籍检索词], '
              '"seed_patterns": [繁体证候,用于图谱扩展]}')
    hint = "、".join(onto.all_synonyms()[:40])
    user = (f"查询:{query}\n\n本病种({onto.label})常见古籍词参考:{hint}\n\n"
            f"输出 JSON,字段:{schema}")
    try:
        data = client.complete_json(sys, user)
        # 合并规则结果兜底,避免 LLM 漏词。规则侧对简体查询做繁体归一后再匹配。
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

def span_grounded(span: str, text: str) -> Optional[bool]:
    """引用忠实度的确定性核验:evidence_span 是否为原文的逐字连续子串
    (忽略空白差异)。None=无 span 可核;True/False=核验结果。

    这是 attributable generation 评测(ALCE 等)中"引文必须支撑断言"原则的
    最低但可完全确定性执行的形式:span 若非原文子串,则一定是改写或编造。
    """
    if not span:
        return None
    return _WS.sub("", str(span)) in _WS.sub("", text)


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
        # 源头修复:LLM 给出的 span 若非原文子串(改写/编造),立即用规则 span 替换,
        # 保证进入卡片的引文永远逐字来自原文。
        if not d.get("evidence_span") or span_grounded(d["evidence_span"], text) is False:
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
    # 句级共现判定:仅当核心词句均处被排除语境(hard)才判 negativeMatch;
    # 排除词只出现在他句(soft)时保留正常映射(裁判层会降级并提示复核)。
    scoped = onto.exclusions_scoped(text)
    for b in onto.bridges:
        if any(a in text for a in b.get("ancient", [])):
            phenos.extend(b.get("modern_phenotype", []))
            r = _MATCH_RANK.get(b.get("match", "relatedMatch"), 1)
            if r > best_rank:
                best_rank, best_type = r, b.get("match", "relatedMatch")
    if scoped["flags"] and scoped["hard"]:
        best_type = "negativeMatch"
    return {
        "modern_phenotypes": list(dict.fromkeys(phenos)),
        "mapping_type": best_type,
        "confidence": round(min(0.95, 0.4 + 0.15 * len(set(phenos))), 2),
    }


# ---------------------------------------------------------------------------
# 2.5 Reranker(LLM 交叉编码器精排)
# ---------------------------------------------------------------------------

def rerank_llm(client: Optional[LLMClient], query: str,
               candidates: List[dict]) -> Optional[Dict[int, float]]:
    """对候选做一次 LLM 相关性精排(交叉编码器式:query × 每条原文一起判分)。

    candidates: [{"id": passage_id, "cite": 出处, "text": 原文}]
    返回 {passage_id: relevance∈[0,1]};client 为 None 或失败返回 None(pipeline 保持
    确定性加权排序)。一次调用批量评分,控制成本。
    """
    if client is None or not candidates:
        return None
    sys = ("你是中医古籍检索的交叉编码器重排器。针对查询,逐条判断古籍原文与查询意图的"
           "相关程度(0~1),需综合病名/表型/证候/治法是否契合,并识别貌似相关实则语境"
           "不符(如危候预后、外伤、他病)的条文给低分。只依据所给原文判分。")
    lines = [f'{c["id"]}\t{c["cite"]}\t{c["text"][:160]}' for c in candidates]
    user = ("查询:" + query + "\n\n候选(id<TAB>出处<TAB>原文):\n"
            + "\n".join(lines) +
            '\n\n输出 JSON 数组,每条 {"id": 原样id, "relevance": 0~1}。只输出数组。')
    try:
        arr = client.complete_json(sys, user)
        if isinstance(arr, dict):        # 容忍 {"results":[...]} 包裹
            arr = arr.get("results") or arr.get("data") or []
        out = {}
        for item in arr if isinstance(arr, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                pid = int(item.get("id"))
                rel = max(0.0, min(1.0, float(item.get("relevance"))))
            except (TypeError, ValueError):
                continue
            out[pid] = rel
        return out or None
    except Exception:
        return None


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
    # 句级共现判定的硬/软排除(见 Ontology.exclusions_scoped);
    # 缺省 True 保守处理(旧调用方未提供时不放宽)。
    excl_hard = bool(card_data.get("exclusion_hard", True))
    has_symptom = bool(card_data.get("ancient_disease_terms")
                       or card_data.get("manifestations"))
    has_pattern = bool(card_data.get("patterns"))
    has_treat = bool(card_data.get("therapeutic_principles")
                     or card_data.get("formulas_or_herbs"))
    n_pheno = len(card_data.get("modern_phenotypes") or [])
    mtype = card_data.get("mapping_type", "")

    if excl and excl_hard and not (has_pattern and has_treat):
        grade, reason = "E", f"命中排除项({'、'.join(excl)}),核心词所在句均处被排除语境。"
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
    # 软排除(排除词在他句,核心词句干净):不一票否决,但 A 降为 B,
    # 提示同段含混杂语境,需专家复核后方可作核心证据。
    if excl and not excl_hard and grade == "A":
        grade = "B"
        reason += f"(同段含排除词 {'、'.join(excl)},但不与核心词同句;降为支持证据待复核)"
    # 方证极性:方剂被明确禁忌(negate)时,不能作"该方治此病"的高等级证据——
    # 封顶为 C(背景/反证);辨证使用(conditional)封顶为 B(需据证甄别)。
    polarity = card_data.get("formula_polarity", "neutral")
    if polarity == "negate" and grade in ("A", "B"):
        grade = "C"
        reason += "(方剂在原文中被禁忌/否定,作反证或背景,不宜作处方证据)"
    elif polarity == "conditional" and grade == "A":
        grade = "B"
        reason += "(方剂为辨证使用,随证可用可禁,降为支持证据待甄别)"
    base = {"A": 0.88, "B": 0.72, "C": 0.55, "D": 0.35, "E": 0.15}[grade]
    penalty = 0.05 * len(excl) if excl_hard else 0.02 * len(excl)
    score = round(base + 0.03 * min(3, n_pheno) - penalty, 3)
    return {"grade": grade, "relevance_score": max(0.0, min(1.0, score)), "reason": reason}


# ---------------------------------------------------------------------------
# 4. Verifier(幻觉/断言核验)
# ---------------------------------------------------------------------------

def verify(client: Optional[LLMClient], evidence_span: str, card_data: dict,
           full_text: str, onto: Ontology = None) -> dict:
    # 确定性前置核验:span 非原文子串 → 直接判不通过,LLM 无权放行。
    # (extract 已在源头修复,此处为纵深防御,防任何路径漏进改写的引文。)
    if span_grounded(evidence_span, full_text) is False:
        return {"grounded": False,
                "note": "evidence_span 非原文逐字子串(引用忠实度核验失败)"}
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
