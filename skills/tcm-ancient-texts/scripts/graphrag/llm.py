"""LLM 客户端抽象:统一 chat 与结构化 JSON 输出,后端可切换。

支持 provider:
  litellm  — 经 litellm SDK 统一路由(百余种后端)
  azure    — Azure OpenAI(openai SDK 的 AzureOpenAI 客户端)
  poe      — Poe OpenAI 兼容端点(openai SDK, base_url=https://api.poe.com/v1)
  openai   — 任意 OpenAI 兼容端点(含自建/vLLM)
  rule     — 不走网络,由 agents 层的确定性规则实现(此处不构造客户端)

所有 SDK 懒加载:未安装对应 provider 的 SDK 时,仅在实际使用时报清晰错误,
不影响 import graphrag 或使用 rule provider。
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class LLMClient:
    """统一接口。子类只需实现 _chat(messages, json_mode)->str。"""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    # -- 子类实现 --
    def _chat(self, messages, json_mode: bool) -> str:  # pragma: no cover
        raise NotImplementedError

    # -- 公共 API --
    def complete(self, system: str, user: str, **kw) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return self._chat(messages, json_mode=False)

    def complete_json(self, system: str, user: str, *, retries: int = 1) -> object:
        """要求模型输出 JSON,并稳健解析。失败自动修复重试。"""
        sys_json = system + ("\n\n只输出一个 JSON,不要任何解释、前后缀或 Markdown 代码块。")
        messages = [{"role": "system", "content": sys_json},
                    {"role": "user", "content": user}]
        last_err = None
        for attempt in range(retries + 1):
            raw = self._chat(messages, json_mode=True)
            try:
                return _extract_json(raw)
            except ValueError as e:
                last_err = e
                messages.append({"role": "assistant", "content": (raw or "")[:2000]})
                messages.append({"role": "user",
                                 "content": "上一次输出不是合法 JSON。请只返回一个合法 JSON 对象。"})
        raise LLMError(f"模型未能返回合法 JSON: {last_err}")


def _is_format_error(exc: Exception) -> bool:
    """判断异常是否因 response_format(json_mode)不被支持而起。
    仅对这类错误做"去掉 json_mode 重试",避免把鉴权/超时/限流也重试一遍。"""
    msg = str(exc).lower()
    return any(k in msg for k in (
        "response_format", "json_object", "json mode", "not supported",
        "unsupported", "invalid parameter", "unrecognized", "bad request", "400"))


def _extract_json(text: str):
    """从模型输出中提取第一个 JSON 值,容忍代码块与前后缀噪声。"""
    if text is None:
        raise ValueError("空响应")
    t = text.strip()
    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # 扫描平衡的 {..} 或 [..];某个平衡组解析失败时,从下一个 opener 继续找
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        while start >= 0:
            depth = 0
            in_str = False
            esc = False
            end = -1
            for i in range(start, len(t)):
                c = t[i]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == opener:
                        depth += 1
                    elif c == closer:
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
            if end >= 0:
                try:
                    return json.loads(t[start:end + 1])
                except json.JSONDecodeError:
                    pass
            start = t.find(opener, start + 1)
    raise ValueError(f"无法解析 JSON:{text[:200]!r}")


# ---------------------------------------------------------------------------
# 各 provider
# ---------------------------------------------------------------------------

class LiteLLMClient(LLMClient):
    def _chat(self, messages, json_mode):
        try:
            import litellm
        except ImportError as e:
            raise LLMError("需要 litellm:pip install litellm") from e
        kw = dict(model=self.cfg.model, messages=messages,
                  temperature=self.cfg.temperature,
                  max_tokens=self.cfg.max_tokens, timeout=self.cfg.timeout)
        if self.cfg.api_key:
            kw["api_key"] = self.cfg.api_key
        if self.cfg.api_base:
            kw["api_base"] = self.cfg.api_base
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        kw.update(self.cfg.extra)
        try:
            resp = litellm.completion(**kw)
        except Exception as e:
            if json_mode and "response_format" in kw and _is_format_error(e):
                kw.pop("response_format", None)
                resp = litellm.completion(**kw)
            else:
                raise
        return resp["choices"][0]["message"]["content"]


class _OpenAICompatClient(LLMClient):
    """openai SDK 的通用 OpenAI 兼容后端(用于 openai / poe)。"""
    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError("需要 openai:pip install openai") from e
        kw = {"api_key": self.cfg.api_key or "EMPTY"}
        if self.cfg.api_base:
            kw["base_url"] = self.cfg.api_base
        kw["timeout"] = self.cfg.timeout
        return OpenAI(**kw)

    def _chat(self, messages, json_mode):
        client = self._client()
        kw = dict(model=self.cfg.model, messages=messages,
                  temperature=self.cfg.temperature,
                  max_tokens=self.cfg.max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        kw.update(self.cfg.extra)
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:
            if json_mode and "response_format" in kw and _is_format_error(e):
                kw.pop("response_format", None)
                resp = client.chat.completions.create(**kw)
            else:
                raise
        return resp.choices[0].message.content


class AzureClient(LLMClient):
    def _chat(self, messages, json_mode):
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise LLMError("需要 openai>=1.0:pip install openai") from e
        if not self.cfg.api_base:
            raise LLMError("Azure 需要 AZURE_OPENAI_ENDPOINT")
        client = AzureOpenAI(api_key=self.cfg.api_key,
                             azure_endpoint=self.cfg.api_base,
                             api_version=self.cfg.api_version or "2024-06-01",
                             timeout=self.cfg.timeout)
        model = self.cfg.deployment or self.cfg.model
        if not model:
            raise LLMError("Azure 需要 AZURE_OPENAI_DEPLOYMENT")
        kw = dict(model=model, messages=messages,
                  temperature=self.cfg.temperature,
                  max_tokens=self.cfg.max_tokens)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        kw.update(self.cfg.extra)
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:
            if json_mode and "response_format" in kw and _is_format_error(e):
                kw.pop("response_format", None)
                resp = client.chat.completions.create(**kw)
            else:
                raise
        return resp.choices[0].message.content


_REGISTRY = {
    "litellm": LiteLLMClient,
    "azure": AzureClient,
    "poe": _OpenAICompatClient,
    "openai": _OpenAICompatClient,
}


def make_client(cfg: LLMConfig) -> Optional[LLMClient]:
    """构造 LLM 客户端。provider=='rule' 返回 None(agents 走确定性规则)。"""
    if cfg.provider == "rule":
        return None
    cls = _REGISTRY.get(cfg.provider)
    if not cls:
        raise LLMError(f"未知 provider: {cfg.provider}("
                       f"可选: rule/litellm/azure/poe/openai)")
    return cls(cfg)


def probe(cfg: LLMConfig) -> str:
    """轻量连通性自检,返回人类可读状态。"""
    if cfg.provider == "rule":
        return "rule provider:离线规则引擎,无需 API key,始终可用。"
    client = make_client(cfg)
    try:
        out = client.complete("你是连通性测试。", "回复两个字:就绪")
        return f"{cfg.provider} ({cfg.model or cfg.deployment}) 可用,返回:{out.strip()[:20]}"
    except Exception as e:
        return f"{cfg.provider} 不可用:{e}"
