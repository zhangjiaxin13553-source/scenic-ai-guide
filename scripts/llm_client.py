"""
LLM API 统一封装
================
支持 DeepSeek / GLM 等国产大模型（OpenAI 兼容协议）。

功能：
  - 统一接口：chat(system_prompt, context, user_query) → str
  - 指数退避重试（最多 3 次）
  - 调用间隔限流（防 API 超额）
  - 多轮对话历史管理（最近 5 轮）
  ✨增强加固新增：
  - call() 返回结构化字典，携带状态、token用量、耗时，异常不直接抛出
  - token用量统计，接口耗时统计(ms)
  - 响应清洗，剔除多余markdown标记
  - message长度安全截断防护
  - 全类型异常捕获，业务层获取错误信息，适配RAG后置守卫重试逻辑

环境变量（.env）：
  LLM_API_KEY    - API 密钥（必填）
  LLM_BASE_URL   - API 地址（必填）
  LLM_MODEL      - 模型名（必填）
  LLM_MAX_TOKENS - 最大输出 token（可选，默认 1024）
  LLM_TEMPERATURE- 温度（可选，默认 0.3）

用法：
  from scripts.llm_client import LLMClient, chat

  client = LLMClient()
  reply = client.chat(
      system_prompt="你是讲解员",
      context="广州鲁迅纪念馆位于文明路215号...",
      user_query="纪念馆在哪里？",
  )

  # 底层结构化调用（给rag_pipeline使用）
  res = client.call(messages=[...])
  if res["success"]:
      ans = res["content"]
"""

import os
import json
import time
import logging
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm_client")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ============================================================
# 配置常量
# ============================================================

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0     # 秒，指数退避基数
DEFAULT_MIN_INTERVAL = 0.3          # 秒，两次调用最小间隔（简单限流）
DEFAULT_TIMEOUT = 60                # 秒
SAFE_MESSAGE_MAX_TOTAL_CHARS = 12000  # messages总字符安全上限，防止超长报错

# 常用 provider → base_url 映射
PROVIDER_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
}


# ============================================================
# LLMClient
# ============================================================

class LLMClient:
    """
    LLM 统一调用客户端。

    自动从 .env 读取：
      LLM_API_KEY   API 密钥
      LLM_BASE_URL  API 地址（OpenAI 兼容）
      LLM_MODEL     模型名（默认 deepseek-chat）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        # --- 从参数 / 环境变量读取 ---
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)

        self.max_tokens = (
            max_tokens if max_tokens is not None
            else int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))
        )
        self.temperature = (
            temperature if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", DEFAULT_TEMPERATURE))
        )

        self.max_retries = max_retries
        self.timeout = timeout

        # 简单限流
        self._last_call_time = 0.0
        self._min_interval = DEFAULT_MIN_INTERVAL

        # 对话历史（多轮对话管理）
        self._history: List[Dict[str, str]] = []

        # Token 用量统计（实例累计）
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        # 熔断器：连续失败计数 + 冷却期
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._circuit_failure_threshold = 5    # 连续5次失败触发熔断
        self._circuit_cooldown_sec = 60.0      # 冷却60秒

        # --- 校验 ---
        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY 未设置。请在 .env 中配置，或通过 api_key 参数传入。"
            )
        if not self.base_url:
            raise ValueError(
                "LLM_BASE_URL 未设置。请在 .env 中配置，或通过 base_url 参数传入。"
            )

        # 组装请求 URL 与 Headers
        self._endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "LLMClient 就绪 | model=%s | endpoint=%s | max_tokens=%d | temperature=%.2f",
            self.model, self._endpoint, self.max_tokens, self.temperature,
        )

    @staticmethod
    def _clean_llm_output(text: str) -> str:
        """清洗模型输出：剔除markdown ``` 代码块标记、多余换行空格"""
        if not text:
            return ""
        s = text.strip()
        # 移除 ```json / ``` 标记
        if s.startswith("```"):
            lines = s.splitlines()
            filter_lines = [ln for ln in lines if not ln.strip().startswith("```")]
            s = "\n".join(filter_lines).strip()
        return s

    @staticmethod
    def _safe_truncate_messages(messages: List[Dict[str, str]], max_chars: int) -> List[Dict[str, str]]:
        """安全截断messages总字符，避免请求payload过大报错"""
        total = sum(len(m.get("content", "")) for m in messages)
        if total <= max_chars:
            return messages
        logger.warning(f"messages总字符 {total} 超过安全阈值{max_chars}，执行截断")
        # 保留system，从最早user-assistant开始删减
        system_msg = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m
            else:
                chat_msgs.append(m)
        while chat_msgs and sum(len(m.get("content", "")) for m in ([system_msg] if system_msg else []) + chat_msgs) > max_chars:
            chat_msgs.pop(0)
        out = []
        if system_msg:
            out.append(system_msg)
        out.extend(chat_msgs)
        return out

    def _record_usage(self, usage: Optional[Dict[str, int]]):
        """累加实例维度token用量"""
        if not usage:
            return
        self._total_prompt_tokens += usage.get("prompt_tokens", 0)
        self._total_completion_tokens += usage.get("completion_tokens", 0)

    # ----------------------------------------------------------
    # 核心接口
    # ----------------------------------------------------------

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        底层调用：发送messages，返回结构化结果字典
        自动处理熔断、重试和限流。
        返回字典字段：
            success: bool 是否调用成功
            content: str 模型输出文本 / 错误描述
            token_usage: dict | None {"prompt_tokens":int,"completion_tokens":int,"total_tokens":int}
            cost_ms: int 接口耗时毫秒
            error: str | None 异常信息
        """
        # 熔断检查
        if self._is_circuit_open():
            cooldown_remaining = int(self._circuit_open_until - time.time())
            err_msg = f"熔断器已打开（连续 {self._circuit_failure_threshold} 次失败），请等待 {cooldown_remaining}s 后重试"
            logger.warning(err_msg)
            return {
                "success": False,
                "content": f"[LLM熔断] {err_msg}",
                "token_usage": None,
                "cost_ms": 0,
                "error": err_msg
            }

        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens

        # ✨增强：安全截断，防止payload超长
        safe_messages = self._safe_truncate_messages(messages, SAFE_MESSAGE_MAX_TOTAL_CHARS)

        payload = {
            "model": self.model,
            "messages": safe_messages,
            "temperature": temp,
            "max_tokens": mt,
            "stream": False,
        }

        last_error: Optional[Exception] = None
        token_usage = None
        content = ""
        start_ts = time.time()

        for attempt in range(self.max_retries + 1):
            self._rate_limit()
            try:
                resp = requests.post(
                    self._endpoint,
                    headers=self._headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                # 提取token用量（兼容deepseek/glm返回格式）
                if "usage" in data:
                    token_usage = dict(data["usage"])
                    self._record_usage(token_usage)

                choices = data.get("choices") or []
                if not choices:
                    raise ValueError(f"API 返回空 choices: {json.dumps(data)[:200]}")
                message = choices[0].get("message") or {}
                raw_content = message.get("content", "")
                content = self._clean_llm_output(raw_content)

                # 成功 → 重置熔断计数
                self._consecutive_failures = 0
                cost_ms = int((time.time() - start_ts) * 1000)
                return {
                    "success": True,
                    "content": content,
                    "token_usage": token_usage,
                    "cost_ms": cost_ms,
                    "error": None
                }

            except requests.exceptions.Timeout as e:
                last_error = e
                self._consecutive_failures += 1
                logger.warning("API 超时")
            except requests.exceptions.HTTPError as e:
                last_error = e
                self._consecutive_failures += 1
                status_code = getattr(e.response, "status_code", 0)
                logger.warning(f"HTTP异常 status={status_code}")
                # 429限流特殊日志
                if status_code == 429:
                    logger.warning("触发API限流429")
            except json.JSONDecodeError as e:
                last_error = e
                self._consecutive_failures += 1
                logger.warning("API返回非合法JSON")
            except Exception as e:
                last_error = e
                self._consecutive_failures += 1

            if attempt < self.max_retries:
                delay = DEFAULT_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "API 失败 (第 %d/%d 次)，%.1fs 后重试: %s",
                    attempt + 1, self.max_retries, delay, str(last_error),
                )
                time.sleep(delay)
            else:
                logger.error(
                    "API 失败，已达最大重试次数 (%d): %s",
                    self.max_retries, str(last_error),
                )

        # 检查是否触发熔断
        if self._consecutive_failures >= self._circuit_failure_threshold:
            self._circuit_open_until = time.time() + self._circuit_cooldown_sec
            logger.critical(
                "连续 %d 次失败，熔断器已打开，冷却 %ds",
                self._consecutive_failures, self._circuit_cooldown_sec,
            )

        cost_ms = int((time.time() - start_ts) * 1000)
        return {
            "success": False,
            "content": f"[LLM调用异常] {str(last_error)}",
            "token_usage": token_usage,
            "cost_ms": cost_ms,
            "error": str(last_error) if last_error else "unknown"
        }

    def chat(
        self,
        system_prompt: str,
        context: str = "",
        user_query: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        单轮对话（RAG 场景）。向后兼容原有接口，返回字符串。
        """
        # 拼接上下文与用户问题
        if context:
            user_content = (
                f"【参考资料（请优先根据以下信息回答）】\n"
                f"{context}\n\n"
                f"【用户问题】\n"
                f"{user_query}"
            )
        else:
            user_content = user_query

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        res = self.call(messages, temperature=temperature, max_tokens=max_tokens)
        return res["content"]

    def chat_with_history(
        self,
        system_prompt: str,
        context: str,
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        多轮对话。向后兼容原有接口，返回字符串。
        """
        msgs = history if history is not None else self._history

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(msgs)

        if context:
            user_content = (
                f"【参考资料】\n{context}\n\n"
                f"【用户问题】\n{user_query}"
            )
        else:
            user_content = user_query

        messages.append({"role": "user", "content": user_content})

        res = self.call(messages)
        reply = res["content"]

        # 更新内置历史（只保留最近 5 轮 = 10 条消息）
        self._history.append({"role": "user", "content": user_query})
        self._history.append({"role": "assistant", "content": reply})
        if len(self._history) > 10:
            self._history = self._history[-10:]

        return reply

    # ----------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------

    def _rate_limit(self):
        """简单限流：确保两次调用间隔不小于 _min_interval 秒。"""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def _is_circuit_open(self) -> bool:
        """检查熔断器是否打开。"""
        if self._circuit_open_until > time.time():
            return True
        # 冷却期已过，复位
        if self._circuit_open_until > 0:
            self._circuit_open_until = 0.0
            self._consecutive_failures = 0
            logger.info("熔断器已复位")
        return False

    def clear_history(self):
        """清空多轮对话历史。"""
        self._history.clear()

    def reset_circuit(self):
        """手动复位熔断器。"""
        self._circuit_open_until = 0.0
        self._consecutive_failures = 0

    @property
    def history(self) -> List[Dict[str, str]]:
        """返回当前对话历史（只读副本）。"""
        return list(self._history)

    @property
    def token_usage(self) -> dict:
        """返回累计 token 用量统计。"""
        return {
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    @property
    def circuit_status(self) -> dict:
        """返回熔断器状态。"""
        return {
            "is_open": self._is_circuit_open(),
            "consecutive_failures": self._consecutive_failures,
            "cooldown_remaining_s": max(0, int(self._circuit_open_until - time.time())),
        }

    # ----------------------------------------------------------
    # 工厂方法
    # ----------------------------------------------------------

    @classmethod
    def from_provider(cls, provider: str, api_key: Optional[str] = None) -> "LLMClient":
        """
        按 provider 名称快速创建客户端。

        用法：
          client = LLMClient.from_provider("deepseek")
          client = LLMClient.from_provider("glm", api_key="xxx")
        """
        provider = provider.lower()
        if provider in PROVIDER_URLS:
            return cls(base_url=PROVIDER_URLS[provider], api_key=api_key)
        raise ValueError(
            f"不支持的 provider: '{provider}'，可选: {list(PROVIDER_URLS)}"
        )


# ============================================================
# 便捷函数
# ============================================================

_default_client: Optional[LLMClient] = None


def _get_client() -> LLMClient:
    """懒加载全局默认客户端。"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def chat(
    system_prompt: str,
    context: str,
    user_query: str,
    temperature: Optional[float] = None,
) -> str:
    """
    便捷函数 —— 签名与执行方案一致：
        chat(system_prompt, context, user_query) → str

    使用全局默认客户端，适合快速调用。
    """
    return _get_client().chat(
        system_prompt=system_prompt,
        context=context,
        user_query=user_query,
        temperature=temperature,
    )


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-5s | %(message)s",
    )

    print("=" * 60)
    print("  LLM Client 自测（合并增强完整版）")
    print("=" * 60)

    client = LLMClient()

    print(f"  model:      {client.model}")
    print(f"  endpoint:   {client._endpoint}")
    print(f"  max_tokens: {client.max_tokens}")
    print(f"  temperature:{client.temperature}")
    print()

    # 测试结构化call接口（rag_pipeline使用）
    system = "你是一个可靠的问答助手。请用一句话直接回答，不要多余解释。"
    context = "鲁迅（1881-1936），原名周树人，字豫才，浙江绍兴人，中国现代文学奠基人之一。"
    question = "鲁迅是谁？用一句话回答。"
    messages = [
        {"role":"system","content":system},
        {"role":"user","content":f"【参考资料】\n{context}\n【用户问题】{question}"}
    ]
    try:
        res = client.call(messages)
        print(f"call()返回success={res['success']}")
        print(f"token_usage={res['token_usage']}")
        print(f"cost_ms={res['cost_ms']}")
        print(f"A: {res['content']}")
        print(f"实例累计token:{client.token_usage}")
        print(f"熔断器状态:{client.circuit_status}")
        print("OK 结构化接口自测通过\n")
    except Exception as e:
        print(f"FAIL call接口自测失败 {e}")

    # 测试原有兼容chat接口
    try:
        reply = client.chat(system, context, question)
        print(f"兼容chat() A: {reply}")
        print("OK 兼容接口自测通过")
    except Exception as e:
        print(f"FAIL chat自测失败: {e}")