import time
import threading
import requests
from typing import Optional, Tuple, Dict
from prompts.fallback_responses import get_fallback_response, FaultType, RoleType

# ===================== 配置常量 =====================
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
TOTAL_MAX_TIMEOUT = 45
FAILURE_THRESHOLD = 3
COOLDOWN_SECONDS = 30

# ===================== 熔断器状态管理 =====================
class CircuitBreaker:
    def __init__(self):
        self.failure_count: int = 0
        self.is_open: bool = False
        self.open_time: float = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """判断当前是否允许发起API请求"""
        with self._lock:
            if not self.is_open:
                return True
            # 冷却时间结束，重置熔断器
            if time.time() - self.open_time >= COOLDOWN_SECONDS:
                self.is_open = False
                self.failure_count = 0
                return True
            return False

    def record_success(self):
        """调用成功，清空失败计数"""
        with self._lock:
            self.failure_count = 0

    def record_failure(self):
        """调用失败，累计计数，达到阈值则打开熔断"""
        with self._lock:
            self.failure_count += 1
            if self.failure_count >= FAILURE_THRESHOLD:
                self.is_open = True
                self.open_time = time.time()

# 全局单例熔断器
_global_circuit = CircuitBreaker()

# ===================== 带分层超时的API请求封装 =====================
def safe_llm_api_call(
    api_url: str,
    headers: Dict,
    payload: Dict,
    role: RoleType,
    circuit: CircuitBreaker = _global_circuit
) -> Tuple[str, bool]:
    """
    执行带熔断+分层超时的LLM API调用
    :param role: 当前角色 luxun / guide，用于选取兜底话术
    :return: (返回文本, 是否正常成功)
    """
    # 熔断拦截：故障类型 circuit_break
    if not circuit.allow_request():
        fallback_msg = get_fallback_response(fault_type="circuit_break", role=role)
        return fallback_msg, False

    start_total = time.time()
    try:
        # requests分层超时：(连接超时, 读取超时)
        resp = requests.post(
            url=api_url,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
        )
        # 全局总超时兜底校验
        if time.time() - start_total > TOTAL_MAX_TIMEOUT:
            raise TimeoutError("全局总调用超时45s上限")

        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        circuit.record_success()
        return answer, True

    # 连接超时
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
        circuit.record_failure()
        fallback_msg = get_fallback_response(fault_type="timeout", role=role)
        return fallback_msg, False
    # 读取超时
    except requests.exceptions.ReadTimeout:
        circuit.record_failure()
        fallback_msg = get_fallback_response(fault_type="timeout", role=role)
        return fallback_msg, False
    # 全局总超时
    except TimeoutError:
        circuit.record_failure()
        fallback_msg = get_fallback_response(fault_type="timeout", role=role)
        return fallback_msg, False
    # HTTP异常/其他接口异常归为 api_error
    except requests.exceptions.HTTPError:
        circuit.record_failure()
        fallback_msg = get_fallback_response(fault_type="api_error", role=role)
        return fallback_msg, False
    # 其余全部未知异常归为 api_error
    except Exception:
        circuit.record_failure()
        fallback_msg = get_fallback_response(fault_type="api_error", role=role)
        return fallback_msg, False

# ===================== 获取熔断器状态【输出中文】 =====================
def get_circuit_status() -> Dict:
    """返回熔断器中文状态，用于Gradio前端展示"""
    cb = _global_circuit
    with cb._lock:
        remain_cool = max(0, COOLDOWN_SECONDS - (time.time() - cb.open_time)) if cb.is_open else 0
        if cb.is_open:
            fuse_status_text = "🔴 已熔断(冷却中)"
        else:
            fuse_status_text = "🟢 正常可用"
        return {
            "熔断状态": fuse_status_text,
            "连续失败次数": cb.failure_count,
            "冷却剩余秒数": round(remain_cool, 1)
        }