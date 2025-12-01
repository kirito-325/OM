"""大模型调用模块 - 提供统一的LLM调用接口"""

import requests
import json
import re
from typing import List, Dict, Any, Optional


# 默认API配置
DEFAULT_API_URL = "http://129.227.88.34:19101/v1/chat/completions"
DEFAULT_MODEL = "Qwen3-32B"


def call_llm(
    messages: List[Dict[str, str]],
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    response_format: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    调用大模型API

    Args:
        messages: 消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]
        api_url: API地址
        model: 模型名称
        max_tokens: 最大输出token数
        temperature: 温度参数
        response_format: 响应格式，如 {"type": "json_object"}

    Returns:
        解析后的JSON字典，如果解析失败返回空字典
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # 如果指定了响应格式
    if response_format:
        payload["response_format"] = response_format
    else:
        # 默认使用JSON格式
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")

        # 尝试解析JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                return json.loads(m.group(0))
            return {}

    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return {}


def call_llm_text(
    messages: List[Dict[str, str]],
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0
) -> str:
    """
    调用大模型API，返回原始文本

    Args:
        messages: 消息列表
        api_url: API地址
        model: 模型名称
        max_tokens: 最大输出token数
        temperature: 温度参数

    Returns:
        模型返回的原始文本
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return ""
