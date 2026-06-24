"""钢材表面缺陷质检解释与报告生成。

模块采用“规则知识库兜底 + OpenAI 兼容接口可选增强”的方式：
- 默认不需要联网、不需要密钥，直接生成稳定中文质检报告。
- 设置环境变量后，可调用兼容 Chat Completions 的大模型接口生成自然语言报告。
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class DefectKnowledge:
    class_name: str
    class_name_zh: str
    base_risk: str
    description: str
    possible_causes: list[str]
    recommendations: list[str]


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_url: str
    model: str
    api_key: str | None
    endpoint_type: str
    source: str


DEFECT_KNOWLEDGE: dict[str, DefectKnowledge] = {
    "crazing": DefectKnowledge(
        class_name="crazing",
        class_name_zh="龟裂",
        base_risk="高",
        description="表面出现细密网状或线状裂纹，通常会削弱材料表面连续性，并可能在后续加工或服役中扩展。",
        possible_causes=["轧制或冷却过程应力集中", "材料表面氧化或组织不均", "后续加工中局部变形过大"],
        recommendations=["对该批次进行重点复检", "检查轧制温度、冷却速度和张力控制记录", "必要时降低该材料等级或进行返修处理"],
    ),
    "inclusion": DefectKnowledge(
        class_name="inclusion",
        class_name_zh="夹杂",
        base_risk="中高",
        description="表面存在非金属夹杂或异物痕迹，可能影响钢材表面质量和局部力学性能。",
        possible_causes=["冶炼或连铸过程中杂质控制不足", "轧制前坯料表面清理不充分", "生产线环境带入异物"],
        recommendations=["追溯原料和连铸工艺记录", "抽检同批次相邻样品", "加强坯料表面清理和生产线洁净度控制"],
    ),
    "patches": DefectKnowledge(
        class_name="patches",
        class_name_zh="斑块",
        base_risk="中",
        description="表面出现局部块状色差或纹理异常，通常反映表面氧化、污染或局部处理不均。",
        possible_causes=["表面氧化层分布不均", "酸洗或清洗过程不充分", "局部污染或辊面状态异常"],
        recommendations=["检查表面清洗和酸洗参数", "对异常区域进行外观复核", "结合后续用途判断是否需要返修"],
    ),
    "pitted_surface": DefectKnowledge(
        class_name="pitted_surface",
        class_name_zh="麻点表面",
        base_risk="中",
        description="表面存在点状凹坑或粗糙区域，可能影响涂装、镀层质量和外观一致性。",
        possible_causes=["表面氧化皮压入或脱落", "酸洗过度或局部腐蚀", "轧辊表面磨损导致压痕"],
        recommendations=["检查酸洗和除鳞工艺", "排查轧辊磨损或污染情况", "对需要涂装的产品提高复检等级"],
    ),
    "rolled-in_scale": DefectKnowledge(
        class_name="rolled-in_scale",
        class_name_zh="轧入氧化皮",
        base_risk="中高",
        description="氧化皮在轧制过程中被压入钢材表面，会造成局部表面缺陷并影响后续加工质量。",
        possible_causes=["加热后除鳞不彻底", "轧制过程中氧化皮未及时清除", "除鳞水压力或喷嘴状态异常"],
        recommendations=["检查高压水除鳞系统", "核对加热炉温度和停留时间", "对同批次钢材进行连续抽检"],
    ),
    "scratches": DefectKnowledge(
        class_name="scratches",
        class_name_zh="划痕",
        base_risk="中",
        description="表面存在方向性线状损伤，通常与输送、轧辊、导卫或搬运过程中的机械摩擦有关。",
        possible_causes=["输送辊或导卫部件划伤", "卷曲或搬运过程发生摩擦", "生产线局部异物造成连续刮擦"],
        recommendations=["检查输送辊、导卫和卷曲设备表面状态", "观察划痕是否呈连续规律分布", "对外观要求高的产品进行返修或降级处理"],
    ),
}


def _is_local_url(api_url: str) -> bool:
    host = urlparse(api_url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("192.168.")


def _codex_dir() -> Path:
    return Path.home() / ".codex"


def _read_local_codex_config() -> tuple[dict, str | None]:
    config_path = _codex_dir() / "config.toml"
    auth_path = _codex_dir() / "auth.json"
    config: dict = {}
    api_key: str | None = None

    if config_path.exists():
        try:
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError):
            config = {}

    if auth_path.exists():
        try:
            auth = json.loads(auth_path.read_text(encoding="utf-8"))
            raw_key = auth.get("OPENAI_API_KEY")
            if isinstance(raw_key, str) and raw_key.strip():
                api_key = raw_key.strip()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            api_key = None

    return config, api_key


def _normalize_api_url(
    api_url: str | None,
    provider: str,
    api_key: str | None,
    wire_api: str | None = None,
) -> str | None:
    if api_url:
        cleaned = api_url.rstrip("/")
        if cleaned.endswith("/chat/completions") or cleaned.endswith("/responses"):
            return cleaned
        if cleaned.endswith("/v1"):
            if wire_api == "responses" or "api.9e.lv" in cleaned:
                return f"{cleaned}/responses"
            return f"{cleaned}/chat/completions"
        return cleaned
    if provider == "codex" and api_key:
        return "https://api.9e.lv/v1/responses"
    if provider == "openai" and api_key:
        return "https://api.openai.com/v1/responses"
    return None


def get_llm_config() -> LLMConfig | None:
    """读取本地可用的大模型接口配置。

    优先级：
    1. CODEX_API_URL / CODEX_API_KEY / CODEX_MODEL
    2. LLM_API_URL / LLM_API_KEY / LLM_MODEL
    3. OPENAI_API_URL 或 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
    """

    if os.getenv("CODEX_API_URL") or os.getenv("CODEX_MODEL") or os.getenv("CODEX_API_KEY"):
        provider = "codex"
        raw_url = os.getenv("CODEX_API_URL")
        api_key = os.getenv("CODEX_API_KEY")
        model = os.getenv("CODEX_MODEL") or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        wire_api = os.getenv("CODEX_WIRE_API") or os.getenv("LLM_WIRE_API")
        source = "environment"
    else:
        codex_config, codex_api_key = _read_local_codex_config()
        codex_provider = codex_config.get("model_provider")
        codex_base_url = codex_config.get("model_providers", {}).get(codex_provider, {}).get("base_url")
        codex_wire_api = codex_config.get("model_providers", {}).get(codex_provider, {}).get("wire_api")
        codex_requires_auth = codex_config.get("model_providers", {}).get(codex_provider, {}).get("requires_openai_auth")
        if codex_provider and codex_base_url and codex_wire_api and (codex_api_key or not codex_requires_auth):
            provider = "codex"
            raw_url = codex_base_url
            api_key = codex_api_key
            model = codex_config.get("model") or "gpt-5.5"
            wire_api = codex_wire_api
            source = "local_codex_config"
        elif os.getenv("LLM_API_URL") or os.getenv("LLM_MODEL") or os.getenv("LLM_API_KEY"):
            provider = "llm"
            raw_url = os.getenv("LLM_API_URL")
            api_key = os.getenv("LLM_API_KEY")
            model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
            wire_api = os.getenv("LLM_WIRE_API")
            source = "environment"
        else:
            provider = "openai"
            raw_url = os.getenv("OPENAI_API_URL") or os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
            wire_api = os.getenv("OPENAI_WIRE_API") or "responses"
            source = "environment"

    api_url = _normalize_api_url(raw_url, provider, api_key, wire_api)
    if not api_url:
        return None
    if not api_key and not _is_local_url(api_url):
        return None

    endpoint_type = "responses" if api_url.rstrip("/").endswith("/responses") else "chat_completions"
    return LLMConfig(
        provider=provider,
        api_url=api_url,
        model=model,
        api_key=api_key,
        endpoint_type=endpoint_type,
        source=source,
    )


def llm_config_status() -> dict[str, str | bool]:
    config = get_llm_config()
    if not config:
        return {
            "available": False,
            "message": "未检测到可用外部大模型接口配置，当前使用本地知识库规则。",
        }
    return {
        "available": True,
        "provider": config.provider,
        "api_url": config.api_url,
        "model": config.model,
        "endpoint_type": config.endpoint_type,
        "source": config.source,
        "has_api_key": bool(config.api_key),
        "message": f"已检测到 {config.provider} 接口配置，模型：{config.model}",
    }


def normalize_risk(base_risk: str, confidence: float) -> str:
    if confidence < 0.6:
        return "需人工复核"
    if confidence < 0.8:
        return f"{base_risk}，但模型置信度偏低，建议复核"
    return base_risk


def build_prompt(prediction: dict) -> str:
    class_name = prediction["predicted_class"]
    knowledge = DEFECT_KNOWLEDGE[class_name]
    top_k_text = "；".join(
        f"{item['class_name_zh']}({item['confidence']:.2%})" for item in prediction.get("top_k", [])
    )
    return (
        "你是钢材表面质量检测工程师。请根据模型识别结果生成一段中文质检报告，"
        "要求包含缺陷解释、可能成因、风险等级、处理建议，语言专业但简洁。\n\n"
        f"模型预测类别：{knowledge.class_name_zh} ({knowledge.class_name})\n"
        f"预测置信度：{prediction['confidence']:.2%}\n"
        f"推理耗时：{prediction.get('inference_time_ms', 0):.3f} ms\n"
        f"Top-K 结果：{top_k_text}\n"
        f"知识库说明：{knowledge.description}\n"
        f"常见成因：{'、'.join(knowledge.possible_causes)}\n"
        f"建议措施：{'、'.join(knowledge.recommendations)}\n"
    )


def _build_payload(prompt: str, config: LLMConfig) -> dict:
    if config.endpoint_type == "responses":
        return {
            "model": config.model,
            "input": [
                {"role": "system", "content": "你是严谨的工业质检报告助手。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "你是严谨的工业质检报告助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }


def _extract_text(data: dict, endpoint_type: str) -> str | None:
    if endpoint_type == "responses":
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        for item in data.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return None
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content.strip() if isinstance(content, str) else None


def call_openai_compatible(prompt: str) -> tuple[str | None, str | None]:
    config = get_llm_config()
    if not config:
        return None, "未检测到可用接口配置。"

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    request = urllib.request.Request(
        config.api_url,
        data=json.dumps(_build_payload(prompt, config)).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="ignore")[:300]
        return None, f"接口返回 HTTP {error.code}: {details}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return None, f"接口调用失败: {error}"

    text = _extract_text(data, config.endpoint_type)
    if not text:
        return None, "接口响应中未解析到文本内容。"
    return text, None


def generate_rule_report(prediction: dict) -> dict:
    class_name = prediction["predicted_class"]
    knowledge = DEFECT_KNOWLEDGE[class_name]
    confidence = float(prediction["confidence"])
    risk = normalize_risk(knowledge.base_risk, confidence)
    summary = (
        f"系统识别该样本主要缺陷为“{knowledge.class_name_zh}”，"
        f"模型置信度为 {confidence:.2%}，综合风险等级为“{risk}”。"
        f"{knowledge.description}"
    )
    return {
        "mode": "rule_fallback",
        "defect_class": knowledge.class_name,
        "defect_name": knowledge.class_name_zh,
        "risk_level": risk,
        "summary": summary,
        "description": knowledge.description,
        "possible_causes": knowledge.possible_causes,
        "recommendations": knowledge.recommendations,
    }


def generate_quality_report(prediction: dict, enable_llm: bool = False) -> dict:
    report = generate_rule_report(prediction)
    prompt = build_prompt(prediction)
    report["llm_prompt"] = prompt

    if enable_llm:
        llm_text, llm_error = call_openai_compatible(prompt)
        if llm_text:
            report["mode"] = "llm_enhanced"
            report["llm_report"] = llm_text
        else:
            report["llm_error"] = llm_error
            report["llm_report"] = "外部大模型暂不可用，已使用本地知识库规则生成报告。"
    return report
