#!/usr/bin/env python3
"""
DeepSeek API 客户端封装
兼容 OpenAI API 格式
"""

import os
import json
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_PRO = "deepseek-v4-pro"
DEEPSEEK_MODEL_FLASH = "deepseek-v4-flash"


def get_client() -> Optional["OpenAI"]:
    """获取 DeepSeek API 客户端。"""
    if OpenAI is None:
        raise RuntimeError(
            "缺少 openai 依赖。请运行: pip install openai"
        )
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY 环境变量。\n"
            "请设置: export DEEPSEEK_API_KEY='your-key'"
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=120.0)


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEEPSEEK_MODEL_PRO,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    thinking: bool = False,
    reasoning_effort: str = "medium",
) -> str:
    """
    调用 DeepSeek Chat Completion API。
    
    Args:
        messages: 消息列表，格式 [{"role": "system"/"user"/"assistant", "content": "..."}]
        model: 模型名称，默认 deepseek-v4-pro
        temperature: 创造性温度，0-2
        max_tokens: 最大输出 token 数
        stream: 是否流式输出
        thinking: 是否启用思考模式（reasoning）
        reasoning_effort: 思考努力程度: low/medium/high
    
    Returns:
        模型的文本回复内容
    """
    client = get_client()
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if thinking:
        kwargs["thinking"] = {"type": "enabled"}
        kwargs["reasoning_effort"] = reasoning_effort
    
    response = client.chat.completions.create(**kwargs)
    
    if stream:
        # 流式输出需要迭代处理，这里返回空字符串占位
        content = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        return content
    
    return response.choices[0].message.content


def chat_completion_json(
    messages: List[Dict[str, str]],
    model: str = DEEPSEEK_MODEL_PRO,
    temperature: float = 0.5,
    max_tokens: Optional[int] = None,
    thinking: bool = False,
) -> dict:
    """
    调用 DeepSeek API，强制返回 JSON 格式。
    在 messages 末尾自动追加 JSON 格式要求。
    """
    # 复制 messages 避免修改原列表
    msgs = list(messages)
    msgs.append({
        "role": "user",
        "content": (
            "请严格按照 JSON 格式输出，不要包含 markdown 代码块标记（如 ```json），"
            "直接输出纯 JSON 字符串。确保 JSON 格式合法，可以被 Python json.loads 解析。"
        )
    })
    
    content = chat_completion(
        messages=msgs,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
    )
    
    # 清理可能的 markdown 代码块
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # 去掉第一行 ```json 和最后一行 ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"DeepSeek 返回了非法 JSON: {e}\n原始内容:\n{content}")


# ─── 便捷封装：特定场景的 LLM 调用 ───

def expand_persona(user_input: str) -> dict:
    """
    根据用户简要输入，用 LLM 自动扩展生成完整的五层框架 persona 细节。
    返回 JSON 格式的结构化数据。
    使用 deepseek-v4-flash 模型（快速且足够好）。
    """
    system_prompt = """你是一位资深用户研究专家，擅长基于「五层设计框架」构建虚拟用户。

五层框架如下：
1. 角色定义层：demographics（人口统计）+ psychographics（心理特征）+ behavioral_traits（行为特征）+ context（上下文）
2. 大模型条件化：将 persona 转化为 System Prompt
3. 场景注入层：当前场景、初始态度、时间压力、参与动机、之前接触程度
4. 记忆系统层：初始信任度、情绪状态、接触次数
5. 行为引擎层：注意力关键词、怀疑触发词、知识边界

你的任务：根据用户的简要描述，生成一个**具体、真实、有细节**的虚拟用户。

关键原则：
- 每个字段都要具体，不能泛泛而谈
- 量化指标（怀疑度、价格敏感度）要合理
- 痛点要真实，不能是"太复杂"这种空话
- 竞争产品要具体提到名字
- 技术栈要具体到工具名
- 场景要有时间压力、动机冲突

输出严格 JSON，字段名如下：
{
  "demographics": {"age", "role", "company_size", "industry", "location", "years_experience"},
  "psychographics": {"goals", "frustrations", "decision_style", "tech_stack", "budget_authority"},
  "behavioral_traits": {"communication", "skepticism_level", "price_sensitivity", "risk_tolerance"},
  "context": {"current_problem", "recent_changes", "team_pressure", "competitive_exposure"},
  "scene_context": {"scene_description", "initial_attitude", "time_pressure", "participation_motivation", "prior_exposure"},
  "behavior_engine": {"attention_keywords", "skepticism_triggers", "knowledge_boundary"}
}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请基于以下描述生成完整的虚拟用户定义：\n\n{user_input}"}
    ]
    
    return chat_completion_json(messages, model=DEEPSEEK_MODEL_FLASH, temperature=0.7)


def optimize_system_prompt(raw_prompt: str) -> str:
    """
    用 LLM 优化 System Prompt，让它更有"人味"、更自然、更严格。
    防止 LLM 过于配合，确保角色行为一致。
    使用 deepseek-v4-flash 模型（快速且足够好）。
    """
    system_prompt = """你是一位顶尖的 prompt engineering 专家，专门优化 LLM 角色扮演的 System Prompt。

你的任务：把一份结构化的 persona 定义，改写成一份**高质量、有说服力、有"人味"**的 System Prompt。

优化要求：
1. **自然口语化**：不要像简历条目一样罗列，要像真人在描述自己的状态和情绪
2. **增加内心独白**：加入"我现在在想...""我其实有点..."这类内心活动
3. **强化约束**：明确写出"你不会...""你讨厌...""如果...你会..."
4. **防止过度配合**：加上"你不是来帮忙的，你是来评估产品的""如果产品不行，你会直接说"
5. **场景感**：让角色明确知道自己在什么场景下对话
6. **具体反应模式**：不是"不耐烦"，而是"你会打断对方，说'说重点'"

输出：只返回优化后的 System Prompt 文本，不要有任何解释、不要加 markdown 代码块。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请优化以下 System Prompt：\n\n{raw_prompt}"}
    ]
    
    return chat_completion(messages, model=DEEPSEEK_MODEL_FLASH, temperature=0.8)


def critique_prompt(system_prompt: str) -> dict:
    """
    用 LLM 对 System Prompt 做 Self-Critique，评估质量并给出改进建议。
    返回 JSON：{"score": 1-10, "issues": [...], "suggestions": [...]}
    使用 deepseek-v4-flash 模型（快速且足够好）。
    """
    system_prompt_text = """你是一位严格的 prompt quality reviewer，专门评估虚拟用户的 System Prompt 质量。

请从以下维度评估：
1. **真实性**：角色是否像真人？有没有 generic/模板感？
2. **严格性**：LLM 会不会太配合？约束够不够强？
3. **一致性**：角色属性之间有没有矛盾？
4. **场景感**：角色是否清楚自己在什么场景下？
5. **可操作性**：这个 prompt 直接给 LLM 用，效果会好吗？

输出 JSON 格式：
{
  "score": 1-10 的整数,
  "issues": ["问题1", "问题2", ...],
  "suggestions": ["建议1", "建议2", ...],
  "strengths": ["优点1", "优点2", ...]
}"""

    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": f"请评估以下 System Prompt 的质量：\n\n{system_prompt}"}
    ]
    
    return chat_completion_json(messages, model=DEEPSEEK_MODEL_FLASH, temperature=0.3)
