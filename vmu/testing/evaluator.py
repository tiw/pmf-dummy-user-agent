"""
对话评估器

用 LLM 评估被测 agent 与虚拟人的对话质量。
"""

import json
from typing import Any, Callable, Dict, List, Optional

from ..models import PersonaInstance
from .tester import TestTurn


class ConversationEvaluator:
    """
    用 LLM 评估 agent 与虚拟人的对话质量。

    评估维度：
    - 角色一致性：虚拟人是否始终符合设定
    - 对话流畅度：对话是否自然流畅
    - Agent 表现：被测 agent 是否专业、有效
    - 需求满足度：虚拟人的需求是否被满足
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        self.llm_client = llm_client
        if self.llm_client is None:
            try:
                from deepseek_client import chat_completion
                self.llm_client = chat_completion
            except ImportError:
                pass

    def evaluate(
        self,
        persona: PersonaInstance,
        turns: List[TestTurn],
        agent_name: str = "agent",
    ) -> Dict[str, Any]:
        """
        评估一次完整对话。

        Returns:
            {
                "overall_score": 1-10,
                "dimensions": {
                    "persona_consistency": {"score": ..., "comment": ...},
                    "conversation_flow": {"score": ..., "comment": ...},
                    "agent_effectiveness": {"score": ..., "comment": ...},
                    "need_fulfillment": {"score": ..., "comment": ...},
                },
                "highlights": ["亮点1", ...],
                "issues": ["问题1", ...],
                "suggestions": ["建议1", ...],
            }
        """
        if not self.llm_client:
            return self._fallback_evaluation(persona, turns)

        # 构建对话文本
        conversation_text = self._format_conversation(turns)

        # 构建 persona 描述
        persona_desc = self._format_persona(persona)

        prompt = f"""你是一位专业的对话质量评估专家。请评估以下 Agent 与虚拟用户的对话。

## 虚拟用户设定
{persona_desc}

## 对话记录
{conversation_text}

## 评估要求

请从以下四个维度评估（每项 1-10 分）：

1. **角色一致性 (persona_consistency)**：虚拟用户是否始终符合其设定？有没有突然变成客服或助手的情况？
2. **对话流畅度 (conversation_flow)**：对话是否自然？有没有逻辑断裂、重复、或尴尬的地方？
3. **Agent 有效性 (agent_effectiveness)**：被测 Agent 是否专业？有没有理解用户需求？回应是否有效？
4. **需求满足度 (need_fulfillment)**：虚拟用户的需求有没有被满足？Agent 是否解决了用户的问题？

同时请给出：
- 亮点（highlights）
- 问题（issues）
- 改进建议（suggestions）

请严格按以下 JSON 格式输出，不要包含 markdown 代码块：
{{
  "overall_score": 0-10 的整数,
  "dimensions": {{
    "persona_consistency": {{"score": 0-10, "comment": "具体评价"}},
    "conversation_flow": {{"score": 0-10, "comment": "具体评价"}},
    "agent_effectiveness": {{"score": 0-10, "comment": "具体评价"}},
    "need_fulfillment": {{"score": 0-10, "comment": "具体评价"}}
  }},
  "highlights": ["..."],
  "issues": ["..."],
  "suggestions": ["..."]
}}"""

        try:
            response = self.llm_client(
                messages=[
                    {"role": "system", "content": "你是一位专业的对话质量评估专家，擅长分析人机对话。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return self._parse_evaluation(response)
        except Exception as e:
            return {
                "overall_score": 0,
                "error": str(e),
                "dimensions": {},
                "highlights": [],
                "issues": [f"评估失败: {e}"],
                "suggestions": [],
            }

    async def aevaluate(
        self,
        persona: PersonaInstance,
        turns: List[TestTurn],
        agent_name: str = "agent",
    ) -> Dict[str, Any]:
        """evaluate 的异步版本，使用异步 LLM 客户端。"""
        if not self.llm_client:
            return self._fallback_evaluation(persona, turns)

        conversation_text = self._format_conversation(turns)
        persona_desc = self._format_persona(persona)

        prompt = f"""你是一位专业的对话质量评估专家。请评估以下 Agent 与虚拟用户的对话。

## 虚拟用户设定
{persona_desc}

## 对话记录
{conversation_text}

## 评估要求

请从以下四个维度评估（每项 1-10 分）：

1. **角色一致性 (persona_consistency)**：虚拟用户是否始终符合其设定？有没有突然变成客服或助手的情况？
2. **对话流畅度 (conversation_flow)**：对话是否自然？有没有逻辑断裂、重复、或尴尬的地方？
3. **Agent 有效性 (agent_effectiveness)**：被测 Agent 是否专业？有没有理解用户需求？回应是否有效？
4. **需求满足度 (need_fulfillment)**：虚拟用户的需求有没有被满足？Agent 是否解决了用户的问题？

同时请给出：
- 亮点（highlights）
- 问题（issues）
- 改进建议（suggestions）

请严格按以下 JSON 格式输出，不要包含 markdown 代码块：
{{
  "overall_score": 0-10 的整数,
  "dimensions": {{
    "persona_consistency": {{"score": 0-10, "comment": "具体评价"}},
    "conversation_flow": {{"score": 0-10, "comment": "具体评价"}},
    "agent_effectiveness": {{"score": 0-10, "comment": "具体评价"}},
    "need_fulfillment": {{"score": 0-10, "comment": "具体评价"}}
  }},
  "highlights": ["..."],
  "issues": ["..."],
  "suggestions": ["..."]
}}"""

        try:
            response = await self.llm_client(
                messages=[
                    {"role": "system", "content": "你是一位专业的对话质量评估专家，擅长分析人机对话。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return self._parse_evaluation(response)
        except Exception as e:
            return {
                "overall_score": 0,
                "error": str(e),
                "dimensions": {},
                "highlights": [],
                "issues": [f"评估失败: {e}"],
                "suggestions": [],
            }

    def quick_score(self, turns: List[TestTurn]) -> float:
        """
        快速评分：基于对话长度、虚拟人记忆变化等简单指标。
        不需要 LLM，适合实时反馈。
        """
        if not turns:
            return 0.0

        scores = []

        # 1. 对话轮数得分（更多轮数通常表示更好的互动）
        scores.append(min(len(turns) / 5.0, 1.0) * 2.5)

        # 2. 回复长度得分（双方都有实质性回复）
        avg_agent_len = sum(len(t.agent_message) for t in turns) / len(turns)
        avg_user_len = sum(len(t.user_response) for t in turns) / len(turns)
        length_score = min((avg_agent_len + avg_user_len) / 200.0, 1.0) * 2.5
        scores.append(length_score)

        # 3. 信任度变化（如果有的话）
        trust_scores = []
        for t in turns:
            trust = t.metadata.get("trust_level")
            if trust is not None:
                trust_scores.append(trust)
        if trust_scores and len(trust_scores) > 1:
            trust_change = trust_scores[-1] - trust_scores[0]
            # 信任度上升是积极的（但如果起点很高，保持也行）
            trust_score = min(max(trust_change + 0.3, 0), 1.0) * 2.5
            scores.append(trust_score)
        else:
            scores.append(1.25)

        # 4. 情绪多样性（对话中有情绪变化表示互动有效）
        emotions = set()
        for t in turns:
            emotion = t.metadata.get("emotional_state")
            if emotion:
                emotions.add(emotion)
        emotion_score = min(len(emotions) / 3.0, 1.0) * 2.5
        scores.append(emotion_score)

        return round(sum(scores), 1)

    # ═══════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════

    def _format_conversation(self, turns: List[TestTurn]) -> str:
        lines = []
        for t in turns:
            lines.append(f"Agent: {t.agent_message}")
            lines.append(f"User:  {t.user_response}")
            lines.append("")
        return "\n".join(lines)

    def _format_persona(self, persona: PersonaInstance) -> str:
        d = persona.demographics
        bt = persona.behavioral_traits
        ps = persona.psychographics
        lines = [
            f"- 姓名：{persona.name}",
            f"- 角色：{d.role}，{d.age} 岁，{d.industry}",
            f"- 沟通风格：{bt.communication}",
            f"- 怀疑度：{bt.skepticism_level}",
            f"- 价格敏感度：{bt.price_sensitivity}",
            f"- 决策风格：{ps.decision_style}",
            f"- 目标：{', '.join(ps.goals)}",
            f"- 痛点：{', '.join(ps.frustrations)}",
        ]
        return "\n".join(lines)

    def _parse_evaluation(self, text: str) -> Dict[str, Any]:
        """解析 LLM 返回的评估 JSON"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            # 确保字段完整
            return {
                "overall_score": data.get("overall_score", 0),
                "dimensions": data.get("dimensions", {}),
                "highlights": data.get("highlights", []),
                "issues": data.get("issues", []),
                "suggestions": data.get("suggestions", []),
            }
        except json.JSONDecodeError:
            return {
                "overall_score": 0,
                "raw_response": text,
                "dimensions": {},
                "highlights": [],
                "issues": ["评估 JSON 解析失败"],
                "suggestions": [],
            }

    def _fallback_evaluation(self, persona: PersonaInstance, turns: List[TestTurn]) -> Dict[str, Any]:
        """没有 LLM 时的降级评估"""
        quick = self.quick_score(turns)
        return {
            "overall_score": round(quick),
            "note": "基于启发式规则的快速评分（无 LLM 评估）",
            "dimensions": {
                "persona_consistency": {"score": round(quick), "comment": "无法深度评估，需 LLM"},
                "conversation_flow": {"score": round(quick), "comment": "无法深度评估，需 LLM"},
                "agent_effectiveness": {"score": round(quick), "comment": "无法深度评估，需 LLM"},
                "need_fulfillment": {"score": round(quick), "comment": "无法深度评估，需 LLM"},
            },
            "highlights": [],
            "issues": ["未配置 LLM，无法生成深度评估"],
            "suggestions": ["配置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY 以启用深度评估"],
        }
