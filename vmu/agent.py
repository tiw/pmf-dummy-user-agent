"""
LLM Agent 交互接口
封装与 LLM 的对话，管理 message history，支持单轮/多轮交互
"""

from typing import Any, Dict, List, Optional

from .models import InteractionResult, Message, MemoryState, PersonaInstance
from .prompts import PromptRenderer


class PersonaAgent:
    """
    虚拟人 Agent：将 PersonaInstance 包装为可与 LLM 交互的 Agent。
    
    使用方式：
        agent = PersonaAgent(instance, llm_client=chat_completion)
        result = agent.interact("你好，我想给你介绍我们的产品...")
        print(result.response)
    """
    
    def __init__(
        self,
        instance: PersonaInstance,
        llm_client=None,
        system_prompt_override: Optional[str] = None,
        memory_update_callback=None,
        auto_persist: bool = False,
        storage=None,
    ):
        """
        Args:
            instance: PersonaInstance 实例
            llm_client: LLM 调用函数，签名 fn(messages, **kwargs) -> str
            system_prompt_override: 覆盖默认 system prompt
            memory_update_callback: 记忆更新回调，签名 fn(instance, new_memory) -> None
            auto_persist: 是否在每次交互后自动持久化实例状态
            storage: 持久化存储（auto_persist=True 时需要）
        """
        self.instance = instance
        self.llm_client = llm_client
        self.memory_update_callback = memory_update_callback
        self.auto_persist = auto_persist
        self.storage = storage
        
        # 确定 system prompt
        if system_prompt_override:
            self.system_prompt = system_prompt_override
        elif instance.system_prompt:
            self.system_prompt = instance.system_prompt
        else:
            self.system_prompt = PromptRenderer.render_system_prompt(instance)
    
    def _build_messages(self, user_input: str, include_history: bool = True) -> List[Dict[str, str]]:
        """构建发送给 LLM 的 messages"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if include_history and self.instance.message_history:
            for msg in self.instance.message_history:
                messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": user_input})
        return messages
    
    def interact(
        self,
        user_input: str,
        include_history: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **llm_kwargs
    ) -> InteractionResult:
        """
        与虚拟人进行一次交互。
        
        Args:
            user_input: 外部输入（如销售话术、产品描述等）
            include_history: 是否携带历史对话
            temperature: LLM 温度
            max_tokens: 最大输出长度
            **llm_kwargs: 传递给 LLM 客户端的其他参数
        
        Returns:
            InteractionResult 包含回复内容和更新后的状态
        """
        if self.llm_client is None:
            raise RuntimeError(
                "PersonaAgent 未配置 llm_client。"
                "请传入一个 callable，如 deepseek_client.chat_completion"
            )
        
        # 记录用户输入
        self.instance.message_history.append(
            Message(role="user", content=user_input)
        )
        
        # 构建 messages 并调用 LLM
        messages = self._build_messages(user_input, include_history)
        response_text = self.llm_client(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **llm_kwargs
        )
        
        # 记录助手回复
        self.instance.message_history.append(
            Message(role="assistant", content=response_text)
        )
        
        # 简单记忆更新（接触次数 +1）
        updated_memory = self._update_memory(user_input, response_text)
        
        # 回调
        if self.memory_update_callback:
            self.memory_update_callback(self.instance, updated_memory)
        
        # 自动持久化
        if self.auto_persist and self.storage:
            self.storage.save("instances", self.instance.instance_id, self.instance.model_dump())
        
        return InteractionResult(
            instance_id=self.instance.instance_id,
            response=response_text,
            updated_memory=updated_memory,
            metadata={
                "history_length": len(self.instance.message_history),
                "temperature": temperature,
            }
        )
    
    def _update_memory(self, user_input: str, response: str) -> MemoryState:
        """根据交互内容更新记忆状态（简单版）"""
        mem = self.instance.memory
        new_exposure = mem.exposure_count + 1
        
        # 简单的信任度调整：根据对话长度和关键词粗略判断
        trust_delta = 0.0
        positive_words = ["好", "不错", "有用", "解决", "合适", "满意", "喜欢"]
        negative_words = ["不行", "没用", "贵", "复杂", "麻烦", "失望", "不"]
        
        for w in positive_words:
            if w in response:
                trust_delta += 0.02
        for w in negative_words:
            if w in response:
                trust_delta -= 0.02
        
        new_trust = max(0.0, min(1.0, mem.trust_level + trust_delta))
        
        # 情绪简单判断
        emotional_state = mem.emotional_state
        if trust_delta > 0.05:
            emotional_state = "interested"
        elif trust_delta < -0.05:
            emotional_state = "frustrated"
        
        new_memory = MemoryState(
            trust_level=round(new_trust, 3),
            emotional_state=emotional_state,
            exposure_count=new_exposure,
        )
        self.instance.memory = new_memory
        return new_memory
    
    def reset_history(self):
        """清空对话历史（保留 system prompt 和 memory）"""
        self.instance.message_history = []
    
    def get_history(self) -> List[Message]:
        """获取对话历史"""
        return list(self.instance.message_history)
    
    def export_conversation(self) -> str:
        """导出对话为文本格式"""
        lines = [f"# 对话记录：{self.instance.name} ({self.instance.instance_id})", ""]
        for msg in self.instance.message_history:
            role_label = {"user": "👤 用户", "assistant": "🤖 AI", "system": "⚙️ 系统"}.get(msg.role, msg.role)
            lines.append(f"**{role_label}**：{msg.content}")
            lines.append("")
        return "\n".join(lines)


# ───────────────────────────────────────────────
# 群聊交互引擎（串行思考模式）
# ───────────────────────────────────────────────

import random


def build_group_chat_prompt(
    instance,
    all_participants,
    chat_history,
    user_message,
):
    """
    为群聊构建注入到 system prompt 末尾的上下文指令。
    串行模式下，chat_history 包含之前参与者的回复。
    """
    participant_list = "\n".join(
        [f"- {p.name}（{p.demographics.role}，{p.type_id}）" for p in all_participants]
    )

    history_lines = []
    for entry in chat_history:
        speaker = entry.get("speaker", "某人")
        content = entry.get("content", "")
        history_lines.append(f"[{speaker}]: {content}")
    history_text = "\n".join(history_lines) if history_lines else "（暂无历史）"

    return f"""

=== 群聊上下文 ===
你现在在一个群聊中。群聊参与者包括：
{participant_list}

群聊历史：
{history_text}

现在 [用户] 发送了新消息："{user_message}"

作为 {instance.name}（{instance.demographics.role}），请先内心思考这条消息和你是否相关、你是否有强烈的观点需要表达、或者你是否需要回应其他人的发言，然后决定是否回复。

注意：
- 如果消息和你无关，你没有强烈观点，且不需要补充，请保持沉默（PASS）
- 如果消息触发了你的关注点，或者你想补充/反驳其他人的观点，请回复（REPLY）
- 回复要符合你的身份特征（怀疑度、沟通风格等）

请严格按以下格式回复（不要添加任何其他内容）：
THINKING: （你的内心思考过程，1-2句话）
DECISION: REPLY 或 PASS
CONTENT: （如果 DECISION 是 REPLY，写出你的发言；如果是 PASS，写"无"）
"""


def _parse_group_chat_response(text):
    """解析 LLM 返回的群聊决策格式"""
    text = text.strip()
    result = {"thinking": "", "decision": "PASS", "content": ""}

    # 提取 THINKING
    thinking_match = text.split("THINKING:")
    if len(thinking_match) > 1:
        thinking_part = thinking_match[1].split("DECISION:")[0] if "DECISION:" in thinking_match[1] else thinking_match[1]
        result["thinking"] = thinking_part.strip()

    # 提取 DECISION
    decision_match = text.split("DECISION:")
    if len(decision_match) > 1:
        decision_part = decision_match[1].split("CONTENT:")[0] if "CONTENT:" in decision_match[1] else decision_match[1]
        decision_val = decision_part.strip().upper()
        if "REPLY" in decision_val:
            result["decision"] = "REPLY"
        else:
            result["decision"] = "PASS"

    # 提取 CONTENT
    content_match = text.split("CONTENT:")
    if len(content_match) > 1:
        result["content"] = content_match[1].strip()

    return result


class GroupChatEngine:
    """
    群聊引擎（串行思考模式）。

    流程：
    1. 用户发送消息
    2. 随机排序参与者
    3. 逐个遍历：构建群聊上下文 -> LLM 决策 -> 如果 REPLY 则追加到共享历史
    4. 后面的参与者可以看到前面人的回复
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def run_turn(
        self,
        scene_id,
        participants,
        user_message,
        temperature=0.7,
        max_history=10,
        storage=None,
    ):
        """
        执行一轮群聊交互。

        Returns:
            List of turn dicts
        """
        if not self.llm_client:
            raise RuntimeError("GroupChatEngine 未配置 llm_client")

        # 随机排序参与者，增加不确定性
        shuffled = list(participants)
        random.shuffle(shuffled)

        # 群聊共享历史
        shared_history = self._build_shared_history(participants, max_history)

        turns = []
        # 记录本轮中已有回复（供后续参与者看到）
        round_replies = []

        for inst in shuffled:
            # 构建该参与者的完整上下文
            group_prompt = build_group_chat_prompt(
                instance=inst,
                all_participants=participants,
                chat_history=shared_history + round_replies,
                user_message=user_message,
            )

            # 构建 messages
            messages = [
                {"role": "system", "content": inst.system_prompt + group_prompt},
            ]

            # 发送给 LLM
            try:
                raw_response = self.llm_client(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=800,
                )
            except Exception as e:
                turns.append({
                    "instance_id": inst.instance_id,
                    "name": inst.name,
                    "type_id": inst.type_id,
                    "decision": "PASS",
                    "reasoning": f"LLM 调用失败: {str(e)}",
                    "reply": "",
                    "updated_memory": None,
                })
                continue

            # 解析回复
            parsed = _parse_group_chat_response(raw_response)
            decision = parsed["decision"]
            reasoning = parsed["thinking"]
            reply = parsed["content"] if decision == "REPLY" else ""

            # 清理 reply
            if reply.lower() in ("无", "none", "pass", ""):
                decision = "PASS"
                reply = ""

            # 如果决定回复，记录到 round_replies 供后续参与者看到
            if decision == "REPLY" and reply:
                round_replies.append({"speaker": inst.name, "content": reply})

                # 更新该实例的 message_history
                inst.message_history.append(Message(role="user", content=user_message))
                inst.message_history.append(Message(role="assistant", content=reply))

                # 更新记忆
                updated_memory = self._update_memory_simple(inst, user_message, reply)

                # 持久化
                if storage:
                    storage.save("instances", inst.instance_id, inst.model_dump())
            else:
                updated_memory = None

            turns.append({
                "instance_id": inst.instance_id,
                "name": inst.name,
                "type_id": inst.type_id,
                "decision": decision,
                "reasoning": reasoning,
                "reply": reply,
                "updated_memory": updated_memory.model_dump() if updated_memory else None,
            })

        return turns

    def _build_shared_history(self, participants, max_entries):
        """
        从所有参与者的 message_history 中提取最近的群聊历史。
        """
        all_entries = []
        for p in participants:
            for msg in p.message_history:
                if msg.role == "assistant":
                    all_entries.append({
                        "speaker": p.name,
                        "content": msg.content,
                        "ts": msg.timestamp,
                    })
                elif msg.role == "user":
                    all_entries.append({
                        "speaker": "用户",
                        "content": msg.content,
                        "ts": msg.timestamp,
                    })
        # 去重（相同 speaker+content），然后取最后 max_entries 条
        seen = set()
        unique = []
        for e in all_entries:
            key = (e["speaker"], e["content"])
            if key not in seen:
                seen.add(key)
                unique.append({"speaker": e["speaker"], "content": e["content"]})
        return unique[-max_entries:] if len(unique) > max_entries else unique

    def _update_memory_simple(self, inst, user_input, response):
        """简化版记忆更新"""
        mem = inst.memory
        new_exposure = mem.exposure_count + 1
        trust_delta = 0.0
        positive_words = ["好", "不错", "有用", "解决", "合适", "满意", "喜欢"]
        negative_words = ["不行", "没用", "贵", "复杂", "麻烦", "失望", "不"]
        for w in positive_words:
            if w in response:
                trust_delta += 0.02
        for w in negative_words:
            if w in response:
                trust_delta -= 0.02
        new_trust = max(0.0, min(1.0, mem.trust_level + trust_delta))
        emotional_state = mem.emotional_state
        if trust_delta > 0.05:
            emotional_state = "interested"
        elif trust_delta < -0.05:
            emotional_state = "frustrated"
        new_memory = MemoryState(
            trust_level=round(new_trust, 3),
            emotional_state=emotional_state,
            exposure_count=new_exposure,
        )
        inst.memory = new_memory
        return new_memory
