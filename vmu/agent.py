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
