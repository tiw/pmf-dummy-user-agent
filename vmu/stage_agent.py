"""
阶段感知的虚拟人 Agent。

继承 PersonaAgent，在销售对话中注入阶段感知的行为指导。
"""

from typing import Any, Dict, List, Optional

from .agent import PersonaAgent
from .behavior_engine import UserBehaviorEngine
from .models import InteractionResult, MemoryState, Message


class StageAwarePersonaAgent(PersonaAgent):
    """
    阶段感知虚拟人 Agent。

    与标准 PersonaAgent 的区别：
    1. 每次交互前检测销售人员当前所处的销售阶段
    2. 根据阶段 + persona 性格生成行为骨架
    3. 将行为指导注入 system prompt
    4. 跟踪已透露的信息，逐步释放私有信息
    """

    def __init__(
        self,
        instance,
        llm_client=None,
        system_prompt_override: Optional[str] = None,
        memory_update_callback=None,
        auto_persist: bool = False,
        storage=None,
    ):
        super().__init__(
            instance=instance,
            llm_client=llm_client,
            system_prompt_override=system_prompt_override,
            memory_update_callback=memory_update_callback,
            auto_persist=auto_persist,
            storage=storage,
        )
        self.behavior_engine = UserBehaviorEngine(instance)

    def _build_messages(self, user_input: str, include_history: bool = True) -> List[Dict[str, str]]:
        """
        构建发送给 LLM 的 messages，注入阶段感知的行为指导。
        """
        # 检测当前销售阶段（user_input 是销售人员的消息）
        stage = self.behavior_engine.detect_stage(user_input)

        # 生成行为指导
        behavior_instruction = self.behavior_engine.get_behavior_instruction(stage, user_input)

        # 基础 system prompt
        base_system = self.system_prompt

        # 注入行为指导
        enhanced_system = base_system + "\n" + behavior_instruction

        messages = [{"role": "system", "content": enhanced_system}]

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
        与虚拟人进行一次阶段感知的交互。
        """
        # 记录销售人员输入
        self.instance.message_history.append(
            Message(role="user", content=user_input)
        )

        # 构建 messages（已注入阶段感知指导）
        messages = self._build_messages(user_input, include_history)

        # 调用 LLM
        response_text = self.llm_client(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **llm_kwargs
        )

        # 记录虚拟人回复
        self.instance.message_history.append(
            Message(role="assistant", content=response_text)
        )

        # 更新已透露的信息
        self.behavior_engine.update_collected(response_text)

        # 记忆更新（保留原有逻辑 + 阶段信息）
        updated_memory = self._update_memory(user_input, response_text)

        # 在 metadata 中加入阶段信息
        stage_stats = self.behavior_engine.get_stage_stats()

        if self.memory_update_callback:
            self.memory_update_callback(self.instance, updated_memory)

        if self.auto_persist and self.storage:
            self.storage.save("instances", self.instance.instance_id, self.instance.model_dump())

        return InteractionResult(
            instance_id=self.instance.instance_id,
            response=response_text,
            updated_memory=updated_memory,
            metadata={
                "history_length": len(self.instance.message_history),
                "temperature": temperature,
                "current_stage": stage_stats["current_stage"],
                "stage_history": stage_stats["stage_history"],
                "collected": stage_stats["collected"],
                "stage_coverage": stage_stats["stage_coverage"],
            }
        )

    def get_stage_stats(self) -> Dict[str, Any]:
        """获取当前阶段统计"""
        return self.behavior_engine.get_stage_stats()
