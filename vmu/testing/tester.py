"""
测试引擎核心

管理测试会话，协调被测 agent 和虚拟人之间的交互。
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ..manager import PersonaManager
from ..agent import PersonaAgent
from ..models import PersonaInstance, Scene, Message


@dataclass
class TestTurn:
    """单轮交互记录"""
    round_num: int
    agent_message: str       # 被测 agent 发给虚拟人的话
    user_response: str       # 虚拟人回复
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSession:
    """一次测试会话的完整状态"""
    session_id: str
    agent_name: str
    persona_instance: PersonaInstance
    turns: List[TestTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "running"  # running / completed / error
    error_message: Optional[str] = None
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "persona": {
                "instance_id": self.persona_instance.instance_id,
                "name": self.persona_instance.name,
                "type_id": self.persona_instance.type_id,
            },
            "turns": [
                {
                    "round": t.round_num,
                    "agent": t.agent_message,
                    "user": t.user_response,
                    "metadata": t.metadata,
                }
                for t in self.turns
            ],
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "error_message": self.error_message,
            "summary": self.summary,
            "total_turns": len(self.turns),
        }


class DummyUserTester:
    """
    虚拟人测试器：用虚拟人测试外部 Agent 的核心引擎。

    使用示例：
        tester = DummyUserTester()

        # 测试单个 agent
        result = tester.test_agent(
            agent=my_agent_function,
            persona_type="anxious_buyer",
            opening="你好，我是销售小李，想给您介绍一下我们的产品...",
            rounds=5,
        )

        # 测试场景中的多个虚拟人
        result = tester.test_scene(
            agent=my_agent_function,
            scene_id="scene_xxx",
            opening="大家好，欢迎参加今天的产品演示...",
            rounds=3,
        )
    """

    def __init__(
        self,
        manager: Optional[PersonaManager] = None,
        llm_client: Optional[Callable] = None,
    ):
        self.manager = manager or PersonaManager()
        self.llm_client = llm_client
        self._sessions: Dict[str, TestSession] = {}

        # 如果未传入 llm_client，尝试自动导入
        if self.llm_client is None:
            try:
                from deepseek_client import chat_completion
                self.llm_client = chat_completion
            except ImportError:
                pass

    # ═══════════════════════════════════════════════
    # 核心：单虚拟人测试
    # ═══════════════════════════════════════════════

    def test_agent(
        self,
        agent: Union[Callable[[str], str], "AgentAdapter"],
        persona_type: str,
        name: Optional[str] = None,
        opening: Optional[str] = None,
        rounds: int = 5,
        auto_continue: bool = False,
        temperature: float = 0.7,
        scene_overrides: Optional[Dict[str, Any]] = None,
    ) -> TestSession:
        """
        用单个虚拟人测试一个 agent。

        Args:
            agent: 被测 agent，可以是函数或 AgentAdapter
            persona_type: 使用的虚拟人类型 ID（如 "anxious_buyer"）
            name: 虚拟人实例名称（可选）
            opening: agent 的开场白（可选，默认由虚拟人先开口）
            rounds: 对话轮数
            auto_continue: 是否让虚拟人自动追问（如果 agent 回复太简短）
            temperature: LLM 温度
            scene_overrides: 场景覆盖参数

        Returns:
            TestSession 包含完整对话记录
        """
        # 确保预设类型存在
        types = self.manager.list_types()
        if not types:
            self.manager.create_preset_types()

        # 实例化虚拟人
        instance = self.manager.instantiate(
            type_id=persona_type,
            name=name,
            scene_overrides=scene_overrides,
        )
        if not instance:
            raise ValueError(f"PersonaType '{persona_type}' 不存在，请先创建")

        # 标准化 agent
        sender = self._normalize_agent(agent)

        # 创建会话
        session = TestSession(
            session_id=f"test_{uuid.uuid4().hex[:8]}",
            agent_name=getattr(agent, "__name__", getattr(agent, "name", "unknown_agent")),
            persona_instance=instance,
        )
        self._sessions[session.session_id] = session

        # 创建虚拟人 Agent
        if self.llm_client is None:
            raise RuntimeError(
                "DummyUserTester 需要 llm_client 才能运行测试。"
                "请传入 chat_completion 函数，或配置 DEEPSEEK_API_KEY/DASHSCOPE_API_KEY"
            )

        persona_agent = PersonaAgent(
            instance=instance,
            llm_client=self.llm_client,
            auto_persist=True,
            storage=self.manager.storage,
        )

        # 执行对话轮次
        current_message = opening

        for i in range(rounds):
            try:
                if current_message is None:
                    # 如果没有消息，让虚拟人主动开口
                    current_message = "（用户没有说话，等待你开口）"

                # 虚拟人回复
                result = persona_agent.interact(
                    current_message,
                    include_history=True,
                    temperature=temperature,
                )
                user_response = result.response

                # 记录
                turn = TestTurn(
                    round_num=i + 1,
                    agent_message=current_message,
                    user_response=user_response,
                    metadata={
                        "trust_level": instance.memory.trust_level,
                        "emotional_state": instance.memory.emotional_state,
                    },
                )
                session.turns.append(turn)

                # 被测 agent 回复虚拟人
                agent_reply = sender(user_response)

                # 检查是否需要自动追问
                if auto_continue and len(agent_reply.strip()) < 10:
                    # agent 回复太短，虚拟人会继续追问
                    current_message = agent_reply + "\n\n（你觉得呢？有什么想法？）"
                else:
                    current_message = agent_reply

            except Exception as e:
                session.status = "error"
                session.error_message = str(e)
                break

        session.status = "completed"
        return session

    # ═══════════════════════════════════════════════
    # 核心：场景测试（多虚拟人）
    # ═══════════════════════════════════════════════

    def test_scene(
        self,
        agent: Union[Callable[[str], str], "AgentAdapter"],
        scene_id: str,
        opening: Optional[str] = None,
        rounds: int = 3,
        temperature: float = 0.7,
        mode: str = "sequential",  # sequential / broadcast
    ) -> List[TestSession]:
        """
        在场景中测试 agent，与多个虚拟人分别对话。

        Args:
            agent: 被测 agent
            scene_id: 场景 ID
            opening: 开场白
            rounds: 每个虚拟人的对话轮数
            temperature: LLM 温度
            mode: sequential=逐个测试, broadcast=同时发消息给所有人

        Returns:
            List[TestSession] 每个虚拟人一个会话
        """
        scene = self.manager.get_scene(scene_id)
        if not scene:
            raise ValueError(f"Scene '{scene_id}' 不存在")

        participants = self.manager.get_scene_instances(scene_id)
        if not participants:
            # 尝试实例化
            self.manager.instantiate_scene(scene_id)
            participants = self.manager.get_scene_instances(scene_id)

        if not participants:
            raise ValueError(f"场景 '{scene_id}' 没有可测试的虚拟人")

        sessions = []
        sender = self._normalize_agent(agent)

        for participant in participants:
            session = self._run_single_participant(
                sender=sender,
                participant=participant,
                opening=opening,
                rounds=rounds,
                temperature=temperature,
            )
            sessions.append(session)

        return sessions

    def test_group_chat(
        self,
        agent: Union[Callable[[str], str], "AgentAdapter"],
        scene_id: str,
        opening: str,
        rounds: int = 3,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        群聊模式测试：agent 面对一群虚拟人，进行多轮群聊。

        使用 GroupChatEngine 来实现真正的群聊动态。
        """
        from ..agent import GroupChatEngine

        scene = self.manager.get_scene(scene_id)
        if not scene:
            raise ValueError(f"Scene '{scene_id}' 不存在")

        participants = self.manager.get_scene_instances(scene_id)
        if not participants:
            self.manager.instantiate_scene(scene_id)
            participants = self.manager.get_scene_instances(scene_id)

        if not self.llm_client:
            raise RuntimeError("test_group_chat 需要 llm_client")

        sender = self._normalize_agent(agent)
        engine = GroupChatEngine(llm_client=self.llm_client)

        all_turns = []
        current_message = opening

        for r in range(rounds):
            # agent 发消息给群组
            # 群聊引擎处理所有虚拟人的回复
            group_turns = engine.run_turn(
                scene_id=scene_id,
                participants=participants,
                user_message=current_message,
                temperature=temperature,
                storage=self.manager.storage,
            )

            all_turns.append({
                "round": r + 1,
                "agent_message": current_message,
                "group_replies": group_turns,
            })

            # 构建 agent 的「观察」—— 把群聊内容发给 agent，让它决定下一条消息
            chat_summary = self._summarize_group_turn(group_turns)
            try:
                agent_next = sender(chat_summary)
            except Exception as e:
                agent_next = f"[Agent 调用失败: {e}]"

            current_message = agent_next

        return {
            "scene_id": scene_id,
            "rounds": rounds,
            "turns": all_turns,
            "participants": [
                {"id": p.instance_id, "name": p.name, "type": p.type_id}
                for p in participants
            ],
        }

    # ═══════════════════════════════════════════════
    # 交互式测试（HTTP API 场景）
    # ═══════════════════════════════════════════════

    def create_session(
        self,
        agent: Union[Callable[[str], str], "AgentAdapter"],
        persona_type: str,
        name: Optional[str] = None,
        scene_overrides: Optional[Dict[str, Any]] = None,
    ) -> TestSession:
        """
        创建一个交互式测试会话（用于 HTTP API 模式）。
        返回 session，后续通过 send_message / get_response 交互。
        """
        types = self.manager.list_types()
        if not types:
            self.manager.create_preset_types()

        instance = self.manager.instantiate(
            type_id=persona_type,
            name=name,
            scene_overrides=scene_overrides,
        )
        if not instance:
            raise ValueError(f"PersonaType '{persona_type}' 不存在")

        session = TestSession(
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            agent_name=getattr(agent, "__name__", getattr(agent, "name", "unknown_agent")),
            persona_instance=instance,
        )
        self._sessions[session.session_id] = session
        return session

    def send_to_session(
        self,
        session_id: str,
        agent_message: str,
        temperature: float = 0.7,
    ) -> str:
        """
        向指定测试会话发送一条 agent 消息，返回虚拟人的回复。
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' 不存在")

        if self.llm_client is None:
            raise RuntimeError("需要 llm_client")

        persona_agent = PersonaAgent(
            instance=session.persona_instance,
            llm_client=self.llm_client,
            auto_persist=True,
            storage=self.manager.storage,
        )

        result = persona_agent.interact(
            agent_message,
            include_history=True,
            temperature=temperature,
        )

        turn = TestTurn(
            round_num=len(session.turns) + 1,
            agent_message=agent_message,
            user_response=result.response,
            metadata={
                "trust_level": session.persona_instance.memory.trust_level,
                "emotional_state": session.persona_instance.memory.emotional_state,
            },
        )
        session.turns.append(turn)
        return result.response

    async def asend_to_session(
        self,
        session_id: str,
        agent_message: str,
        temperature: float = 0.7,
    ) -> str:
        """send_to_session 的异步版本，使用异步 LLM 客户端。"""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' 不存在")

        if self.llm_client is None:
            raise RuntimeError("需要 llm_client")

        persona_agent = PersonaAgent(
            instance=session.persona_instance,
            llm_client=self.llm_client,
            auto_persist=True,
            storage=self.manager.storage,
        )

        result = await persona_agent.ainteract(
            agent_message,
            include_history=True,
            temperature=temperature,
        )

        turn = TestTurn(
            round_num=len(session.turns) + 1,
            agent_message=agent_message,
            user_response=result.response,
            metadata={
                "trust_level": session.persona_instance.memory.trust_level,
                "emotional_state": session.persona_instance.memory.emotional_state,
            },
        )
        session.turns.append(turn)
        return result.response

    def get_session(self, session_id: str) -> Optional[TestSession]:
        """获取会话状态"""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[TestSession]:
        """列出所有活跃会话"""
        return list(self._sessions.values())

    def close_session(self, session_id: str) -> bool:
        """关闭并清理会话"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.status = "completed"
            return True
        return False

    # ═══════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════

    def _normalize_agent(self, agent) -> Callable[[str], str]:
        """将被测 agent 标准化为 send(message) -> response 函数"""
        # 如果已经是 AgentAdapter
        if hasattr(agent, "send") and callable(agent.send):
            return agent.send

        # 如果是普通函数
        if callable(agent):
            return agent

        # 如果是有方法的对象
        return AgentAdapter.from_object(agent).send

    def _run_single_participant(
        self,
        sender: Callable[[str], str],
        participant: PersonaInstance,
        opening: Optional[str],
        rounds: int,
        temperature: float,
    ) -> TestSession:
        """对单个虚拟人执行测试"""
        session = TestSession(
            session_id=f"test_{uuid.uuid4().hex[:8]}",
            agent_name="agent",
            persona_instance=participant,
        )
        self._sessions[session.session_id] = session

        persona_agent = PersonaAgent(
            instance=participant,
            llm_client=self.llm_client,
            auto_persist=True,
            storage=self.manager.storage,
        )

        current_message = opening

        for i in range(rounds):
            try:
                if current_message is None:
                    current_message = "（用户没有说话）"

                result = persona_agent.interact(
                    current_message,
                    include_history=True,
                    temperature=temperature,
                )

                turn = TestTurn(
                    round_num=i + 1,
                    agent_message=current_message,
                    user_response=result.response,
                    metadata={
                        "trust_level": participant.memory.trust_level,
                        "emotional_state": participant.memory.emotional_state,
                    },
                )
                session.turns.append(turn)

                # agent 回复
                current_message = sender(result.response)

            except Exception as e:
                session.status = "error"
                session.error_message = str(e)
                break

        session.status = "completed"
        return session

    def _summarize_group_turn(self, group_turns: List[Dict[str, Any]]) -> str:
        """把群聊一轮的多个回复总结成一条发给 agent 的消息"""
        lines = ["群聊中大家的发言："]
        for turn in group_turns:
            if turn.get("decision") == "REPLY" and turn.get("reply"):
                lines.append(f"  [{turn['name']}]: {turn['reply']}")
        if len(lines) == 1:
            lines.append("  （没有人发言）")
        lines.append("\n请作为主持人/销售继续对话。")
        return "\n".join(lines)
