"""
VMU Testing —— 用虚拟人测试外部 Agent

让 pmf-dummy-user 成为其他 agent 的「测试用户」，
提供标准化的测试基础设施。

使用方式：
    from vmu.testing import DummyUserTester, AgentAdapter

    # 方式1：测试一个函数
    def my_agent(message: str) -> str:
        return f"收到: {message}"

    tester = DummyUserTester()
    result = tester.test_agent(
        agent=my_agent,
        persona_type="anxious_buyer",
        rounds=3,
    )
    print(result.report)

    # 方式2：测试一个 HTTP 服务
    adapter = AgentAdapter.http("http://localhost:3000/chat")
    result = tester.test_agent(agent=adapter, ...)
"""

from .adapters import AgentAdapter
from .tester import DummyUserTester, TestSession
from .evaluator import ConversationEvaluator
from .report import TestReport

__all__ = [
    "AgentAdapter",
    "DummyUserTester",
    "TestSession",
    "ConversationEvaluator",
    "TestReport",
]
