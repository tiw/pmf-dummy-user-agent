#!/usr/bin/env python3
"""
示例：用虚拟人测试外部 Agent

这个示例展示了如何用 pmf-dummy-user 的虚拟人来测试你自己的 agent。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vmu.testing import DummyUserTester, AgentAdapter
from vmu.testing.evaluator import ConversationEvaluator
from vmu.testing.report import TestReport


# ═══════════════════════════════════════════════
# 示例 1：测试一个函数型 Agent
# ═══════════════════════════════════════════════

def my_sales_agent(message: str) -> str:
    """
    这是一个简单的销售 agent 示例。
    在实际使用中，这里可以是你的真实 agent。
    """
    # 模拟一个还不太成熟的销售 agent
    responses = [
        "您好！感谢您对我们产品的关注。我们的项目管理工具可以帮助您提高效率。",
        "我们的核心功能包括任务分配、进度追踪和团队协作。",
        "价格方面，我们提供灵活的订阅方案，月费只要 99 元起。",
        "很多客户都在用，反馈非常好。您要不要先试用一下？",
        "如果您还有其他问题，随时联系我！",
    ]
    # 简单的轮询回复（实际 agent 应该是智能的）
    idx = min(hash(message) % len(responses), len(responses) - 1)
    return responses[idx]


def demo_test_function_agent():
    """演示：测试一个函数型 agent"""
    print("=" * 70)
    print("🧪 示例 1：测试函数型 Agent")
    print("=" * 70)

    # 初始化测试器
    tester = DummyUserTester()

    # 运行测试：用焦虑型买家测试销售 agent
    print("\n📋 测试配置：")
    print("   Agent: my_sales_agent")
    print("   虚拟人: anxious_buyer (焦虑型买家)")
    print("   轮数: 3")
    print("\n⏳ 开始测试...\n")

    try:
        session = tester.test_agent(
            agent=my_sales_agent,
            persona_type="anxious_buyer",
            opening="你好，我听说你们有个项目管理工具？",
            rounds=3,
        )

        print(f"✅ 测试完成！会话 ID: {session.session_id}")
        print(f"   总轮数: {len(session.turns)}")
        print(f"   最终信任度: {session.persona_instance.memory.trust_level:.2f}")
        print(f"   最终情绪: {session.persona_instance.memory.emotional_state}")

        # 打印对话记录
        print("\n📜 对话记录：")
        for turn in session.turns:
            print(f"\n  ── 第 {turn.round_num} 轮 ──")
            print(f"  Agent: {turn.agent_message}")
            print(f"  User:  {turn.user_response}")

        # 评估
        print("\n🔍 生成评估报告...")
        evaluator = ConversationEvaluator()
        evaluation = evaluator.evaluate(
            persona=session.persona_instance,
            turns=session.turns,
            agent_name="my_sales_agent",
        )

        print(f"\n📊 综合得分: {evaluation.get('overall_score', 'N/A')} / 10")
        dims = evaluation.get("dimensions", {})
        for dim_name, dim_data in dims.items():
            if isinstance(dim_data, dict):
                print(f"   {dim_name}: {dim_data.get('score', 'N/A')} - {dim_data.get('comment', '')}")

        if evaluation.get("issues"):
            print("\n⚠️  发现的问题:")
            for issue in evaluation["issues"][:3]:
                print(f"   • {issue}")

        if evaluation.get("suggestions"):
            print("\n💡 改进建议:")
            for sgt in evaluation["suggestions"][:3]:
                print(f"   • {sgt}")

        # 生成完整报告
        report = TestReport(session=session, evaluation=evaluation)
        print("\n📄 Markdown 报告预览（前 500 字符）：")
        print(report.to_markdown()[:500] + "...")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════
# 示例 2：测试 HTTP API 型 Agent
# ═══════════════════════════════════════════════

def demo_test_http_agent():
    """演示：测试一个 HTTP API 型 agent"""
    print("\n" + "=" * 70)
    print("🌐 示例 2：测试 HTTP API 型 Agent")
    print("=" * 70)

    # 这里假设你有一个运行中的 agent 服务
    # 实际使用时，替换为你的 agent URL
    agent_url = "http://localhost:3000/chat"

    print(f"\n📋 假设 agent 运行在: {agent_url}")
    print("   （如果没有运行，此示例会显示连接错误，这是正常的）")

    try:
        # 创建 HTTP 适配器
        adapter = AgentAdapter.http(
            url=agent_url,
            message_field="message",
            response_field="response",
        )

        tester = DummyUserTester()

        session = tester.test_agent(
            agent=adapter,
            persona_type="rational_analyst",
            opening="你好，我想了解一下你们产品的数据导出功能。",
            rounds=2,
        )

        print(f"\n✅ HTTP Agent 测试完成！")
        print(f"   总轮数: {len(session.turns)}")

    except Exception as e:
        print(f"\n⚠️  HTTP Agent 测试遇到问题（预期行为，因为服务未运行）：")
        print(f"   错误: {e}")
        print(f"\n💡 提示：先启动你的 agent 服务，然后修改 agent_url 再运行")


# ═══════════════════════════════════════════════
# 示例 3：交互式测试（HTTP API 模式）
# ═══════════════════════════════════════════════

def demo_interactive_testing():
    """演示：交互式测试，模拟外部 agent 逐步与虚拟人对话"""
    print("\n" + "=" * 70)
    print("🎮 示例 3：交互式测试（模拟外部 Agent 逐步调用）")
    print("=" * 70)

    tester = DummyUserTester()

    # 创建一个测试会话
    session = tester.create_session(
        agent=lambda msg: msg,  # 占位
        persona_type="tech_skeptic",
    )

    print(f"\n📋 创建测试会话: {session.session_id}")
    print(f"   虚拟人: {session.persona_instance.name} ({session.persona_instance.type_id})")

    # 模拟外部 agent 的回复
    agent_replies = [
        "您好！我是技术支持小李，有什么可以帮您的？",
        "我们的系统采用微服务架构，支持水平扩展，已经服务了 500+ 企业客户。",
        "关于数据安全，我们有 SOC2 认证，数据加密传输和存储。",
    ]

    print("\n🔄 开始交互式对话：")
    for i, agent_msg in enumerate(agent_replies):
        print(f"\n  ── 第 {i+1} 轮 ──")
        print(f"  Agent → 虚拟人: {agent_msg}")

        # 外部 agent 发送消息给虚拟人
        user_response = tester.send_to_session(
            session_id=session.session_id,
            agent_message=agent_msg,
        )

        print(f"  虚拟人 → Agent: {user_response}")

    # 获取完整会话状态
    final_session = tester.get_session(session.session_id)
    print(f"\n✅ 交互完成！")
    print(f"   总轮数: {len(final_session.turns)}")
    print(f"   最终信任度: {final_session.persona_instance.memory.trust_level:.2f}")


# ═══════════════════════════════════════════════
# 示例 4：场景测试（多个虚拟人）
# ═══════════════════════════════════════════════

def demo_scene_testing():
    """演示：在场景中测试 agent 面对多个虚拟人"""
    print("\n" + "=" * 70)
    print("🎭 示例 4：场景测试（多个虚拟人）")
    print("=" * 70)

    from vmu import PersonaManager
    from vmu.models import SceneParticipant

    manager = PersonaManager()

    # 确保有预设类型
    if not manager.list_types():
        manager.create_preset_types()

    # 创建一个测试场景
    scene = manager.create_scene(
        name="Agent 压力测试",
        description="同时面对不同类型的用户",
        scenario="一个产品演示会，有不同背景的用户",
        participant_configs=[
            SceneParticipant(type_id="anxious_buyer", count=1),
            SceneParticipant(type_id="rational_analyst", count=1),
            SceneParticipant(type_id="tech_skeptic", count=1),
        ],
    )

    print(f"\n📋 创建测试场景: {scene.name} ({scene.scene_id})")

    # 实例化场景
    manager.instantiate_scene(scene.scene_id)
    participants = manager.get_scene_instances(scene.scene_id)
    print(f"   场景参与者: {len(participants)} 人")
    for p in participants:
        print(f"   - {p.name} ({p.type_id})")

    # 用测试器测试场景中每个虚拟人
    tester = DummyUserTester(manager=manager)

    def simple_agent(msg: str) -> str:
        return "感谢您的反馈，我们会认真考虑的。"

    print(f"\n⏳ 开始场景测试...")

    try:
        sessions = tester.test_scene(
            agent=simple_agent,
            scene_id=scene.scene_id,
            opening="大家好，欢迎参加今天的产品演示。",
            rounds=2,
        )

        print(f"\n✅ 场景测试完成！")
        for s in sessions:
            print(f"\n  ── {s.persona_instance.name} ({s.persona_instance.type_id}) ──")
            print(f"     轮数: {len(s.turns)}")
            print(f"     信任度: {s.persona_instance.memory.trust_level:.2f}")
            for t in s.turns:
                print(f"     轮{t.round_num}: {t.user_response[:60]}...")

    except Exception as e:
        print(f"❌ 场景测试失败: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     🤖 虚拟人测试外部 Agent —— 示例脚本                                 ║
║                                                                      ║
║  这个脚本演示了如何用 pmf-dummy-user 的虚拟人来测试你的 agent。          ║
║  你可以复制这些代码，修改为自己的 agent 进行测试。                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # 运行示例
    demo_test_function_agent()
    demo_test_http_agent()
    demo_interactive_testing()
    demo_scene_testing()

    print("\n" + "=" * 70)
    print("✅ 所有示例演示完成！")
    print("=" * 70)
    print("""
💡 下一步：
   1. 修改 my_sales_agent 函数为你的真实 agent
   2. 或者使用 AgentAdapter.http() 连接你的 HTTP 服务
   3. 运行测试，查看评估报告
   4. 根据建议优化你的 agent

📚 API 文档：
   启动 web_server.py 后访问 http://localhost:8000
   测试 API 端点：/api/v1/testing/*
""")
