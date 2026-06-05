#!/usr/bin/env python3
"""Mock 测试：不调用 LLM，验证旅行场景代码逻辑。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vmu import PersonaManager
from vmu.models import SceneParticipant, Scene
from vmu.stage_agent import StageAwarePersonaAgent
from vmu.testing.scene_evaluator import SceneEvaluator
from vmu.testing.tester import TestSession, TestTurn


def mock_llm(messages, **kwargs):
    """Mock LLM：返回基于行为骨架的固定回复。"""
    system = messages[0]["content"] if messages else ""
    
    # 根据阶段返回不同回复
    if "问候" in system:
        return "你好，我想去泰国玩。"
    if "目的地" in system:
        return "对，就是想去泰国曼谷。"
    if "日期" in system:
        return "计划7月出发，玩5天。"
    if "预算" in system:
        return "预算大概5000元。"
    if "人数" in system:
        return "就我一个人去。"
    if "检查信息" in system:
        return "嗯，应该都说了吧。"
    if "展示方案" in system:
        return "方案一听起来不错，价格能再便宜点吗？"
    if "确认预订" in system:
        return "好的，确认预订。"
    return "好的，没问题。"


def test_travel_engine():
    """测试旅行行为引擎的阶段检测和骨架生成。"""
    from vmu.behavior_engine import UserBehaviorEngine, TRAVEL_STAGES
    from vmu.models import PersonaType, PersonaInstance

    # 创建 mock persona
    ptype = PersonaType(
        type_id="budget_traveler",
        name="预算型旅行者",
        description="test",
        demographics={"age": 26, "role": "test", "company_size": "N/A", "industry": "test", "location": "test", "years_experience": 1},
        psychographics={"goals": [], "frustrations": [], "decision_style": "谨慎", "tech_stack": [], "budget_authority": "self"},
        behavioral_traits={"communication": "直接", "skepticism_level": 0.5, "price_sensitivity": 0.9, "risk_tolerance": "低"},
    )
    instance = PersonaInstance(
        type_id="budget_traveler",
        name="预算型旅行者",
        demographics=ptype.demographics,
        psychographics=ptype.psychographics,
        behavioral_traits=ptype.behavioral_traits,
    )

    engine = UserBehaviorEngine(instance, domain="travel_booking")

    # 测试阶段检测
    assert engine.detect_stage("您好，欢迎咨询旅行！") == "greet"
    assert engine.detect_stage("请问您想去哪里旅行？") == "gather_destination"
    assert engine.detect_stage("计划什么时候出发？") == "gather_dates"
    assert engine.detect_stage("您的预算大概是多少？") == "gather_budget"
    assert engine.detect_stage("一共有几个人？") == "gather_travelers"
    assert engine.detect_stage("信息都齐了吗？") == "check_ready"
    assert engine.detect_stage("我推荐两个方案") == "present_options"
    assert engine.detect_stage("确认预订") == "confirm_booking"
    assert engine.detect_stage("算了，不订了") == "abandon"

    # 测试骨架生成
    skeleton = engine.build_user_skeleton("greet", "你好")
    assert "泰国" in skeleton or "自然" in skeleton

    skeleton = engine.build_user_skeleton("gather_destination", "去哪")
    assert "泰国" in skeleton

    # 测试指令生成
    instruction = engine.get_behavior_instruction("gather_budget", "多少钱")
    assert "当前预订阶段" in instruction
    assert "预算" in instruction
    assert "5000元" in instruction

    # 测试信息收集
    engine.update_collected("预算大概5000元")
    assert engine.collected.get("budget") == True

    # 测试统计
    stats = engine.get_stage_stats()
    assert stats["stage_coverage"] > 0
    print("✅ 旅行行为引擎测试通过")


def test_travel_scene():
    """测试完整旅行场景流程（mock LLM）。"""
    manager = PersonaManager()
    if not manager.list_types():
        manager.create_preset_types()

    # 加载旅行 persona
    import json
    for tt in ["budget_traveler", "luxury_seeker", "anxious_flyer", "spontaneous_explorer"]:
        if not manager.get_type(tt):
            json_path = Path(__file__).parent / "data" / "types" / f"{tt}.json"
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            from vmu.models import PersonaType
            manager.register_type(PersonaType(**data))

    instance = manager.instantiate(type_id="budget_traveler")
    agent = StageAwarePersonaAgent(
        instance=instance,
        llm_client=mock_llm,
        domain="travel_booking",
    )

    messages = [
        "您好，欢迎咨询旅行！",
        "请问您想去哪里？",
        "计划什么时候出发？",
        "预算多少？",
    ]

    session = TestSession(session_id="test", agent_name="MockAgent", persona_instance=instance)
    for i, msg in enumerate(messages):
        result = agent.interact(msg)
        turn = TestTurn(
            round_num=i+1,
            agent_message=msg,
            user_response=result.response,
            metadata=result.metadata,
        )
        session.turns.append(turn)
        print(f"  轮{i+1} [{result.metadata.get('current_stage')}] Agent: {msg[:30]}... → User: {result.response[:40]}...")

    stats = agent.get_stage_stats()
    print(f"\n  阶段历史: {' → '.join(stats['unique_stages'])}")
    print(f"  阶段覆盖率: {stats['stage_coverage']:.0%}")
    print(f"  已收集信息: {stats['collected']}")

    assert stats["stage_coverage"] > 0
    assert len(stats["stage_history"]) >= 3
    print("✅ 旅行场景流程测试通过")


def test_scene_evaluator():
    """测试场景评估器。"""
    from vmu.testing.scene_evaluator import SceneEvaluator
    from vmu.models import Scene, SceneParticipant, PersonaType, PersonaInstance

    scene = Scene(
        name="测试场景",
        description="test",
        scenario="test",
        expected_stages=["greet", "gather_destination", "gather_dates"],
        success_criteria={"min_overall_score": 5.0, "min_stage_coverage": 0.5, "min_trust_level": 0.0, "required_stages": ["greet"]},
        evaluation_config={"stage_coverage_weight": 0.35, "info_release_weight": 0.30, "trust_weight": 0.20, "consistency_weight": 0.15},
        participant_configs=[SceneParticipant(type_id="budget_traveler", count=1, expected_stages=["greet", "gather_destination"], min_info_release={"destination": True})],
    )

    evaluator = SceneEvaluator()
    
    # 创建一个有有效 persona_instance 的 session
    ptype = PersonaType(
        type_id="budget_traveler",
        name="预算型旅行者",
        description="test",
        demographics={"age": 26, "role": "test", "company_size": "N/A", "industry": "test", "location": "test", "years_experience": 1},
        psychographics={"goals": [], "frustrations": [], "decision_style": "谨慎", "tech_stack": [], "budget_authority": "self"},
        behavioral_traits={"communication": "直接", "skepticism_level": 0.5, "price_sensitivity": 0.9, "risk_tolerance": "低"},
    )
    instance = PersonaInstance(
        type_id="budget_traveler",
        name="预算型旅行者",
        demographics=ptype.demographics,
        psychographics=ptype.psychographics,
        behavioral_traits=ptype.behavioral_traits,
    )
    session = TestSession(session_id="test", agent_name="test", persona_instance=instance)
    
    result = evaluator.evaluate_session(session, scene, scene.participant_configs[0])
    print(f"✅ 场景评估器测试通过 — 得分: {result.overall_score}")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 旅行场景 Mock 测试")
    print("=" * 60)

    test_travel_engine()
    print()
    test_travel_scene()
    print()
    test_scene_evaluator()
    print()
    print("🎉 所有测试通过！")
