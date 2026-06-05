#!/usr/bin/env python3
"""
旅行预订场景化测试：定义场景预期标准，评估旅行客服 agent 表现。

演示了如何用 Scene + SceneEvaluator 做结构化 agent 评估（旅行预订 Domain）：
1. 定义场景（预期阶段、成功标准、评估配置）
2. 为不同 persona 设置行为覆盖参数
3. 运行测试，收集对话数据
4. 用 SceneEvaluator 生成结构化评估报告

用法:
    cd /Users/ting/work/pmf-dummy-user-agent
    python examples/test_travel_scene.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any, Dict, List

from vmu import PersonaManager
from vmu.models import SceneParticipant, Scene
from vmu.stage_agent import StageAwarePersonaAgent
from vmu.testing.scene_evaluator import SceneEvaluator
from vmu.testing.tester import TestSession, TestTurn


# ───────────────────────────────────────────────
# 模拟旅行客服 Agent
# ───────────────────────────────────────────────

class TravelAgent:
    """
    旅行客服 agent，按预订流程推进对话。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.round = 0
        self.stage_order = [
            "greet",
            "gather_destination",
            "gather_dates",
            "gather_budget",
            "gather_travelers",
            "check_ready",
            "present_options",
            "confirm_booking",
        ]
        self.used_scripts: Dict[str, set] = {s: set() for s in self.stage_order}

    def __call__(self, user_msg: str) -> str:
        import random
        rng = random.Random(self.seed + self.round)

        stage_idx = min(self.round, len(self.stage_order) - 1)
        stage = self.stage_order[stage_idx]

        lower = user_msg.lower()
        if any(w in lower for w in ["算了", "不需要", "不用", "没兴趣", "拒绝", "不聊了", "不订了", "取消"]):
            return "完全理解，那我就不打扰了。如果您以后有旅行需求，随时联系我！"

        if any(w in lower for w in ["确认", "定下来", "预订", "下单", "付款"]):
            stage = "confirm_booking"

        scripts = self.SCRIPTS.get(stage, ["好的，您还有什么想了解的吗？"])

        available = [i for i in range(len(scripts)) if i not in self.used_scripts[stage]]
        if not available:
            available = list(range(len(scripts)))
            self.used_scripts[stage].clear()

        idx = rng.choice(available)
        self.used_scripts[stage].add(idx)
        reply = scripts[idx]

        self.round += 1
        return reply

    SCRIPTS = {
        "greet": [
            "您好！我是旅行顾问小美，很高兴为您服务。请问您计划去哪里旅行呢？",
            "你好呀！欢迎来到环球旅行，我是您的专属顾问。今天想帮您规划一次怎样的旅程呢？",
        ],
        "gather_destination": [
            "好的！那请问您计划什么时候出发，玩几天呢？",
            "明白了。那您大概想什么时候去呢？是周末短途还是长假深度游？",
        ],
        "gather_dates": [
            "了解了。那您的预算是多少呢？这样我可以推荐合适的方案。",
            "好的，那请问您的预算范围大概是多少？人均预算也可以。",
        ],
        "gather_budget": [
            "了解了。那请问一共有几个人出行呢？有没有老人或小孩需要特别安排？",
            "好的。那请问出行人数是多少呢？同行人员的年龄情况方便说一下吗？",
        ],
        "gather_travelers": [
            "好的，信息都齐了，我帮您确认一下：目的地、时间、预算、人数都明确了，对吗？",
            "明白。那我汇总一下您的需求，您看看有没有遗漏或需要调整的地方？",
        ],
        "check_ready": [
            "好的！根据您的需求，我推荐两个方案：方案A是【5天4晚自由行，含机票+酒店】，方案B是【精品小团，全程导游】。您更倾向哪种？",
            "基于您的预算和需求，我为您筛选了：方案一【经济型，机票+青旅】；方案二【舒适型，机票+四星酒店】。您看看？",
        ],
        "present_options": [
            "没问题！那我现在帮您确认预订。请问您是用信用卡还是支付宝付款呢？",
            "好的，那我为您锁定这个方案。您确认后我需要收取20%定金，剩余款项出发前7天付清。",
        ],
        "confirm_booking": [
            "预订成功！您的订单号是 TRV2026001。行程单和注意事项我会发到您邮箱。祝您旅途愉快！",
            "太好了！已经为您确认预订。客服微信已添加，有任何问题随时联系。期待您的旅行！",
        ],
    }


# ───────────────────────────────────────────────
# 旅行场景定义
# ───────────────────────────────────────────────

def create_travel_scene() -> Scene:
    """创建一个标准的旅行预订测试场景。"""
    return Scene(
        name="旅行预订全流程测试",
        description="测试旅行客服 agent 是否能按标准流程与不同类型客户完成预订对话",
        scenario="一个旅行客服 agent 需要通过多轮对话，与客户建立信任、收集需求、推荐方案、完成预订",
        expected_stages=[
            "greet",
            "gather_destination",
            "gather_dates",
            "gather_budget",
            "gather_travelers",
            "check_ready",
            "present_options",
            "confirm_booking",
        ],
        success_criteria={
            "min_overall_score": 5.0,
            "min_stage_coverage": 0.5,
            "min_trust_level": 0.2,
            "required_stages": ["gather_destination", "present_options"],
        },
        evaluation_config={
            "stage_coverage_weight": 0.35,
            "info_release_weight": 0.30,
            "trust_weight": 0.20,
            "consistency_weight": 0.15,
        },
        participant_configs=[
            SceneParticipant(
                type_id="budget_traveler",
                count=1,
                expected_stages=["greet", "gather_destination", "gather_dates", "gather_budget", "present_options", "confirm_booking"],
                min_info_release={"destination": True, "budget": True, "dates": True},
                behavior_overrides={"hesitation": 0.4, "info_release": "normal", "price_sensitive": True},
            ),
            SceneParticipant(
                type_id="luxury_seeker",
                count=1,
                expected_stages=["greet", "gather_destination", "gather_dates", "gather_budget", "present_options", "confirm_booking"],
                min_info_release={"destination": True, "budget": True, "dates": True},
                behavior_overrides={"hesitation": 0.1, "info_release": "fast", "price_sensitive": False, "quick_decider": True},
            ),
            SceneParticipant(
                type_id="anxious_flyer",
                count=1,
                expected_stages=["greet", "gather_destination", "gather_dates", "gather_budget", "gather_travelers", "present_options", "confirm_booking"],
                min_info_release={"destination": True, "budget": True, "dates": True, "travelers": True},
                behavior_overrides={"hesitation": 0.7, "info_release": "slow", "risk_averse": True},
            ),
            SceneParticipant(
                type_id="spontaneous_explorer",
                count=1,
                expected_stages=["greet", "gather_destination", "gather_dates", "gather_budget", "present_options", "confirm_booking"],
                min_info_release={"destination": True, "budget": True},
                behavior_overrides={"hesitation": 0.2, "info_release": "fast", "quick_decider": True},
            ),
        ],
    )


# ───────────────────────────────────────────────
# 测试执行
# ───────────────────────────────────────────────

def run_travel_scene_test(
    scene: Scene,
    rounds: int = 8,
    temperature: float = 0.7,
) -> List[TestSession]:
    """
    对场景中的每个参与者运行测试。
    """
    from deepseek_client import chat_completion

    manager = PersonaManager()
    if not manager.list_types():
        manager.create_preset_types()

    # 确保旅行 persona 类型已注册
    travel_types = ["budget_traveler", "luxury_seeker", "anxious_flyer", "spontaneous_explorer"]
    for tt in travel_types:
        if not manager.get_type(tt):
            # 从 JSON 文件加载
            import json
            json_path = Path(__file__).parent.parent / "data" / "types" / f"{tt}.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                from vmu.models import PersonaType
                manager.register_type(PersonaType(**data))
                print(f"  📂 从文件加载 persona 类型: {tt}")
            else:
                print(f"  ⚠️  找不到 persona 类型文件: {json_path}")

    sessions = []

    for config in scene.participant_configs:
        print(f"\n{'='*60}")
        print(f"🎭 测试: {config.type_id}")
        print(f"   行为覆盖: {config.behavior_overrides}")
        print(f"   预期阶段: {' → '.join(config.expected_stages)}")
        print(f"{'='*60}")

        instance = manager.instantiate(type_id=config.type_id)

        # 创建阶段感知 Agent（传入旅行 domain）
        agent = StageAwarePersonaAgent(
            instance=instance,
            llm_client=chat_completion,
            behavior_overrides=config.behavior_overrides,
            private_info_override=config.private_info_override,
            domain="travel_booking",
        )

        travel_agent = TravelAgent(seed=42)
        session = TestSession(
            session_id=f"travel_{scene.scene_id}_{config.type_id}",
            agent_name="TravelAgent",
            persona_instance=instance,
        )

        current_msg = travel_agent.SCRIPTS["greet"][0]

        for i in range(rounds):
            try:
                result = agent.interact(current_msg, temperature=temperature)
                stage_stats = agent.get_stage_stats()

                turn = TestTurn(
                    round_num=i + 1,
                    agent_message=current_msg,
                    user_response=result.response,
                    metadata={
                        "trust_level": instance.memory.trust_level,
                        "emotional_state": instance.memory.emotional_state,
                        "current_stage": stage_stats.get("current_stage"),
                        "collected": stage_stats.get("collected"),
                    },
                )
                session.turns.append(turn)

                print(f"\n  轮{i+1} [{stage_stats.get('current_stage', '?')}]")
                print(f"  Agent: {current_msg[:60]}...")
                print(f"  User:  {result.response[:70]}...")

                current_msg = travel_agent(result.response)
                if "不打扰" in current_msg:
                    print(f"\n  💬 客服 agent 主动结束对话")
                    break

            except Exception as e:
                print(f"\n  ❌ 第{i+1}轮出错: {e}")
                session.status = "error"
                session.error_message = str(e)
                break

        session.status = "completed"
        sessions.append(session)

    return sessions


# ───────────────────────────────────────────────
# 评估报告
# ───────────────────────────────────────────────

def print_travel_scene_report(scene: Scene, sessions: List[TestSession]):
    """打印旅行场景评估报告。"""
    evaluator = SceneEvaluator()

    print(f"\n{'='*70}")
    print(f"📊 场景评估报告: {scene.name}")
    print(f"{'='*70}")
    print(f"场景描述: {scene.description}")
    print(f"预期阶段: {' → '.join(scene.expected_stages)}")
    print(f"通过标准:")
    for k, v in scene.success_criteria.items():
        print(f"  - {k}: {v}")

    all_passed = True
    results = []

    for session in sessions:
        config = next(
            (c for c in scene.participant_configs if c.type_id == session.persona_instance.type_id),
            None,
        )
        result = evaluator.evaluate_session(session, scene, config)
        results.append(result)
        if not result.passed:
            all_passed = False

        print(f"\n{'─'*70}")
        print(f"🧑 {session.persona_instance.type_id} — {'✅ 通过' if result.passed else '❌ 未通过'}")
        print(f"{'─'*70}")
        print(f"  综合得分:     {result.overall_score}/10")
        print(f"  阶段覆盖率:   {result.stage_coverage.coverage_rate:.0%} "
              f"({len(result.stage_coverage.unique_actual)}/{len(result.stage_coverage.expected)})")
        print(f"  经历阶段:     {' → '.join(result.stage_coverage.unique_actual)}")
        if result.stage_coverage.missing_stages:
            print(f"  缺失阶段:     {', '.join(result.stage_coverage.missing_stages)}")
        print(f"  信息释放率:   {result.info_release.release_rate:.0%}")
        if result.info_release.missing_items:
            print(f"  缺失信息:     {', '.join(result.info_release.missing_items)}")
        print(f"  信任度轨迹:   {result.trust_trajectory.initial:.2f} → {result.trust_trajectory.final:.2f} "
              f"({result.trust_trajectory.trend})")
        if result.issues:
            print(f"\n  ⚠️  发现的问题:")
            for issue in result.issues:
                print(f"     • {issue}")
        if result.suggestions:
            print(f"\n  💡 改进建议:")
            for sgt in result.suggestions:
                print(f"     • {sgt}")

    print(f"\n{'='*70}")
    print(f"📋 场景总结: {'✅ 全部通过' if all_passed else '❌ 部分未通过'}")
    print(f"{'='*70}")

    # 跨 persona 对比
    print(f"\n【跨 Persona 对比】")
    print(f"  {'Persona':<25} {'得分':>6} {'阶段覆盖':>10} {'信息释放':>10} {'信任趋势':>10} {'结果':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for r in results:
        status = "通过" if r.passed else "未通过"
        print(f"  {r.persona_type:<25} {r.overall_score:>6.1f} "
              f"{r.stage_coverage.coverage_rate:>9.0%} {r.info_release.release_rate:>9.0%} "
              f"{r.trust_trajectory.trend:>10} {status:>6}")

    return results


# ───────────────────────────────────────────────
# 主入口
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="场景化测试旅行客服 agent")
    parser.add_argument("--rounds", type=int, default=8, help="每 persona 对话轮数")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM 温度")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     ✈️  场景化测试: 旅行预订全流程评估                                   ║
║                                                                      ║
║  测试4种不同性格的旅行者对同一个客服 agent 的反应                         ║
║  评估维度: 阶段覆盖率 / 信息释放 / 信任度轨迹 / 行为一致性                 ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    scene = create_travel_scene()
    print(f"📋 场景: {scene.name}")
    print(f"   参与者: {len(scene.participant_configs)} 种 persona")
    for c in scene.participant_configs:
        print(f"   - {c.type_id}: behavior={c.behavior_overrides}, expected={' → '.join(c.expected_stages)}")

    print(f"\n⏳ 开始测试...")
    try:
        sessions = run_travel_scene_test(scene, rounds=args.rounds, temperature=args.temp)

        results = print_travel_scene_report(scene, sessions)

        import json
        report_path = "/tmp/travel_scene_evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
        print(f"\n📄 详细 JSON 报告已保存: {report_path}")

        # ── 生成可视化图表 ──
        print(f"\n🎨 生成可视化图表...")
        from vmu.testing.visualizer import EvaluationVisualizer

        viz = EvaluationVisualizer(output_dir="reports")
        chart_paths = viz.generate_full_report(
            results=results,
            sessions=sessions,
            scene_stages=scene.expected_stages,
        )

        print(f"\n📊 图表已生成:")
        for name, path in chart_paths.items():
            print(f"  • {name}: {path}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
