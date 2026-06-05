#!/usr/bin/env python3
"""
场景化测试示例：定义场景预期标准，评估销售 agent 表现。

演示了如何用 Scene + SceneEvaluator 做结构化 agent 评估：
1. 定义场景（预期阶段、成功标准、评估配置）
2. 为不同 persona 设置行为覆盖参数
3. 运行测试，收集对话数据
4. 用 SceneEvaluator 生成结构化评估报告

用法：
    cd /Users/ting/work/pmf-dummy-user-agent
    python examples/test_with_scene.py
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
# 模拟销售 Agent（升级版：按场景推进，减少重复）
# ───────────────────────────────────────────────

class SmartSalesAgent:
    """
    升级版销售 agent，尝试按场景阶段推进，减少重复话术。
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.round = 0
        self.stage_order = [
            "initial_contact",
            "need_discovery",
            "solution_presentation",
            "objection_handling",
            "pricing_discussion",
            "decision",
        ]
        self.used_scripts: Dict[str, set] = {s: set() for s in self.stage_order}

    def __call__(self, user_msg: str) -> str:
        import random
        rng = random.Random(self.seed + self.round)

        stage_idx = min(self.round, len(self.stage_order) - 1)
        stage = self.stage_order[stage_idx]

        lower = user_msg.lower()
        if any(w in lower for w in ["算了", "不需要", "不用", "没兴趣", "拒绝", "不聊了"]):
            return "完全理解，那我就不打扰了。如果以后有需求随时联系！"

        if any(w in lower for w in ["试用", "可以试", "poc", "试点", "demo"]):
            stage = "decision"

        scripts = self.SCRIPTS.get(stage, ["好的，您还有什么想了解的吗？"])

        # 尽量选择没用过的脚本
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
        "initial_contact": [
            "您好！我是效率云的小李，我们专注帮助增长团队打通审批和数据看板。方便聊2分钟吗？",
            "你好，我是小李。我们做了一个专门给增长团队用的协作工具，核心是减少审批卡点。想了解一下你们现在的流程吗？",
        ],
        "need_discovery": [
            "感谢回复！想快速了解一下——你们现在从'想做个活动'到'真正上线'，中间要过几层审批？最卡的是哪一环？",
            "明白了。你们团队现在用什么工具做任务分配和进度同步？过程中最耽误时间的是审批还是执行？",
        ],
        "solution_presentation": [
            "了解了。我们产品有两个模块可能适合你们：一个是'审批流自动化'，可以把现在3天的审批压缩到2小时；另一个是'增长看板'，活动数据实时汇总。你们更关注哪一块？",
            "基于你说的卡点，我设计了一个轻量方案：先上审批自动化模块，和飞书/钉钉打通，实施周期1周。你们IT需要介入吗？",
        ],
        "objection_handling": [
            "理解，安全是底线。我们支持数据本地化存储，已有SOC2认证。另外可以先用沙箱环境跑一周，不影响现有系统。",
            "你们担心的这个问题我们很多客户一开始也有。实际上我们是通过API和现有系统对接，不迁移数据，试错成本很低。",
        ],
        "pricing_discussion": [
            "价格方面，标准版199/人/月，增长团队版299/人/月（含审批自动化+看板）。按年付有85折。你们6个人的话，年付大概1.5万。",
            "我们也提供按月付，前30天免费。如果一个月内团队觉得没用，随时停，没有任何绑定。",
        ],
        "decision": [
            "太好了！那我这边准备试用协议，明天发给你们。需要我同步拉个客户成功群吗？",
            "没问题，那我让同事发POC方案给你。你这边决策流程大概需要多久？需要我配合准备什么材料？",
        ],
    }


# ───────────────────────────────────────────────
# 场景定义
# ───────────────────────────────────────────────

def create_b2b_sales_scene() -> Scene:
    """创建一个标准的 B2B 销售测试场景。"""
    return Scene(
        name="B2B SaaS 销售全流程测试",
        description="测试销售 agent 是否能按标准 B2B 流程与不同类型客户完成对话",
        scenario="一个销售 agent 需要通过多轮对话，与潜在客户建立信任、挖掘需求、展示方案、处理异议、达成决策",
        expected_stages=[
            "initial_contact",
            "need_discovery",
            "solution_presentation",
            "objection_handling",
            "pricing_discussion",
            "decision",
        ],
        success_criteria={
            "min_overall_score": 6.0,
            "min_stage_coverage": 0.6,
            "min_trust_level": 0.2,
            "required_stages": ["need_discovery", "solution_presentation"],
        },
        evaluation_config={
            "stage_coverage_weight": 0.35,
            "info_release_weight": 0.30,
            "trust_weight": 0.20,
            "consistency_weight": 0.15,
        },
        participant_configs=[
            SceneParticipant(
                type_id="anxious_buyer",
                count=1,
                expected_stages=["initial_contact", "need_discovery", "objection_handling", "pricing_discussion", "decision"],
                min_info_release={"pain_points": True, "budget_range": True, "team_size": True},
                # 场景特定调参：在采购场景下，焦虑型买家更加犹豫
                behavior_overrides={"hesitation": 0.7, "info_release": "slow"},
            ),
            SceneParticipant(
                type_id="rational_analyst",
                count=1,
                expected_stages=["initial_contact", "need_discovery", "solution_presentation", "pricing_discussion", "decision"],
                min_info_release={"pain_points": True, "current_tools": True, "budget_range": True},
                # 场景特定调参：分析师会更快释放信息，但要求数据支撑
                behavior_overrides={"hesitation": 0.2, "info_release": "fast"},
            ),
            SceneParticipant(
                type_id="tech_skeptic",
                count=1,
                expected_stages=["initial_contact", "need_discovery", "solution_presentation", "objection_handling", "decision"],
                min_info_release={"pain_points": True, "current_tools": True, "timeline": True},
                # 场景特定调参：技术怀疑者在安全评审场景下更加谨慎
                behavior_overrides={"hesitation": 0.6, "risk_averse": True, "tech_challenger": True},
            ),
            SceneParticipant(
                type_id="impulsive_decider",
                count=1,
                expected_stages=["initial_contact", "need_discovery", "solution_presentation", "decision"],
                min_info_release={"pain_points": True, "timeline": True, "budget_range": True},
                # 场景特定调参：冲动决策者会快速推进，但容易反悔
                behavior_overrides={"hesitation": 0.1, "quick_decider": True, "info_release": "fast"},
            ),
        ],
    )


# ───────────────────────────────────────────────
# 测试执行
# ───────────────────────────────────────────────

def run_scene_test(
    scene: Scene,
    rounds: int = 6,
    temperature: float = 0.7,
) -> List[TestSession]:
    """
    对场景中的每个参与者运行测试。
    """
    from deepseek_client import chat_completion

    manager = PersonaManager()
    if not manager.list_types():
        manager.create_preset_types()

    sessions = []
    sales_agent = SmartSalesAgent(seed=42)

    for config in scene.participant_configs:
        print(f"\n{'='*60}")
        print(f"🎭 测试: {config.type_id}")
        print(f"   行为覆盖: {config.behavior_overrides}")
        print(f"   预期阶段: {' → '.join(config.expected_stages)}")
        print(f"{'='*60}")

        # 实例化虚拟人
        instance = manager.instantiate(type_id=config.type_id)

        # 创建阶段感知 Agent（传入场景特定的行为覆盖）
        agent = StageAwarePersonaAgent(
            instance=instance,
            llm_client=chat_completion,
            behavior_overrides=config.behavior_overrides,
            private_info_override=config.private_info_override,
        )

        # 创建会话
        session = TestSession(
            session_id=f"scene_{scene.scene_id}_{config.type_id}",
            agent_name="SmartSalesAgent",
            persona_instance=instance,
        )

        # 开场白
        current_msg = sales_agent.SCRIPTS["initial_contact"][0]

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
                print(f"  Sales: {current_msg[:60]}...")
                print(f"  User:  {result.response[:70]}...")

                # 销售 agent 回复
                current_msg = sales_agent(result.response)
                if "不打扰" in current_msg:
                    print(f"\n  💬 销售 agent 主动结束对话")
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

def print_scene_report(scene: Scene, sessions: List[TestSession]):
    """打印场景评估报告。"""
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
    print(f"  {'Persona':<20} {'得分':>6} {'阶段覆盖':>10} {'信息释放':>10} {'信任趋势':>10} {'结果':>6}")
    print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for r in results:
        status = "通过" if r.passed else "未通过"
        print(f"  {r.persona_type:<20} {r.overall_score:>6.1f} "
              f"{r.stage_coverage.coverage_rate:>9.0%} {r.info_release.release_rate:>9.0%} "
              f"{r.trust_trajectory.trend:>10} {status:>6}")

    return results


# ───────────────────────────────────────────────
# 主入口
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="场景化测试销售 agent")
    parser.add_argument("--rounds", type=int, default=6, help="每 persona 对话轮数")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM 温度")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     🎭 场景化测试: B2B SaaS 销售全流程评估                              ║
║                                                                      ║
║  测试4种不同性格的客户对同一个销售 agent 的反应                         ║
║  评估维度: 阶段覆盖率 / 信息释放 / 信任度轨迹 / 行为一致性               ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # 创建场景
    scene = create_b2b_sales_scene()
    print(f"📋 场景: {scene.name}")
    print(f"   参与者: {len(scene.participant_configs)} 种 persona")
    for c in scene.participant_configs:
        print(f"   - {c.type_id}: behavior={c.behavior_overrides}, expected={' → '.join(c.expected_stages)}")

    # 运行测试
    print(f"\n⏳ 开始测试...")
    try:
        sessions = run_scene_test(scene, rounds=args.rounds, temperature=args.temp)

        # 生成报告
        results = print_scene_report(scene, sessions)

        # 导出 JSON 报告（可选）
        import json
        report_path = "/tmp/scene_evaluation_report.json"
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
