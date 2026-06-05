#!/usr/bin/env python3
"""
对比测试：自由发挥 vs 阶段感知的虚拟人

验证问题：
1. 阶段感知是否让虚拟人行为更可预测？
2. 是否覆盖更多销售阶段？
3. 信息释放是否符合性格设定？
4. 对话轮数是否有差异？

用法：
    cd /Users/ting/work/pmf-dummy-user-agent
    python examples/compare_agent_modes.py --persona anxious_buyer --rounds 6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any, Dict, List

from vmu import PersonaManager
from vmu.agent import PersonaAgent
from vmu.stage_agent import StageAwarePersonaAgent


# ───────────────────────────────────────────────
# 模拟销售 Agent（有明确阶段推进意图）
# ───────────────────────────────────────────────

class MockSalesAgent:
    """
    模拟一个"努力按销售流程推进"的销售 agent。
    每轮按预设脚本推进一个阶段。
    """

    SCRIPTS = {
        "initial_contact": [
            "您好！我是 XX 公司的小李，关注到贵团队在项目管理方面可能有一些提升空间，想简单聊两句？",
            "您好！打扰一下，我是做企业效率工具的，方便了解您这边目前的工作流程吗？",
        ],
        "need_discovery": [
            "感谢回复！想了解一下，您团队目前在协作流程上有什么痛点吗？比如任务分配、进度追踪这些？",
            "明白了。那您这边目前用的是什么工具？团队大概多少人？预算范围大概在什么水平？",
        ],
        "solution_presentation": [
            "了解了。我们产品正好针对这些场景做了优化，比如自动任务分配、实时看板、和飞书/钉钉的深度集成。您看这些功能对你们有帮助吗？",
            "基于您说的情况，我为您设计了一个方案：核心模块 + 定制化看板 + 培训服务。整体实施周期 2 周。",
        ],
        "objection_handling": [
            "理解您的顾虑。关于稳定性，我们有 99.9% SLA，支持私有化部署。数据方面通过 SOC2 认证。",
            "您担心的这个问题我们很多客户一开始也有。实际上迁移过程我们有专门的客户成功团队全程跟进。",
        ],
        "pricing_discussion": [
            "价格方面，标准版 299/人/月，企业版 599/人/月。如果按年付有 8 折优惠。您团队 12 人的话，年付大概 3 万左右。",
            "我们也提供 30 天免费试用，试用期间会有专人对接。如果满意再签约，不满意随时停。",
        ],
        "decision": [
            "好的，那我这边准备一下试用协议。您看这周方便走流程吗？需要我联系您的 CTO 吗？",
            "太棒了！那我马上让同事发 POC 方案给您。您这边决策流程大概需要多久？",
        ],
    }

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

    def __call__(self, user_msg: str) -> str:
        import random
        rng = random.Random(self.seed + self.round)

        # 根据轮数选择阶段（模拟销售推进）
        stage_idx = min(self.round, len(self.stage_order) - 1)
        stage = self.stage_order[stage_idx]

        # 如果用户明确拒绝或要求暂停，调整
        lower = user_msg.lower()
        if any(w in lower for w in ["算了", "不需要", "不用", "没兴趣", "拒绝"]):
            return "完全理解，那我就不打扰了。如果以后有需求随时联系！"

        # 如果用户要求试用，提前进入决策
        if any(w in lower for w in ["试用", "可以试", "poc", "试点"]):
            stage = "decision"

        # 随机选择该阶段的一条话术
        scripts = self.SCRIPTS.get(stage, ["好的，您还有什么想了解的吗？"])
        reply = rng.choice(scripts)

        self.round += 1
        return reply


# ───────────────────────────────────────────────
# 对比测试核心逻辑
# ───────────────────────────────────────────────

def run_comparison(
    persona_type: str,
    rounds: int = 6,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    对同一个 persona，分别用 Mode A（自由发挥）和 Mode B（阶段感知）测试。
    """
    from deepseek_client import chat_completion

    manager = PersonaManager()
    if not manager.list_types():
        manager.create_preset_types()

    results = {}

    # ── Mode A: 原始 PersonaAgent（自由发挥）──
    print(f"\n{'='*60}")
    print(f"🅰️  Mode A: 自由发挥 ({persona_type})")
    print(f"{'='*60}")

    instance_a = manager.instantiate(type_id=persona_type, name=f"{persona_type}_A")
    agent_a = PersonaAgent(instance=instance_a, llm_client=chat_completion)
    sales_a = MockSalesAgent(seed=42)

    turns_a = []
    current_msg = sales_a.SCRIPTS["initial_contact"][0]

    for i in range(rounds):
        result = agent_a.interact(current_msg, temperature=temperature)
        turns_a.append({
            "round": i + 1,
            "sales_stage": sales_a.stage_order[min(i, len(sales_a.stage_order) - 1)],
            "sales_msg": current_msg[:80],
            "user_response": result.response,
            "trust": result.updated_memory.trust_level if result.updated_memory else None,
            "emotion": result.updated_memory.emotional_state if result.updated_memory else None,
        })
        print(f"\n  轮{i+1} [{turns_a[-1]['sales_stage']}]")
        print(f"  Sales: {current_msg[:60]}...")
        print(f"  User:  {result.response[:80]}...")

        current_msg = sales_a(result.response)
        if "不打扰" in current_msg:
            break

    results["mode_a"] = {
        "persona": persona_type,
        "turns": turns_a,
        "final_trust": instance_a.memory.trust_level,
        "final_emotion": instance_a.memory.emotional_state,
        "total_rounds": len(turns_a),
    }

    # ── Mode B: StageAwarePersonaAgent（阶段感知）──
    print(f"\n{'='*60}")
    print(f"🅱️  Mode B: 阶段感知 ({persona_type})")
    print(f"{'='*60}")

    instance_b = manager.instantiate(type_id=persona_type, name=f"{persona_type}_B")
    agent_b = StageAwarePersonaAgent(instance=instance_b, llm_client=chat_completion)
    sales_b = MockSalesAgent(seed=42)  # 同样的销售脚本

    turns_b = []
    current_msg = sales_b.SCRIPTS["initial_contact"][0]

    for i in range(rounds):
        result = agent_b.interact(current_msg, temperature=temperature)
        stage_stats = agent_b.get_stage_stats()

        turns_b.append({
            "round": i + 1,
            "sales_stage": sales_b.stage_order[min(i, len(sales_b.stage_order) - 1)],
            "detected_stage": stage_stats["current_stage"],
            "sales_msg": current_msg[:80],
            "user_response": result.response,
            "trust": result.updated_memory.trust_level if result.updated_memory else None,
            "emotion": result.updated_memory.emotional_state if result.updated_memory else None,
            "collected": stage_stats["collected"],
            "stage_coverage": stage_stats["stage_coverage"],
        })
        print(f"\n  轮{i+1} [Sales:{turns_b[-1]['sales_stage']} | Detected:{turns_b[-1]['detected_stage']}]")
        print(f"  Sales: {current_msg[:60]}...")
        print(f"  User:  {result.response[:80]}...")
        if stage_stats["collected"]:
            print(f"  Collected: {stage_stats['collected']}")

        current_msg = sales_b(result.response)
        if "不打扰" in current_msg:
            break

    results["mode_b"] = {
        "persona": persona_type,
        "turns": turns_b,
        "final_trust": instance_b.memory.trust_level,
        "final_emotion": instance_b.memory.emotional_state,
        "total_rounds": len(turns_b),
        "stage_coverage": stage_stats["stage_coverage"],
        "unique_stages": stage_stats["unique_stages"],
        "final_collected": stage_stats["collected"],
    }

    return results


# ───────────────────────────────────────────────
# 评估与报告
# ───────────────────────────────────────────────

def print_comparison_report(results: Dict[str, Any]):
    """打印对比报告"""
    mode_a = results["mode_a"]
    mode_b = results["mode_b"]

    print(f"\n{'='*60}")
    print("📊 对比测试报告")
    print(f"{'='*60}")

    print(f"\n【基础指标】")
    print(f"  对话轮数:     A={mode_a['total_rounds']}  vs  B={mode_b['total_rounds']}")
    print(f"  最终信任度:   A={mode_a['final_trust']:.2f}  vs  B={mode_b['final_trust']:.2f}")
    print(f"  最终情绪:     A={mode_a['final_emotion']}  vs  B={mode_b['final_emotion']}")

    print(f"\n【Mode B 阶段感知专属指标】")
    print(f"  阶段覆盖率:   {mode_b['stage_coverage']*100:.0f}% ({len(mode_b['unique_stages'])}/6 个阶段)")
    print(f"  经历的阶段:   {' → '.join(mode_b['unique_stages'])}")
    print(f"  信息释放:     {mode_b['final_collected']}")

    print(f"\n【逐轮对比】")
    max_rounds = max(mode_a['total_rounds'], mode_b['total_rounds'])
    for i in range(max_rounds):
        a_msg = mode_a['turns'][i]['user_response'][:50] if i < len(mode_a['turns']) else "(无)"
        b_msg = mode_b['turns'][i]['user_response'][:50] if i < len(mode_b['turns']) else "(无)"
        a_stage = mode_a['turns'][i]['sales_stage'] if i < len(mode_a['turns']) else "-"
        b_stage = mode_b['turns'][i]['sales_stage'] if i < len(mode_b['turns']) else "-"
        b_detected = mode_b['turns'][i]['detected_stage'] if i < len(mode_b['turns']) else "-"

        print(f"\n  轮{i+1}")
        print(f"    A [{a_stage}]: {a_msg}...")
        print(f"    B [{b_stage}|detected:{b_detected}]: {b_msg}...")

    print(f"\n{'='*60}")
    print("💡 关键观察")
    print(f"{'='*60}")

    # 自动分析
    observations = []

    # 观察1: 阶段覆盖
    if mode_b['stage_coverage'] < 0.5:
        observations.append("⚠️ 阶段覆盖率偏低，销售 agent 可能跳跃阶段或虚拟人没有按阶段回应")
    else:
        observations.append(f"✅ 覆盖了 {len(mode_b['unique_stages'])} 个销售阶段")

    # 观察2: 信息释放
    collected = mode_b['final_collected']
    if collected:
        released_count = sum(1 for v in collected.values() if v)
        observations.append(f"✅ 虚拟人释放了 {released_count}/{len(collected)} 类信息")
    else:
        observations.append("⚠️ 没有检测到任何信息释放")

    # 观察3: 情绪一致性
    if mode_a['final_emotion'] != mode_b['final_emotion']:
        observations.append(f"📝 情绪结果不同：自由发挥→{mode_a['final_emotion']}，阶段感知→{mode_b['final_emotion']}")

    for obs in observations:
        print(f"  {obs}")

    print(f"\n{'='*60}")
    print("✅ 对比测试完成！")
    print(f"{'='*60}")


# ───────────────────────────────────────────────
# 主入口
# ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对比测试虚拟人的两种模式")
    parser.add_argument(
        "--persona", default="anxious_buyer",
        choices=["anxious_buyer", "rational_analyst", "tech_skeptic", "impulsive_decider"],
        help="测试的 persona 类型"
    )
    parser.add_argument("--rounds", type=int, default=6, help="对话轮数")
    parser.add_argument("--temp", type=float, default=0.7, help="LLM 温度")

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     🧪 虚拟人模式对比测试                                               ║
║                                                                      ║
║  Mode A: 原始 PersonaAgent（system prompt 自由发挥）                    ║
║  Mode B: StageAwarePersonaAgent（阶段感知 + 行为骨架）                  ║
║                                                                      ║
║  销售 Agent: MockSalesAgent（按标准 B2B 流程推进）                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    try:
        results = run_comparison(
            persona_type=args.persona,
            rounds=args.rounds,
            temperature=args.temp,
        )
        print_comparison_report(results)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
