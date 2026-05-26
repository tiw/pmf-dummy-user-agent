#!/usr/bin/env python3
"""
场景运行示例

演示：
1. 创建预设人格类型
2. 创建一个产品评估场景
3. 实例化场景中的多个虚拟人
4. 与每个虚拟人进行交互
"""

import sys
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from vmu import PersonaManager, PersonaAgent
from vmu.models import SceneParticipant

# 如果配置了 LLM，可以取消下面的注释进行真实 LLM 交互
# from deepseek_client import chat_completion


def demo_without_llm():
    """无 LLM 的演示：展示系统结构和 prompt"""
    print("=" * 60)
    print("🎭 虚拟人管理系统 —— 场景运行示例")
    print("=" * 60)
    
    # 1. 初始化管理器
    manager = PersonaManager()
    
    # 2. 创建预设类型
    print("\n📦 创建预设人格类型...")
    types = manager.create_preset_types()
    for pt in types:
        print(f"   ✅ {pt.name} ({pt.type_id})")
    
    # 3. 创建一个场景
    print("\n🎬 创建场景：B2B SaaS 产品演示会")
    scene = manager.create_scene(
        name="B2B SaaS 产品演示会",
        description="一个 5 人的产品演示会议，模拟真实决策委员会",
        scenario=(
            "你们是一家中型企业的决策团队，正在评估一款新的项目管理 SaaS 工具。"
            "演示者将在 30 分钟内展示产品核心功能。"
            "你们各自有不同的关注点和决策风格。"
        ),
        participant_configs=[
            SceneParticipant(
                type_id="anxious_buyer",
                count=2,
                scene_specific_prompt_addition="你特别担心数据迁移风险和供应商稳定性。",
            ),
            SceneParticipant(
                type_id="rational_analyst",
                count=1,
                scene_specific_prompt_addition="你负责评估产品的数据导出能力和 API 文档质量。",
            ),
            SceneParticipant(
                type_id="tech_skeptic",
                count=1,
                scene_specific_prompt_addition="你需要确认产品能否与现有的 Oracle 系统集成。",
            ),
            SceneParticipant(
                type_id="impulsive_decider",
                count=1,
                scene_specific_prompt_addition="你已经被繁琐的现有流程折磨了很久，希望快速看到改变。",
            ),
        ],
        shared_context={
            "product_name": "ProjectFlow Pro",
            "product_type": "项目管理 SaaS",
            "demo_duration_minutes": 30,
            "company_current_tool": "Excel + 邮件",
        },
    )
    print(f"   ✅ 场景创建：{scene.name} ({scene.scene_id})")
    
    # 4. 实例化场景
    print("\n👥 实例化场景参与者...")
    instantiated = manager.instantiate_scene(scene.scene_id)
    if not instantiated:
        print("❌ 场景实例化失败")
        return
    
    participants = manager.get_scene_instances(scene.scene_id)
    print(f"   ✅ 共生成 {len(participants)} 个虚拟人实例：")
    
    for p in participants:
        print(f"\n   ── {p.name} ({p.type_id}) ──")
        print(f"      角色：{p.demographics.role}，{p.demographics.age} 岁")
        print(f"      怀疑度：{p.behavioral_traits.skepticism_level:.2f}")
        print(f"      价格敏感度：{p.behavioral_traits.price_sensitivity:.2f}")
        print(f"      差异化参数：{p.variation}")
    
    # 5. 展示每个实例的 system prompt
    print("\n" + "=" * 60)
    print("📝 各虚拟人的 System Prompt 预览")
    print("=" * 60)
    
    for p in participants:
        print(f"\n{'─' * 50}")
        print(f"👤 {p.name} ({p.type_id})")
        print(f"{'─' * 50}")
        # 只打印前 800 字符
        preview = p.system_prompt[:800] + "..." if len(p.system_prompt) > 800 else p.system_prompt
        print(preview)
    
    # 6. 演示 LLM 交互接口（如果配置了 LLM）
    print("\n" + "=" * 60)
    print("🤖 LLM 交互接口演示")
    print("=" * 60)
    
    try:
        from deepseek_client import chat_completion
        has_llm = True
        print("   ✅ 检测到 LLM 客户端")
    except ImportError:
        has_llm = False
        print("   ⚠️ 未检测到 LLM 客户端，跳过交互演示")
    
    if has_llm:
        test_message = "你好，我是 ProjectFlow Pro 的产品经理，今天想给你们演示我们的项目管理工具。"
        
        for p in participants[:2]:  # 只演示前两个
            print(f"\n👤 与 {p.name} 交互：")
            print(f"   输入：{test_message}")
            
            agent = PersonaAgent(instance=p, llm_client=chat_completion)
            try:
                result = agent.interact(test_message)
                print(f"   回复：{result.response[:200]}...")
                print(f"   信任度变化：{p.memory.trust_level:.2f}")
            except Exception as e:
                print(f"   ⚠️ 交互失败：{e}")
    
    # 7. 保存状态
    print("\n" + "=" * 60)
    print("💾 持久化状态")
    print("=" * 60)
    print(f"   数据保存在 data/ 目录下：")
    print(f"   - types/      : {len(manager.list_types())} 个类型")
    print(f"   - instances/  : {len(manager.list_instances())} 个实例")
    print(f"   - scenes/     : {len(manager.list_scenes())} 个场景")
    
    print("\n✅ 演示完成！")


def demo_api_style():
    """
    以 API 风格演示管理操作。
    展示所有 CRUD 接口。
    """
    print("\n" + "=" * 60)
    print("🔧 API 风格操作演示")
    print("=" * 60)
    
    manager = PersonaManager()
    
    # 创建类型
    from vmu.models import Demographics, Psychographics, BehavioralTraits
    
    pt = manager.create_type(
        type_id="budget_controller",
        name="预算控制者",
        description="关注成本效益，对价格极其敏感",
        demographics=Demographics(age=45, role="财务总监"),
        psychographics=Psychographics(
            goals=["控制成本", "提高预算使用效率"],
            frustrations=["部门超支", "ROI 不清晰"],
        ),
        behavioral_traits=BehavioralTraits(
            skepticism_level=0.6,
            price_sensitivity=0.95,
        ),
        variation_config={
            "demographics.age": {"range": [38, 55]},
            "behavioral_traits.price_sensitivity": {"range": [0.8, 1.0]},
        },
    )
    print(f"✅ 创建类型：{pt.name}")
    
    # 实例化 3 个
    for i in range(3):
        inst = manager.instantiate("budget_controller", variation_seed=i * 100)
        if inst:
            print(f"   👤 实例 {i+1}: {inst.name}, 年龄 {inst.demographics.age}, "
                  f"价格敏感度 {inst.behavioral_traits.price_sensitivity:.2f}")
    
    # 查询
    all_instances = manager.list_instances("budget_controller")
    print(f"\n📊 查询结果：budget_controller 类型共有 {len(all_instances)} 个实例")
    
    # 创建场景
    scene = manager.create_scene(
        name="预算审批会议",
        scenario="财务部正在评估各部门提交的新工具采购申请",
        participant_configs=[
            SceneParticipant(type_id="budget_controller", count=3),
        ],
    )
    print(f"\n✅ 创建场景：{scene.name}")
    
    # 实例化场景
    manager.instantiate_scene(scene.scene_id)
    scene_instances = manager.get_scene_instances(scene.scene_id)
    print(f"   👥 场景内实例：{len(scene_instances)} 人")
    for si in scene_instances:
        print(f"      - {si.name} ({si.demographics.role})")


if __name__ == "__main__":
    demo_without_llm()
    demo_api_style()
