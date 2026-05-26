"""
Prompt 模板渲染器
将 PersonaType 模板 + 变异参数 → 实际 system prompt
"""

import random
import re
from typing import Any, Dict, List, Optional

from .models import (
    BehaviorEngine,
    BehavioralTraits,
    Context,
    Demographics,
    MemoryState,
    PersonaInstance,
    PersonaType,
    Psychographics,
    SceneContext,
)


class PromptRenderer:
    """Prompt 渲染器：负责组装 system prompt"""
    
    @staticmethod
    def render_system_prompt(instance: PersonaInstance) -> str:
        """
        将 PersonaInstance 渲染为最终的 system prompt。
        优先使用 type 模板，如果为空则使用默认模板。
        """
        if instance.system_prompt:
            # 已经有预渲染的 prompt
            return instance.system_prompt
        
        # 使用默认模板组装
        return PromptRenderer._default_template(instance)
    
    @staticmethod
    def _default_template(inst: PersonaInstance) -> str:
        """默认 system prompt 模板"""
        d = inst.demographics
        ps = inst.psychographics
        bt = inst.behavioral_traits
        ctx = inst.context
        s = inst.scene_context
        be = inst.behavior_engine
        m = inst.memory
        
        lines = [
            f"# 虚拟用户身份设定：{d.role}",
            "",
            f"你叫 {inst.name}，是一位 {d.age} 岁的 {d.role}，在 {d.industry} 行业工作 {d.years_experience} 年。",
            f"所在公司规模：{d.company_size}，地点：{d.location}。",
            "",
            "## 你的状态",
            f"- 当前最大的痛点：{ctx.current_problem}",
            f"- 最近的变化：{ctx.recent_changes}",
            f"- 团队压力：{ctx.team_pressure}",
        ]
        if ctx.competitive_exposure:
            lines.append(f"- 竞品经历：{', '.join(ctx.competitive_exposure)}")
        
        lines.extend(["", "## 你的工作目标"])
        for g in ps.goals:
            lines.append(f"- {g}")
        
        lines.extend(["", "## 你的 frustrations"])
        for f in ps.frustrations:
            lines.append(f"- {f}")
        
        lines.extend([
            "", "## 你的沟通风格",
            f"- {bt.communication}",
            f"- 怀疑程度：{'高' if bt.skepticism_level > 0.6 else '中' if bt.skepticism_level > 0.3 else '低'}（{bt.skepticism_level:.2f}）",
            f"- 价格敏感度：{'高' if bt.price_sensitivity > 0.6 else '中' if bt.price_sensitivity > 0.3 else '低'}（{bt.price_sensitivity:.2f}）",
            f"- 风险承受度：{bt.risk_tolerance}",
            f"- 决策风格：{ps.decision_style}",
        ])
        
        if ps.tech_stack:
            lines.extend(["", "## 你的技术/工具栈"])
            for t in ps.tech_stack:
                lines.append(f"- {t}")
        
        lines.extend([
            "", "## 当前场景",
            f"- 场景：{s.scene_description}",
            f"- 初始态度：{s.initial_attitude}",
            f"- 时间压力：{s.time_pressure}",
            f"- 参与动机：{s.participation_motivation}",
            f"- 之前接触：{s.prior_exposure}",
            "", "## 对话规则",
            "- 你的回复必须符合上述身份和场景，不能偏离角色。",
            "- 如果对方使用模糊语言，你会要求给具体例子。",
            "- 如果对方提到你关心的关键词，你会高度关注并深入追问。",
        ])
        if be.skepticism_triggers:
            lines.append(f"- 如果对方使用 {', '.join(be.skepticism_triggers)} 这类词汇，你会表现出怀疑或反感。")
        lines.extend([
            "- 你不会主动暴露公司敏感信息。",
            f"- 你的采购权限：{ps.budget_authority}。",
            "- 如果产品不解决你的问题，你会直接说'这不解决我的问题'。",
            "", "## 记忆状态（初始）",
            f"- 信任度：{m.trust_level:.2f}/1.0",
            f"- 情绪：{m.emotional_state}",
            f"- 接触次数：{m.exposure_count}",
        ])
        
        if be.knowledge_boundary:
            lines.extend([
                "", "## 知识边界",
                f"- {be.knowledge_boundary}",
            ])
        
        return "\n".join(lines)
    
    @staticmethod
    def apply_variation(
        base: Any,
        variation: Dict[str, Any],
    ) -> Any:
        """
        将变异参数应用到基础模型上，返回新的模型实例。
        支持嵌套路径，如 {"demographics.age": 35}
        """
        if not variation:
            return base
        
        data = base.model_dump()
        for key, value in variation.items():
            if "." in key:
                parts = key.split(".")
                target = data
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
            else:
                data[key] = value
        
        return base.__class__(**data)


class VariationGenerator:
    """
    根据 variation_config 自动生成差异化的实例参数。
    """
    
    @staticmethod
    def generate(config: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        """
        根据配置生成一组变异参数。
        
        config 格式示例：
        {
            "demographics.age": {"range": [25, 45]},
            "behavioral_traits.skepticism_level": {"range": [0.2, 0.8]},
            "psychographics.goals": {"options": [["效率"], ["质量", "效率"], ["创新"]]},
        }
        """
        if seed is not None:
            random.seed(seed)
        
        variation = {}
        for key, spec in config.items():
            if "range" in spec:
                low, high = spec["range"]
                if isinstance(low, int) and isinstance(high, int):
                    variation[key] = random.randint(low, high)
                else:
                    variation[key] = round(random.uniform(low, high), 2)
            elif "options" in spec:
                variation[key] = random.choice(spec["options"])
            elif "values" in spec:
                variation[key] = random.choice(spec["values"])
        
        return variation
    
    @staticmethod
    def generate_name(role: str, index: int) -> str:
        """根据角色生成一个示例名称"""
        names_pool = {
            "产品经理": ["张晓明", "李思远", "王建国", "陈浩然", "刘子轩"],
            "工程师": ["赵文博", "孙志强", "周凯文", "吴天宇", "郑一凡"],
            "设计师": ["林小美", "黄雅琪", "徐静怡", "马丽娜", "朱雨桐"],
            "运营": ["何晓峰", "高宇航", "罗振宇", "梁嘉伟", "宋文博"],
            "销售": ["唐丽丽", "许文强", "邓建国", "冯晓波", "程思远"],
            "总监": ["曹德明", "彭志强", "潘宏伟", "袁国华", "蒋志明"],
            "CTO": ["沈凯文", "陆天宇", "姚志强", "卢晓东", "钱文涛"],
        }
        pool = names_pool.get(role, ["用户A", "用户B", "用户C", "用户D", "用户E"])
        return pool[index % len(pool)]
