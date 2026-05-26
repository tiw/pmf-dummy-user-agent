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
        """根据角色生成一个真实中文名称"""
        # 大量真实中文姓名，按角色特征分配
        male_names = [
            "张伟", "王强", "李明", "刘洋", "陈浩", "杨帆", "赵磊", "黄勇", "周杰", "吴涛",
            "徐鹏", "孙伟", "马超", "朱军", "胡斌", "郭明", "何健", "高翔", "林峰", "罗辉",
            "郑宇", "梁波", "谢军", "宋凯", "唐勇", "许刚", "韩涛", "冯磊", "曹阳", "彭亮",
            "曾伟", "肖杰", "田波", "董强", "袁明", "潘峰", "于洋", "蒋磊", "蔡勇", "贾浩",
            "魏鹏", "薛刚", "叶波", "阎杰", "余亮", "潘明", "杜强", "戴勇", "夏阳", "钟辉",
            "汪峰", "田凯", "任杰", "姜波", "范明", "方亮", "石磊", "姚洋", "谭刚", "廖辉",
            "邹涛", "熊伟", "金波", "陆明", "郝勇", "孔阳", "白杰", "崔亮", "康峰", "毛波",
            "邱强", "秦凯", "江明", "孟阳", "龙辉", "万波", "段杰", "雷亮", "钱勇", "汤涛",
            "尹伟", "黎峰", "易波", "常杰", "武明", "乔阳", "贺刚", "赖凯", "龚辉", "文峰",
        ]
        female_names = [
            "李娜", "王芳", "张敏", "刘静", "陈丽", "杨秀", "赵燕", "黄玲", "周婷", "吴霞",
            "徐倩", "孙洁", "马丽", "朱莉", "胡娜", "郭慧", "何娟", "高艳", "林琳", "罗琴",
            "郑雪", "梁莹", "谢瑶", "宋欣", "唐颖", "许薇", "韩璐", "冯倩", "曹蕾", "彭娟",
            "曾燕", "肖婷", "田静", "董丽", "袁敏", "潘玲", "于洁", "蒋芳", "蔡秀", "贾娜",
            "魏霞", "薛莹", "叶琴", "阎瑶", "余欣", "潘颖", "杜薇", "戴璐", "夏倩", "钟蕾",
            "汪娟", "田燕", "任婷", "姜静", "范丽", "方敏", "石玲", "姚洁", "谭芳", "廖秀",
            "邹娜", "熊霞", "金莹", "陆琴", "郝瑶", "孔欣", "白颖", "崔薇", "康璐", "毛倩",
            "邱蕾", "秦娟", "江燕", "孟婷", "龙静", "万丽", "段敏", "雷玲", "钱洁", "汤芳",
            "尹秀", "黎娜", "易霞", "常莹", "武琴", "乔瑶", "贺欣", "赖颖", "龚薇", "文璐",
        ]

        # 角色 -> (名字池, 性别倾向)
        # 性别倾向: M=偏男性, F=偏女性, B=均衡
        role_mapping = {
            "产品经理": (male_names + female_names, "B"),
            "中小团队负责人": (male_names + female_names, "B"),
            "数据分析师": (male_names + female_names, "B"),
            "技术总监": (male_names, "M"),
            "增长运营": (female_names + male_names, "B"),
            "财务总监": (female_names + male_names, "B"),
            "工程师": (male_names + female_names, "M"),
            "设计师": (female_names + male_names, "F"),
            "运营": (female_names + male_names, "B"),
            "销售": (female_names + male_names, "B"),
            "总监": (male_names + female_names, "M"),
            "CTO": (male_names, "M"),
            "用户": (male_names + female_names, "B"),
        }

        import random
        pool, gender = role_mapping.get(role, (male_names + female_names, "B"))

        # 用 role + index 作为种子，保证同一角色同一下标得到相同名字
        # 但不同角色不会冲突
        seed = hash(f"{role}_{index}") % (2**31)
        rng = random.Random(seed)

        # 随机打乱后取第 index 个，避免相邻实例名字太像
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        return shuffled[index % len(shuffled)]
