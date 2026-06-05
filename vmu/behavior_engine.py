"""
B2B 销售场景下的用户行为引擎。

基于 workflow-in-llm 的"代码控制骨架 + LLM 润色"方法论，
为虚拟人提供阶段感知的结构化行为指导。
"""

import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .models import Message, PersonaInstance


# ───────────────────────────────────────────────
# 行为参数模型（支持显式定义 + 动态映射）
# ───────────────────────────────────────────────

class BehaviorParams(BaseModel):
    """用户行为参数。可从 persona 字段动态映射，也可显式覆盖。"""
    info_release: str = Field("normal", description="信息释放速度: fast/normal/slow")
    hesitation: float = Field(0.3, ge=0.0, le=1.0, description="犹豫度，越高越不愿意一次性给全信息")
    price_sensitive: bool = Field(False, description="是否对价格敏感")
    risk_averse: bool = Field(False, description="是否风险厌恶")
    tech_challenger: bool = Field(False, description="是否会在技术层面挑战销售")
    quick_decider: bool = Field(False, description="是否快速决策")
    skepticism_level: float = Field(0.5, ge=0.0, le=1.0, description="怀疑程度")
    communication_style: str = Field("", description="沟通风格描述")

    def apply_overrides(self, overrides: Dict[str, Any]) -> "BehaviorParams":
        """应用外部覆盖参数，返回新实例。"""
        if not overrides:
            return self
        data = self.model_dump()
        for key, value in overrides.items():
            if key in data:
                data[key] = value
        return BehaviorParams(**data)


# ───────────────────────────────────────────────
# B2B 销售阶段定义
# ───────────────────────────────────────────────

SALES_STAGES = [
    "initial_contact",      # 初次接触
    "need_discovery",       # 需求挖掘
    "solution_presentation",# 方案展示
    "objection_handling",   # 异议处理
    "pricing_discussion",   # 价格/试用谈判
    "decision",             # 决策确认
]

STAGE_TRANSITIONS = {
    "initial_contact": ["need_discovery"],
    "need_discovery": ["solution_presentation", "need_discovery"],
    "solution_presentation": ["objection_handling", "pricing_discussion", "solution_presentation"],
    "objection_handling": ["pricing_discussion", "objection_handling", "solution_presentation"],
    "pricing_discussion": ["decision", "pricing_discussion", "objection_handling"],
    "decision": ["decision"],
}


def detect_sales_stage(sales_msg: str, current_stage: Optional[str] = None) -> str:
    """
    根据销售人员的最新消息判断当前销售阶段。
    规则引擎，简单高效。
    """
    msg = sales_msg.lower()

    # 决策相关（最高优先级）
    if any(w in msg for w in ["确认", "签约", "合同", "付款", "定下来", "推进", "拍板", "走流程"]):
        return "decision"

    # 价格/试用
    if any(w in msg for w in ["价格", "费用", "多少钱", "预算", "折扣", "优惠", "试用", "pilot", "demo", "po", "采购"]):
        return "pricing_discussion"

    # 异议处理
    if any(w in msg for w in ["担心", "顾虑", "问题", "风险", "安全", "兼容", "集成", "但是", "不过", "挑战", "困难"]):
        return "objection_handling"

    # 方案展示
    if any(w in msg for w in ["功能", "特性", "可以", "支持", "方案", "解决", "产品", "展示", "介绍", "演示", "能力", "优势"]):
        return "solution_presentation"

    # 需求挖掘：必须是主动提问了解客户情况（疑问词 + 业务词）
    has_question = any(w in msg for w in ["怎么", "什么", "如何", "为什么", "吗", "呢", "多少", "哪些"])
    has_business = any(w in msg for w in ["痛点", "现状", "需求", "团队", "流程", "工作", "用", "情况"])
    if has_question and has_business:
        return "need_discovery"

    # 简短问候语判定为初次接触（排除上面已匹配的）
    if len(msg) < 50 and any(w in msg for w in ["您好", "你好", "打扰", "请问"]):
        return "initial_contact"

    # 默认保持当前阶段或初始接触
    return current_stage if current_stage in SALES_STAGES else "initial_contact"


# ───────────────────────────────────────────────
# 性格参数映射
# ───────────────────────────────────────────────

def extract_behavior_params(
    persona: PersonaInstance,
    overrides: Optional[Dict[str, Any]] = None,
) -> BehaviorParams:
    """
    从 PersonaInstance 的现有字段映射出行为参数。
    零侵入：不需要修改 models.py。
    支持通过 overrides 显式覆盖任何参数。
    """
    bt = persona.behavioral_traits
    ps = persona.psychographics

    # 从 communication 风格映射 info_release
    comm = bt.communication.lower()
    if any(w in comm for w in ["直接", "快节奏", "快", "简洁"]):
        info_release = "fast"
    elif any(w in comm for w in ["反复确认", "细节", "追问", "谨慎"]):
        info_release = "slow"
    else:
        info_release = "normal"

    # 犹豫度 = 怀疑度 × 0.5 + 决策风格因子
    hesitation = bt.skepticism_level * 0.5
    if "谨慎" in ps.decision_style or "数据驱动" in ps.decision_style:
        hesitation += 0.15
    hesitation = min(1.0, hesitation)

    # 价格敏感度
    price_sensitive = bt.price_sensitivity > 0.6

    # 风险厌恶
    risk_averse = bt.risk_tolerance in ["极低", "低"] or bt.skepticism_level > 0.7

    # 技术挑战倾向
    tech_challenger = bt.skepticism_level > 0.6 and "技术" in persona.demographics.role.lower()

    # 快速决策倾向
    quick_decider = "直觉" in ps.decision_style or "冲动" in ps.decision_style

    params = BehaviorParams(
        info_release=info_release,
        hesitation=round(hesitation, 2),
        price_sensitive=price_sensitive,
        risk_averse=risk_averse,
        tech_challenger=tech_challenger,
        quick_decider=quick_decider,
        skepticism_level=bt.skepticism_level,
        communication_style=bt.communication,
    )

    # 应用显式覆盖（场景特定调参）
    if overrides:
        params = params.apply_overrides(overrides)

    return params


# ───────────────────────────────────────────────
# 私有信息模板（按 persona type）
# ───────────────────────────────────────────────

PRIVATE_INFO_TEMPLATES = {
    "anxious_buyer": {
        "pain_points": ["之前被供应商坑过，担心再次选错", "团队对换工具抵触", "老板给的预算卡得很紧"],
        "current_tools": ["Notion", "飞书文档", "腾讯会议"],
        "team_size": "12人",
        "budget_range": "3-5万/年",
        "timeline": "年底前必须上线",
        "decision_makers": ["我（有建议权）", "老板（最终审批）"],
        "success_criteria": "团队成员愿意用，不出故障",
    },
    "rational_analyst": {
        "pain_points": ["数据散落在5个系统里", "手工报表每周花8小时", "缺乏实时指标看板"],
        "current_tools": ["SQL", "Python", "Tableau", "Excel"],
        "team_size": "8人数据团队",
        "budget_range": "10-20万/年",
        "timeline": "Q3完成POC，Q4全面推广",
        "decision_makers": ["我（有部门预算权）"],
        "success_criteria": "ROI可量化，数据准确率>99%",
    },
    "tech_skeptic": {
        "pain_points": ["新技术集成成本被低估", "之前用开源方案踩过坑", "担心vendor锁定"],
        "current_tools": ["Java", "Oracle", "Linux", "自研脚本"],
        "team_size": "30人技术部",
        "budget_range": "20-50万/年",
        "timeline": "明年Q1，前提是安全评审通过",
        "decision_makers": ["我（有较大技术预算权）", "CTO（重大项目）"],
        "success_criteria": "系统稳定性99.9%，支持私有化部署",
    },
    "impulsive_decider": {
        "pain_points": ["流程太慢耽误增长", "审批繁琐错失机会", "竞品已经在用了"],
        "current_tools": ["Figma", "剪映", "各种SaaS", "Google Analytics"],
        "team_size": "6人增长小队",
        "budget_range": "1-3万/年（小额快速决策）",
        "timeline": "这个月就要用",
        "decision_makers": ["我（有小额快速决策权）"],
        "success_criteria": "上手快，能看到数据提升",
    },
}


def generate_private_info(type_id: str, variation_seed: Optional[int] = None) -> Dict[str, Any]:
    """根据 persona type 生成私有信息（可随机变异）"""
    template = PRIVATE_INFO_TEMPLATES.get(type_id, PRIVATE_INFO_TEMPLATES["anxious_buyer"])

    if variation_seed is not None:
        rng = random.Random(variation_seed)
        # 简单的变异：随机打乱 pain_points 顺序
        pain_points = list(template.get("pain_points", []))
        rng.shuffle(pain_points)
        result = dict(template)
        result["pain_points"] = pain_points
        return result

    return dict(template)


# ───────────────────────────────────────────────
# 用户行为骨架生成器
# ───────────────────────────────────────────────

class UserBehaviorEngine:
    """
    用户行为引擎：根据销售阶段和 persona 性格，生成用户行为骨架。

    核心原则（来自 workflow-in-llm）：
    - 代码控制"用户应该做什么/说什么"
    - LLM 只负责把骨架翻译成自然语言
    """

    def __init__(
        self,
        persona: PersonaInstance,
        private_info: Optional[Dict[str, Any]] = None,
        behavior_overrides: Optional[Dict[str, Any]] = None,
        private_info_override: Optional[Dict[str, Any]] = None,
    ):
        self.persona = persona
        self.params = extract_behavior_params(persona, overrides=behavior_overrides)
        base_private = private_info or generate_private_info(persona.type_id)
        # 应用场景特定的私有信息覆盖
        if private_info_override:
            merged = dict(base_private)
            merged.update(private_info_override)
            self.private_info = merged
        else:
            self.private_info = base_private
        self.collected: Dict[str, bool] = {}
        self.stage_history: List[str] = []
        self.current_stage: Optional[str] = None

    def detect_stage(self, sales_msg: str) -> str:
        """检测当前销售阶段"""
        stage = detect_sales_stage(sales_msg, self.current_stage)
        if stage != self.current_stage:
            self.current_stage = stage
            self.stage_history.append(stage)
        return stage

    def build_user_skeleton(self, stage: str, sales_msg: str) -> str:
        """
        根据阶段和性格生成用户行为骨架。
        返回一个指令字符串，告诉 LLM 用户本回合应该做什么。
        """
        p = self.params
        pi = self.private_info

        # 已透露的信息追踪
        revealed_pain = self.collected.get("pain_points", False)
        revealed_budget = self.collected.get("budget_range", False)
        revealed_timeline = self.collected.get("timeline", False)
        revealed_tools = self.collected.get("current_tools", False)

        # ── 初次接触 ──
        if stage == "initial_contact":
            if p.info_release == "fast":
                return (
                    "礼貌简短回应，快速进入正题。可以说'你好，我了解到你们的产品，"
                    "想了解一下是否能解决我们【{}】的问题'。"
                    .format(pi["pain_points"][0][:15])
                )
            return (
                "保持谨慎，简短回应。只说'你好'或'有什么事'，"
                "不要主动透露任何需求或痛点，等对方先开口。"
            )

        # ── 需求挖掘 ──
        if stage == "need_discovery":
            if revealed_pain:
                # 已经说过一些痛点
                if p.hesitation > 0.4:
                    return (
                        "确认之前说的痛点，但保留核心细节。可以说'之前说的那些是表象，"
                        "更深入的问题是...'然后透露第二个痛点（保留最关键的不说）。"
                    )
                return (
                    "进一步展开痛点，补充具体场景和数据。透露第二个痛点，"
                    "并说明当前使用的工具是【{}】，团队【{}】。".format(
                        "、".join(pi["current_tools"][:2]), pi["team_size"]
                    )
                )

            # 第一次被问需求
            if p.hesitation > 0.5:
                return (
                    "只透露最表面的痛点，不说核心问题。可以说'最近确实有点效率问题，"
                    "但具体情况我还在整理'。不要提具体工具或预算。"
                )
            if p.info_release == "fast":
                return (
                    "直接说出核心痛点和团队现状。包括：主要痛点是【{}】，"
                    "团队【{}】，目前用【{}】。".format(
                        pi["pain_points"][0], pi["team_size"],
                        "、".join(pi["current_tools"][:2])
                    )
                )
            return (
                "描述主要痛点，但保留部分细节。可以说'我们团队在【{}】方面遇到一些问题，"
                "目前用【{}】，感觉不太够用'。".format(
                    pi["pain_points"][0][:10], "、".join(pi["current_tools"][:2])
                )
            )

        # ── 方案展示 ──
        if stage == "solution_presentation":
            if p.tech_challenger:
                return (
                    "对展示的功能表示怀疑，从技术角度提出挑战。可以问'这个功能在高并发下表现如何？"
                    "你们的数据存储方案是什么？有客户案例吗？'"
                )
            if p.skepticism_level > 0.6:
                return (
                    "保持怀疑，要求看到与自己场景相关的部分。可以说'这个功能听起来不错，"
                    "但我们的场景是【{}】，你们有类似案例吗？'".format(pi["pain_points"][0][:15])
                )
            if p.info_release == "fast":
                return (
                    "积极回应，表达对某些功能的兴趣，追问落地细节。可以说'这个功能正是我们需要的，"
                    "实施周期多久？需要团队配合做什么？'"
                )
            return (
                "谨慎评估，要求看到与自己场景相关的演示。可以说'能不能针对我们【{}】的场景"
                "具体说一下？我需要看到实际效果才能判断。'".format(pi["pain_points"][0][:10])
            )

        # ── 异议处理 ──
        if stage == "objection_handling":
            if p.risk_averse:
                return (
                    "强调稳定性和风险顾虑，要求更多保障。可以说'我理解你们的方案，"
                    "但我们最担心的是【{}】，你们有什么保障措施？需要试运行。'".format(
                        random.choice(["稳定性", "数据安全", "迁移成本", "团队学习成本"])
                    )
                )
            if p.tech_challenger:
                return (
                    "提出技术层面的深度质疑。可以说'我关心的是技术架构层面的问题："
                    "API 兼容性、数据迁移方案、故障恢复机制。能详细说说吗？'"
                )
            if p.hesitation > 0.4:
                return (
                    "表达顾虑，但给台阶。可以说'你说的有道理，但我还是有些担心..."
                    "能不能给我一些参考资料或者让我和团队商量一下？'"
                )
            return (
                "表达一些顾虑，但保持开放。可以说'这些确实是我们的关注点，"
                "你们在这方面有什么具体做法？'"
            )

        # ── 价格/试用谈判 ──
        if stage == "pricing_discussion":
            if p.price_sensitive:
                if not revealed_budget:
                    return (
                        "表示预算紧张，试探价格底线。可以说'我们的预算比较有限，"
                        "大概【{}】，你们这个方案在这个范围内能做吗？'".format(pi["budget_range"])
                    )
                return (
                    "要求折扣或更灵活的付费方案。可以说'这个价格超出我们预期了，"
                    "能不能按季度付？或者给初创团队一些优惠？'"
                )
            if p.quick_decider and not p.price_sensitive:
                return (
                    "如果价格合理，表示可以接受，询问签约流程。可以说'价格可以接受，"
                    "如果没问题的话我们这周就能定下来。'"
                )
            if not revealed_budget:
                return (
                    "要求详细报价，不急于表态。可以说'我需要看到详细报价单，"
                    "包括实施费用和后续维护成本。我们的预算范围是【{}】。'".format(pi["budget_range"])
                )
            return (
                "要求 ROI 计算和对比分析。可以说'这个价格我需要内部评估，"
                "你们能提供 ROI 计算器或者同行对比数据吗？'"
            )

        # ── 决策 ──
        if stage == "decision":
            if p.risk_averse and p.hesitation > 0.3:
                return (
                    "表示需要再考虑，要求试用或参考客户。可以说'我还需要再想想，"
                    "能不能先给我们一个小范围试用？或者我能不能联系你们的现有客户聊聊？'"
                )
            if p.quick_decider and not p.risk_averse:
                return (
                    "明确表示可以推进，询问下一步。可以说'我觉得可以，"
                    "接下来是什么流程？需要我这边准备什么材料？'"
                )
            if p.skepticism_level > 0.6:
                return (
                    "谨慎表示有意向，但设条件。可以说'如果试用效果达到预期，"
                    "我们可以推进。但我需要看到具体的 success criteria。'"
                )
            return (
                "谨慎表示有意向，但需要内部确认。可以说'我个人觉得方向是对的，"
                "但我需要走内部审批流程，大概需要{}时间。'".format(
                    random.choice(["一周", "两周", "两三天"])
                )
            )

        return "自然回应对方的话。"

    def get_behavior_instruction(self, stage: str, sales_msg: str) -> str:
        """
        生成完整的 behavior instruction，用于注入 system prompt。
        """
        skeleton = self.build_user_skeleton(stage, sales_msg)

        lines = [
            "",
            "=== 当前销售阶段 ===",
            f"销售人员当前处于【{self._stage_name_cn(stage)}】阶段。",
            "",
            "=== 你的行为指导 ===",
            skeleton,
            "",
            "=== 你的私有信息（未全部透露给销售）===",
            f"- 核心痛点：{self.private_info.get('pain_points', ['未知'])[0]}",
            f"- 团队规模：{self.private_info.get('team_size', '未知')}",
            f"- 预算范围：{self.private_info.get('budget_range', '未知')}",
            f"- 时间要求：{self.private_info.get('timeline', '未知')}",
            f"- 现有工具：{', '.join(self.private_info.get('current_tools', [])[:3])}",
            "",
            "=== 性格参数 ===",
            f"- 信息释放速度：{self.params.info_release}",
            f"- 犹豫度：{self.params.hesitation:.0%}",
            f"- 价格敏感：{'是' if self.params.price_sensitive else '否'}",
            f"- 风险厌恶：{'是' if self.params.risk_averse else '否'}",
            "",
            "【铁律】",
            "1. 严格按行为指导执行，不要偏离角色。",
            "2. 你的回复必须符合当前销售阶段——如果对方在问需求，你就说痛点；如果对方在展示方案，你就评估或质疑。",
            "3. 不要一次性透露所有私有信息，按信息释放速度逐步透露。",
            "4. 语言自然、口语化，像真实的微信/钉钉对话。",
        ]
        return "\n".join(lines)

    def update_collected(self, user_response: str):
        """
        从虚拟人的回复中简单检测已透露的信息。
        PoC 版本：关键词匹配。
        """
        r = user_response.lower()

        # 痛点相关
        pain_keywords = ["效率", "痛点", "问题", "困难", "麻烦", "慢", "乱", "散", "坑"]
        if any(k in r for k in pain_keywords):
            self.collected["pain_points"] = True

        # 预算相关
        budget_keywords = ["预算", "万", "元", "价格", "费用", "钱", "投资"]
        if any(k in r for k in budget_keywords):
            self.collected["budget_range"] = True

        # 时间相关
        time_keywords = ["时间", "期限", "年底", "q", "季度", "月", "周", "年前", "尽快"]
        if any(k in r for k in time_keywords):
            self.collected["timeline"] = True

        # 工具相关
        tool_keywords = ["用", "工具", "系统", "平台", "软件", "notion", "excel", "python", "sql"]
        if any(k in r for k in tool_keywords):
            self.collected["current_tools"] = True

        # 团队规模
        team_keywords = ["人", "团队", "部门", "同事"]
        if any(k in r for k in team_keywords):
            self.collected["team_size"] = True

    def get_stage_stats(self) -> Dict[str, Any]:
        """获取阶段统计"""
        return {
            "current_stage": self.current_stage,
            "stage_history": self.stage_history.copy(),
            "collected": self.collected.copy(),
            "stage_coverage": len(set(self.stage_history)) / len(SALES_STAGES),
            "unique_stages": list(dict.fromkeys(self.stage_history)),
        }

    @staticmethod
    def _stage_name_cn(stage: str) -> str:
        mapping = {
            "initial_contact": "初次接触",
            "need_discovery": "需求挖掘",
            "solution_presentation": "方案展示",
            "objection_handling": "异议处理",
            "pricing_discussion": "价格谈判",
            "decision": "决策确认",
        }
        return mapping.get(stage, stage)
