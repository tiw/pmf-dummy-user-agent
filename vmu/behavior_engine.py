"""
用户行为引擎 — 支持多 Domain（B2B 销售 + 旅行预订）。

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


# ═══════════════════════════════════════════════
# Domain 1: B2B 销售
# ═══════════════════════════════════════════════

B2B_STAGES = [
    "initial_contact",
    "need_discovery",
    "solution_presentation",
    "objection_handling",
    "pricing_discussion",
    "decision",
]

B2B_PRIVATE_INFO = {
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


def _detect_b2b_stage(msg: str, current_stage: Optional[str] = None) -> str:
    m = msg.lower()
    if any(w in m for w in ["确认", "签约", "合同", "付款", "定下来", "推进", "拍板", "走流程"]):
        return "decision"
    if any(w in m for w in ["价格", "费用", "多少钱", "预算", "折扣", "优惠", "试用", "pilot", "demo", "po", "采购"]):
        return "pricing_discussion"
    if any(w in m for w in ["担心", "顾虑", "问题", "风险", "安全", "兼容", "集成", "但是", "不过", "挑战", "困难"]):
        return "objection_handling"
    if any(w in m for w in ["功能", "特性", "可以", "支持", "方案", "解决", "产品", "展示", "介绍", "演示", "能力", "优势"]):
        return "solution_presentation"
    has_q = any(w in m for w in ["怎么", "什么", "如何", "为什么", "吗", "呢", "多少", "哪些"])
    has_biz = any(w in m for w in ["痛点", "现状", "需求", "团队", "流程", "工作", "用", "情况"])
    if has_q and has_biz:
        return "need_discovery"
    if len(m) < 50 and any(w in m for w in ["您好", "你好", "打扰", "请问"]):
        return "initial_contact"
    return current_stage if current_stage in B2B_STAGES else "initial_contact"


# ═══════════════════════════════════════════════
# Domain 2: 旅行预订
# ═══════════════════════════════════════════════

TRAVEL_STAGES = [
    "greet",
    "gather_destination",
    "gather_dates",
    "gather_budget",
    "gather_travelers",
    "check_ready",
    "present_options",
    "confirm_booking",
    "abandon",
    "end_success",
    "end_abandon",
]

TRAVEL_PRIVATE_INFO = {
    "budget_traveler": {
        "destination": "泰国",
        "month": "7月",
        "days": "5天",
        "budget": "5000元",
        "travelers": "2人",
        "special_need": "无",
        "pain_points": ["预算有限", "希望性价比高", "不想被坑"],
    },
    "luxury_seeker": {
        "destination": "马尔代夫",
        "month": "10月",
        "days": "7天",
        "budget": "50000元",
        "travelers": "情侣两人",
        "special_need": "蜜月",
        "pain_points": ["追求独特体验", "不在乎价格", "要求服务到位"],
    },
    "anxious_flyer": {
        "destination": "日本",
        "month": "3月",
        "days": "4天",
        "budget": "20000元",
        "travelers": "一家四口（带老人小孩）",
        "special_need": "有老人、需要保险",
        "pain_points": ["担心航班延误", "怕转机麻烦", "担心安全问题", "需要详细行程"],
    },
    "spontaneous_explorer": {
        "destination": "还没定",
        "month": "灵活",
        "days": "3-5天",
        "budget": "10000元",
        "travelers": "朋友3人",
        "special_need": "自由行",
        "pain_points": ["不想做攻略", "希望说走就走", "喜欢意外惊喜"],
    },
}


def _detect_travel_stage(msg: str, current_stage: Optional[str] = None) -> str:
    m = msg.lower()

    # 放弃（最高优先级）
    if any(w in m for w in ["算了", "不需要", "不用", "没兴趣", "不订了", "取消"]):
        return "abandon"

    # 检查信息是否足够（在 confirm_booking 之前，避免"确认一下"被误判）
    if any(w in m for w in ["确认一下", "核对", "汇总", "信息齐了", "齐全", "够了"]):
        return "check_ready"

    # 确认预订（排除"确认一下"这种信息核对场景）
    confirm_words = ["确认预订", "确认订单", "预订", "下单", "付款", "签合同", "定下来", "现在就定"]
    # 只有当消息里包含确认类词汇但不包含"一下"时，才认为是 confirm_booking
    if any(w in m for w in confirm_words) or ("确认" in m and "一下" not in m and "是否" not in m):
        return "confirm_booking"

    # 展示方案
    if any(w in m for w in ["方案", "推荐", "选择", "路线", "行程", "酒店", "景点", "套餐"]):
        return "present_options"

    # 问候（放在目的地/日期/预算之前，避免"您好，请问去哪"被误判为 gather_destination）
    if any(w in m for w in ["您好", "你好", "欢迎", "嗨", "hello", "顾问", "小美", "很高兴"]):
        return "greet"

    # 人数
    if any(w in m for w in ["几个人", "人数", "同行", "孩子", "老人", "带小孩", "情侣"]):
        return "gather_travelers"

    # 预算
    if any(w in m for w in ["预算", "多少钱", "价格", "费用", "花费", "人均"]):
        return "gather_budget"

    # 日期
    if any(w in m for w in ["什么时候", "日期", "几号", "几天", "出发", "回程", "行程天数"]):
        return "gather_dates"

    # 目的地
    if any(w in m for w in ["哪里", "目的地", "去哪", "城市", "国家", "想去"]):
        return "gather_destination"

    return current_stage if current_stage in TRAVEL_STAGES else "greet"


# ───────────────────────────────────────────────
# 性格参数映射（Domain 通用）
# ───────────────────────────────────────────────

def extract_behavior_params(
    persona: PersonaInstance,
    overrides: Optional[Dict[str, Any]] = None,
) -> BehaviorParams:
    bt = persona.behavioral_traits
    ps = persona.psychographics

    comm = bt.communication.lower()
    if any(w in comm for w in ["直接", "快节奏", "快", "简洁"]):
        info_release = "fast"
    elif any(w in comm for w in ["反复确认", "细节", "追问", "谨慎"]):
        info_release = "slow"
    else:
        info_release = "normal"

    hesitation = bt.skepticism_level * 0.5
    if "谨慎" in ps.decision_style or "数据驱动" in ps.decision_style:
        hesitation += 0.15
    hesitation = min(1.0, hesitation)

    price_sensitive = bt.price_sensitivity > 0.6
    risk_averse = bt.risk_tolerance in ["极低", "低"] or bt.skepticism_level > 0.7
    tech_challenger = bt.skepticism_level > 0.6 and "技术" in persona.demographics.role.lower()
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
    if overrides:
        params = params.apply_overrides(overrides)
    return params


# ───────────────────────────────────────────────
# 用户行为引擎
# ───────────────────────────────────────────────

class UserBehaviorEngine:
    """
    用户行为引擎：支持 B2B 销售和旅行预订两种 Domain。

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
        domain: str = "b2b_sales",
    ):
        self.persona = persona
        self.domain = domain
        self.params = extract_behavior_params(persona, overrides=behavior_overrides)

        # 选择私有信息模板
        if domain == "travel_booking":
            base_private = private_info or TRAVEL_PRIVATE_INFO.get(
                persona.type_id, TRAVEL_PRIVATE_INFO["budget_traveler"]
            )
        else:
            base_private = private_info or B2B_PRIVATE_INFO.get(
                persona.type_id, B2B_PRIVATE_INFO["anxious_buyer"]
            )

        if private_info_override:
            merged = dict(base_private)
            merged.update(private_info_override)
            self.private_info = merged
        else:
            self.private_info = base_private

        self.collected: Dict[str, bool] = {}
        self.stage_history: List[str] = []
        self.current_stage: Optional[str] = None

    # ── 阶段检测 ──

    def detect_stage(self, sales_msg: str) -> str:
        if self.domain == "travel_booking":
            stage = _detect_travel_stage(sales_msg, self.current_stage)
        else:
            stage = _detect_b2b_stage(sales_msg, self.current_stage)
        if stage != self.current_stage:
            self.current_stage = stage
            self.stage_history.append(stage)
        return stage

    # ── 行为骨架 ──

    def build_user_skeleton(self, stage: str, sales_msg: str) -> str:
        if self.domain == "travel_booking":
            return self._build_travel_skeleton(stage, sales_msg)
        return self._build_b2b_skeleton(stage, sales_msg)

    def _build_b2b_skeleton(self, stage: str, sales_msg: str) -> str:
        p = self.params
        pi = self.private_info
        revealed_pain = self.collected.get("pain_points", False)
        revealed_budget = self.collected.get("budget_range", False)

        if stage == "initial_contact":
            if p.info_release == "fast":
                return f"礼貌简短回应，快速进入正题。可以说'你好，我了解到你们的产品，想了解一下是否能解决我们【{pi['pain_points'][0][:15]}】的问题'。"
            return "保持谨慎，简短回应。只说'你好'或'有什么事'，不要主动透露任何需求或痛点，等对方先开口。"

        if stage == "need_discovery":
            if revealed_pain:
                if p.hesitation > 0.4:
                    return "确认之前说的痛点，但保留核心细节。可以说'之前说的那些是表象，更深入的问题是...'然后透露第二个痛点（保留最关键的不说）。"
                return f"进一步展开痛点，补充具体场景和数据。透露第二个痛点，并说明当前使用的工具是【{'、'.join(pi['current_tools'][:2])}】，团队【{pi['team_size']}】。"
            if p.hesitation > 0.5:
                return "只透露最表面的痛点，不说核心问题。可以说'最近确实有点效率问题，但具体情况我还在整理'。不要提具体工具或预算。"
            if p.info_release == "fast":
                return f"直接说出核心痛点和团队现状。包括：主要痛点是【{pi['pain_points'][0]}】，团队【{pi['team_size']}】，目前用【{'、'.join(pi['current_tools'][:2])}】。"
            return f"描述主要痛点，但保留部分细节。可以说'我们团队在【{pi['pain_points'][0][:10]}】方面遇到一些问题，目前用【{'、'.join(pi['current_tools'][:2])}】，感觉不太够用'。"

        if stage == "solution_presentation":
            if p.tech_challenger:
                return "对展示的功能表示怀疑，从技术角度提出挑战。可以问'这个功能在高并发下表现如何？你们的数据存储方案是什么？有客户案例吗？'"
            if p.skepticism_level > 0.6:
                return f"保持怀疑，要求看到与自己场景相关的部分。可以说'这个功能听起来不错，但我们的场景是【{pi['pain_points'][0][:15]}】，你们有类似案例吗？'"
            if p.info_release == "fast":
                return "积极回应，表达对某些功能的兴趣，追问落地细节。可以说'这个功能正是我们需要的，实施周期多久？需要团队配合做什么？'"
            return f"谨慎评估，要求看到与自己场景相关的演示。可以说'能不能针对我们【{pi['pain_points'][0][:10]}】的场景具体说一下？我需要看到实际效果才能判断。'"

        if stage == "objection_handling":
            if p.risk_averse:
                return f"强调稳定性和风险顾虑，要求更多保障。可以说'我理解你们的方案，但我们最担心的是【{random.choice(['稳定性', '数据安全', '迁移成本', '团队学习成本'])}】，你们有什么保障措施？需要试运行。'"
            if p.tech_challenger:
                return "提出技术层面的深度质疑。可以说'我关心的是技术架构层面的问题：API 兼容性、数据迁移方案、故障恢复机制。能详细说说吗？'"
            if p.hesitation > 0.4:
                return "表达顾虑，但给台阶。可以说'你说的有道理，但我还是有些担心...能不能给我一些参考资料或者让我和团队商量一下？'"
            return "表达一些顾虑，但保持开放。可以说'这些确实是我们的关注点，你们在这方面有什么具体做法？'"

        if stage == "pricing_discussion":
            if p.price_sensitive:
                if not revealed_budget:
                    return f"表示预算紧张，试探价格底线。可以说'我们的预算比较有限，大概【{pi['budget_range']}】，你们这个方案在这个范围内能做吗？'"
                return "要求折扣或更灵活的付费方案。可以说'这个价格超出我们预期了，能不能按季度付？或者给初创团队一些优惠？'"
            if p.quick_decider and not p.price_sensitive:
                return "如果价格合理，表示可以接受，询问签约流程。可以说'价格可以接受，如果没问题的话我们这周就能定下来。'"
            if not revealed_budget:
                return f"要求详细报价，不急于表态。可以说'我需要看到详细报价单，包括实施费用和后续维护成本。我们的预算范围是【{pi['budget_range']}】。'"
            return "要求 ROI 计算和对比分析。可以说'这个价格我需要内部评估，你们能提供 ROI 计算器或者同行对比数据吗？'"

        if stage == "decision":
            if p.risk_averse and p.hesitation > 0.3:
                return "表示需要再考虑，要求试用或参考客户。可以说'我还需要再想想，能不能先给我们一个小范围试用？或者我能不能联系你们的现有客户聊聊？'"
            if p.quick_decider and not p.risk_averse:
                return "明确表示可以推进，询问下一步。可以说'我觉得可以，接下来是什么流程？需要我这边准备什么材料？'"
            if p.skepticism_level > 0.6:
                return "谨慎表示有意向，但设条件。可以说'如果试用效果达到预期，我们可以推进。但我需要看到具体的 success criteria。'"
            return f"谨慎表示有意向，但需要内部确认。可以说'我个人觉得方向是对的，但我需要走内部审批流程，大概需要{random.choice(['一周', '两周', '两三天'])}时间。'"

        return "自然回应对方的话。"

    def _build_travel_skeleton(self, stage: str, sales_msg: str) -> str:
        p = self.params
        pi = self.private_info

        revealed_dest = self.collected.get("destination", False)
        revealed_dates = self.collected.get("dates", False)
        revealed_budget = self.collected.get("budget", False)
        revealed_travelers = self.collected.get("travelers", False)

        # ── 问候 ──
        if stage == "greet":
            if p.info_release == "fast":
                dest = pi.get("destination", "某个地方")
                return f"热情回应，直接表明旅行意向。可以说'你好，我想预订一次去【{dest}】的旅行。'"
            return "简单回应问候，不急于暴露需求。可以说'你好，我想了解一下旅行预订。'"

        # ── 收集目的地 ──
        if stage == "gather_destination":
            if revealed_dest:
                return f"确认目的地。可以说'对，就是想去【{pi['destination']}】。'"
            if p.hesitation > 0.4:
                return f"透露目的地但留有余地。可以说'可能想去【{pi['destination']}】吧，但还在考虑其他城市。'"
            return f"直接说出目的地。可以说'想去【{pi['destination']}】。'"

        # ── 收集日期 ──
        if stage == "gather_dates":
            if revealed_dates:
                return f"确认日期。可以说'对，【{pi['month']}】出发，玩【{pi['days']}】。'"
            if p.hesitation > 0.4:
                return f"说大概时间但不完全确定。可以说'大概【{pi['month']}】吧，天数还没完全想好，可能【{pi['days']}】左右。'"
            return f"直接给出日期。可以说'计划【{pi['month']}】出发，玩【{pi['days']}】。'"

        # ── 收集预算 ──
        if stage == "gather_budget":
            if revealed_budget:
                return f"确认预算。可以说'对，预算大概【{pi['budget']}】。'"
            if p.price_sensitive:
                return f"表示预算紧张，试探价格底线。可以说'预算比较有限，大概【{pi['budget']}】，有没有什么性价比高的方案？'"
            if p.quick_decider:
                return f"直接说预算。可以说'预算【{pi['budget']}】左右，只要体验好就行。'"
            return f"给出预算范围。可以说'预算大概【{pi['budget']}】，看看能安排什么样的行程。'"

        # ── 收集人数 ──
        if stage == "gather_travelers":
            if revealed_travelers:
                return f"确认人数。可以说'对，【{pi['travelers']}】。'"
            if p.hesitation > 0.3:
                return f"透露人数但补充细节。可以说'【{pi['travelers']}】，其中有老人/小孩，需要安排方便一点的行程。'"
            return f"直接说人数。可以说'【{pi['travelers']}】一起出行。'"

        # ── 检查信息 ──
        if stage == "check_ready":
            return "确认信息是否齐全。可以说'我看看...目的地、日期、预算、人数都说了，应该差不多了吧？'"

        # ── 展示方案 ──
        if stage == "present_options":
            if p.price_sensitive:
                return "关注性价比。可以说'这个方案听起来不错，但价格能不能再优惠一点？或者有没有更经济的替代方案？'"
            if p.risk_averse:
                return "关注安全和保障。可以说'行程安排看起来可以，但我想确认一下：有没有保险？航班如果延误怎么处理？'"
            if p.quick_decider:
                return "快速表态。可以说'方案一不错，就这个吧，赶紧定下来。'"
            return "审慎评估。可以说'两个方案都不错，我再比较一下。能不能给一份详细的行程单？'"

        # ── 确认预订 ──
        if stage == "confirm_booking":
            if p.risk_averse and p.hesitation > 0.3:
                return "要求更多保障。可以说'我再确认一下：如果临时去不了，能退改吗？取消政策是什么？'"
            if p.quick_decider:
                return "快速确认。可以说'没问题，确认预订。什么时候付款？'"
            return "谨慎确认。可以说'好的，我确认一下细节...对，就按这个方案。'"

        # ── 放弃 ──
        if stage == "abandon":
            return "礼貌表示暂时不订。可以说'不好意思，暂时先不订了，我再考虑考虑。'"

        # ── 结束 ──
        if stage in ("end_success", "end_abandon"):
            return "礼貌结束对话。可以说'好的，谢谢，期待旅行！'"

        return "自然回应对方的话。"

    # ── 行为指令 ──

    def get_behavior_instruction(self, stage: str, sales_msg: str) -> str:
        if self.domain == "travel_booking":
            return self._get_travel_instruction(stage, sales_msg)
        return self._get_b2b_instruction(stage, sales_msg)

    def _get_b2b_instruction(self, stage: str, sales_msg: str) -> str:
        skeleton = self._build_b2b_skeleton(stage, sales_msg)
        pi = self.private_info
        lines = [
            "",
            "=== 当前销售阶段 ===",
            f"销售人员当前处于【{self._stage_name_cn(stage)}】阶段。",
            "",
            "=== 你的行为指导 ===",
            skeleton,
            "",
            "=== 你的私有信息（未全部透露给销售）===",
            f"- 核心痛点：{pi.get('pain_points', ['未知'])[0]}",
            f"- 团队规模：{pi.get('team_size', '未知')}",
            f"- 预算范围：{pi.get('budget_range', '未知')}",
            f"- 时间要求：{pi.get('timeline', '未知')}",
            f"- 现有工具：{', '.join(pi.get('current_tools', [])[:3])}",
            "",
            "=== 性格参数 ===",
            f"- 信息释放速度：{self.params.info_release}",
            f"- 犹豫度：{self.params.hesitation:.0%}",
            f"- 价格敏感：{'是' if self.params.price_sensitive else '否'}",
            f"- 风险厌恶：{'是' if self.params.risk_averse else '否'}",
            "",
            "【铁律】",
            "1. 严格按行为指导执行，不要偏离角色。",
            "2. 你的回复必须符合当前销售阶段。",
            "3. 不要一次性透露所有私有信息，按信息释放速度逐步透露。",
            "4. 语言自然、口语化，像真实的微信/钉钉对话。",
        ]
        return "\n".join(lines)

    def _get_travel_instruction(self, stage: str, sales_msg: str) -> str:
        skeleton = self._build_travel_skeleton(stage, sales_msg)
        pi = self.private_info
        lines = [
            "",
            "=== 当前预订阶段 ===",
            f"客服当前处于【{self._travel_stage_name_cn(stage)}】阶段。",
            "",
            "=== 你的行为指导 ===",
            skeleton,
            "",
            "=== 你的私有信息（未全部透露给客服）===",
            f"- 目的地：{pi.get('destination', '未知')}",
            f"- 出发时间：{pi.get('month', '未知')}",
            f"- 天数：{pi.get('days', '未知')}",
            f"- 预算：{pi.get('budget', '未知')}",
            f"- 人数：{pi.get('travelers', '未知')}",
            f"- 特殊需求：{pi.get('special_need', '无')}",
            "",
            "=== 性格参数 ===",
            f"- 信息释放速度：{self.params.info_release}",
            f"- 犹豫度：{self.params.hesitation:.0%}",
            f"- 价格敏感：{'是' if self.params.price_sensitive else '否'}",
            f"- 风险厌恶：{'是' if self.params.risk_averse else '否'}",
            "",
            "【铁律】",
            "1. 严格按行为指导执行，不要偏离角色。",
            "2. 你的回复必须符合当前预订阶段。",
            "3. 不要一次性透露所有私有信息，按信息释放速度逐步透露。",
            "4. 语言自然、口语化，像真实的微信/客服对话。",
            "5. 客服问什么你答什么，不要主动推荐景点或酒店。",
        ]
        return "\n".join(lines)

    # ── 信息收集更新 ──

    def update_collected(self, user_response: str):
        if self.domain == "travel_booking":
            self._update_travel_collected(user_response)
        else:
            self._update_b2b_collected(user_response)

    def _update_b2b_collected(self, user_response: str):
        r = user_response.lower()
        pain_keywords = ["效率", "痛点", "问题", "困难", "麻烦", "慢", "乱", "散", "坑"]
        if any(k in r for k in pain_keywords):
            self.collected["pain_points"] = True
        budget_keywords = ["预算", "万", "元", "价格", "费用", "钱", "投资"]
        if any(k in r for k in budget_keywords):
            self.collected["budget_range"] = True
        time_keywords = ["时间", "期限", "年底", "q", "季度", "月", "周", "年前", "尽快"]
        if any(k in r for k in time_keywords):
            self.collected["timeline"] = True
        tool_keywords = ["用", "工具", "系统", "平台", "软件", "notion", "excel", "python", "sql"]
        if any(k in r for k in tool_keywords):
            self.collected["current_tools"] = True
        team_keywords = ["人", "团队", "部门", "同事"]
        if any(k in r for k in team_keywords):
            self.collected["team_size"] = True

    def _update_travel_collected(self, user_response: str):
        r = user_response.lower()
        dest_keywords = ["日本", "泰国", "马尔代夫", "巴黎", "新西兰", "巴厘岛", "去哪", "目的地"]
        if any(k in r for k in dest_keywords):
            self.collected["destination"] = True
        date_keywords = ["月", "号", "天", "出发", "回程", "行程", " week", "weekend"]
        if any(k in r for k in date_keywords):
            self.collected["dates"] = True
        budget_keywords = ["预算", "万", "元", "价格", "费用", "钱", "人均", "花费"]
        if any(k in r for k in budget_keywords):
            self.collected["budget"] = True
        traveler_keywords = ["人", "口", "小孩", "老人", "情侣", "朋友", "一家", "亲子"]
        if any(k in r for k in traveler_keywords):
            self.collected["travelers"] = True

    # ── 统计 ──

    def get_stage_stats(self) -> Dict[str, Any]:
        stages = TRAVEL_STAGES if self.domain == "travel_booking" else B2B_STAGES
        return {
            "current_stage": self.current_stage,
            "stage_history": self.stage_history.copy(),
            "collected": self.collected.copy(),
            "stage_coverage": len(set(self.stage_history)) / len(stages),
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

    @staticmethod
    def _travel_stage_name_cn(stage: str) -> str:
        mapping = {
            "greet": "问候",
            "gather_destination": "收集目的地",
            "gather_dates": "收集日期",
            "gather_budget": "收集预算",
            "gather_travelers": "收集人数",
            "check_ready": "检查信息",
            "present_options": "展示方案",
            "confirm_booking": "确认预订",
            "abandon": "放弃",
            "end_success": "预订成功",
            "end_abandon": "结束",
        }
        return mapping.get(stage, stage)
