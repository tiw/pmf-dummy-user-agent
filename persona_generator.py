#!/usr/bin/env python3
"""
虚拟用户生成 Agent —— LLM 增强版
基于「五层设计框架」+ DeepSeek API

用法:
  python persona_generator.py          # 交互模式（推荐快速模式）
  python persona_generator.py --guide  # 仅显示需要收集的信息清单
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from deepseek_client import (
    expand_persona,
    optimize_system_prompt,
    critique_prompt,
    chat_completion,
    DEEPSEEK_MODEL_PRO,
    DEEPSEEK_MODEL_FLASH,
)


# ───────────────────────────────────────────────
# 数据模型
# ───────────────────────────────────────────────

@dataclass
class Demographics:
    age: int
    role: str
    company_size: str
    industry: str
    location: str
    years_experience: int


@dataclass
class Psychographics:
    goals: List[str]
    frustrations: List[str]
    decision_style: str
    tech_stack: List[str]
    budget_authority: str


@dataclass
class BehavioralTraits:
    communication: str
    skepticism_level: float
    price_sensitivity: float
    risk_tolerance: str


@dataclass
class Context:
    current_problem: str
    recent_changes: str
    team_pressure: str
    competitive_exposure: List[str]


@dataclass
class Persona:
    persona_id: str
    demographics: Demographics
    psychographics: Psychographics
    behavioral_traits: BehavioralTraits
    context: Context


@dataclass
class SceneContext:
    scene_description: str
    initial_attitude: str
    time_pressure: str
    participation_motivation: str
    prior_exposure: str


@dataclass
class MemoryState:
    trust_level: float
    emotional_state: str
    exposure_count: int = 0


@dataclass
class BehaviorEngine:
    attention_keywords: List[str]
    skepticism_triggers: List[str]
    knowledge_boundary: str


@dataclass
class SyntheticUser:
    persona: Persona
    scene: SceneContext
    memory: MemoryState
    behavior: BehaviorEngine
    system_prompt: str = ""
    critique_result: Optional[dict] = None


# ───────────────────────────────────────────────
# 辅助函数
# ───────────────────────────────────────────────

def _ask(question: str, default: str = "") -> str:
    if default:
        prompt = f"{question} [{default}]: "
    else:
        prompt = f"{question}: "
    answer = input(prompt).strip()
    return answer if answer else default


def _ask_int(question: str, default: int = 0) -> int:
    raw = _ask(question, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _ask_float(question: str, default: float = 0.5) -> float:
    raw = _ask(question, str(default))
    try:
        v = float(raw)
        return max(0.0, min(1.0, v))
    except ValueError:
        return default


def _ask_list(question: str) -> List[str]:
    raw = _ask(question)
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _ask_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(question + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "是", "确认", "ok")


def _sanitize_id(text: str) -> str:
    import re
    return re.sub(r'[^\w\u4e00-\u9fff]+', '_', text).strip('_')


# ───────────────────────────────────────────────
# 模式 A：LLM 快速扩展模式
# ───────────────────────────────────────────────

def mode_llm_expand() -> SyntheticUser:
    """用户输入简要描述，LLM 自动扩展为完整 persona。"""
    print()
    print("─" * 50)
    print("🚀 LLM 快速扩展模式")
    print("─" * 50)
    print("你只需描述你想模拟的用户，DeepSeek 会自动生成完整细节。")
    print()
    print("示例描述：")
    print('  "一个在大厂工作了5年的产品经理，对数据分析工具很挑剔，')
    print('   之前用过 Tableau 觉得太贵，现在在看国产替代方案。')
    print('   他正在参加一个 B2B SaaS 的 demo 会议，时间很紧。"')
    print()
    
    product_name = _ask("首先，你的产品/服务名称是什么")
    product_type = _ask("产品类型 (如: B2B SaaS, 工具 App)")
    
    print()
    print("请用一段话描述你想模拟的用户（越具体越好）：")
    user_description = input("> ").strip()
    
    if not user_description:
        print("描述为空，切换到手动模式...")
        return mode_manual(product_name, product_type)
    
    # 让 LLM 扩展
    print()
    print("🤖 正在调用 DeepSeek 生成 persona 细节...")
    print("   （这可能需要 10-30 秒）")
    print()
    
    try:
        expanded = expand_persona(
            f"产品：{product_name}（{product_type}）\n用户描述：{user_description}"
        )
    except Exception as e:
        print(f"❌ DeepSeek API 调用失败: {e}")
        print("切换到手动模式...")
        return mode_manual(product_name, product_type)
    
    # 解析 LLM 返回的 JSON
    d = expanded.get("demographics", {})
    p = expanded.get("psychographics", {})
    bt = expanded.get("behavioral_traits", {})
    ctx = expanded.get("context", {})
    sc = expanded.get("scene_context", {})
    be = expanded.get("behavior_engine", {})
    
    # 展示给用户确认
    print()
    print("=" * 60)
    print("📋 DeepSeek 生成的 Persona 草案")
    print("=" * 60)
    print()
    print(f"👤 身份：{d.get('role', 'N/A')}，{d.get('age', 'N/A')} 岁，{d.get('industry', 'N/A')}")
    print(f"🏢 公司：{d.get('company_size', 'N/A')}，{d.get('location', 'N/A')}")
    print(f"🎯 目标：{', '.join(p.get('goals', []))}")
    print(f"😤 痛点：{', '.join(p.get('frustrations', []))}")
    print(f"📊 怀疑度：{bt.get('skepticism_level', 'N/A')} | 价格敏感度：{bt.get('price_sensitivity', 'N/A')}")
    print(f"🎬 场景：{sc.get('scene_description', 'N/A')}")
    print(f"🎭 初始态度：{sc.get('initial_attitude', 'N/A')}")
    print()
    
    if _ask_yes_no("是否直接使用这个 persona？（否 = 进入手动修改）", True):
        # 用户确认，直接构建
        persona = Persona(
            persona_id=f"{_sanitize_id(d.get('role', 'user'))}_{d.get('age', 0):02d}",
            demographics=Demographics(
                age=d.get("age", 30),
                role=d.get("role", "用户"),
                company_size=d.get("company_size", "未知"),
                industry=d.get("industry", "未知"),
                location=d.get("location", "未知"),
                years_experience=d.get("years_experience", 3),
            ),
            psychographics=Psychographics(
                goals=p.get("goals", []),
                frustrations=p.get("frustrations", []),
                decision_style=p.get("decision_style", "谨慎"),
                tech_stack=p.get("tech_stack", []),
                budget_authority=p.get("budget_authority", "无采购权"),
            ),
            behavioral_traits=BehavioralTraits(
                communication=bt.get("communication", "直接"),
                skepticism_level=float(bt.get("skepticism_level", 0.5)),
                price_sensitivity=float(bt.get("price_sensitivity", 0.5)),
                risk_tolerance=bt.get("risk_tolerance", "中"),
            ),
            context=Context(
                current_problem=ctx.get("current_problem", "效率低"),
                recent_changes=ctx.get("recent_changes", "无"),
                team_pressure=ctx.get("team_pressure", "一般"),
                competitive_exposure=ctx.get("competitive_exposure", []),
            ),
        )
        
        scene = SceneContext(
            scene_description=sc.get("scene_description", "评估产品"),
            initial_attitude=sc.get("initial_attitude", "中立"),
            time_pressure=sc.get("time_pressure", "一般"),
            participation_motivation=sc.get("participation_motivation", "解决痛点"),
            prior_exposure=sc.get("prior_exposure", "未接触"),
        )
        
        memory = MemoryState(
            trust_level=0.3,
            emotional_state="skeptical",
            exposure_count=0,
        )
        
        behavior = BehaviorEngine(
            attention_keywords=be.get("attention_keywords", ["效率", "价格", "ROI"]),
            skepticism_triggers=be.get("skepticism_triggers", ["赋能", "生态"]),
            knowledge_boundary=be.get("knowledge_boundary", "知道自己擅长什么"),
        )
        
        user = SyntheticUser(persona=persona, scene=scene, memory=memory, behavior=behavior)
        user._product_name = product_name
        user._product_type = product_type
        return user
    else:
        # 用户想修改，进入手动模式但预填值
        return mode_manual(product_name, product_type, expanded)


# ───────────────────────────────────────────────
# 模式 B：手动逐字段填写模式
# ───────────────────────────────────────────────

def mode_manual(product_name: str, product_type: str, prefill: Optional[dict] = None) -> SyntheticUser:
    """交互式逐字段收集，支持预填值。"""
    
    def p(path: str, default=None):
        """从预填数据中获取值。"""
        if not prefill:
            return default
        keys = path.split(".")
        val = prefill
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            else:
                return default
        return val if val is not None else default
    
    print()
    print("─" * 50)
    print("✏️  手动填写模式")
    print("─" * 50)
    
    d = p("demographics", {})
    ps = p("psychographics", {})
    bt = p("behavioral_traits", {})
    ctx = p("context", {})
    sc = p("scene_context", {})
    be = p("behavior_engine", {})
    
    role = _ask("职位/角色", d.get("role", ""))
    industry = _ask("所在行业", d.get("industry", "互联网/科技"))
    company_size = _ask("公司规模", d.get("company_size", "50-500人"))
    location = _ask("所在城市/地区", d.get("location", "一线城市"))
    age = _ask_int("年龄", d.get("age", 32))
    years_exp = _ask_int("工作经验(年)", d.get("years_experience", 5))
    
    goals = _ask_list("工作目标（逗号分隔）") or ps.get("goals", ["提高效率"])
    frustrations = _ask_list("当前痛点（逗号分隔）") or ps.get("frustrations", ["工具复杂"])
    decision_style = _ask("决策风格", ps.get("decision_style", "数据驱动，谨慎"))
    tech_stack = _ask_list("常用技术/工具（逗号分隔）") or ps.get("tech_stack", [])
    budget_authority = _ask("采购权限", ps.get("budget_authority", "有建议权，无最终决策权"))
    
    scene_desc = _ask("模拟场景描述", sc.get("scene_description", f"正在评估是否采用 {product_name}"))
    initial_attitude = _ask("初始态度", sc.get("initial_attitude", "怀疑但愿意了解"))
    time_pressure = _ask("时间压力", sc.get("time_pressure", "中等"))
    participation_motivation = _ask("参与动机", sc.get("participation_motivation", "解决当前痛点"))
    prior_exposure = _ask("之前对产品的接触程度", sc.get("prior_exposure", "听说过，未深入试用"))
    
    communication = _ask("沟通风格", bt.get("communication", "直接，注重效率"))
    skepticism = _ask_float("怀疑程度 (0-1)", bt.get("skepticism_level", 0.6))
    price_sensitivity = _ask_float("价格敏感度 (0-1)", bt.get("price_sensitivity", 0.5))
    risk_tolerance = _ask("风险承受度", bt.get("risk_tolerance", "中低"))
    
    attention_keywords = _ask_list("最关心的关键词（逗号分隔）") or be.get("attention_keywords", ["效率", "价格"])
    skepticism_triggers = _ask_list("触发反感的词汇") or be.get("skepticism_triggers", ["赋能", "生态"])
    
    current_problem = _ask("当前最迫切的问题", ctx.get("current_problem", frustrations[0] if frustrations else ""))
    recent_changes = _ask("最近的工作/环境变化", ctx.get("recent_changes", "无重大变化"))
    team_pressure = _ask("团队/公司当前压力", ctx.get("team_pressure", "业绩压力"))
    competitive_exposure = _ask_list("竞品使用经历") or ctx.get("competitive_exposure", [])
    
    persona = Persona(
        persona_id=f"{_sanitize_id(role)}_{age:02d}",
        demographics=Demographics(
            age=age, role=role, company_size=company_size,
            industry=industry, location=location, years_experience=years_exp,
        ),
        psychographics=Psychographics(
            goals=goals, frustrations=frustrations,
            decision_style=decision_style, tech_stack=tech_stack,
            budget_authority=budget_authority,
        ),
        behavioral_traits=BehavioralTraits(
            communication=communication, skepticism_level=skepticism,
            price_sensitivity=price_sensitivity, risk_tolerance=risk_tolerance,
        ),
        context=Context(
            current_problem=current_problem or "工作中存在效率瓶颈",
            recent_changes=recent_changes, team_pressure=team_pressure,
            competitive_exposure=competitive_exposure,
        ),
    )
    
    scene = SceneContext(
        scene_description=scene_desc, initial_attitude=initial_attitude,
        time_pressure=time_pressure, participation_motivation=participation_motivation,
        prior_exposure=prior_exposure,
    )
    
    memory = MemoryState(trust_level=0.3, emotional_state="skeptical", exposure_count=0)
    
    behavior = BehaviorEngine(
        attention_keywords=attention_keywords,
        skepticism_triggers=skepticism_triggers,
        knowledge_boundary="知道自己擅长什么，不会假装懂不熟悉的领域",
    )
    
    user = SyntheticUser(persona=persona, scene=scene, memory=memory, behavior=behavior)
    user._product_name = product_name
    user._product_type = product_type
    return user


# ───────────────────────────────────────────────
# System Prompt 构建（模板版）
# ───────────────────────────────────────────────

def build_raw_system_prompt(user: SyntheticUser, product_name: str, product_type: str) -> str:
    """先用模板生成一版结构化的 raw prompt，供 LLM 优化。"""
    p = user.persona
    d = p.demographics
    ps = p.psychographics
    bt = p.behavioral_traits
    ctx = p.context
    s = user.scene
    
    lines = [
        f"# 虚拟用户身份设定：{d.role}",
        "",
        f"你是一位 {d.age} 岁的 {d.role}，在 {d.industry} 行业工作 {d.years_experience} 年。",
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
    
    lines.extend(["", "## 你的 frustrates"])
    for f in ps.frustrations:
        lines.append(f"- {f}")
    
    lines.extend([
        "", "## 你的沟通风格",
        f"- {bt.communication}",
        f"- 怀疑程度：{'高' if bt.skepticism_level > 0.6 else '中' if bt.skepticism_level > 0.3 else '低'}（{bt.skepticism_level}）",
        f"- 价格敏感度：{'高' if bt.price_sensitivity > 0.6 else '中' if bt.price_sensitivity > 0.3 else '低'}（{bt.price_sensitivity}）",
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
        f"- 如果对方使用 {', '.join(user.behavior.skepticism_triggers)} 这类词汇，你会表现出怀疑或反感。",
        "- 你不会主动暴露公司敏感信息。",
        f"- 你没有最终采购权（{ps.budget_authority}），不会说'立即购买'。",
        "- 如果产品不解决你的问题，你会直接说'这不解决我的问题'。",
        "", "## 记忆状态（初始）",
        f"- 信任度：{user.memory.trust_level}/1.0",
        f"- 情绪：{user.memory.emotional_state}",
        f"- 接触次数：{user.memory.exposure_count}",
    ])
    
    return "\n".join(lines)


# ───────────────────────────────────────────────
# 输出格式化
# ───────────────────────────────────────────────

class PersonaExporter:
    @staticmethod
    def to_yaml(user: SyntheticUser) -> str:
        p = user.persona
        lines = [
            f"# 虚拟用户定义：{p.demographics.role}",
            f"# 框架版本：五层设计框架 v2.0 (LLM增强版)",
            "",
            "# ═══════════════════════════════════════════════",
            "# Layer 1: 角色定义层 (Persona Definition)",
            "# ═══════════════════════════════════════════════",
            "",
            f"persona_id: \"{p.persona_id}\"",
            "",
            "demographics:",
            f"  age: {p.demographics.age}",
            f"  role: \"{p.demographics.role}\"",
            f"  company_size: \"{p.demographics.company_size}\"",
            f"  industry: \"{p.demographics.industry}\"",
            f"  location: \"{p.demographics.location}\"",
            f"  years_experience: {p.demographics.years_experience}",
            "",
            "psychographics:",
            f"  goals: {json.dumps(p.psychographics.goals, ensure_ascii=False)}",
            f"  frustrations: {json.dumps(p.psychographics.frustrations, ensure_ascii=False)}",
            f"  decision_style: \"{p.psychographics.decision_style}\"",
            f"  tech_stack: {json.dumps(p.psychographics.tech_stack, ensure_ascii=False)}",
            f"  budget_authority: \"{p.psychographics.budget_authority}\"",
            "",
            "behavioral_traits:",
            f"  communication: \"{p.behavioral_traits.communication}\"",
            f"  skepticism_level: {p.behavioral_traits.skepticism_level}",
            f"  price_sensitivity: {p.behavioral_traits.price_sensitivity}",
            f"  risk_tolerance: \"{p.behavioral_traits.risk_tolerance}\"",
            "",
            "context:",
            f"  current_problem: \"{p.context.current_problem}\"",
            f"  recent_changes: \"{p.context.recent_changes}\"",
            f"  team_pressure: \"{p.context.team_pressure}\"",
            f"  competitive_exposure: {json.dumps(p.context.competitive_exposure, ensure_ascii=False)}",
            "",
            "# ═══════════════════════════════════════════════",
            "# Layer 2: 大模型条件化 (LLM Conditioning)",
            "# ═══════════════════════════════════════════════",
            "",
            "system_prompt: |",
        ]
        for line in user.system_prompt.splitlines():
            lines.append(f"  {line}")
        
        lines.extend([
            "",
            "# ═══════════════════════════════════════════════",
            "# Layer 3: 场景注入层 (Scene Context)",
            "# ═══════════════════════════════════════════════",
            "",
            "scene_context:",
            f"  scene_description: \"{user.scene.scene_description}\"",
            f"  initial_attitude: \"{user.scene.initial_attitude}\"",
            f"  time_pressure: \"{user.scene.time_pressure}\"",
            f"  participation_motivation: \"{user.scene.participation_motivation}\"",
            f"  prior_exposure: \"{user.scene.prior_exposure}\"",
            "",
            "# ═══════════════════════════════════════════════",
            "# Layer 4: 记忆系统层 (Memory & State)",
            "# ═══════════════════════════════════════════════",
            "",
            "memory_state:",
            f"  trust_level: {user.memory.trust_level}",
            f"  emotional_state: \"{user.memory.emotional_state}\"",
            f"  exposure_count: {user.memory.exposure_count}",
            "",
            "# ═══════════════════════════════════════════════",
            "# Layer 5: 行为引擎层 (Behavior Engine)",
            "# ═══════════════════════════════════════════════",
            "",
            "behavior_engine:",
            f"  attention_keywords: {json.dumps(user.behavior.attention_keywords, ensure_ascii=False)}",
            f"  skepticism_triggers: {json.dumps(user.behavior.skepticism_triggers, ensure_ascii=False)}",
            f"  knowledge_boundary: \"{user.behavior.knowledge_boundary}\"",
            "",
            "# ═══════════════════════════════════════════════",
            "# 一致性守卫规则 (Consistency Guard)",
            "# ═══════════════════════════════════════════════",
            "",
            "consistency_rules:",
            "  - \"怀疑度 > 0.6 时，不会突然变得非常感兴趣\"",
            "  - \"没有采购权时，不会说'立即购买'\"",
            "  - \"不会主动暴露公司敏感信息\"",
            "  - \"不会假装懂不熟悉的领域\"",
        ])
        
        if user.critique_result:
            cr = user.critique_result
            lines.extend([
                "",
                "# ═══════════════════════════════════════════════",
                "# LLM 质量评估 (Self-Critique)",
                "# ═══════════════════════════════════════════════",
                "",
                f"llm_critique:",
                f"  score: {cr.get('score', 'N/A')}",
                f"  strengths: {json.dumps(cr.get('strengths', []), ensure_ascii=False)}",
                f"  issues: {json.dumps(cr.get('issues', []), ensure_ascii=False)}",
                f"  suggestions: {json.dumps(cr.get('suggestions', []), ensure_ascii=False)}",
            ])
        
        return "\n".join(lines)
    
    @staticmethod
    def save(user: SyntheticUser, out_dir: Path = Path("generated")) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = _sanitize_id(user.persona.persona_id)
        
        yaml_path = out_dir / f"{base_name}.yaml"
        md_path = out_dir / f"{base_name}.md"
        
        yaml_path.write_text(PersonaExporter.to_yaml(user), encoding="utf-8")
        md_path.write_text(
            f"# 虚拟用户：{user.persona.demographics.role}\n\n"
            f"## System Prompt（可直接用于 LLM）\n\n"
            f"```markdown\n{user.system_prompt}\n```\n",
            encoding="utf-8",
        )
        
        print(f"📁 已保存到:")
        print(f"   YAML : {yaml_path}")
        print(f"   MD   : {md_path}")
        return yaml_path


# ───────────────────────────────────────────────
# 辅助：信息清单
# ───────────────────────────────────────────────

INFO_GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║         生成虚拟用户前，你需要准备的信息清单                            ║
╠══════════════════════════════════════════════════════════════════════╣

【快速模式 — 只需一段话】
  描述你想模拟的用户，包括：
  - 职位、行业、公司规模
  - 当前痛点和动机
  - 对产品的态度（怀疑/开放/急迫）
  - 测试场景（demo 会议、自助试用等）

【手动模式 — 需要准备】
  1. 产品名称 & 类型
  2. 目标用户：职位、行业、公司规模、地区、年龄
  3. 用户动机：工作目标、痛点、技术栈、决策风格
  4. 测试场景：场景描述、初始态度、时间压力
  5. 行为特征：沟通风格、怀疑度、价格敏感度
  6. 关键词：触发关注的、触发反感的

═══════════════════════════════════════════════════════════════════════
"""


# ───────────────────────────────────────────────
# 主入口
# ───────────────────────────────────────────────

def check_api_key():
    """检查 DEEPSEEK_API_KEY 是否配置。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("❌ 未找到 DEEPSEEK_API_KEY 环境变量")
        print("   请设置: export DEEPSEEK_API_KEY='your-api-key'")
        print()
        print("没有 API Key 也能运行，但只能使用手动模式（无 LLM 增强）")
        return False
    return True


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("--guide", "-g", "guide"):
        print(INFO_GUIDE)
        return
    
    has_api = check_api_key()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           🤖 虚拟用户生成 Agent —— LLM 增强版                          ║
║      基于「五层设计框架」+ DeepSeek API 构建 LLM 模拟用户                ║
╠══════════════════════════════════════════════════════════════════════╣
║  用法: python persona_generator.py --guide   查看信息清单               ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # 选择模式
    if has_api:
        use_llm = _ask_yes_no("🚀 是否使用 LLM 快速扩展模式？（推荐）", True)
    else:
        print("⚠️  未配置 DEEPSEEK_API_KEY，进入手动模式")
        use_llm = False
    
    # 收集信息
    if use_llm:
        user = mode_llm_expand()
    else:
        product_name = _ask("你的产品/服务名称是什么")
        product_type = _ask("产品类型", "B2B SaaS")
        user = mode_manual(product_name, product_type)
    
    product_name = getattr(user, '_product_name', '') or '产品'
    product_type = getattr(user, '_product_type', '') or 'SaaS'
    
    # 生成 raw system prompt
    print()
    print("📝 生成基础 System Prompt...")
    raw_prompt = build_raw_system_prompt(user, product_name, product_type)
    
    # LLM 优化
    if has_api:
        print()
        print("🤖 调用 DeepSeek 优化 System Prompt（增加人味、强化约束）...")
        print("   （这可能需要 10-20 秒）")
        try:
            optimized = optimize_system_prompt(raw_prompt)
            user.system_prompt = optimized
            print("   ✅ 优化完成")
        except Exception as e:
            print(f"   ⚠️ 优化失败，使用基础版本: {e}")
            user.system_prompt = raw_prompt
        
        # Self-Critique
        print()
        print("🔍 调用 DeepSeek 进行质量评估（Self-Critique）...")
        try:
            critique = critique_prompt(user.system_prompt)
            user.critique_result = critique
            score = critique.get('score', 'N/A')
            print(f"   ✅ 评估完成 —— 质量得分: {score}/10")
            issues = critique.get('issues', [])
            if issues:
                print(f"   ⚠️  发现 {len(issues)} 个问题:")
                for i in issues:
                    print(f"      • {i}")
        except Exception as e:
            print(f"   ⚠️ 评估失败: {e}")
            user.critique_result = None
    else:
        user.system_prompt = raw_prompt
    
    # 输出
    print()
    print("═" * 60)
    print("📋 最终 System Prompt（可直接用于 LLM）")
    print("═" * 60)
    print()
    print(user.system_prompt)
    print()
    
    # 保存
    PersonaExporter.save(user)
    
    print()
    print("═" * 60)
    print("✅ 虚拟用户生成完成！")
    print("═" * 60)
    print()
    print("使用方式：")
    print("  1. 将 generated/*.md 中的 System Prompt 粘贴到 LLM 的 system prompt")
    print("  2. 开始对话测试你的产品/服务")
    print("  3. 根据对话表现，调整 persona 中的量化指标")
    print()


if __name__ == "__main__":
    main()
