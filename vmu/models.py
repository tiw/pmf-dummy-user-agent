"""
VMU 核心数据模型
使用 Pydantic v2 定义所有实体
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ───────────────────────────────────────────────
# Layer 1: 基础结构（复用现有概念）
# ───────────────────────────────────────────────

class Demographics(BaseModel):
    age: int = 30
    role: str = "用户"
    company_size: str = "未知"
    industry: str = "未知"
    location: str = "未知"
    years_experience: int = 3


class Psychographics(BaseModel):
    goals: List[str] = Field(default_factory=list)
    frustrations: List[str] = Field(default_factory=list)
    decision_style: str = "谨慎"
    tech_stack: List[str] = Field(default_factory=list)
    budget_authority: str = "无采购权"


class BehavioralTraits(BaseModel):
    communication: str = "直接"
    skepticism_level: float = Field(0.5, ge=0.0, le=1.0)
    price_sensitivity: float = Field(0.5, ge=0.0, le=1.0)
    risk_tolerance: str = "中"


class Context(BaseModel):
    current_problem: str = "效率低"
    recent_changes: str = "无"
    team_pressure: str = "一般"
    competitive_exposure: List[str] = Field(default_factory=list)


class SceneContext(BaseModel):
    scene_description: str = "评估产品"
    initial_attitude: str = "中立"
    time_pressure: str = "一般"
    participation_motivation: str = "解决痛点"
    prior_exposure: str = "未接触"


class MemoryState(BaseModel):
    trust_level: float = Field(0.3, ge=0.0, le=1.0)
    emotional_state: str = "skeptical"
    exposure_count: int = 0


class BehaviorEngine(BaseModel):
    attention_keywords: List[str] = Field(default_factory=lambda: ["效率", "价格", "ROI"])
    skepticism_triggers: List[str] = Field(default_factory=lambda: ["赋能", "生态"])
    knowledge_boundary: str = "知道自己擅长什么，不会假装懂不熟悉的领域"


# ───────────────────────────────────────────────
# Layer 2: 核心实体
# ───────────────────────────────────────────────

class PersonaType(BaseModel):
    """
    人格类型模板。
    
    一种「类型」定义了一类人的共同特征，
    比如「焦虑型买家」「理性分析师」「技术怀疑者」。
    同类型可以实例化出多个有差异的个体。
    """
    type_id: str = Field(..., description="类型唯一标识，如 anxious_buyer")
    name: str = Field(..., description="显示名称")
    description: str = Field("", description="类型描述")
    
    # 五层框架模板
    demographics: Demographics = Field(default_factory=Demographics)
    psychographics: Psychographics = Field(default_factory=Psychographics)
    behavioral_traits: BehavioralTraits = Field(default_factory=BehavioralTraits)
    context: Context = Field(default_factory=Context)
    scene_context: SceneContext = Field(default_factory=SceneContext)
    behavior_engine: BehaviorEngine = Field(default_factory=BehaviorEngine)
    
    # 系统 prompt（可以是模板字符串，支持 Jinja2 风格变量）
    system_prompt_template: str = Field(
        "",
        description="System prompt 模板。可用变量: {{name}}, {{age}}, {{role}}, 以及所有字段"
    )
    
    # 变异配置：定义如何从模板生成有差异的实例
    variation_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="变异配置，如 {'age': {'range': [25, 45]}, 'skepticism_level': {'range': [0.3, 0.8]}}"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        d = super().model_dump(**kwargs)
        # 处理 datetime 序列化
        for key in ("created_at", "updated_at"):
            if key in d and isinstance(d[key], datetime):
                d[key] = d[key].isoformat()
        return d


class PersonaInstance(BaseModel):
    """
    人格实例。
    
    基于 PersonaType 模板创建的特定个体，
    带有具体的差异化参数和对话状态。
    """
    instance_id: str = Field(default_factory=lambda: f"inst_{uuid.uuid4().hex[:8]}")
    type_id: str = Field(..., description="所属 PersonaType")
    name: str = Field(..., description="实例名称，如 张三")
    
    # 差异化参数（覆盖模板的值）
    variation: Dict[str, Any] = Field(default_factory=dict, description="差异化参数")
    
    # 实际属性（由模板 + 变异生成）
    demographics: Demographics = Field(default_factory=Demographics)
    psychographics: Psychographics = Field(default_factory=Psychographics)
    behavioral_traits: BehavioralTraits = Field(default_factory=BehavioralTraits)
    context: Context = Field(default_factory=Context)
    scene_context: SceneContext = Field(default_factory=SceneContext)
    behavior_engine: BehaviorEngine = Field(default_factory=BehaviorEngine)
    memory: MemoryState = Field(default_factory=MemoryState)
    
    # 最终 system prompt（已渲染）
    system_prompt: str = ""
    
    # 对话历史
    message_history: List[Message] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        d = super().model_dump(**kwargs)
        for key in ("created_at", "updated_at"):
            if key in d and isinstance(d[key], datetime):
                d[key] = d[key].isoformat()
        return d


class Message(BaseModel):
    """单条消息"""
    role: str = Field(..., description="system / user / assistant")
    content: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        d = super().model_dump(**kwargs)
        if "timestamp" in d and isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        return d


class SceneParticipant(BaseModel):
    """场景参与者配置"""
    type_id: str = Field(..., description="使用哪种 PersonaType")
    count: int = Field(1, ge=1, description="该类型需要生成几个实例")
    variation_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="该场景下对该类型的额外覆盖参数"
    )
    scene_specific_prompt_addition: str = Field(
        "",
        description="该场景对该类型参与者的额外 prompt 补充"
    )
    # ── 新增：场景评估相关 ──
    behavior_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="场景特定的行为参数覆盖，如 {'hesitation': 0.8, 'info_release': 'slow'}"
    )
    private_info_override: Dict[str, Any] = Field(
        default_factory=dict,
        description="场景特定的私有信息覆盖"
    )
    expected_stages: List[str] = Field(
        default_factory=list,
        description="该参与者预期经历的销售阶段，用于评估 agent 表现"
    )
    min_info_release: Dict[str, Any] = Field(
        default_factory=dict,
        description="最低信息释放要求，如 {'pain_points': True, 'budget_range': True}"
    )


class Scene(BaseModel):
    """
    场景。
    
    定义一个虚拟人交互场景，包含：
    - 场景上下文
    - 参与者配置（类型 + 数量）
    - 实际生成的实例引用
    """
    scene_id: str = Field(default_factory=lambda: f"scene_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="场景名称")
    description: str = Field("", description="场景描述")
    
    # 场景上下文
    scenario: str = Field(
        "",
        description="场景剧本/背景描述，会注入到所有参与者的 prompt 中"
    )
    
    # 参与者配置（用于实例化）
    participant_configs: List[SceneParticipant] = Field(default_factory=list)
    
    # 已实例化的参与者
    participant_instance_ids: List[str] = Field(default_factory=list)
    
    # 共享记忆/上下文（所有参与者可见）
    shared_context: Dict[str, Any] = Field(default_factory=dict)

    # ── 新增：场景评估标准 ──
    expected_stages: List[str] = Field(
        default_factory=list,
        description="场景下销售 agent 应该覆盖的销售阶段"
    )
    success_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="成功标准，如 {'min_stage_coverage': 0.8, 'min_trust_level': 0.5}"
    )
    evaluation_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="评估配置，如 {'require_all_stages': False, 'info_release_weight': 0.3}"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        d = super().model_dump(**kwargs)
        for key in ("created_at", "updated_at"):
            if key in d and isinstance(d[key], datetime):
                d[key] = d[key].isoformat()
        return d


class InteractionResult(BaseModel):
    """单次交互结果"""
    instance_id: str
    response: str
    updated_memory: Optional[MemoryState] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroupChatTurn(BaseModel):
    """群聊中单个虚拟人的一轮决策"""
    instance_id: str
    name: str
    type_id: str
    decision: str = Field(..., description="REPLY 或 PASS")
    reasoning: str = ""
    reply: str = ""
    updated_memory: Optional[MemoryState] = None
