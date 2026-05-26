#!/usr/bin/env python3
"""
PersonaForge Web Server
FastAPI + 静态文件服务
复用 deepseek_client 进行所有 LLM 调用
"""

import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import local modules
from deepseek_client import (
    expand_persona,
    optimize_system_prompt,
    critique_prompt,
    chat_completion_json,
)

# ─── Pydantic Models ───

class Demographics(BaseModel):
    age: int
    role: str
    company_size: str
    industry: str
    location: str
    years_experience: int


class Psychographics(BaseModel):
    goals: List[str]
    frustrations: List[str]
    decision_style: str
    tech_stack: List[str]
    budget_authority: str


class BehavioralTraits(BaseModel):
    communication: str
    skepticism_level: float
    price_sensitivity: float
    risk_tolerance: str


class Context(BaseModel):
    current_problem: str
    recent_changes: str
    team_pressure: str
    competitive_exposure: List[str]


class PersonaData(BaseModel):
    demographics: Demographics
    psychographics: Psychographics
    behavioral_traits: BehavioralTraits
    context: Context


class SceneData(BaseModel):
    scene_description: str
    initial_attitude: str
    time_pressure: str
    participation_motivation: str
    prior_exposure: str


class BehaviorData(BaseModel):
    attention_keywords: List[str]
    skepticism_triggers: List[str]
    knowledge_boundary: str = "知道自己擅长什么，不会假装懂不熟悉的领域"


class LlmGenerateRequest(BaseModel):
    mode: str  # "llm" or "manual"
    product_name: str
    product_type: str
    user_description: Optional[str] = None
    persona: Optional[PersonaData] = None
    scene: Optional[SceneData] = None
    behavior: Optional[BehaviorData] = None


# ─── App ───

app = FastAPI(title="PersonaForge API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).parent / "web"

# API routes registered BEFORE static files

# ─── Helpers ───

def _sanitize_id(text: str) -> str:
    return re.sub(r'[^\w\u4e00-\u9fff]+', '_', text).strip('_')


def _build_raw_system_prompt(persona: dict, scene: dict, behavior: dict, memory: dict, product_name: str) -> str:
    d = persona["demographics"]
    ps = persona["psychographics"]
    bt = persona["behavioral_traits"]
    ctx = persona["context"]
    s = scene

    lines = [
        f"# 虚拟用户身份设定：{d['role']}",
        "",
        f"你是一位 {d['age']} 岁的 {d['role']}，在 {d['industry']} 行业工作 {d['years_experience']} 年。",
        f"所在公司规模：{d['company_size']}，地点：{d['location']}。",
        "",
        "## 你的状态",
        f"- 当前最大的痛点：{ctx['current_problem']}",
        f"- 最近的变化：{ctx['recent_changes']}",
        f"- 团队压力：{ctx['team_pressure']}",
    ]
    if ctx.get("competitive_exposure"):
        lines.append(f"- 竞品经历：{', '.join(ctx['competitive_exposure'])}")

    lines.extend(["", "## 你的工作目标"])
    for g in ps["goals"]:
        lines.append(f"- {g}")

    lines.extend(["", "## 你的 frustrates"])
    for f in ps["frustrations"]:
        lines.append(f"- {f}")

    lines.extend([
        "", "## 你的沟通风格",
        f"- {bt['communication']}",
        f"- 怀疑程度：{'高' if bt['skepticism_level'] > 0.6 else '中' if bt['skepticism_level'] > 0.3 else '低'}（{bt['skepticism_level']}）",
        f"- 价格敏感度：{'高' if bt['price_sensitivity'] > 0.6 else '中' if bt['price_sensitivity'] > 0.3 else '低'}（{bt['price_sensitivity']}）",
        f"- 风险承受度：{bt['risk_tolerance']}",
        f"- 决策风格：{ps['decision_style']}",
    ])

    if ps.get("tech_stack"):
        lines.extend(["", "## 你的技术/工具栈"])
        for t in ps["tech_stack"]:
            lines.append(f"- {t}")

    lines.extend([
        "", "## 当前场景",
        f"- 场景：{s['scene_description']}",
        f"- 初始态度：{s['initial_attitude']}",
        f"- 时间压力：{s['time_pressure']}",
        f"- 参与动机：{s['participation_motivation']}",
        f"- 之前接触：{s['prior_exposure']}",
        "", "## 对话规则",
        "- 你的回复必须符合上述身份和场景，不能偏离角色。",
        "- 如果对方使用模糊语言，你会要求给具体例子。",
        "- 如果对方提到你关心的关键词，你会高度关注并深入追问。",
        f"- 如果对方使用 {', '.join(behavior['attention_keywords'])} 这类词汇，你会表现出怀疑或反感。",
        "- 你不会主动暴露公司敏感信息。",
        f"- 你没有最终采购权（{ps['budget_authority']}），不会说'立即购买'。",
        "- 如果产品不解决你的问题，你会直接说'这不解决我的问题'。",
        "", "## 记忆状态（初始）",
        f"- 信任度：{memory['trust_level']}/1.0",
        f"- 情绪：{memory['emotional_state']}",
        f"- 接触次数：{memory['exposure_count']}",
    ])

    return "\n".join(lines)


# ─── API Routes ───

@app.post("/api/generate")
async def generate(req: LlmGenerateRequest):
    if not os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        raise HTTPException(status_code=500, detail="未配置 LLM API Key，请设置 DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY")

    product_name = req.product_name or "产品"
    product_type = req.product_type or "SaaS"

    if req.mode == "llm":
        # ── LLM 快速扩展模式 ──
        if not req.user_description:
            raise HTTPException(status_code=400, detail="user_description 不能为空")

        try:
            expanded = expand_persona(
                f"产品：{product_name}（{product_type}）\n用户描述：{req.user_description}"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM 扩展失败: {str(e)}")

        d = expanded.get("demographics", {})
        p = expanded.get("psychographics", {})
        bt = expanded.get("behavioral_traits", {})
        ctx = expanded.get("context", {})
        sc = expanded.get("scene_context", {})
        be = expanded.get("behavior_engine", {})

        persona = {
            "persona_id": f"{_sanitize_id(d.get('role', 'user'))}_{d.get('age', 0):02d}",
            "demographics": {
                "age": d.get("age", 30),
                "role": d.get("role", "用户"),
                "company_size": d.get("company_size", "未知"),
                "industry": d.get("industry", "未知"),
                "location": d.get("location", "未知"),
                "years_experience": d.get("years_experience", 3),
            },
            "psychographics": {
                "goals": p.get("goals", ["提高效率"]),
                "frustrations": p.get("frustrations", ["工具复杂"]),
                "decision_style": p.get("decision_style", "谨慎"),
                "tech_stack": p.get("tech_stack", []),
                "budget_authority": p.get("budget_authority", "无采购权"),
            },
            "behavioral_traits": {
                "communication": bt.get("communication", "直接"),
                "skepticism_level": float(bt.get("skepticism_level", 0.5)),
                "price_sensitivity": float(bt.get("price_sensitivity", 0.5)),
                "risk_tolerance": bt.get("risk_tolerance", "中"),
            },
            "context": {
                "current_problem": ctx.get("current_problem", "效率低"),
                "recent_changes": ctx.get("recent_changes", "无"),
                "team_pressure": ctx.get("team_pressure", "一般"),
                "competitive_exposure": ctx.get("competitive_exposure", []),
            },
        }
        scene = {
            "scene_description": sc.get("scene_description", "评估产品"),
            "initial_attitude": sc.get("initial_attitude", "中立"),
            "time_pressure": sc.get("time_pressure", "一般"),
            "participation_motivation": sc.get("participation_motivation", "解决痛点"),
            "prior_exposure": sc.get("prior_exposure", "未接触"),
        }
        behavior = {
            "attention_keywords": be.get("attention_keywords", ["效率", "价格", "ROI"]),
            "skepticism_triggers": be.get("skepticism_triggers", ["赋能", "生态"]),
            "knowledge_boundary": be.get("knowledge_boundary", "知道自己擅长什么"),
        }

    else:
        # ── 手动模式 ──
        if not req.persona or not req.scene or not req.behavior:
            raise HTTPException(status_code=400, detail="手动模式需要提供 persona、scene、behavior")

        pd = req.persona.demographics
        persona = {
            "persona_id": f"{_sanitize_id(pd.role)}_{pd.age:02d}",
            "demographics": pd.model_dump(),
            "psychographics": req.persona.psychographics.model_dump(),
            "behavioral_traits": req.persona.behavioral_traits.model_dump(),
            "context": req.persona.context.model_dump(),
        }
        scene = req.scene.model_dump()
        behavior = req.behavior.model_dump()

    # ── 通用后续：生成 raw prompt → LLM 优化 → Self-Critique ──
    memory = {"trust_level": 0.3, "emotional_state": "skeptical", "exposure_count": 0}

    raw_prompt = _build_raw_system_prompt(persona, scene, behavior, memory, product_name)

    # LLM 优化
    try:
        optimized = optimize_system_prompt(raw_prompt)
        system_prompt = optimized
    except Exception as e:
        system_prompt = raw_prompt

    # Self-Critique
    critique = None
    try:
        critique = critique_prompt(system_prompt)
    except Exception:
        pass

    return {
        "persona": persona,
        "scene": scene,
        "memory": memory,
        "behavior": behavior,
        "system_prompt": system_prompt,
        "critique": critique,
    }


# ═══════════════════════════════════════════════
# VMU API Routes (v1)
# ═══════════════════════════════════════════════

from vmu import PersonaManager, PersonaAgent
from vmu.models import (
    Demographics, Psychographics, BehavioralTraits,
    Context, SceneContext, BehaviorEngine,
    SceneParticipant,
)

manager = PersonaManager()

# ─── Helper ───

def _inst_to_dict(inst):
    """将 PersonaInstance 转为可 JSON 序列化的 dict"""
    d = inst.model_dump()
    # message_history 可能很长，截断预览
    if "message_history" in d:
        d["message_history_count"] = len(d["message_history"])
        d["message_history_preview"] = [
            {"role": m["role"], "content": m["content"][:200]}
            for m in d["message_history"][-5:]
        ]
    return d


# ─── PersonaType ───

class TypeCreateRequest(BaseModel):
    type_id: str
    name: str
    description: str = ""
    demographics: Optional[dict] = None
    psychographics: Optional[dict] = None
    behavioral_traits: Optional[dict] = None
    context: Optional[dict] = None
    scene_context: Optional[dict] = None
    behavior_engine: Optional[dict] = None
    system_prompt_template: str = ""
    variation_config: Optional[dict] = None


class TypeUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    demographics: Optional[dict] = None
    psychographics: Optional[dict] = None
    behavioral_traits: Optional[dict] = None
    context: Optional[dict] = None
    scene_context: Optional[dict] = None
    behavior_engine: Optional[dict] = None
    system_prompt_template: Optional[str] = None
    variation_config: Optional[dict] = None


@app.get("/api/v1/types")
async def list_types():
    return {"types": [pt.model_dump() for pt in manager.list_types()]}


@app.post("/api/v1/types")
async def create_type(req: TypeCreateRequest):
    pt = manager.create_type(
        type_id=req.type_id,
        name=req.name,
        description=req.description,
        demographics=Demographics(**req.demographics) if req.demographics else None,
        psychographics=Psychographics(**req.psychographics) if req.psychographics else None,
        behavioral_traits=BehavioralTraits(**req.behavioral_traits) if req.behavioral_traits else None,
        context=Context(**req.context) if req.context else None,
        scene_context=SceneContext(**req.scene_context) if req.scene_context else None,
        behavior_engine=BehaviorEngine(**req.behavior_engine) if req.behavior_engine else None,
        system_prompt_template=req.system_prompt_template,
        variation_config=req.variation_config or {},
    )
    return {"type": pt.model_dump()}


@app.get("/api/v1/types/{type_id}")
async def get_type(type_id: str):
    pt = manager.get_type(type_id)
    if not pt:
        raise HTTPException(status_code=404, detail="Type not found")
    return {"type": pt.model_dump()}


@app.put("/api/v1/types/{type_id}")
async def update_type(type_id: str, req: TypeUpdateRequest):
    pt = manager.get_type(type_id)
    if not pt:
        raise HTTPException(status_code=404, detail="Type not found")
    updates = req.model_dump(exclude_unset=True)
    if not updates:
        return {"type": pt.model_dump()}
    updated = manager.update_type(type_id, **updates)
    return {"type": updated.model_dump() if updated else None}


@app.delete("/api/v1/types/{type_id}")
async def delete_type(type_id: str):
    ok = manager.delete_type(type_id)
    return {"deleted": ok}


# ─── PersonaInstance ───

class InstanceCreateRequest(BaseModel):
    type_id: str
    name: Optional[str] = None
    variation: Optional[dict] = None
    variation_seed: Optional[int] = None


@app.get("/api/v1/instances")
async def list_instances(type_id: Optional[str] = None):
    instances = manager.list_instances(type_id)
    return {"instances": [_inst_to_dict(i) for i in instances]}


@app.post("/api/v1/instances")
async def create_instance(req: InstanceCreateRequest):
    inst = manager.instantiate(
        type_id=req.type_id,
        name=req.name,
        variation=req.variation,
        variation_seed=req.variation_seed,
    )
    if not inst:
        raise HTTPException(status_code=404, detail="Type not found")
    return {"instance": _inst_to_dict(inst)}


@app.get("/api/v1/instances/{instance_id}")
async def get_instance(instance_id: str):
    inst = manager.get_instance(instance_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Instance not found")
    return {"instance": _inst_to_dict(inst)}


@app.delete("/api/v1/instances/{instance_id}")
async def delete_instance(instance_id: str):
    ok = manager.delete_instance(instance_id)
    return {"deleted": ok}


# ─── Instance Interaction ───

class InteractRequest(BaseModel):
    message: str
    include_history: bool = True
    temperature: float = 0.7


@app.post("/api/v1/instances/{instance_id}/interact")
async def interact(instance_id: str, req: InteractRequest):
    inst = manager.get_instance(instance_id)
    if not inst:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    from deepseek_client import chat_completion
    agent = PersonaAgent(
        instance=inst,
        llm_client=chat_completion,
        auto_persist=True,
        storage=manager.storage,
    )
    try:
        result = agent.interact(
            req.message,
            include_history=req.include_history,
            temperature=req.temperature,
        )
        return {
            "instance_id": result.instance_id,
            "response": result.response,
            "memory": result.updated_memory.model_dump() if result.updated_memory else None,
            "metadata": result.metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM interaction failed: {str(e)}")


# ─── Scene ───

class SceneCreateRequest(BaseModel):
    name: str
    description: str = ""
    scenario: str = ""
    participant_configs: List[dict] = Field(default_factory=list)
    shared_context: Optional[dict] = None


@app.get("/api/v1/scenes")
async def list_scenes():
    return {"scenes": [s.model_dump() for s in manager.list_scenes()]}


@app.post("/api/v1/scenes")
async def create_scene(req: SceneCreateRequest):
    configs = [SceneParticipant(**c) for c in req.participant_configs]
    scene = manager.create_scene(
        name=req.name,
        description=req.description,
        scenario=req.scenario,
        participant_configs=configs,
        shared_context=req.shared_context or {},
    )
    return {"scene": scene.model_dump()}


@app.get("/api/v1/scenes/{scene_id}")
async def get_scene(scene_id: str):
    scene = manager.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    # 同时返回参与者详情
    participants = manager.get_scene_instances(scene_id)
    data = scene.model_dump()
    data["participants"] = [_inst_to_dict(p) for p in participants]
    return {"scene": data}


@app.post("/api/v1/scenes/{scene_id}/instantiate")
async def instantiate_scene(scene_id: str):
    scene = manager.instantiate_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    participants = manager.get_scene_instances(scene_id)
    return {
        "scene": scene.model_dump(),
        "participants": [_inst_to_dict(p) for p in participants],
    }


@app.delete("/api/v1/scenes/{scene_id}")
async def delete_scene(scene_id: str):
    ok = manager.delete_scene(scene_id)
    return {"deleted": ok}


# ─── Presets ───

@app.post("/api/v1/presets")
async def create_presets():
    types = manager.create_preset_types()
    return {"types": [pt.model_dump() for pt in types]}


# ─── Dashboard Stats ───

@app.get("/api/v1/stats")
async def get_stats():
    return {
        "types": len(manager.list_types()),
        "instances": len(manager.list_instances()),
        "scenes": len(manager.list_scenes()),
    }


# Mount static files AFTER all API routes
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

# ─── Main ───

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
