"""
Agent-Friendly 基础设施

让外部 Agent 能够：
1. Self-Discover: 自主发现服务能力
2. Intent-Driven: 用高层意图驱动，而非记忆精确 API
3. Navigate: 从响应中获取下一步建议
4. Query: 用自然语言查询"我该怎么用"

设计原则：
- 机器可读：返回结构化 JSON，LLM 可直接消费
- 语义丰富：每个端点都有 when_to_use 说明
- 状态感知：响应中包含基于当前状态的导航建议
- 零先验：agent 不需要提前知道 API 细节
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════
# 1. 能力注册表 —— 机器可读的服务描述
# ═══════════════════════════════════════════════

CAPABILITIES_REGISTRY = {
    "service": {
        "name": "PersonaForge",
        "description": (
            "虚拟用户生成与测试平台。你可以用它：\n"
            "1) 创建和管理虚拟用户(persona)——模拟真实用户的行为和性格\n"
            "2) 搭建交互场景(scene)——让多个虚拟人在同一背景下互动\n"
            "3) 测试你自己的 Agent/销售机器人/客服机器人——获得真实的用户反馈和评估报告"
        ),
        "version": "2.0",
        "agent_friendly_version": "1.0",
        "base_url": "/api/v1",
    },
    "concepts": [
        {
            "name": "PersonaType",
            "description": "人格类型模板，比如'焦虑型买家'、'理性分析师'。每种类型可以生成多个有差异的实例。",
            "analogy": "相当于一个'用户画像原型'",
            "key_fields": ["type_id", "name", "description", "behavioral_traits", "variation_config"],
        },
        {
            "name": "PersonaInstance",
            "description": "基于 PersonaType 生成的具体虚拟人个体，有名字、记忆、对话历史，可以和你对话。",
            "analogy": "相当于一个'具体的测试用户'",
            "key_fields": ["instance_id", "name", "type_id", "memory", "message_history"],
        },
        {
            "name": "Scene",
            "description": "场景，定义了多个虚拟人参与者的交互环境，比如'产品演示会'。",
            "analogy": "相当于一个'测试场景/会议室'",
            "key_fields": ["scene_id", "name", "scenario", "participant_configs", "participant_instance_ids"],
        },
        {
            "name": "TestingSession",
            "description": "测试会话，记录你的 Agent 与一个虚拟人的完整对话过程和评估结果。",
            "analogy": "相当于一次'用户测试访谈'",
            "key_fields": ["session_id", "persona", "turns", "status"],
        },
    ],
    "endpoints": [
        # ─── 能力发现 ───
        {
            "path": "/api/v1/capabilities",
            "method": "GET",
            "description": "获取完整的服务能力目录（就是这个文档）",
            "when_to_use": "当你第一次连接此服务，或不确定该调用哪个 API 时",
            "parameters": {},
            "returns": "能力注册表 JSON",
        },
        {
            "path": "/api/v1/intent",
            "method": "POST",
            "description": "用高层意图描述你想做什么，服务自动执行对应操作",
            "when_to_use": "当你知道目标但不想记具体 API 路径时，比如'我想测试我的销售 agent'",
            "parameters": {
                "intent": "意图名称（见 supported_intents）",
                "params": "意图所需的参数 dict",
            },
            "returns": "执行结果",
        },
        {
            "path": "/api/v1/query",
            "method": "POST",
            "description": "用自然语言提问，获取建议的 API 调用序列",
            "when_to_use": "当你想问'我该怎么测试我的 agent？'这类问题时",
            "parameters": {
                "question": "自然语言问题",
                "context": "可选的当前上下文",
            },
            "returns": "建议的操作步骤和对应的 API 调用",
        },
        # ─── PersonaType ───
        {
            "path": "/api/v1/types",
            "method": "GET",
            "description": "列出所有人格类型模板",
            "when_to_use": "你想看看有哪些虚拟人类型可用，比如焦虑型买家、理性分析师",
            "parameters": {},
            "returns": "PersonaType 列表",
        },
        {
            "path": "/api/v1/types",
            "method": "POST",
            "description": "创建新的人格类型模板",
            "when_to_use": "你需要一个项目里还没有的虚拟人类型时",
            "parameters": {
                "type_id": "唯一标识，如 budget_controller",
                "name": "显示名称",
                "description": "描述",
                "demographics": "人口统计",
                "psychographics": "心理特征",
                "behavioral_traits": "行为特征",
                "variation_config": "变异配置，定义实例化时的差异范围",
            },
            "returns": "创建的 PersonaType",
        },
        {
            "path": "/api/v1/types/{type_id}",
            "method": "GET",
            "description": "获取某个人格类型的详情",
            "when_to_use": "你想了解某个类型的具体特征",
            "parameters": {"type_id": "路径参数"},
            "returns": "PersonaType 详情",
        },
        {
            "path": "/api/v1/presets",
            "method": "POST",
            "description": "一键创建 4 种预设人格类型（焦虑型买家、理性分析师、技术怀疑者、冲动决策者）",
            "when_to_use": "快速开始测试，不需要自己定义类型",
            "parameters": {},
            "returns": "创建的类型列表",
        },
        # ─── PersonaInstance ───
        {
            "path": "/api/v1/instances",
            "method": "POST",
            "description": "基于某个 PersonaType 创建一个具体的虚拟人实例",
            "when_to_use": "你需要一个可以和它对话的具体虚拟人",
            "parameters": {
                "type_id": "基于哪个类型",
                "name": "实例名称（可选，自动分配）",
                "variation": "手动变异参数（可选）",
                "variation_seed": "随机种子（可选，用于复现）",
            },
            "returns": "PersonaInstance",
        },
        {
            "path": "/api/v1/instances",
            "method": "GET",
            "description": "列出所有虚拟人实例",
            "when_to_use": "你想看看已经创建了哪些虚拟人",
            "parameters": {"type_id": "可选，按类型过滤"},
            "returns": "PersonaInstance 列表",
        },
        {
            "path": "/api/v1/instances/{instance_id}",
            "method": "GET",
            "description": "获取某个虚拟人实例的详情",
            "when_to_use": "你想查看某个虚拟人的当前状态、记忆、对话历史",
            "parameters": {"instance_id": "路径参数"},
            "returns": "PersonaInstance 详情",
        },
        {
            "path": "/api/v1/instances/{instance_id}/interact",
            "method": "POST",
            "description": "与某个虚拟人进行一次对话",
            "when_to_use": "你想和虚拟人说话，看它怎么回复",
            "parameters": {
                "instance_id": "路径参数",
                "message": "你说的话",
                "include_history": "是否携带历史（默认 true）",
                "temperature": "温度（默认 0.7）",
            },
            "returns": "虚拟人的回复、更新后的记忆、元数据",
        },
        # ─── Scene ───
        {
            "path": "/api/v1/scenes",
            "method": "POST",
            "description": "创建一个场景",
            "when_to_use": "你需要多个虚拟人在同一背景下互动",
            "parameters": {
                "name": "场景名称",
                "description": "描述",
                "scenario": "场景剧本/背景",
                "participant_configs": "List[{type_id: str, count: int}] — 参与者配置",
            },
            "returns": "Scene",
        },
        {
            "path": "/api/v1/scenes/{scene_id}/instantiate",
            "method": "POST",
            "description": "实例化场景，根据 participant_configs 生成所有虚拟人",
            "when_to_use": "创建场景后，需要生成具体的虚拟人参与者",
            "parameters": {"scene_id": "路径参数"},
            "returns": "更新后的 Scene 和参与者列表",
        },
        {
            "path": "/api/v1/scenes/{scene_id}/group-chat",
            "method": "POST",
            "description": "在场景中进行群聊：发送一条消息，所有虚拟人根据角色决定是否回复",
            "when_to_use": "你想观察多个虚拟人对同一条消息的不同反应",
            "parameters": {
                "scene_id": "路径参数",
                "message": "消息内容",
                "temperature": "温度",
            },
            "returns": "每个虚拟人的决策和回复",
        },
        # ─── Testing ───
        {
            "path": "/api/v1/testing/sessions",
            "method": "POST",
            "description": "创建测试会话，让你的 Agent 和一个虚拟人进行测试对话",
            "when_to_use": "开始测试你的 Agent 时的第一步",
            "parameters": {
                "persona_type": "使用哪种虚拟人类型",
                "name": "虚拟人名称（可选）",
                "scene_overrides": "场景覆盖参数（可选）",
            },
            "returns": "session_id 和虚拟人信息",
        },
        {
            "path": "/api/v1/testing/sessions/{session_id}/message",
            "method": "POST",
            "description": "向测试会话发送消息（你的 Agent → 虚拟人），返回虚拟人回复",
            "when_to_use": "测试过程中的核心交互：把你的 Agent 的输出发给虚拟人",
            "parameters": {
                "session_id": "路径参数",
                "message": "你的 Agent 说的话",
                "temperature": "温度",
            },
            "returns": "虚拟人的回复、信任度、情绪状态",
        },
        {
            "path": "/api/v1/testing/sessions/{session_id}/evaluate",
            "method": "POST",
            "description": "对测试会话进行 LLM 评估",
            "when_to_use": "对话结束后，你想知道你的 Agent 表现如何",
            "parameters": {"session_id": "路径参数"},
            "returns": "评估结果",
        },
        {
            "path": "/api/v1/testing/sessions/{session_id}/report",
            "method": "GET",
            "description": "获取测试报告",
            "when_to_use": "你想看完整的测试报告",
            "parameters": {
                "session_id": "路径参数",
                "fmt": "格式：json / markdown / console（默认 json）",
            },
            "returns": "测试报告",
        },
        {
            "path": "/api/v1/testing/sessions/{session_id}",
            "method": "DELETE",
            "description": "关闭测试会话",
            "when_to_use": "测试结束后清理",
            "parameters": {"session_id": "路径参数"},
            "returns": {"closed": True},
        },
        # ─── Legacy ───
        {
            "path": "/api/generate",
            "method": "POST",
            "description": "（旧版）生成单个 Persona 的完整定义和 System Prompt",
            "when_to_use": "你只需要生成一个 persona 定义，不需要后续的实例化和对话管理",
            "parameters": {
                "mode": "'llm' 或 'manual'",
                "product_name": "产品名",
                "product_type": "产品类型",
                "user_description": "用户描述（llm 模式）",
            },
            "returns": "persona、scene、system_prompt、critique",
        },
    ],
    "workflows": [
        {
            "name": "快速开始：测试你的 Agent（推荐）",
            "description": "最快上手的路径，5 步完成一次 Agent 测试",
            "steps": [
                {"order": 1, "action": "创建预设类型", "endpoint": "POST /api/v1/presets"},
                {"order": 2, "action": "创建测试会话", "endpoint": "POST /api/v1/testing/sessions", "body": {"persona_type": "anxious_buyer"}},
                {"order": 3, "action": "发送消息", "endpoint": "POST /api/v1/testing/sessions/{session_id}/message", "body": {"message": "你的 Agent 的回复"}},
                {"order": 4, "action": "（重复第 3 步）多轮对话"},
                {"order": 5, "action": "获取报告", "endpoint": "GET /api/v1/testing/sessions/{session_id}/report"},
            ],
        },
        {
            "name": "自定义虚拟人类型后测试",
            "description": "先创建你自己的虚拟人类型，再用它测试 Agent",
            "steps": [
                {"order": 1, "action": "创建类型", "endpoint": "POST /api/v1/types"},
                {"order": 2, "action": "创建实例", "endpoint": "POST /api/v1/instances"},
                {"order": 3, "action": "直接对话", "endpoint": "POST /api/v1/instances/{instance_id}/interact"},
            ],
        },
        {
            "name": "场景测试：多虚拟人",
            "description": "在一个场景中同时与多个不同类型的虚拟人互动",
            "steps": [
                {"order": 1, "action": "创建预设类型", "endpoint": "POST /api/v1/presets"},
                {"order": 2, "action": "创建场景", "endpoint": "POST /api/v1/scenes"},
                {"order": 3, "action": "实例化场景", "endpoint": "POST /api/v1/scenes/{scene_id}/instantiate"},
                {"order": 4, "action": "群聊", "endpoint": "POST /api/v1/scenes/{scene_id}/group-chat"},
            ],
        },
    ],
    "supported_intents": {
        "discover": {
            "description": "发现服务能力",
            "example_params": {},
        },
        "list_persona_types": {
            "description": "列出所有虚拟人类型",
            "example_params": {},
        },
        "create_preset_personas": {
            "description": "一键创建 4 种预设虚拟人类型",
            "example_params": {},
        },
        "create_persona_type": {
            "description": "创建自定义虚拟人类型",
            "example_params": {"type_id": "my_type", "name": "我的类型", "description": "..."},
        },
        "instantiate_persona": {
            "description": "创建一个具体的虚拟人实例",
            "example_params": {"type_id": "anxious_buyer", "name": "小王"},
        },
        "interact_with_persona": {
            "description": "与一个虚拟人对话",
            "example_params": {"instance_id": "inst_xxx", "message": "你好"},
        },
        "create_scene": {
            "description": "创建场景",
            "example_params": {"name": "产品演示", "scenario": "...", "participant_configs": [{"type_id": "anxious_buyer", "count": 2}]},
        },
        "instantiate_scene": {
            "description": "实例化场景",
            "example_params": {"scene_id": "scene_xxx"},
        },
        "group_chat": {
            "description": "场景群聊",
            "example_params": {"scene_id": "scene_xxx", "message": "大家好"},
        },
        "test_agent": {
            "description": "创建测试会话，开始测试你的 Agent",
            "example_params": {"persona_type": "anxious_buyer", "name": "测试用户"},
        },
        "send_test_message": {
            "description": "向测试会话发送消息",
            "example_params": {"session_id": "session_xxx", "message": "你的 Agent 的回复"},
        },
        "get_test_report": {
            "description": "获取测试报告",
            "example_params": {"session_id": "session_xxx"},
        },
        "evaluate_session": {
            "description": "评估测试会话",
            "example_params": {"session_id": "session_xxx"},
        },
        "close_test_session": {
            "description": "关闭测试会话",
            "example_params": {"session_id": "session_xxx"},
        },
        "generate_persona": {
            "description": "（旧版）生成 persona 定义",
            "example_params": {"mode": "llm", "product_name": "MyApp", "user_description": "..."},
        },
    },
}


# ═══════════════════════════════════════════════
# 2. 响应导航构建器 —— 让响应自带下一步建议
# ═══════════════════════════════════════════════

class AgentFriendlyResponseBuilder:
    """
    给 API 响应添加 agent 导航元数据：
    - _links: HATEOAS 风格的相关资源链接
    - suggested_actions: 基于当前状态的下一步建议
    """

    @classmethod
    def wrap(
        cls,
        data: Any,
        endpoint_path: str,
        method: str = "GET",
        path_params: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """包装响应数据，添加导航元数据"""
        result = {
            "data": data,
            "_links": cls._build_links(endpoint_path, method, path_params, query_params, data),
            "suggested_actions": cls._build_suggested_actions(endpoint_path, method, path_params, data),
        }
        return result

    @classmethod
    def _build_links(
        cls,
        path: str,
        method: str,
        path_params: Optional[Dict[str, str]],
        query_params: Optional[Dict[str, str]],
        data: Any,
    ) -> Dict[str, Any]:
        """构建 HATEOAS 链接"""
        links = {"self": {"href": cls._fill_params(path, path_params), "method": method}}

        # 根据端点类型添加相关链接
        if path == "/api/v1/capabilities":
            links["intent"] = {"href": "/api/v1/intent", "method": "POST"}
            links["query"] = {"href": "/api/v1/query", "method": "POST"}

        elif path == "/api/v1/types" and method == "GET":
            links["create_type"] = {"href": "/api/v1/types", "method": "POST"}
            links["presets"] = {"href": "/api/v1/presets", "method": "POST"}

        elif path == "/api/v1/types" and method == "POST":
            type_id = _safe_get(data, "type", "type_id")
            if type_id:
                links["self"] = {"href": f"/api/v1/types/{type_id}", "method": "GET"}
                links["instantiate"] = {"href": "/api/v1/instances", "method": "POST"}

        elif path.startswith("/api/v1/types/") and method == "GET":
            type_id = _extract_path_param(path, "/api/v1/types/")
            links["instantiate"] = {
                "href": "/api/v1/instances",
                "method": "POST",
                "body_template": {"type_id": type_id},
            }
            links["update"] = {"href": f"/api/v1/types/{type_id}", "method": "PUT"}
            links["delete"] = {"href": f"/api/v1/types/{type_id}", "method": "DELETE"}

        elif path == "/api/v1/instances" and method == "GET":
            links["create_instance"] = {"href": "/api/v1/instances", "method": "POST"}

        elif path == "/api/v1/instances" and method == "POST":
            inst_id = _safe_get(data, "instance", "instance_id")
            if inst_id:
                links["self"] = {"href": f"/api/v1/instances/{inst_id}", "method": "GET"}
                links["interact"] = {
                    "href": f"/api/v1/instances/{inst_id}/interact",
                    "method": "POST",
                    "body_template": {"message": "string"},
                }

        elif path.startswith("/api/v1/instances/") and path.endswith("/interact"):
            inst_id = path.replace("/api/v1/instances/", "").replace("/interact", "")
            links["continue"] = {
                "href": f"/api/v1/instances/{inst_id}/interact",
                "method": "POST",
                "body_template": {"message": "string"},
            }
            links["instance_detail"] = {"href": f"/api/v1/instances/{inst_id}", "method": "GET"}

        elif path == "/api/v1/scenes" and method == "POST":
            scene_id = _safe_get(data, "scene", "scene_id")
            if scene_id:
                links["self"] = {"href": f"/api/v1/scenes/{scene_id}", "method": "GET"}
                links["instantiate"] = {
                    "href": f"/api/v1/scenes/{scene_id}/instantiate",
                    "method": "POST",
                }
                links["group_chat"] = {
                    "href": f"/api/v1/scenes/{scene_id}/group-chat",
                    "method": "POST",
                    "body_template": {"message": "string"},
                }

        elif path.startswith("/api/v1/scenes/") and path.endswith("/instantiate"):
            scene_id = path.replace("/api/v1/scenes/", "").replace("/instantiate", "")
            links["group_chat"] = {
                "href": f"/api/v1/scenes/{scene_id}/group-chat",
                "method": "POST",
                "body_template": {"message": "string"},
            }
            links["scene_detail"] = {"href": f"/api/v1/scenes/{scene_id}", "method": "GET"}

        elif path.startswith("/api/v1/scenes/") and path.endswith("/group-chat"):
            scene_id = path.replace("/api/v1/scenes/", "").replace("/group-chat", "")
            links["continue"] = {
                "href": f"/api/v1/scenes/{scene_id}/group-chat",
                "method": "POST",
                "body_template": {"message": "string"},
            }

        elif path == "/api/v1/testing/sessions" and method == "POST":
            session_id = _safe_get(data, "session_id")
            if session_id:
                links["send_message"] = {
                    "href": f"/api/v1/testing/sessions/{session_id}/message",
                    "method": "POST",
                    "body_template": {"message": "string"},
                }
                links["get_report"] = {
                    "href": f"/api/v1/testing/sessions/{session_id}/report",
                    "method": "GET",
                }
                links["close"] = {
                    "href": f"/api/v1/testing/sessions/{session_id}",
                    "method": "DELETE",
                }
                links["self"] = {"href": f"/api/v1/testing/sessions/{session_id}", "method": "GET"}

        elif path.startswith("/api/v1/testing/sessions/") and path.endswith("/message"):
            session_id = path.replace("/api/v1/testing/sessions/", "").replace("/message", "")
            links["continue"] = {
                "href": f"/api/v1/testing/sessions/{session_id}/message",
                "method": "POST",
                "body_template": {"message": "string"},
            }
            links["evaluate"] = {
                "href": f"/api/v1/testing/sessions/{session_id}/evaluate",
                "method": "POST",
            }
            links["report"] = {
                "href": f"/api/v1/testing/sessions/{session_id}/report",
                "method": "GET",
            }
            links["close"] = {
                "href": f"/api/v1/testing/sessions/{session_id}",
                "method": "DELETE",
            }

        elif path.startswith("/api/v1/testing/sessions/") and path.endswith("/evaluate"):
            session_id = path.replace("/api/v1/testing/sessions/", "").replace("/evaluate", "")
            links["report"] = {
                "href": f"/api/v1/testing/sessions/{session_id}/report",
                "method": "GET",
            }
            links["close"] = {
                "href": f"/api/v1/testing/sessions/{session_id}",
                "method": "DELETE",
            }

        elif path.startswith("/api/v1/testing/sessions/") and method == "GET" and not path.endswith(("/message", "/evaluate", "/report")):
            session_id = path.replace("/api/v1/testing/sessions/", "")
            links["send_message"] = {
                "href": f"/api/v1/testing/sessions/{session_id}/message",
                "method": "POST",
                "body_template": {"message": "string"},
            }
            links["report"] = {
                "href": f"/api/v1/testing/sessions/{session_id}/report",
                "method": "GET",
            }
            links["evaluate"] = {
                "href": f"/api/v1/testing/sessions/{session_id}/evaluate",
                "method": "POST",
            }
            links["close"] = {
                "href": f"/api/v1/testing/sessions/{session_id}",
                "method": "DELETE",
            }

        return links

    @classmethod
    def _build_suggested_actions(
        cls,
        path: str,
        method: str,
        path_params: Optional[Dict[str, str]],
        data: Any,
    ) -> List[Dict[str, Any]]:
        """构建建议的下一步操作"""
        actions = []

        # 能力发现页面
        if path == "/api/v1/capabilities":
            actions.append({
                "action": "create_presets",
                "description": "快速创建 4 种预设虚拟人类型，立即开始测试",
                "method": "POST",
                "href": "/api/v1/presets",
            })
            actions.append({
                "action": "query_usage",
                "description": "不知道怎么做？用自然语言提问",
                "method": "POST",
                "href": "/api/v1/query",
                "body_template": {"question": "我想测试我的销售 agent，该怎么做？"},
            })

        # 类型列表
        elif path == "/api/v1/types" and method == "GET":
            types = _safe_get(data, "types", default=[])
            if not types:
                actions.append({
                    "action": "create_presets",
                    "description": "还没有任何类型，一键创建 4 种预设类型",
                    "method": "POST",
                    "href": "/api/v1/presets",
                })
            else:
                actions.append({
                    "action": "create_instance",
                    "description": "基于某个类型创建具体虚拟人",
                    "method": "POST",
                    "href": "/api/v1/instances",
                    "body_template": {"type_id": "选择一个类型的 type_id"},
                })
                actions.append({
                    "action": "start_test",
                    "description": "直接创建测试会话",
                    "method": "POST",
                    "href": "/api/v1/testing/sessions",
                    "body_template": {"persona_type": types[0].get("type_id", "anxious_buyer") if types else "anxious_buyer"},
                })

        # 预设创建完成
        elif path == "/api/v1/presets" and method == "POST":
            types = _safe_get(data, "types", default=[])
            type_ids = [t.get("type_id", "") for t in types]
            actions.append({
                "action": "start_test",
                "description": "创建一个测试会话，开始测试你的 Agent",
                "method": "POST",
                "href": "/api/v1/testing/sessions",
                "body_template": {"persona_type": type_ids[0] if type_ids else "anxious_buyer"},
            })
            actions.append({
                "action": "create_scene",
                "description": "创建一个多虚拟人场景",
                "method": "POST",
                "href": "/api/v1/scenes",
                "body_template": {
                    "name": "产品演示会",
                    "scenario": "评估一款新的 SaaS 工具",
                    "participant_configs": [{"type_id": tid, "count": 1} for tid in type_ids[:3]],
                },
            })

        # 实例创建完成
        elif path == "/api/v1/instances" and method == "POST":
            inst_id = _safe_get(data, "instance", "instance_id")
            if inst_id:
                actions.append({
                    "action": "interact",
                    "description": "开始和这个虚拟人对话",
                    "method": "POST",
                    "href": f"/api/v1/instances/{inst_id}/interact",
                    "body_template": {"message": "你好"},
                })

        # 对话交互
        elif path.startswith("/api/v1/instances/") and path.endswith("/interact"):
            inst_id = path.replace("/api/v1/instances/", "").replace("/interact", "")
            memory = _safe_get(data, "memory")
            if memory:
                actions.append({
                    "action": "continue_chat",
                    "description": f"继续对话（当前信任度: {memory.get('trust_level', '?')}, 情绪: {memory.get('emotional_state', '?')}）",
                    "method": "POST",
                    "href": f"/api/v1/instances/{inst_id}/interact",
                    "body_template": {"message": "string"},
                })
            actions.append({
                "action": "view_instance",
                "description": "查看虚拟人详情和对话历史",
                "method": "GET",
                "href": f"/api/v1/instances/{inst_id}",
            })

        # 场景创建
        elif path == "/api/v1/scenes" and method == "POST":
            scene_id = _safe_get(data, "scene", "scene_id")
            if scene_id:
                actions.append({
                    "action": "instantiate",
                    "description": "实例化场景，生成所有虚拟人参与者",
                    "method": "POST",
                    "href": f"/api/v1/scenes/{scene_id}/instantiate",
                })

        # 场景实例化
        elif path.startswith("/api/v1/scenes/") and path.endswith("/instantiate"):
            scene_id = path.replace("/api/v1/scenes/", "").replace("/instantiate", "")
            actions.append({
                "action": "group_chat",
                "description": "开始群聊，观察多个虚拟人的反应",
                "method": "POST",
                "href": f"/api/v1/scenes/{scene_id}/group-chat",
                "body_template": {"message": "大家好"},
            })

        # 群聊
        elif path.startswith("/api/v1/scenes/") and path.endswith("/group-chat"):
            scene_id = path.replace("/api/v1/scenes/", "").replace("/group-chat", "")
            turns = _safe_get(data, "turns", default=[])
            reply_count = sum(1 for t in turns if t.get("decision") == "REPLY")
            actions.append({
                "action": "continue_chat",
                "description": f"继续群聊（本轮 {reply_count} 人回复）",
                "method": "POST",
                "href": f"/api/v1/scenes/{scene_id}/group-chat",
                "body_template": {"message": "string"},
            })

        # 测试会话创建
        elif path == "/api/v1/testing/sessions" and method == "POST":
            session_id = _safe_get(data, "session_id")
            persona_name = _safe_get(data, "persona", "name")
            if session_id:
                actions.append({
                    "action": "send_message",
                    "description": f"向 {persona_name or '虚拟人'} 发送消息，开始测试",
                    "method": "POST",
                    "href": f"/api/v1/testing/sessions/{session_id}/message",
                    "body_template": {"message": "你的 Agent 的回复"},
                })
                actions.append({
                    "action": "get_report",
                    "description": "查看测试报告",
                    "method": "GET",
                    "href": f"/api/v1/testing/sessions/{session_id}/report",
                })

        # 测试消息发送
        elif path.startswith("/api/v1/testing/sessions/") and path.endswith("/message"):
            session_id = path.replace("/api/v1/testing/sessions/", "").replace("/message", "")
            trust = _safe_get(data, "trust_level")
            emotional = _safe_get(data, "emotional_state")
            turn = _safe_get(data, "turn", 0)
            actions.append({
                "action": "continue_test",
                "description": f"继续测试（第 {turn} 轮，信任度: {trust or '?'}, 情绪: {emotional or '?' }）",
                "method": "POST",
                "href": f"/api/v1/testing/sessions/{session_id}/message",
                "body_template": {"message": "string"},
            })
            actions.append({
                "action": "evaluate",
                "description": "评估当前对话质量",
                "method": "POST",
                "href": f"/api/v1/testing/sessions/{session_id}/evaluate",
            })
            actions.append({
                "action": "get_report",
                "description": "查看完整测试报告",
                "method": "GET",
                "href": f"/api/v1/testing/sessions/{session_id}/report",
            })
            actions.append({
                "action": "close_session",
                "description": "结束测试",
                "method": "DELETE",
                "href": f"/api/v1/testing/sessions/{session_id}",
            })

        # 评估完成
        elif path.startswith("/api/v1/testing/sessions/") and path.endswith("/evaluate"):
            session_id = path.replace("/api/v1/testing/sessions/", "").replace("/evaluate", "")
            actions.append({
                "action": "get_report",
                "description": "查看完整测试报告",
                "method": "GET",
                "href": f"/api/v1/testing/sessions/{session_id}/report",
            })
            actions.append({
                "action": "close_session",
                "description": "结束测试",
                "method": "DELETE",
                "href": f"/api/v1/testing/sessions/{session_id}",
            })

        return actions

    @staticmethod
    def _fill_params(path: str, params: Optional[Dict[str, str]]) -> str:
        if not params:
            return path
        result = path
        for k, v in params.items():
            result = result.replace(f"{{{k}}}", str(v))
        return result


# ═══════════════════════════════════════════════
# 3. 意图路由器 —— 高层意图驱动
# ═══════════════════════════════════════════════

class IntentRequest(BaseModel):
    intent: str = Field(..., description="意图名称")
    params: Dict[str, Any] = Field(default_factory=dict, description="意图参数")


class IntentRouter:
    """
    将高层意图路由到具体的 VMU 操作。
    Agent 不需要知道底层 API，只需要表达意图。
    """

    def __init__(self, manager, llm_client=None):
        self.manager = manager
        self.llm_client = llm_client

    async def route(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """根据意图名称执行对应操作"""
        handler = getattr(self, f"_handle_{intent}", None)
        if handler is None:
            return {
                "success": False,
                "error": f"不支持的意图: '{intent}'",
                "supported_intents": list(CAPABILITIES_REGISTRY["supported_intents"].keys()),
            }
        try:
            result = await handler(params)
            return {"success": True, "intent": intent, "result": result}
        except Exception as e:
            return {"success": False, "intent": intent, "error": str(e)}

    async def _handle_discover(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """发现服务能力"""
        return {"capabilities": CAPABILITIES_REGISTRY}

    async def _handle_list_persona_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        types = self.manager.list_types()
        return {"types": [pt.model_dump() for pt in types]}

    async def _handle_create_preset_personas(self, params: Dict[str, Any]) -> Dict[str, Any]:
        types = self.manager.create_preset_types()
        return {"types": [pt.model_dump() for pt in types], "message": "已创建 4 种预设人格类型"}

    async def _handle_create_persona_type(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .models import Demographics, Psychographics, BehavioralTraits, Context, SceneContext, BehaviorEngine

        pt = self.manager.create_type(
            type_id=params["type_id"],
            name=params["name"],
            description=params.get("description", ""),
            demographics=Demographics(**params.get("demographics", {})) if params.get("demographics") else None,
            psychographics=Psychographics(**params.get("psychographics", {})) if params.get("psychographics") else None,
            behavioral_traits=BehavioralTraits(**params.get("behavioral_traits", {})) if params.get("behavioral_traits") else None,
            context=Context(**params.get("context", {})) if params.get("context") else None,
            scene_context=SceneContext(**params.get("scene_context", {})) if params.get("scene_context") else None,
            behavior_engine=BehaviorEngine(**params.get("behavior_engine", {})) if params.get("behavior_engine") else None,
            system_prompt_template=params.get("system_prompt_template", ""),
            variation_config=params.get("variation_config", {}),
        )
        return {"type": pt.model_dump()}

    async def _handle_instantiate_persona(self, params: Dict[str, Any]) -> Dict[str, Any]:
        inst = self.manager.instantiate(
            type_id=params["type_id"],
            name=params.get("name"),
            variation=params.get("variation"),
            variation_seed=params.get("variation_seed"),
        )
        if not inst:
            raise ValueError(f"PersonaType '{params['type_id']}' 不存在")
        return _inst_to_dict(inst)

    async def _handle_interact_with_persona(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .agent import PersonaAgent
        from deepseek_client import async_chat_completion

        inst = self.manager.get_instance(params["instance_id"])
        if not inst:
            raise ValueError(f"Instance '{params['instance_id']}' 不存在")

        agent = PersonaAgent(
            instance=inst,
            llm_client=async_chat_completion,
            auto_persist=True,
            storage=self.manager.storage,
        )
        result = await agent.ainteract(
            params["message"],
            include_history=params.get("include_history", True),
            temperature=params.get("temperature", 0.7),
        )
        return {
            "instance_id": result.instance_id,
            "response": result.response,
            "memory": result.updated_memory.model_dump() if result.updated_memory else None,
            "metadata": result.metadata,
        }

    async def _handle_create_scene(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .models import SceneParticipant

        configs = [SceneParticipant(**c) for c in params.get("participant_configs", [])]
        scene = self.manager.create_scene(
            name=params["name"],
            description=params.get("description", ""),
            scenario=params.get("scenario", ""),
            participant_configs=configs,
            shared_context=params.get("shared_context", {}),
        )
        return {"scene": scene.model_dump()}

    async def _handle_instantiate_scene(self, params: Dict[str, Any]) -> Dict[str, Any]:
        scene = self.manager.instantiate_scene(params["scene_id"])
        if not scene:
            raise ValueError(f"Scene '{params['scene_id']}' 不存在")
        participants = self.manager.get_scene_instances(params["scene_id"])
        return {
            "scene": scene.model_dump(),
            "participants": [_inst_to_dict(p) for p in participants],
        }

    async def _handle_group_chat(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .agent import GroupChatEngine
        from deepseek_client import async_chat_completion

        scene_id = params["scene_id"]
        scene = self.manager.get_scene(scene_id)
        if not scene:
            raise ValueError(f"Scene '{scene_id}' 不存在")

        participants = self.manager.get_scene_instances(scene_id)
        if not participants:
            raise ValueError("场景没有实例化的参与者")

        engine = GroupChatEngine(llm_client=async_chat_completion)
        turns = await engine.arun_turn(
            scene_id=scene_id,
            participants=participants,
            user_message=params["message"],
            temperature=params.get("temperature", 0.7),
            storage=self.manager.storage,
        )
        return {
            "scene_id": scene_id,
            "user_message": params["message"],
            "turns": turns,
        }

    async def _handle_test_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """创建测试会话"""
        from .testing import DummyUserTester
        from deepseek_client import async_chat_completion

        tester = DummyUserTester(manager=self.manager, llm_client=async_chat_completion)

        # 确保预设类型存在
        types = self.manager.list_types()
        if not types:
            self.manager.create_preset_types()

        session = tester.create_session(
            agent=lambda msg: msg,
            persona_type=params["persona_type"],
            name=params.get("name"),
            scene_overrides=params.get("scene_overrides"),
        )
        return {
            "session_id": session.session_id,
            "persona": {
                "instance_id": session.persona_instance.instance_id,
                "name": session.persona_instance.name,
                "type_id": session.persona_instance.type_id,
            },
            "status": session.status,
            "created_at": session.created_at.isoformat(),
        }

    async def _handle_send_test_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .testing import DummyUserTester
        from deepseek_client import async_chat_completion

        tester = DummyUserTester(manager=self.manager, llm_client=async_chat_completion)
        response = await tester.asend_to_session(
            session_id=params["session_id"],
            agent_message=params["message"],
            temperature=params.get("temperature", 0.7),
        )
        session = tester.get_session(params["session_id"])
        return {
            "session_id": params["session_id"],
            "user_response": response,
            "turn": len(session.turns) if session else 0,
            "trust_level": session.persona_instance.memory.trust_level if session else None,
            "emotional_state": session.persona_instance.memory.emotional_state if session else None,
        }

    async def _handle_get_test_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .testing import DummyUserTester, TestReport

        tester = DummyUserTester(manager=self.manager)
        session = tester.get_session(params["session_id"])
        if not session:
            raise ValueError(f"Session '{params['session_id']}' 不存在")

        report = TestReport.from_session(session)
        fmt = params.get("fmt", "json")
        if fmt == "markdown":
            return {"format": "markdown", "content": report.to_markdown()}
        elif fmt == "console":
            return {"format": "console", "content": report.to_console()}
        else:
            return report.to_dict()

    async def _handle_evaluate_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .testing import DummyUserTester, ConversationEvaluator
        from deepseek_client import async_chat_completion

        tester = DummyUserTester(manager=self.manager)
        session = tester.get_session(params["session_id"])
        if not session:
            raise ValueError(f"Session '{params['session_id']}' 不存在")

        evaluator = ConversationEvaluator(llm_client=async_chat_completion)
        evaluation = await evaluator.aevaluate(
            persona=session.persona_instance,
            turns=session.turns,
            agent_name=session.agent_name,
        )
        return {"session_id": params["session_id"], "evaluation": evaluation}

    async def _handle_close_test_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .testing import DummyUserTester

        tester = DummyUserTester(manager=self.manager)
        ok = tester.close_session(params["session_id"])
        if not ok:
            raise ValueError(f"Session '{params['session_id']}' 不存在")
        return {"closed": True, "session_id": params["session_id"]}

    async def _handle_generate_persona(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用 legacy generate 端点的逻辑"""
        # 这里不做实际 LLM 调用，返回提示让 agent 直接调用 /api/generate
        return {
            "message": "请直接调用 POST /api/generate",
            "endpoint": "/api/generate",
            "method": "POST",
            "body_template": {
                "mode": params.get("mode", "llm"),
                "product_name": params.get("product_name", "产品"),
                "product_type": params.get("product_type", "SaaS"),
                "user_description": params.get("user_description", ""),
            },
        }


# ═══════════════════════════════════════════════
# 4. 自然语言查询引擎
# ═══════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str = Field(..., description="自然语言问题")
    context: Optional[Dict[str, Any]] = Field(default=None, description="当前上下文（可选）")


class NLQueryEngine:
    """
    用自然语言提问，获取建议的 API 调用序列。
    支持两种模式：LLM 增强（需要 API key）和规则 fallback（无需 LLM）。
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        self.llm_client = llm_client

    async def query(self, question: str, current_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理自然语言查询"""
        # 先尝试规则匹配
        rule_result = self._rule_based_query(question)
        if rule_result:
            return rule_result

        # 如果有 LLM，用 LLM 增强
        if self.llm_client and (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")):
            try:
                return await self._llm_query(question, current_context)
            except Exception:
                pass

        # Fallback
        return self._fallback_response(question)

    def _rule_based_query(self, question: str) -> Optional[Dict[str, Any]]:
        """基于关键词的规则匹配"""
        q = question.lower()

        # 测试相关
        if any(k in q for k in ["测试", "test", "测我的", "验证", "跑一下"]):
            if any(k in q for k in ["场景", "scene", "多个", "多人", "群聊", "group"]):
                return {
                    "understanding": "你想在场景中与多个虚拟人进行测试",
                    "suggested_workflow": "场景测试",
                    "steps": [
                        {"action": "创建预设类型", "endpoint": "POST /api/v1/presets"},
                        {"action": "创建场景", "endpoint": "POST /api/v1/scenes", "body_template": {"name": "测试场景", "participant_configs": [{"type_id": "anxious_buyer", "count": 1}]}},
                        {"action": "实例化场景", "endpoint": "POST /api/v1/scenes/{scene_id}/instantiate"},
                        {"action": "群聊", "endpoint": "POST /api/v1/scenes/{scene_id}/group-chat", "body_template": {"message": "你的 Agent 的消息"}},
                    ],
                }
            return {
                "understanding": "你想测试你的 Agent",
                "suggested_workflow": "快速开始测试",
                "steps": [
                    {"action": "创建预设类型（如果没有）", "endpoint": "POST /api/v1/presets"},
                    {"action": "创建测试会话", "endpoint": "POST /api/v1/testing/sessions", "body_template": {"persona_type": "anxious_buyer"}},
                    {"action": "发送消息", "endpoint": "POST /api/v1/testing/sessions/{session_id}/message", "body_template": {"message": "你的 Agent 的回复"}},
                    {"action": "（重复发送消息进行多轮对话）"},
                    {"action": "获取报告", "endpoint": "GET /api/v1/testing/sessions/{session_id}/report"},
                ],
            }

        # 查看/发现
        if any(k in q for k in ["有什么", "有哪些", "list", "show", "查看", "可用"]):
            if any(k in q for k in ["类型", "type", "persona", "用户"]):
                return {
                    "understanding": "你想查看有哪些虚拟人类型",
                    "direct_api": {"endpoint": "GET /api/v1/types", "description": "列出所有人格类型"},
                }
            if any(k in q for k in ["场景", "scene"]):
                return {
                    "understanding": "你想查看有哪些场景",
                    "direct_api": {"endpoint": "GET /api/v1/scenes", "description": "列出所有场景"},
                }
            return {
                "understanding": "你想了解服务能力",
                "direct_api": {"endpoint": "GET /api/v1/capabilities", "description": "获取完整能力目录"},
            }

        # 创建 persona
        if any(k in q for k in ["创建", "新建", "生成", "create", "make", "new"]):
            if any(k in q for k in ["预设", "preset", "默认"]):
                return {
                    "understanding": "你想创建预设虚拟人类型",
                    "direct_api": {"endpoint": "POST /api/v1/presets", "description": "一键创建 4 种预设类型"},
                }
            if any(k in q for k in ["场景", "scene"]):
                return {
                    "understanding": "你想创建场景",
                    "direct_api": {"endpoint": "POST /api/v1/scenes", "description": "创建场景"},
                }
            return {
                "understanding": "你想创建虚拟人类型或实例",
                "options": [
                    {"description": "创建类型模板", "endpoint": "POST /api/v1/types"},
                    {"description": "基于类型创建具体实例", "endpoint": "POST /api/v1/instances", "body_template": {"type_id": "type_id"}},
                ],
            }

        # 对话/交互
        if any(k in q for k in ["对话", "聊天", "chat", "talk", "说话", "聊", "interact"]):
            return {
                "understanding": "你想和虚拟人对话",
                "options": [
                    {"description": "如果你有 instance_id，直接交互", "endpoint": "POST /api/v1/instances/{instance_id}/interact"},
                    {"description": "如果还没有虚拟人，先创建", "steps": [
                        {"endpoint": "POST /api/v1/presets"},
                        {"endpoint": "POST /api/v1/instances", "body_template": {"type_id": "anxious_buyer"}},
                        {"endpoint": "POST /api/v1/instances/{instance_id}/interact"},
                    ]},
                ],
            }

        # 报告/评估
        if any(k in q for k in ["报告", "评估", "评价", "report", "evaluate", "result", "结果"]):
            return {
                "understanding": "你想查看测试报告或评估结果",
                "direct_api": {"endpoint": "GET /api/v1/testing/sessions/{session_id}/report", "query_params": {"fmt": "json"}},
                "note": "需要 session_id。如果不记得，先 GET /api/v1/testing/sessions 列出所有会话。",
            }

        return None

    async def _llm_query(self, question: str, current_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """使用 LLM 进行查询解析"""
        system_prompt = f"""你是一位 API 使用助手。用户用自然语言询问如何使用 PersonaForge API。

服务概述：PersonaForge 是虚拟用户生成与测试平台。
核心概念：PersonaType（类型模板）-> PersonaInstance（具体实例）-> Scene（场景）-> TestingSession（测试会话）。

你的任务：分析用户的问题，给出最简洁、最准确的 API 使用建议。

请按以下 JSON 格式输出：
{{
  "understanding": "对用户意图的 1 句话理解",
  "suggested_workflow": "推荐的工作流名称（如果有）",
  "steps": [
    {{"action": "步骤描述", "endpoint": "METHOD /path", "body_template": {{...}}, "note": "可选说明"}}
  ],
  "direct_api": {{"endpoint": "METHOD /path", "description": "说明"}},
  "tips": ["给 agent 的额外提示"]
}}

可用端点速查：
- GET /api/v1/capabilities — 能力目录
- POST /api/v1/presets — 创建预设类型
- GET /api/v1/types — 列出类型
- POST /api/v1/instances — 创建实例
- POST /api/v1/instances/{{id}}/interact — 对话
- POST /api/v1/scenes — 创建场景
- POST /api/v1/scenes/{{id}}/group-chat — 群聊
- POST /api/v1/testing/sessions — 创建测试会话
- POST /api/v1/testing/sessions/{{id}}/message — 测试消息
- GET /api/v1/testing/sessions/{{id}}/report — 测试报告
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户问题：{question}\n当前上下文：{json.dumps(current_context, ensure_ascii=False) if current_context else '无'}"},
        ]

        response = await self.llm_client(messages=messages, temperature=0.3, max_tokens=1500)

        # 尝试解析 JSON
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            result = json.loads(cleaned)
            result["_source"] = "llm"
            return result
        except json.JSONDecodeError:
            return {
                "understanding": question,
                "suggested_workflow": None,
                "steps": [],
                "llm_raw_response": response,
                "_source": "llm_raw",
            }

    def _fallback_response(self, question: str) -> Dict[str, Any]:
        return {
            "understanding": f"你的问题是：{question}",
            "message": "我没能完全理解你的问题。你可以：",
            "options": [
                {"action": "查看完整能力目录", "endpoint": "GET /api/v1/capabilities"},
                {"action": "查看推荐工作流", "note": "capabilities.workflows 中列出了常见用法"},
                {"action": "使用意图驱动", "endpoint": "POST /api/v1/intent", "body_template": {"intent": "discover", "params": {}}},
            ],
        }


# ═══════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════

def _safe_get(data: Any, *keys, default=None):
    """安全地从嵌套 dict 中获取值"""
    if data is None:
        return default
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _extract_path_param(path: str, prefix: str) -> Optional[str]:
    """从路径中提取参数，如 /api/v1/types/abc -> abc"""
    if path.startswith(prefix):
        rest = path[len(prefix):]
        # 去掉后续路径段
        return rest.split("/")[0]
    return None


def _inst_to_dict(inst) -> Dict[str, Any]:
    """将 PersonaInstance 转为可 JSON 序列化的 dict"""
    d = inst.model_dump()
    if "message_history" in d:
        d["message_history_count"] = len(d["message_history"])
        d["message_history_preview"] = [
            {"role": m["role"], "content": m["content"][:200]}
            for m in d["message_history"][-5:]
        ]
    return d
