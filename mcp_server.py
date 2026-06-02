#!/usr/bin/env python3
"""
PersonaForge MCP Server

让支持 MCP（Model Context Protocol）的 Agent（Claude Desktop、Cursor、Cline 等）
能够零配置发现和使用 PersonaForge 的能力。

使用方式：
1. 在支持 MCP 的客户端配置中添加：
   {
     "mcpServers": {
       "personaforge": {
         "command": "python",
         "args": ["/path/to/mcp_server.py"]
       }
     }
   }
2. Agent 会自动发现 tools，无需手动阅读 API 文档

协议：stdio-based JSON-RPC 2.0（MCP 标准）
"""

import asyncio
import json
import sys
import traceback
from typing import Any, Dict, List, Optional

from vmu import PersonaManager
from vmu.agent_friendly import CAPABILITIES_REGISTRY
from vmu.agent import PersonaAgent, GroupChatEngine
from vmu.testing import DummyUserTester, ConversationEvaluator, TestReport
from deepseek_client import async_chat_completion


# ───────────────────────────────────────────────
# MCP 协议基础
# ───────────────────────────────────────────────

SERVER_INFO = {
    "name": "personaforge",
    "version": "2.0",
}

SERVER_CAPABILITIES = {
    "tools": {"listChanged": False},
}


def send_message(msg: Dict[str, Any]):
    """通过 stdout 发送 JSON-RPC 消息"""
    payload = json.dumps(msg, ensure_ascii=False)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


def send_error(request_id: Any, code: int, message: str, data: Any = None):
    """发送 JSON-RPC 错误响应"""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    send_message({"jsonrpc": "2.0", "id": request_id, "error": error})


def send_result(request_id: Any, result: Any):
    """发送 JSON-RPC 成功响应"""
    send_message({"jsonrpc": "2.0", "id": request_id, "result": result})


# ───────────────────────────────────────────────
# Tools 定义
# ───────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_capabilities",
        "description": "获取 PersonaForge 的完整能力目录。当不确定能做什么时，先调用此工具。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "create_preset_personas",
        "description": "一键创建 4 种预设虚拟人类型（焦虑型买家、理性分析师、技术怀疑者、冲动决策者）。快速开始测试的第一步。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_persona_types",
        "description": "列出所有可用的虚拟人类型模板。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "create_persona_type",
        "description": "创建自定义虚拟人类型模板。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type_id": {"type": "string", "description": "唯一标识，如 'budget_controller'"},
                "name": {"type": "string", "description": "显示名称"},
                "description": {"type": "string", "description": "类型描述"},
                "demographics": {"type": "object", "description": "人口统计信息"},
                "psychographics": {"type": "object", "description": "心理特征"},
                "behavioral_traits": {"type": "object", "description": "行为特征"},
                "variation_config": {"type": "object", "description": "变异配置"},
            },
            "required": ["type_id", "name"],
        },
    },
    {
        "name": "create_persona_instance",
        "description": "基于某个 PersonaType 创建一个具体的虚拟人实例，可以和它对话。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "type_id": {"type": "string", "description": "类型 ID，如 anxious_buyer"},
                "name": {"type": "string", "description": "实例名称（可选）"},
                "variation": {"type": "object", "description": "手动变异参数（可选）"},
                "variation_seed": {"type": "integer", "description": "随机种子（可选）"},
            },
            "required": ["type_id"],
        },
    },
    {
        "name": "interact_with_persona",
        "description": "与某个虚拟人实例进行一次对话。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "instance_id": {"type": "string", "description": "虚拟人实例 ID"},
                "message": {"type": "string", "description": "你说的话"},
                "temperature": {"type": "number", "description": "温度，默认 0.7"},
            },
            "required": ["instance_id", "message"],
        },
    },
    {
        "name": "create_scene",
        "description": "创建一个场景，定义多个虚拟人参与的交互环境。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "场景名称"},
                "description": {"type": "string", "description": "场景描述"},
                "scenario": {"type": "string", "description": "场景剧本/背景"},
                "participant_configs": {
                    "type": "array",
                    "description": "参与者配置列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type_id": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["type_id", "count"],
                    },
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "instantiate_scene",
        "description": "根据场景的 participant_configs 生成所有虚拟人实例。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "场景 ID"},
            },
            "required": ["scene_id"],
        },
    },
    {
        "name": "group_chat",
        "description": "在场景中进行群聊。发送一条消息，所有虚拟人根据角色决定是否回复。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "场景 ID"},
                "message": {"type": "string", "description": "消息内容"},
                "temperature": {"type": "number", "description": "温度，默认 0.7"},
            },
            "required": ["scene_id", "message"],
        },
    },
    {
        "name": "create_test_session",
        "description": "创建一个测试会话，让你的 Agent 和一个虚拟人进行测试对话。这是测试 Agent 的第一步。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "persona_type": {"type": "string", "description": "虚拟人类型 ID，如 anxious_buyer"},
                "name": {"type": "string", "description": "虚拟人名称（可选）"},
                "scene_overrides": {"type": "object", "description": "场景覆盖参数（可选）"},
            },
            "required": ["persona_type"],
        },
    },
    {
        "name": "send_test_message",
        "description": "向测试会话发送消息（你的 Agent → 虚拟人），返回虚拟人回复。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "测试会话 ID"},
                "message": {"type": "string", "description": "你的 Agent 说的话"},
                "temperature": {"type": "number", "description": "温度，默认 0.7"},
            },
            "required": ["session_id", "message"],
        },
    },
    {
        "name": "get_test_report",
        "description": "获取测试会话的完整报告。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "测试会话 ID"},
                "fmt": {"type": "string", "description": "格式：json / markdown / console，默认 json"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "evaluate_test_session",
        "description": "对测试会话进行 LLM 评估，分析对话质量。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "测试会话 ID"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "close_test_session",
        "description": "关闭并清理测试会话。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "测试会话 ID"},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "query_service",
        "description": "用自然语言提问，获取建议的 API 调用序列。不知道怎么做时就用这个。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "自然语言问题，如'我想测试我的销售 agent，该怎么做？'"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "execute_intent",
        "description": "用高层意图直接执行操作。不需要知道具体 API，只需要描述想做什么。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "意图名称，如 test_agent, create_preset_personas, interact_with_persona 等",
                },
                "params": {"type": "object", "description": "意图参数"},
            },
            "required": ["intent"],
        },
    },
]


# ───────────────────────────────────────────────
# Tool 执行逻辑
# ───────────────────────────────────────────────

class MCPToolHandler:
    """处理 MCP tool 调用，直接通过 Python SDK 操作"""

    def __init__(self):
        self.manager = PersonaManager()
        self.tester = DummyUserTester(manager=self.manager, llm_client=async_chat_completion)
        self._ensure_presets()

    def _ensure_presets(self):
        """确保有预设类型可用"""
        types = self.manager.list_types()
        if not types:
            self.manager.create_preset_types()

    async def handle(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await handler(arguments)

    async def _handle_get_capabilities(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"capabilities": CAPABILITIES_REGISTRY}

    async def _handle_create_preset_personas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        types = self.manager.create_preset_types()
        return {
            "message": "已创建 4 种预设人格类型",
            "types": [{"type_id": t.type_id, "name": t.name, "description": t.description} for t in types],
        }

    async def _handle_list_persona_types(self, args: Dict[str, Any]) -> Dict[str, Any]:
        types = self.manager.list_types()
        return {
            "types": [{"type_id": t.type_id, "name": t.name, "description": t.description} for t in types],
        }

    async def _handle_create_persona_type(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from vmu.models import Demographics, Psychographics, BehavioralTraits, Context, SceneContext, BehaviorEngine

        pt = self.manager.create_type(
            type_id=args["type_id"],
            name=args["name"],
            description=args.get("description", ""),
            demographics=Demographics(**args.get("demographics", {})) if args.get("demographics") else None,
            psychographics=Psychographics(**args.get("psychographics", {})) if args.get("psychographics") else None,
            behavioral_traits=BehavioralTraits(**args.get("behavioral_traits", {})) if args.get("behavioral_traits") else None,
            context=Context(**args.get("context", {})) if args.get("context") else None,
            scene_context=SceneContext(**args.get("scene_context", {})) if args.get("scene_context") else None,
            behavior_engine=BehaviorEngine(**args.get("behavior_engine", {})) if args.get("behavior_engine") else None,
            variation_config=args.get("variation_config", {}),
        )
        return {"type": pt.model_dump()}

    async def _handle_create_persona_instance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        inst = self.manager.instantiate(
            type_id=args["type_id"],
            name=args.get("name"),
            variation=args.get("variation"),
            variation_seed=args.get("variation_seed"),
        )
        if not inst:
            raise ValueError(f"PersonaType '{args['type_id']}' 不存在")
        return {
            "instance_id": inst.instance_id,
            "name": inst.name,
            "type_id": inst.type_id,
            "system_prompt_preview": inst.system_prompt[:200] + "..." if len(inst.system_prompt) > 200 else inst.system_prompt,
        }

    async def _handle_interact_with_persona(self, args: Dict[str, Any]) -> Dict[str, Any]:
        inst = self.manager.get_instance(args["instance_id"])
        if not inst:
            raise ValueError(f"Instance '{args['instance_id']}' 不存在")

        agent = PersonaAgent(
            instance=inst,
            llm_client=async_chat_completion,
            auto_persist=True,
            storage=self.manager.storage,
        )
        result = await agent.ainteract(
            args["message"],
            temperature=args.get("temperature", 0.7),
        )
        return {
            "response": result.response,
            "trust_level": result.updated_memory.trust_level if result.updated_memory else None,
            "emotional_state": result.updated_memory.emotional_state if result.updated_memory else None,
            "exposure_count": result.updated_memory.exposure_count if result.updated_memory else None,
        }

    async def _handle_create_scene(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from vmu.models import SceneParticipant

        configs = [SceneParticipant(**c) for c in args.get("participant_configs", [])]
        scene = self.manager.create_scene(
            name=args["name"],
            description=args.get("description", ""),
            scenario=args.get("scenario", ""),
            participant_configs=configs,
        )
        return {"scene": scene.model_dump()}

    async def _handle_instantiate_scene(self, args: Dict[str, Any]) -> Dict[str, Any]:
        scene = self.manager.instantiate_scene(args["scene_id"])
        if not scene:
            raise ValueError(f"Scene '{args['scene_id']}' 不存在")
        participants = self.manager.get_scene_instances(args["scene_id"])
        return {
            "scene_id": scene.scene_id,
            "participant_count": len(participants),
            "participants": [{"instance_id": p.instance_id, "name": p.name, "type_id": p.type_id} for p in participants],
        }

    async def _handle_group_chat(self, args: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = args["scene_id"]
        participants = self.manager.get_scene_instances(scene_id)
        if not participants:
            raise ValueError("场景没有实例化的参与者")

        engine = GroupChatEngine(llm_client=async_chat_completion)
        turns = await engine.arun_turn(
            scene_id=scene_id,
            participants=participants,
            user_message=args["message"],
            temperature=args.get("temperature", 0.7),
            storage=self.manager.storage,
        )
        replies = [t for t in turns if t.get("decision") == "REPLY"]
        return {
            "replies_count": len(replies),
            "replies": [{"name": t["name"], "reply": t["reply"], "reasoning": t["reasoning"]} for t in replies],
            "all_turns": turns,
        }

    async def _handle_create_test_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        session = self.tester.create_session(
            agent=lambda msg: msg,
            persona_type=args["persona_type"],
            name=args.get("name"),
            scene_overrides=args.get("scene_overrides"),
        )
        return {
            "session_id": session.session_id,
            "persona_name": session.persona_instance.name,
            "type_id": session.persona_instance.type_id,
            "status": session.status,
            "next_step": f"使用 send_test_message 向 session_id={session.session_id} 发送消息",
        }

    async def _handle_send_test_message(self, args: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.tester.asend_to_session(
            session_id=args["session_id"],
            agent_message=args["message"],
            temperature=args.get("temperature", 0.7),
        )
        session = self.tester.get_session(args["session_id"])
        return {
            "user_response": response,
            "turn": len(session.turns) if session else 0,
            "trust_level": session.persona_instance.memory.trust_level if session else None,
            "emotional_state": session.persona_instance.memory.emotional_state if session else None,
        }

    async def _handle_get_test_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        session = self.tester.get_session(args["session_id"])
        if not session:
            raise ValueError(f"Session '{args['session_id']}' 不存在")

        report = TestReport.from_session(session)
        fmt = args.get("fmt", "json")
        if fmt == "markdown":
            return {"format": "markdown", "content": report.to_markdown()}
        elif fmt == "console":
            return {"format": "console", "content": report.to_console()}
        else:
            return report.to_dict()

    async def _handle_evaluate_test_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        session = self.tester.get_session(args["session_id"])
        if not session:
            raise ValueError(f"Session '{args['session_id']}' 不存在")

        evaluator = ConversationEvaluator(llm_client=async_chat_completion)
        evaluation = await evaluator.aevaluate(
            persona=session.persona_instance,
            turns=session.turns,
            agent_name=session.agent_name,
        )
        return {"evaluation": evaluation}

    async def _handle_close_test_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        ok = self.tester.close_session(args["session_id"])
        if not ok:
            raise ValueError(f"Session '{args['session_id']}' 不存在")
        return {"closed": True, "session_id": args["session_id"]}

    async def _handle_query_service(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from vmu.agent_friendly import NLQueryEngine
        engine = NLQueryEngine(llm_client=async_chat_completion)
        return await engine.query(args["question"])

    async def _handle_execute_intent(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from vmu.agent_friendly import IntentRouter
        router = IntentRouter(manager=self.manager, llm_client=async_chat_completion)
        return await router.route(args["intent"], args.get("params", {}))


# ───────────────────────────────────────────────
# 主循环
# ───────────────────────────────────────────────

async def main():
    handler = MCPToolHandler()
    initialized = False

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            msg = json.loads(line.strip())
            method = msg.get("method", "")
            msg_id = msg.get("id")
            params = msg.get("params", {})

            # ── initialize ──
            if method == "initialize":
                send_result(msg_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": SERVER_CAPABILITIES,
                    "serverInfo": SERVER_INFO,
                })
                initialized = True
                continue

            # ── notifications/initialized ──
            if method == "notifications/initialized":
                # 无需响应
                continue

            if not initialized:
                send_error(msg_id, -32002, "Server not initialized")
                continue

            # ── tools/list ──
            if method == "tools/list":
                send_result(msg_id, {"tools": TOOLS})
                continue

            # ── tools/call ──
            if method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})
                try:
                    result = await handler.handle(tool_name, arguments)
                    send_result(msg_id, {
                        "content": [
                            {"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}
                        ],
                    })
                except Exception as e:
                    traceback_str = traceback.format_exc()
                    send_result(msg_id, {
                        "content": [
                            {"type": "text", "text": f"Error: {str(e)}\n\n{traceback_str}"}
                        ],
                        "isError": True,
                    })
                continue

            # ── Unknown method ──
            if msg_id is not None:
                send_error(msg_id, -32601, f"Method not found: {method}")

        except json.JSONDecodeError as e:
            send_error(None, -32700, f"Parse error: {str(e)}")
        except Exception as e:
            traceback_str = traceback.format_exc()
            if 'msg_id' in locals() and msg_id is not None:
                send_error(msg_id, -32603, f"Internal error: {str(e)}", {"traceback": traceback_str})
            else:
                # 记录到 stderr，不要破坏 stdout 协议
                print(f"Internal error: {traceback_str}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
