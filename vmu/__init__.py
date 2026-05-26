"""
Virtual User Management (VMU)
虚拟人管理系统

提供：
- PersonaType 定义与管理
- PersonaInstance 实例化（带差异化）
- Scene 场景管理
- Agent LLM 交互接口
"""

from .models import (
    PersonaType,
    PersonaInstance,
    Scene,
    SceneParticipant,
    Message,
    InteractionResult,
)
from .manager import PersonaManager
from .agent import PersonaAgent
from .storage import JsonStorage

__all__ = [
    "PersonaType",
    "PersonaInstance",
    "Scene",
    "SceneParticipant",
    "Message",
    "InteractionResult",
    "PersonaManager",
    "PersonaAgent",
    "JsonStorage",
]
