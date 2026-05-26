"""
JSON 文件持久化层
所有数据以 JSON 形式保存在 data/ 目录下
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


class JsonStorage:
    """
    基于 JSON 文件的简单持久化存储。
    
    目录结构：
        data/
        ├── types/      PersonaType
        ├── instances/  PersonaInstance
        └── scenes/     Scene
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        for sub in ("types", "instances", "scenes"):
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)
    
    def _path(self, entity_type: str, entity_id: str) -> Path:
        """获取某类实体的存储路径"""
        return self.base_dir / entity_type / f"{entity_id}.json"
    
    def _json_default(self, obj):
        """处理 JSON 不支持的类型"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def save(self, entity_type: str, entity_id: str, data: Dict[str, Any]):
        """保存实体"""
        path = self._path(entity_type, entity_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=self._json_default)
    
    def load(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """加载实体"""
        path = self._path(entity_type, entity_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def delete(self, entity_type: str, entity_id: str) -> bool:
        """删除实体，返回是否成功"""
        path = self._path(entity_type, entity_id)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_ids(self, entity_type: str) -> List[str]:
        """列出某类实体的所有 ID"""
        dir_path = self.base_dir / entity_type
        if not dir_path.exists():
            return []
        return [
            p.stem for p in dir_path.glob("*.json")
            if p.is_file()
        ]
    
    def list_all(self, entity_type: str) -> List[Dict[str, Any]]:
        """列出某类实体的所有数据"""
        results = []
        for eid in self.list_ids(entity_type):
            data = self.load(entity_type, eid)
            if data:
                results.append(data)
        return results
