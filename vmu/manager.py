"""
PersonaManager：虚拟人管理中心

职责：
1. PersonaType CRUD
2. PersonaInstance 实例化（带差异化）
3. Scene 场景管理
4. 持久化（通过 Storage）
"""

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
    Scene,
    SceneContext,
    SceneParticipant,
)
from .prompts import PromptRenderer, VariationGenerator
from .storage import JsonStorage


class PersonaManager:
    """
    虚拟人管理器。
    
    所有 VMU 操作的统一入口。
    """
    
    def __init__(self, storage: Optional[JsonStorage] = None):
        self.storage = storage or JsonStorage()
    
    # ═══════════════════════════════════════════════
    # PersonaType 管理
    # ═══════════════════════════════════════════════
    
    def create_type(
        self,
        type_id: str,
        name: str,
        description: str = "",
        demographics: Optional[Demographics] = None,
        psychographics: Optional[Psychographics] = None,
        behavioral_traits: Optional[BehavioralTraits] = None,
        context: Optional[Context] = None,
        scene_context: Optional[SceneContext] = None,
        behavior_engine: Optional[BehaviorEngine] = None,
        system_prompt_template: str = "",
        variation_config: Optional[Dict[str, Any]] = None,
    ) -> PersonaType:
        """创建一个人格类型"""
        pt = PersonaType(
            type_id=type_id,
            name=name,
            description=description,
            demographics=demographics or Demographics(),
            psychographics=psychographics or Psychographics(),
            behavioral_traits=behavioral_traits or BehavioralTraits(),
            context=context or Context(),
            scene_context=scene_context or SceneContext(),
            behavior_engine=behavior_engine or BehaviorEngine(),
            system_prompt_template=system_prompt_template,
            variation_config=variation_config or {},
        )
        self.storage.save("types", pt.type_id, pt.model_dump())
        return pt
    
    def register_type(self, pt: PersonaType) -> PersonaType:
        """注册一个已构建的 PersonaType（从文件加载等场景使用）"""
        self.storage.save("types", pt.type_id, pt.model_dump())
        return pt
    
    def get_type(self, type_id: str) -> Optional[PersonaType]:
        """获取人格类型"""
        data = self.storage.load("types", type_id)
        if not data:
            return None
        return PersonaType(**data)
    
    def list_types(self) -> List[PersonaType]:
        """列出所有人格类型"""
        return [PersonaType(**d) for d in self.storage.list_all("types")]
    
    def update_type(self, type_id: str, **updates) -> Optional[PersonaType]:
        """更新人格类型"""
        pt = self.get_type(type_id)
        if not pt:
            return None
        data = pt.model_dump()
        data.update(updates)
        updated = PersonaType(**data)
        self.storage.save("types", updated.type_id, updated.model_dump())
        return updated
    
    def delete_type(self, type_id: str) -> bool:
        """删除人格类型"""
        return self.storage.delete("types", type_id)
    
    # ═══════════════════════════════════════════════
    # PersonaInstance 实例化
    # ═══════════════════════════════════════════════
    
    def instantiate(
        self,
        type_id: str,
        name: Optional[str] = None,
        variation: Optional[Dict[str, Any]] = None,
        variation_seed: Optional[int] = None,
        scene_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[PersonaInstance]:
        """
        基于 PersonaType 创建一个差异化实例。
        
        Args:
            type_id: 类型 ID
            name: 实例名称（自动分配 if None）
            variation: 手动指定变异参数（优先级高于自动生成）
            variation_seed: 随机种子（用于可复现的随机差异）
            scene_overrides: 场景级别的覆盖参数
        
        Returns:
            PersonaInstance or None（如果 type 不存在）
        """
        pt = self.get_type(type_id)
        if not pt:
            return None
        
        # 生成变异参数
        auto_variation = VariationGenerator.generate(pt.variation_config, seed=variation_seed)
        if variation:
            auto_variation.update(variation)
        if scene_overrides:
            auto_variation.update(scene_overrides)
        
        # 应用变异到整个类型数据（支持嵌套路径如 demographics.age）
        type_data = pt.model_dump()
        for key, value in auto_variation.items():
            if "." in key:
                parts = key.split(".")
                target = type_data
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
            else:
                type_data[key] = value
        
        # 从更新后的数据中提取各子模型
        demo = Demographics(**type_data.get("demographics", {}))
        psycho = Psychographics(**type_data.get("psychographics", {}))
        behave = BehavioralTraits(**type_data.get("behavioral_traits", {}))
        ctx = Context(**type_data.get("context", {}))
        sc = SceneContext(**type_data.get("scene_context", {}))
        be = BehaviorEngine(**type_data.get("behavior_engine", {}))
        
        # 生成名称
        if name is None:
            role = getattr(demo, "role", "用户")
            existing = self.storage.list_ids("instances")
            idx = len(existing)
            name = VariationGenerator.generate_name(role, idx)
        
        # 组装实例
        inst = PersonaInstance(
            type_id=type_id,
            name=name,
            variation=auto_variation,
            demographics=demo,
            psychographics=psycho,
            behavioral_traits=behave,
            context=ctx,
            scene_context=sc,
            behavior_engine=be,
        )
        
        # 渲染 system prompt
        inst.system_prompt = PromptRenderer.render_system_prompt(inst)
        
        # 保存
        self.storage.save("instances", inst.instance_id, inst.model_dump())
        return inst
    
    def get_instance(self, instance_id: str) -> Optional[PersonaInstance]:
        """获取实例"""
        data = self.storage.load("instances", instance_id)
        if not data:
            return None
        return PersonaInstance(**data)
    
    def list_instances(self, type_id: Optional[str] = None) -> List[PersonaInstance]:
        """列出所有实例，可按 type 过滤"""
        all_data = self.storage.list_all("instances")
        if type_id:
            all_data = [d for d in all_data if d.get("type_id") == type_id]
        return [PersonaInstance(**d) for d in all_data]
    
    def update_instance(self, instance_id: str, **updates) -> Optional[PersonaInstance]:
        """更新实例（如更新 memory、history 等）"""
        inst = self.get_instance(instance_id)
        if not inst:
            return None
        data = inst.model_dump()
        data.update(updates)
        updated = PersonaInstance(**data)
        self.storage.save("instances", updated.instance_id, updated.model_dump())
        return updated
    
    def delete_instance(self, instance_id: str) -> bool:
        """删除实例"""
        return self.storage.delete("instances", instance_id)
    
    # ═══════════════════════════════════════════════
    # Scene 场景管理
    # ═══════════════════════════════════════════════
    
    def create_scene(
        self,
        name: str,
        description: str = "",
        scenario: str = "",
        participant_configs: Optional[List[SceneParticipant]] = None,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> Scene:
        """创建一个场景定义"""
        scene = Scene(
            name=name,
            description=description,
            scenario=scenario,
            participant_configs=participant_configs or [],
            shared_context=shared_context or {},
        )
        self.storage.save("scenes", scene.scene_id, scene.model_dump())
        return scene
    
    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """获取场景"""
        data = self.storage.load("scenes", scene_id)
        if not data:
            return None
        return Scene(**data)
    
    def list_scenes(self) -> List[Scene]:
        """列出所有场景"""
        return [Scene(**d) for d in self.storage.list_all("scenes")]
    
    def delete_scene(self, scene_id: str) -> bool:
        """删除场景"""
        return self.storage.delete("scenes", scene_id)
    
    def instantiate_scene(self, scene_id: str) -> Optional[Scene]:
        """
        实例化一个场景：根据 participant_configs 生成所有实例。
        
        返回更新后的 Scene（participant_instance_ids 已填充）。
        """
        scene = self.get_scene(scene_id)
        if not scene:
            return None
        
        instance_ids = []
        
        for cfg in scene.participant_configs:
            pt = self.get_type(cfg.type_id)
            if not pt:
                raise ValueError(f"Scene 引用了不存在的 PersonaType: {cfg.type_id}")
            
            for i in range(cfg.count):
                # 合并场景覆盖参数
                overrides = cfg.variation_overrides.copy()
                
                inst = self.instantiate(
                    type_id=cfg.type_id,
                    variation=overrides,
                    variation_seed=hash(f"{scene_id}_{cfg.type_id}_{i}") % (2**31),
                    scene_overrides={
                        "scene_context.scene_description": scene.scenario or scene.description,
                    },
                )
                
                if inst:
                    # 如果有场景特定的 prompt 补充，追加到 system prompt
                    if cfg.scene_specific_prompt_addition:
                        inst.system_prompt += f"\n\n## 场景特定补充\n{cfg.scene_specific_prompt_addition}\n"
                        self.storage.save("instances", inst.instance_id, inst.model_dump())
                    
                    instance_ids.append(inst.instance_id)
        
        scene.participant_instance_ids = instance_ids
        self.storage.save("scenes", scene.scene_id, scene.model_dump())
        return scene
    
    def get_scene_instances(self, scene_id: str) -> List[PersonaInstance]:
        """获取场景中所有已实例化的参与者"""
        scene = self.get_scene(scene_id)
        if not scene:
            return []
        return [
            inst for iid in scene.participant_instance_ids
            if (inst := self.get_instance(iid)) is not None
        ]
    
    # ═══════════════════════════════════════════════
    # 便捷：批量操作
    # ═══════════════════════════════════════════════
    
    def create_preset_types(self) -> List[PersonaType]:
        """
        创建一些预设的人格类型，方便快速上手。
        返回创建的类型列表。
        """
        presets = []
        
        # 1. 焦虑型买家
        presets.append(self.create_type(
            type_id="anxious_buyer",
            name="焦虑型买家",
            description="对产品决策非常谨慎，害怕选错，需要大量安全感",
            demographics=Demographics(
                age=34, role="中小团队负责人", company_size="10-50人",
                industry="互联网/科技", location="二线城市", years_experience=6,
            ),
            psychographics=Psychographics(
                goals=["降低团队运营风险", "找到可靠的工具"],
                frustrations=["之前被供应商坑过", "预算有限但需求不少"],
                decision_style="极度谨慎，需要同行背书",
                tech_stack=["Notion", "飞书", "腾讯文档"],
                budget_authority="有建议权，需老板审批",
            ),
            behavioral_traits=BehavioralTraits(
                communication="反复确认细节，喜欢追问 edge case",
                skepticism_level=0.75,
                price_sensitivity=0.8,
                risk_tolerance="极低",
            ),
            variation_config={
                "demographics.age": {"range": [28, 42]},
                "demographics.years_experience": {"range": [3, 10]},
                "behavioral_traits.skepticism_level": {"range": [0.6, 0.9]},
                "behavioral_traits.price_sensitivity": {"range": [0.6, 0.95]},
            },
        ))
        
        # 2. 理性分析师
        presets.append(self.create_type(
            type_id="rational_analyst",
            name="理性分析师",
            description="数据驱动，关注 ROI 和可量化指标",
            demographics=Demographics(
                age=32, role="数据分析师", company_size="500人以上",
                industry="金融/咨询", location="一线城市", years_experience=5,
            ),
            psychographics=Psychographics(
                goals=["提升数据洞察效率", "自动化重复报表工作"],
                frustrations=["数据散落在多个系统", "工具学习曲线陡峭"],
                decision_style="数据驱动，要求看到具体数字",
                tech_stack=["SQL", "Python", "Tableau", "Excel"],
                budget_authority="有部门预算决策权",
            ),
            behavioral_traits=BehavioralTraits(
                communication="逻辑清晰，直接要数据和案例",
                skepticism_level=0.5,
                price_sensitivity=0.4,
                risk_tolerance="中",
            ),
            variation_config={
                "demographics.age": {"range": [26, 40]},
                "demographics.years_experience": {"range": [2, 8]},
                "behavioral_traits.skepticism_level": {"range": [0.3, 0.7]},
                "psychographics.goals": {
                    "options": [
                        ["提升数据洞察效率", "自动化重复报表工作"],
                        ["统一数据源", "建立数据指标体系"],
                        ["减少手工操作", "提高数据准确性"],
                    ]
                },
            },
        ))
        
        # 3. 技术怀疑者
        presets.append(self.create_type(
            type_id="tech_skeptic",
            name="技术怀疑者",
            description="对新技术持怀疑态度，偏好成熟方案",
            demographics=Demographics(
                age=38, role="技术总监", company_size="200-1000人",
                industry="传统企业/制造业", location="一线城市", years_experience=12,
            ),
            psychographics=Psychographics(
                goals=["确保系统稳定性", "降低技术债务"],
                frustrations=["被‘革命性’产品忽悠过", "集成成本被低估"],
                decision_style="安全第一，宁可慢一点也要稳",
                tech_stack=["Java", "Oracle", "Linux", "Shell"],
                budget_authority="有较大技术预算决策权",
            ),
            behavioral_traits=BehavioralTraits(
                communication="技术性很强，会直接 challenge 架构",
                skepticism_level=0.85,
                price_sensitivity=0.3,
                risk_tolerance="极低",
            ),
            variation_config={
                "demographics.age": {"range": [33, 48]},
                "demographics.years_experience": {"range": [8, 18]},
                "behavioral_traits.skepticism_level": {"range": [0.7, 0.95]},
            },
        ))
        
        # 4. 冲动决策者
        presets.append(self.create_type(
            type_id="impulsive_decider",
            name="冲动决策者",
            description="凭直觉做决策，追求新鲜感和效率",
            demographics=Demographics(
                age=27, role="增长运营", company_size="50-200人",
                industry="电商/新媒体", location="一线城市", years_experience=3,
            ),
            psychographics=Psychographics(
                goals=["快速验证想法", "抢占市场先机"],
                frustrations=["流程太慢", "审批繁琐"],
                decision_style="直觉驱动，看重第一印象",
                tech_stack=["Figma", "剪映", "各种 SaaS"],
                budget_authority="有小额快速决策权",
            ),
            behavioral_traits=BehavioralTraits(
                communication="快节奏，不耐烦听长篇大论",
                skepticism_level=0.2,
                price_sensitivity=0.5,
                risk_tolerance="高",
            ),
            variation_config={
                "demographics.age": {"range": [23, 32]},
                "demographics.years_experience": {"range": [1, 6]},
                "behavioral_traits.skepticism_level": {"range": [0.1, 0.4]},
                "behavioral_traits.price_sensitivity": {"range": [0.3, 0.7]},
            },
        ))
        
        return presets
