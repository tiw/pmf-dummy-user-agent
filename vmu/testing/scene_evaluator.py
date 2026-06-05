"""
场景评估器

基于 Scene 定义的预期行为标准，对测试会话进行结构化评估。

评估维度：
1. 阶段覆盖率 — 销售 agent 是否按预期推进了各个阶段
2. 信息释放得分 — 虚拟人是否按场景要求释放了关键信息
3. 信任度轨迹 — 对话过程中信任度的变化趋势
4. 行为一致性 — 虚拟人行为是否符合场景设定的性格参数
5. 综合得分 — 加权汇总
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Scene, SceneParticipant
from .tester import TestSession


@dataclass
class StageCoverageReport:
    """阶段覆盖评估"""
    expected: List[str]
    actual: List[str]
    unique_actual: List[str]
    coverage_rate: float = 0.0
    missing_stages: List[str] = field(default_factory=list)
    extra_stages: List[str] = field(default_factory=list)


@dataclass
class InfoReleaseReport:
    """信息释放评估"""
    required: Dict[str, Any]
    actual: Dict[str, bool]
    release_rate: float = 0.0
    missing_items: List[str] = field(default_factory=list)


@dataclass
class TrustTrajectoryReport:
    """信任度轨迹评估"""
    initial: float = 0.3
    final: float = 0.3
    delta: float = 0.0
    trend: str = "flat"  # up / down / flat / volatile
    values: List[float] = field(default_factory=list)


@dataclass
class SceneEvaluationResult:
    """场景评估结果"""
    scene_id: str
    session_id: str
    persona_type: str

    stage_coverage: StageCoverageReport
    info_release: InfoReleaseReport
    trust_trajectory: TrustTrajectoryReport

    overall_score: float = 0.0
    passed: bool = False
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "session_id": self.session_id,
            "persona_type": self.persona_type,
            "overall_score": round(self.overall_score, 2),
            "passed": self.passed,
            "stage_coverage": {
                "expected": self.stage_coverage.expected,
                "actual": self.stage_coverage.actual,
                "coverage_rate": round(self.stage_coverage.coverage_rate, 2),
                "missing": self.stage_coverage.missing_stages,
            },
            "info_release": {
                "required": self.info_release.required,
                "actual": {k: v for k, v in self.info_release.actual.items()},
                "release_rate": round(self.info_release.release_rate, 2),
                "missing": self.info_release.missing_items,
            },
            "trust_trajectory": {
                "initial": self.trust_trajectory.initial,
                "final": self.trust_trajectory.final,
                "delta": round(self.trust_trajectory.delta, 2),
                "trend": self.trust_trajectory.trend,
            },
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class SceneEvaluator:
    """
    场景评估器：根据 Scene 定义的标准评估销售 agent 表现。

    使用示例：
        evaluator = SceneEvaluator()
        result = evaluator.evaluate_session(session, scene, participant_config)
        print(f"得分: {result.overall_score}/10")
        print(f"通过: {'是' if result.passed else '否'}")
    """

    # 默认权重配置
    DEFAULT_WEIGHTS = {
        "stage_coverage": 0.35,
        "info_release": 0.30,
        "trust_trajectory": 0.20,
        "behavior_consistency": 0.15,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)

    def evaluate_session(
        self,
        session: TestSession,
        scene: Scene,
        participant_config: Optional[SceneParticipant] = None,
    ) -> SceneEvaluationResult:
        """
        评估一个测试会话是否符合场景标准。
        """
        # ── 1. 阶段覆盖率评估 ──
        stage_report = self._eval_stage_coverage(session, scene, participant_config)

        # ── 2. 信息释放评估 ──
        info_report = self._eval_info_release(session, participant_config)

        # ── 3. 信任度轨迹评估 ──
        trust_report = self._eval_trust_trajectory(session)

        # ── 4. 综合得分 ──
        overall = self._compute_overall_score(stage_report, info_report, trust_report)

        # ── 5. 通过判定 ──
        passed = self._check_passed(overall, stage_report, info_report, trust_report, scene)

        # ── 6. 生成 issues 和 suggestions ──
        issues, suggestions = self._generate_feedback(stage_report, info_report, trust_report)

        # 提取 persona_type
        persona_type = session.persona_instance.type_id

        return SceneEvaluationResult(
            scene_id=scene.scene_id,
            session_id=session.session_id,
            persona_type=persona_type,
            stage_coverage=stage_report,
            info_release=info_report,
            trust_trajectory=trust_report,
            overall_score=overall,
            passed=passed,
            issues=issues,
            suggestions=suggestions,
            raw_metadata={
                "total_turns": len(session.turns),
                "scene_name": scene.name,
            },
        )

    # ───────────────────────────────────────────────
    # 评估子方法
    # ───────────────────────────────────────────────

    def _eval_stage_coverage(
        self,
        session: TestSession,
        scene: Scene,
        participant_config: Optional[SceneParticipant],
    ) -> StageCoverageReport:
        """评估阶段覆盖率。"""
        # 确定预期阶段
        if participant_config and participant_config.expected_stages:
            expected = participant_config.expected_stages
        elif scene.expected_stages:
            expected = scene.expected_stages
        else:
            from ..behavior_engine import SALES_STAGES
            expected = SALES_STAGES.copy()

        # 从 session turns 的 metadata 中提取实际经历的阶段
        actual = []
        for turn in session.turns:
            meta = turn.metadata or {}
            stage = meta.get("current_stage")
            if stage:
                actual.append(stage)

        unique_actual = list(dict.fromkeys(actual))  # 保持顺序去重

        if not expected:
            coverage_rate = 1.0
            missing = []
        else:
            expected_set = set(expected)
            actual_set = set(unique_actual)
            coverage_rate = len(actual_set & expected_set) / len(expected_set)
            missing = [s for s in expected if s not in actual_set]

        extra = [s for s in unique_actual if s not in expected]

        return StageCoverageReport(
            expected=expected,
            actual=actual,
            unique_actual=unique_actual,
            coverage_rate=coverage_rate,
            missing_stages=missing,
            extra_stages=extra,
        )

    def _eval_info_release(
        self,
        session: TestSession,
        participant_config: Optional[SceneParticipant],
    ) -> InfoReleaseReport:
        """评估信息释放。"""
        # 确定最低信息释放要求
        if participant_config and participant_config.min_info_release:
            required = participant_config.min_info_release
        else:
            required = {
                "pain_points": True,
                "budget_range": True,
            }

        # 从最后一轮的 metadata 中提取已释放的信息
        actual = {}
        if session.turns:
            last_meta = session.turns[-1].metadata or {}
            actual = last_meta.get("collected", {})

        if not required:
            release_rate = 1.0
            missing = []
        else:
            required_keys = [k for k, v in required.items() if v]
            if not required_keys:
                release_rate = 1.0
                missing = []
            else:
                released_count = sum(1 for k in required_keys if actual.get(k))
                release_rate = released_count / len(required_keys)
                missing = [k for k in required_keys if not actual.get(k)]

        return InfoReleaseReport(
            required=required,
            actual=actual,
            release_rate=release_rate,
            missing_items=missing,
        )

    def _eval_trust_trajectory(self, session: TestSession) -> TrustTrajectoryReport:
        """评估信任度轨迹。"""
        values = []
        for turn in session.turns:
            meta = turn.metadata or {}
            trust = meta.get("trust_level")
            if trust is not None:
                values.append(trust)

        if not values:
            return TrustTrajectoryReport(
                initial=session.persona_instance.memory.trust_level,
                final=session.persona_instance.memory.trust_level,
            )

        initial = values[0]
        final = values[-1]
        delta = final - initial

        # 判断趋势
        if len(values) >= 3:
            ups = sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])
            downs = sum(1 for i in range(1, len(values)) if values[i] < values[i - 1])
            if ups > downs and final > initial:
                trend = "up"
            elif downs > ups and final < initial:
                trend = "down"
            elif ups > 0 and downs > 0:
                trend = "volatile"
            else:
                trend = "flat"
        else:
            trend = "up" if delta > 0.05 else "down" if delta < -0.05 else "flat"

        return TrustTrajectoryReport(
            initial=initial,
            final=final,
            delta=delta,
            trend=trend,
            values=values,
        )

    def _compute_overall_score(
        self,
        stage_report: StageCoverageReport,
        info_report: InfoReleaseReport,
        trust_report: TrustTrajectoryReport,
    ) -> float:
        """计算综合得分（0-10）。"""
        w = self.weights

        # 阶段覆盖率得分
        stage_score = stage_report.coverage_rate * 10

        # 信息释放得分
        info_score = info_report.release_rate * 10

        # 信任度得分：最终信任度 + 趋势加分
        trust_base = trust_report.final * 10  # 0-10
        if trust_report.trend == "up":
            trust_bonus = 1.0
        elif trust_report.trend == "down":
            trust_bonus = -1.0
        else:
            trust_bonus = 0.0
        trust_score = max(0, min(10, trust_base + trust_bonus))

        # 行为一致性得分（简化：如果阶段覆盖率和信息释放都高，认为行为一致）
        consistency_score = (stage_report.coverage_rate + info_report.release_rate) / 2 * 10

        overall = (
            stage_score * w["stage_coverage"]
            + info_score * w["info_release"]
            + trust_score * w["trust_trajectory"]
            + consistency_score * w["behavior_consistency"]
        )
        return round(overall, 2)

    def _check_passed(
        self,
        overall: float,
        stage_report: StageCoverageReport,
        info_report: InfoReleaseReport,
        trust_report: TrustTrajectoryReport,
        scene: Scene,
    ) -> bool:
        """判定是否通过场景标准。"""
        criteria = scene.success_criteria or {}

        # 最低综合得分
        min_score = criteria.get("min_overall_score", 5.0)
        if overall < min_score:
            return False

        # 最低阶段覆盖率
        min_coverage = criteria.get("min_stage_coverage", 0.5)
        if stage_report.coverage_rate < min_coverage:
            return False

        # 最低信任度
        min_trust = criteria.get("min_trust_level", 0.0)
        if trust_report.final < min_trust:
            return False

        # 必须覆盖的阶段
        required_stages = criteria.get("required_stages", [])
        actual_set = set(stage_report.unique_actual)
        for stage in required_stages:
            if stage not in actual_set:
                return False

        return True

    def _generate_feedback(
        self,
        stage_report: StageCoverageReport,
        info_report: InfoReleaseReport,
        trust_report: TrustTrajectoryReport,
    ):
        """生成 issues 和 suggestions。"""
        issues = []
        suggestions = []

        # 阶段覆盖率问题
        if stage_report.coverage_rate < 0.5:
            issues.append(f"阶段覆盖率过低 ({stage_report.coverage_rate:.0%})，销售 agent 可能跳跃阶段")
            suggestions.append("建议销售 agent 按标准流程逐步推进，不要跳过需求挖掘直接展示方案")
        elif stage_report.missing_stages:
            issues.append(f"缺少阶段: {', '.join(stage_report.missing_stages)}")

        # 信息释放问题
        if info_report.missing_items:
            issues.append(f"未能获取关键信息: {', '.join(info_report.missing_items)}")
            suggestions.append("建议在需求挖掘阶段更深入地引导客户透露核心痛点和预算范围")

        # 信任度问题
        if trust_report.trend == "down":
            issues.append(f"信任度下降 ({trust_report.initial:.2f} → {trust_report.final:.2f})")
            suggestions.append("建议减少重复话术，及时回应客户的核心关切")
        elif trust_report.trend == "volatile":
            issues.append("信任度波动较大，销售 agent 的回复质量不稳定")

        # 额外阶段
        if stage_report.extra_stages:
            suggestions.append(f"销售 agent 进入了预期外的阶段: {', '.join(stage_report.extra_stages)}")

        return issues, suggestions
