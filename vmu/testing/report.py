"""
测试报告生成
"""

from typing import Any, Dict, List, Optional

from .tester import TestSession
from .evaluator import ConversationEvaluator


class TestReport:
    """
    测试报告生成器。

    将 TestSession + 评估结果 → 结构化报告。
    """

    def __init__(self, session: TestSession, evaluation: Optional[Dict[str, Any]] = None):
        self.session = session
        self.evaluation = evaluation

    @classmethod
    def from_session(
        cls,
        session: TestSession,
        evaluator: Optional[ConversationEvaluator] = None,
    ) -> "TestReport":
        """从会话自动生成报告（含评估）"""
        if evaluator is None:
            evaluator = ConversationEvaluator()
        evaluation = evaluator.evaluate(
            persona=session.persona_instance,
            turns=session.turns,
            agent_name=session.agent_name,
        )
        return cls(session=session, evaluation=evaluation)

    def to_dict(self) -> Dict[str, Any]:
        """生成报告字典"""
        report = {
            "session": self.session.to_dict(),
            "evaluation": self.evaluation,
        }
        # 添加一些统计
        report["stats"] = self._compute_stats()
        return report

    def to_markdown(self) -> str:
        """生成 Markdown 格式报告"""
        s = self.session
        e = self.evaluation or {}
        dims = e.get("dimensions", {})

        lines = [
            f"# 测试报告：{s.agent_name} vs {s.persona_instance.name}",
            "",
            f"- **会话 ID**: `{s.session_id}`",
            f"- **虚拟人**: {s.persona_instance.name} ({s.persona_instance.type_id})",
            f"- **总轮数**: {len(s.turns)}",
            f"- **状态**: {s.status}",
            "",
            "## 对话记录",
            "",
        ]

        for t in s.turns:
            lines.append(f"### 第 {t.round_num} 轮")
            lines.append("")
            lines.append(f"**Agent**: {t.agent_message}")
            lines.append("")
            lines.append(f"**User ({s.persona_instance.name})**: {t.user_response}")
            lines.append("")
            if t.metadata:
                lines.append(f"_信任度: {t.metadata.get('trust_level', 'N/A')}, "
                           f"情绪: {t.metadata.get('emotional_state', 'N/A')}_")
                lines.append("")

        # 评估结果
        if e:
            lines.extend([
                "## 评估结果",
                "",
                f"**综合得分**: {e.get('overall_score', 'N/A')} / 10",
                "",
            ])

            if dims:
                lines.append("| 维度 | 得分 | 评价 |")
                lines.append("|------|------|------|")
                for dim_name, dim_data in dims.items():
                    name_map = {
                        "persona_consistency": "角色一致性",
                        "conversation_flow": "对话流畅度",
                        "agent_effectiveness": "Agent 有效性",
                        "need_fulfillment": "需求满足度",
                    }
                    display_name = name_map.get(dim_name, dim_name)
                    score = dim_data.get("score", "N/A") if isinstance(dim_data, dict) else "N/A"
                    comment = dim_data.get("comment", "") if isinstance(dim_data, dict) else ""
                    lines.append(f"| {display_name} | {score} | {comment} |")
                lines.append("")

            if e.get("highlights"):
                lines.append("### 亮点")
                lines.append("")
                for h in e["highlights"]:
                    lines.append(f"- {h}")
                lines.append("")

            if e.get("issues"):
                lines.append("### 问题")
                lines.append("")
                for i in e["issues"]:
                    lines.append(f"- {i}")
                lines.append("")

            if e.get("suggestions"):
                lines.append("### 建议")
                lines.append("")
                for sgt in e["suggestions"]:
                    lines.append(f"- {sgt}")
                lines.append("")

        # 统计
        stats = self._compute_stats()
        lines.extend([
            "## 统计",
            "",
            f"- 平均 agent 消息长度: {stats['avg_agent_message_length']:.0f} 字",
            f"- 平均 user 回复长度: {stats['avg_user_response_length']:.0f} 字",
            f"- 信任度变化: {stats['trust_start']:.2f} → {stats['trust_end']:.2f} "
            f"({stats['trust_delta']:+.2f})",
            f"- 情绪变化: {stats['emotions']}",
            "",
        ])

        return "\n".join(lines)

    def to_console(self) -> str:
        """生成适合终端显示的简洁报告"""
        s = self.session
        e = self.evaluation or {}
        score = e.get("overall_score", "N/A")

        lines = [
            "=" * 60,
            f"📊 测试报告: {s.agent_name} vs {s.persona_instance.name}",
            "=" * 60,
            f"   会话: {s.session_id}",
            f"   轮数: {len(s.turns)} | 状态: {s.status} | 评分: {score}/10",
            "",
        ]

        for t in s.turns:
            lines.append(f"  轮{t.round_num}: Agent → {t.agent_message[:50]}...")
            lines.append(f"         User  → {t.user_response[:50]}...")
            lines.append("")

        if e.get("issues"):
            lines.append("⚠️  发现的问题:")
            for i in e["issues"][:3]:
                lines.append(f"   • {i}")

        if e.get("suggestions"):
            lines.append("💡 建议:")
            for sgt in e["suggestions"][:3]:
                lines.append(f"   • {sgt}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _compute_stats(self) -> Dict[str, Any]:
        s = self.session
        turns = s.turns

        if not turns:
            return {
                "avg_agent_message_length": 0,
                "avg_user_response_length": 0,
                "trust_start": 0,
                "trust_end": 0,
                "trust_delta": 0,
                "emotions": [],
            }

        avg_agent = sum(len(t.agent_message) for t in turns) / len(turns)
        avg_user = sum(len(t.user_response) for t in turns) / len(turns)

        trust_values = [t.metadata.get("trust_level") for t in turns if "trust_level" in t.metadata]
        trust_start = trust_values[0] if trust_values else 0
        trust_end = trust_values[-1] if trust_values else 0

        emotions = list(set(
            t.metadata.get("emotional_state")
            for t in turns
            if "emotional_state" in t.metadata
        ))

        return {
            "avg_agent_message_length": avg_agent,
            "avg_user_response_length": avg_user,
            "trust_start": trust_start,
            "trust_end": trust_end,
            "trust_delta": trust_end - trust_start,
            "emotions": emotions,
        }
