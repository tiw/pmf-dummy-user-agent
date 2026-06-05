"""
评估结果可视化器

将 SceneEvaluator 的评估结果生成图表，支持：
- 综合评分对比柱状图
- 多维度雷达图
- 信任度轨迹折线图
- 阶段覆盖热力图
- 信息释放进度图
- 完整报告（多子图组合）

依赖: matplotlib, seaborn, numpy
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .scene_evaluator import SceneEvaluationResult
from .tester import TestSession


# ───────────────────────────────────────────────
# 字体与样式配置
# ───────────────────────────────────────────────

def _setup_chinese_font():
    """配置中文字体支持（macOS 优先 PingFang/Hiragino）"""
    candidates = [
        "PingFang SC", "Hiragino Sans GB", "Heiti SC",
        "STHeiti", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei",
    ]
    available = []
    for f in matplotlib.font_manager.fontManager.ttflist:
        if f.name in candidates and f.name not in available:
            available.append(f.name)
    if available:
        matplotlib.rcParams["font.sans-serif"] = available + ["DejaVu Sans"]
        matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


# Seaborn 样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 在 seaborn 之后设置字体（防止被覆盖）
_setup_chinese_font()


# ───────────────────────────────────────────────
# 可视化器
# ───────────────────────────────────────────────

class EvaluationVisualizer:
    """
    评估结果可视化器。

    使用示例:
        viz = EvaluationVisualizer(output_dir="reports")
        viz.generate_full_report(results, sessions, scene)
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────
    # 单图方法
    # ──────────────────────────────────────────

    def plot_overall_comparison(
        self,
        results: List[SceneEvaluationResult],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """
        综合评分对比柱状图。
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        personas = [r.persona_type for r in results]
        scores = [r.overall_score for r in results]
        colors = ["#2ecc71" if r.passed else "#e74c3c" for r in results]

        bars = ax.barh(personas, scores, color=colors, edgecolor="white", height=0.6)

        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_xlim(0, 10)
        ax.axvline(x=6.0, color="#95a5a6", linestyle="--", linewidth=1, label="通过线 (6.0)")
        ax.set_xlabel("综合得分", fontsize=12)
        ax.set_title("销售 Agent 跨 Persona 评估对比", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")

        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "01_overall_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_radar(
        self,
        results: List[SceneEvaluationResult],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """
        多维度雷达图（所有 persona 叠加）。
        """
        categories = ["阶段覆盖", "信息释放", "信任度", "行为一致性"]
        N = len(categories)

        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        for idx, r in enumerate(results):
            values = [
                r.stage_coverage.coverage_rate * 10,
                r.info_release.release_rate * 10,
                r.trust_trajectory.final * 10,
                r.stage_coverage.coverage_rate * r.info_release.release_rate * 10,
            ]
            values += values[:1]  # 闭合

            ax.plot(angles, values, "o-", linewidth=2, label=r.persona_type, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_title("多维度能力雷达图", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "02_radar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_trust_trajectory(
        self,
        sessions: List[TestSession],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """
        信任度轨迹折线图。
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(sessions)))

        for idx, session in enumerate(sessions):
            turns = session.turns
            rounds = [t.round_num for t in turns]
            trusts = [t.metadata.get("trust_level", 0.3) for t in turns]

            if not rounds:
                continue

            ax.plot(
                rounds,
                trusts,
                marker="o",
                linewidth=2.5,
                markersize=8,
                label=session.persona_instance.type_id,
                color=colors[idx],
            )

        ax.set_xlabel("对话轮次", fontsize=12)
        ax.set_ylabel("信任度", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_title("信任度变化轨迹", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "03_trust_trajectory.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_stage_coverage_heatmap(
        self,
        results: List[SceneEvaluationResult],
        scene_stages: List[str],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """
        阶段覆盖热力图。
        """
        personas = [r.persona_type for r in results]
        stages = scene_stages

        # 构建矩阵: 1=覆盖, 0=未覆盖
        matrix = np.zeros((len(personas), len(stages)))
        for i, r in enumerate(results):
            actual_set = set(r.stage_coverage.unique_actual)
            for j, stage in enumerate(stages):
                if stage in actual_set:
                    matrix[i, j] = 1

        fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.5), max(5, len(personas) * 0.8)))

        cmap = sns.color_palette(["#ecf0f1", "#2ecc71"], as_cmap=True)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap=cmap,
            xticklabels=stages,
            yticklabels=personas,
            cbar=False,
            linewidths=1,
            linecolor="white",
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
        )

        ax.set_title("阶段覆盖热力图 (绿色=已覆盖)", fontsize=14, fontweight="bold")
        ax.set_xlabel("销售阶段", fontsize=12)
        ax.set_ylabel("Persona", fontsize=12)

        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "04_stage_coverage_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def plot_info_release(
        self,
        results: List[SceneEvaluationResult],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """
        信息释放进度图（水平条形图）。
        """
        fig, ax = plt.subplots(figsize=(10, max(5, len(results) * 1.2)))

        all_keys = set()
        for r in results:
            all_keys.update(r.info_release.required.keys())
        all_keys = sorted(all_keys)

        y_positions = []
        bar_heights = []
        colors = []
        labels = []
        tick_labels = []

        y = 0
        for r in results:
            for key in all_keys:
                y_positions.append(y)
                required = r.info_release.required.get(key, False)
                actual = r.info_release.actual.get(key, False)

                if required and actual:
                    bar_heights.append(1)
                    colors.append("#2ecc71")
                    labels.append(f"{r.persona_type}: {key} ✅")
                elif required and not actual:
                    bar_heights.append(1)
                    colors.append("#e74c3c")
                    labels.append(f"{r.persona_type}: {key} ❌")
                else:
                    bar_heights.append(1)
                    colors.append("#bdc3c7")
                    labels.append(f"{r.persona_type}: {key} (未要求)")

                tick_labels.append(key)
                y += 1
            y += 0.5  # persona 之间留空隙

        ax.barh(y_positions, bar_heights, color=colors, edgecolor="white", height=0.8)

        # 设置 y 轴标签
        ax.set_yticks([i for i in range(len(y_positions))])
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_xlim(0, 1.2)
        ax.set_xticks([])
        ax.set_title("信息释放进度 (绿色=已释放 / 红色=缺失)", fontsize=14, fontweight="bold")

        # 添加 persona 分组标签
        y_cursor = 0
        for r in results:
            mid = y_cursor + len(all_keys) / 2 - 0.5
            ax.text(
                1.05,
                mid,
                r.persona_type,
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#2c3e50",
            )
            y_cursor += len(all_keys) + 0.5

        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, "05_info_release.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    # ──────────────────────────────────────────
    # 完整报告
    # ──────────────────────────────────────────

    def generate_full_report(
        self,
        results: List[SceneEvaluationResult],
        sessions: List[TestSession],
        scene_stages: List[str],
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        生成完整可视化报告，返回所有图表路径。
        """
        out = output_dir or self.output_dir
        os.makedirs(out, exist_ok=True)

        paths = {}
        paths["overall"] = self.plot_overall_comparison(results)
        paths["radar"] = self.plot_radar(results)
        paths["trust"] = self.plot_trust_trajectory(sessions)
        paths["heatmap"] = self.plot_stage_coverage_heatmap(results, scene_stages)
        paths["info"] = self.plot_info_release(results)
        paths["combined"] = self._plot_combined_dashboard(results, sessions, scene_stages, out)

        return paths

    def _plot_combined_dashboard(
        self,
        results: List[SceneEvaluationResult],
        sessions: List[TestSession],
        scene_stages: List[str],
        output_dir: str,
    ) -> str:
        """
        组合仪表盘：2×3 子图布局。
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # 1. 综合得分 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        personas = [r.persona_type for r in results]
        scores = [r.overall_score for r in results]
        colors = ["#2ecc71" if r.passed else "#e74c3c" for r in results]
        ax1.barh(personas, scores, color=colors)
        ax1.set_xlim(0, 10)
        ax1.axvline(x=6.0, color="#95a5a6", linestyle="--")
        ax1.set_title("综合得分", fontweight="bold")

        # 2. 信任度轨迹 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        colors2 = plt.cm.tab10(np.linspace(0, 1, len(sessions)))
        for idx, session in enumerate(sessions):
            turns = session.turns
            if not turns:
                continue
            rounds = [t.round_num for t in turns]
            trusts = [t.metadata.get("trust_level", 0.3) for t in turns]
            ax2.plot(rounds, trusts, marker="o", label=session.persona_instance.type_id, color=colors2[idx])
        ax2.set_ylim(0, 1)
        ax2.set_title("信任度轨迹", fontweight="bold")
        ax2.legend(fontsize=8)

        # 3. 雷达图 (右上)
        ax3 = fig.add_subplot(gs[0, 2], polar=True)
        categories = ["阶段覆盖", "信息释放", "信任度", "一致性"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        colors3 = plt.cm.tab10(np.linspace(0, 1, len(results)))
        for idx, r in enumerate(results):
            vals = [
                r.stage_coverage.coverage_rate * 10,
                r.info_release.release_rate * 10,
                r.trust_trajectory.final * 10,
                r.stage_coverage.coverage_rate * r.info_release.release_rate * 10,
            ]
            vals += vals[:1]
            ax3.plot(angles, vals, "o-", label=r.persona_type, color=colors3[idx], linewidth=1.5)
            ax3.fill(angles, vals, alpha=0.08, color=colors3[idx])
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories, fontsize=9)
        ax3.set_ylim(0, 10)
        ax3.set_title("能力雷达", fontweight="bold", pad=15)

        # 4. 阶段覆盖热力图 (左下，跨2列)
        ax4 = fig.add_subplot(gs[1, :2])
        matrix = np.zeros((len(results), len(scene_stages)))
        for i, r in enumerate(results):
            actual_set = set(r.stage_coverage.unique_actual)
            for j, stage in enumerate(scene_stages):
                matrix[i, j] = 1 if stage in actual_set else 0
        cmap = sns.color_palette(["#ecf0f1", "#2ecc71"], as_cmap=True)
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap=cmap, xticklabels=scene_stages,
                    yticklabels=personas, cbar=False, linewidths=1, linecolor="white", ax=ax4,
                    annot_kws={"size": 11, "weight": "bold"})
        ax4.set_title("阶段覆盖热力图", fontweight="bold")

        # 5. 信息释放率 (右下)
        ax5 = fig.add_subplot(gs[1, 2])
        release_rates = [r.info_release.release_rate * 100 for r in results]
        ax5.barh(personas, release_rates, color=["#3498db"] * len(personas))
        ax5.set_xlim(0, 100)
        for i, v in enumerate(release_rates):
            ax5.text(v + 2, i, f"{v:.0f}%", va="center", fontsize=10)
        ax5.set_title("信息释放率", fontweight="bold")

        fig.suptitle("销售 Agent 评估仪表盘", fontsize=16, fontweight="bold", y=0.98)

        path = os.path.join(output_dir, "00_dashboard.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
