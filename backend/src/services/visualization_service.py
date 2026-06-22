"""
可视化服务模块
实现训练过程中的适应度曲线生成和实时显示
"""
import os
import io
import base64
import uuid
import glob
from typing import List, Optional, Dict, Any, Tuple, Literal
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def _compute_stats(values: List[float]) -> Dict[str, Any]:
    """计算统计数据：mean, std, min, max, median, passed(>=200)"""
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "count": 0,
            "passed": False
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "count": int(len(arr)),
        "passed": bool(float(np.mean(arr)) >= 200.0)
    }


def _fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """将matplotlib Figure转换为Base64字符串"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def _fig_save_to_file(fig: plt.Figure, output_dir: str, filename: str, dpi: int = 150) -> str:
    """保存Figure到文件，返回绝对路径"""
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.abspath(os.path.join(output_dir, filename))
    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Chart saved to {full_path}")
    return full_path


class VisualizationService:
    """
    可视化服务
    生成训练过程中的适应度曲线、仪表板、对比图等
    """

    def __init__(self, output_dir: str = "plots", base_url: str = "/plots"):
        """
        Args:
            output_dir: 图表输出目录
            base_url: 静态文件服务的基础URL路径
        """
        self.output_dir = os.path.abspath(output_dir)
        self.base_url = base_url.rstrip("/")
        os.makedirs(self.output_dir, exist_ok=True)

        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        logger.info(f"VisualizationService initialized, output_dir={self.output_dir}, base_url={base_url}")

    def _make_file_url(self, filename: str) -> str:
        """根据文件名构造访问URL"""
        return f"{self.base_url}/{filename}"

    def _generate_filename(self, chart_type: str, task_id: Optional[str] = None, suffix: str = "") -> str:
        """生成唯一文件名"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = uuid.uuid4().hex[:8]
        tid = task_id[:8] if task_id else "custom"
        extra = f"_{suffix}" if suffix else ""
        return f"{chart_type}_{tid}_{ts}_{rand}{extra}.png"

    def generate(
        self,
        chart_type: Literal["fitness_curve", "dashboard", "progress", "comparison"],
        data: Dict[str, Any],
        task_id: Optional[str] = None,
        window_size: int = 10,
        save_to_plots: bool = False,
        fmt: Literal["base64", "file_url", "both"] = "base64",
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show_avg: bool = True
    ) -> Dict[str, Any]:
        """
        统一的图表生成入口

        Args:
            chart_type: 图表类型
            data: 图表数据字典（根据chart_type不同有不同键）
            task_id: 关联任务ID（可选）
            window_size: 移动平均窗口大小
            save_to_plots: 是否保存到plots/目录
            fmt: 输出格式
            title: 自定义标题
            xlabel: 自定义X轴标签
            ylabel: 自定义Y轴标签
            show_avg: 是否显示移动平均线（fitness_curve用）

        Returns:
            包含base64/file_url/file_path/stats等信息的字典
        """
        fig: Optional[plt.Figure] = None
        stats: Dict[str, Any] = {}
        width_px: int = 0
        height_px: int = 0

        if chart_type == "fitness_curve":
            fitness_history = data.get("fitness_history", [])
            if not fitness_history:
                raise ValueError("fitness_history is required for fitness_curve")
            stats = _compute_stats(fitness_history)
            stats["window_size"] = window_size
            stats["data_points"] = len(fitness_history)
            fig, ax = plt.subplots(figsize=(10, 6))
            width_px, height_px = 1500, 900
            self._draw_fitness_curve(
                ax, fitness_history,
                title=title or f"Fitness Curve - {task_id[:8] if task_id else 'Custom'}",
                xlabel=xlabel or "Generation",
                ylabel=ylabel or "Fitness",
                window_size=window_size,
                show_avg=show_avg
            )

        elif chart_type == "dashboard":
            episode_rewards = data.get("episode_rewards", [])
            if not episode_rewards:
                raise ValueError("episode_rewards is required for dashboard")
            policy_losses = data.get("policy_losses", [])
            value_losses = data.get("value_losses", [])
            temperatures = data.get("temperatures", [])
            stats = _compute_stats(episode_rewards)
            stats["has_policy_losses"] = bool(policy_losses)
            stats["has_value_losses"] = bool(value_losses)
            stats["has_temperatures"] = bool(temperatures)
            fig, _ = plt.subplots(2, 2, figsize=(14, 10))
            width_px, height_px = 2100, 1500
            self._draw_training_dashboard(
                fig, episode_rewards, policy_losses, value_losses, temperatures,
                title=title or f"Training Dashboard - {task_id[:8] if task_id else 'Custom'}"
            )

        elif chart_type == "progress":
            fitness_history = data.get("fitness_history", [])
            if not fitness_history:
                raise ValueError("fitness_history is required for progress")
            avg_fitness_history = data.get("avg_fitness_history")
            stats = _compute_stats(fitness_history)
            stats["data_points"] = len(fitness_history)
            fig, ax = plt.subplots(figsize=(10, 6))
            width_px, height_px = 1500, 900
            self._draw_genetic_progress(
                ax, fitness_history, avg_fitness_history,
                title=title or f"Genetic Progress - {task_id[:8] if task_id else 'Custom'}"
            )

        elif chart_type == "comparison":
            diff_rewards = data.get("diff_rewards", [])
            non_diff_rewards = data.get("non_diff_rewards", [])
            if not diff_rewards or not non_diff_rewards:
                raise ValueError("diff_rewards and non_diff_rewards are required for comparison")
            diff_stats = _compute_stats(diff_rewards)
            non_diff_stats = _compute_stats(non_diff_rewards)
            stats = {
                "differentiable": diff_stats,
                "non_differentiable": non_diff_stats,
                "performance_gap": diff_stats["mean"] - non_diff_stats["mean"]
            }
            fig, _ = plt.subplots(1, 2, figsize=(12, 5))
            width_px, height_px = 1800, 750
            self._draw_comparison_chart(
                fig, diff_rewards, non_diff_rewards,
                title=title or "Network Comparison"
            )
        else:
            raise ValueError(f"Unknown chart_type: {chart_type}")

        result: Dict[str, Any] = {
            "stats": stats,
            "width": width_px,
            "height": height_px,
        }

        filename = self._generate_filename(chart_type, task_id)

        if save_to_plots or fmt in ("file_url", "both"):
            if fig is not None:
                fpath = _fig_save_to_file(fig, self.output_dir, filename)
                result["file_path"] = fpath
                result["file_url"] = self._make_file_url(filename)

        if fmt in ("base64", "both"):
            if fig is not None:
                result["image_base64"] = _fig_to_base64(fig)

        if fig is not None:
            plt.close(fig)

        return result

    def generate_comparison(
        self,
        diff_rewards: List[float],
        non_diff_rewards: List[float],
        save_to_plots: bool = False,
        fmt: Literal["base64", "file_url", "both"] = "base64",
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成对比可视化（箱线图 + 直方图 + 组合图）

        Args:
            diff_rewards: 可微网络奖励列表
            non_diff_rewards: 不可微网络奖励列表
            save_to_plots: 是否保存到plots/目录
            fmt: 输出格式
            title: 自定义标题

        Returns:
            包含各图表的base64/url/path及统计信息
        """
        diff_stats = _compute_stats(diff_rewards)
        non_diff_stats = _compute_stats(non_diff_rewards)
        performance_gap = diff_stats["mean"] - non_diff_stats["mean"]

        result: Dict[str, Any] = {
            "differentiable_stats": diff_stats,
            "non_differentiable_stats": non_diff_stats,
            "performance_gap": performance_gap,
        }

        # 1. 独立箱线图
        fig_box, ax_box = plt.subplots(figsize=(8, 6))
        self._draw_boxplot(ax_box, diff_rewards, non_diff_rewards, title=title or "Reward Boxplot")
        fn_box = self._generate_filename("comparison_boxplot", None, "box")
        self._attach_output(fig_box, result, "boxplot", fn_box, save_to_plots, fmt)
        plt.close(fig_box)

        # 2. 独立直方图
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        self._draw_histogram(ax_hist, diff_rewards, non_diff_rewards, title=title or "Reward Histogram")
        fn_hist = self._generate_filename("comparison_histogram", None, "hist")
        self._attach_output(fig_hist, result, "histogram", fn_hist, save_to_plots, fmt)
        plt.close(fig_hist)

        # 3. 组合图
        fig_comb, axes_comb = plt.subplots(1, 2, figsize=(12, 5))
        self._draw_boxplot(axes_comb[0], diff_rewards, non_diff_rewards, title="Boxplot")
        self._draw_histogram(axes_comb[1], diff_rewards, non_diff_rewards, title="Histogram")
        fig_comb.suptitle(title or "Network Comparison", fontsize=14)
        fig_comb.tight_layout()
        fn_comb = self._generate_filename("comparison_combined", None, "combined")
        self._attach_output(fig_comb, result, "combined", fn_comb, save_to_plots, fmt)
        plt.close(fig_comb)

        return result

    def _attach_output(
        self,
        fig: plt.Figure,
        result: Dict[str, Any],
        key_prefix: str,
        filename: str,
        save_to_plots: bool,
        fmt: str
    ) -> None:
        """将图像输出附加到结果字典（base64、file_url、file_path）"""
        need_save = save_to_plots or fmt in ("file_url", "both")
        if need_save:
            fpath = _fig_save_to_file(fig, self.output_dir, filename)
            result[f"{key_prefix}_path"] = fpath
            result[f"{key_prefix}_url"] = self._make_file_url(filename)
        if fmt in ("base64", "both"):
            result[f"{key_prefix}_base64"] = _fig_to_base64(fig)

    # ==================== 绘图核心方法 ====================

    def _draw_fitness_curve(
        self,
        ax: plt.Axes,
        fitness_history: List[float],
        title: str,
        xlabel: str,
        ylabel: str,
        window_size: int = 10,
        show_avg: bool = True
    ) -> None:
        """在给定Axes上绘制适应度曲线"""
        generations = list(range(1, len(fitness_history) + 1))
        ax.plot(generations, fitness_history, 'b-', alpha=0.6, label='Best Fitness', linewidth=1)

        if show_avg and len(fitness_history) >= window_size:
            moving_avg = self._moving_average(fitness_history, window_size)
            avg_x = list(range(window_size, len(fitness_history) + 1))
            ax.plot(avg_x, moving_avg, 'r-', label=f'Moving Avg ({window_size})', linewidth=2)

        max_fitness = max(fitness_history)
        max_gen = fitness_history.index(max_fitness) + 1
        ax.scatter([max_gen], [max_fitness], color='green', s=100, zorder=5, label=f'Max: {max_fitness:.2f}')
        ax.annotate(f'{max_fitness:.2f}', (max_gen, max_fitness),
                    textcoords="offset points", xytext=(0, 10), ha='center')

        ax.axhline(y=200, color='orange', linestyle='--', linewidth=2, label='Target (200)')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        y_min = min(min(fitness_history), 0) - 50
        y_max = max(max(fitness_history), 250) + 50
        ax.set_ylim(y_min, y_max)

    def _draw_training_dashboard(
        self,
        fig: plt.Figure,
        episode_rewards: List[float],
        policy_losses: List[float],
        value_losses: List[float],
        temperatures: List[float],
        title: str
    ) -> None:
        """绘制训练仪表板（2x2子图）"""
        axes = fig.axes
        ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

        if episode_rewards:
            episodes = list(range(1, len(episode_rewards) + 1))
            ax1.plot(episodes, episode_rewards, 'b-', alpha=0.5, linewidth=0.5)
            if len(episode_rewards) >= 100:
                avg = self._moving_average(episode_rewards, 100)
                ax1.plot(range(100, len(episode_rewards) + 1), avg, 'r-', linewidth=2, label='Avg (100)')
            ax1.axhline(y=200, color='orange', linestyle='--', label='Target (200)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.set_title('Episode Rewards (no data)')

        if policy_losses:
            ax2.plot(policy_losses, 'g-', linewidth=1)
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.set_title('Policy Loss')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_title('Policy Loss (no data)')

        if value_losses:
            ax3.plot(value_losses, 'm-', linewidth=1)
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.set_title('Value Loss')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.set_title('Value Loss (no data)')

        if temperatures:
            ax4.plot(temperatures, 'c-', linewidth=2)
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Temperature')
            ax4.set_title('Temperature Annealing')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.set_title('Temperature Annealing (no data)')

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()

    def _draw_genetic_progress(
        self,
        ax: plt.Axes,
        fitness_history: List[float],
        avg_fitness_history: Optional[List[float]] = None,
        title: str = "Genetic Algorithm Progress"
    ) -> None:
        """绘制遗传算法进度图"""
        generations = list(range(1, len(fitness_history) + 1))
        ax.plot(generations, fitness_history, 'b-', linewidth=2, label='Best Fitness')

        if avg_fitness_history and len(avg_fitness_history) == len(fitness_history):
            ax.plot(generations, avg_fitness_history, 'g--', linewidth=1, alpha=0.7, label='Avg Fitness')

        ax.axhline(y=200, color='orange', linestyle='--', linewidth=2, label='Target (200)')
        ax.fill_between(generations, fitness_history, alpha=0.2)

        if fitness_history:
            max_fitness = max(fitness_history)
            max_gen = fitness_history.index(max_fitness) + 1
            ax.scatter([max_gen], [max_fitness], color='red', s=100, zorder=5)
            ax.annotate(f'Best: {max_fitness:.2f}', (max_gen, max_fitness),
                        textcoords="offset points", xytext=(10, 10), ha='left',
                        fontsize=12, fontweight='bold')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def _draw_boxplot(
        self,
        ax: plt.Axes,
        diff_rewards: List[float],
        non_diff_rewards: List[float],
        title: str = "Reward Boxplot"
    ) -> None:
        """在给定Axes上绘制箱线图"""
        data = [diff_rewards, non_diff_rewards]
        ax.boxplot(data, tick_labels=['Differentiable', 'Non-Differentiable'])
        ax.axhline(y=200, color='orange', linestyle='--', label='Target (200)')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _draw_histogram(
        self,
        ax: plt.Axes,
        diff_rewards: List[float],
        non_diff_rewards: List[float],
        title: str = "Reward Histogram"
    ) -> None:
        """在给定Axes上绘制直方图"""
        ax.hist(diff_rewards, bins=20, alpha=0.5, label='Differentiable', color='blue')
        ax.hist(non_diff_rewards, bins=20, alpha=0.5, label='Non-Differentiable', color='green')
        ax.axvline(x=200, color='orange', linestyle='--', label='Target (200)')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _draw_comparison_chart(
        self,
        fig: plt.Figure,
        diff_rewards: List[float],
        non_diff_rewards: List[float],
        title: str = "Network Comparison"
    ) -> None:
        """绘制组合对比图（箱线图+直方图）"""
        axes = fig.axes
        self._draw_boxplot(axes[0], diff_rewards, non_diff_rewards, title="Reward Distribution")
        self._draw_histogram(axes[1], diff_rewards, non_diff_rewards, title="Reward Histogram")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()

    # ==================== 兼容旧接口 ====================

    def generate_fitness_curve(
        self,
        fitness_history: List[float],
        title: str = "Fitness Curve",
        xlabel: str = "Generation",
        ylabel: str = "Fitness",
        save_path: Optional[str] = None,
        show_avg: bool = True,
        window_size: int = 10
    ) -> str:
        """
        旧版兼容接口：生成适应度曲线图并返回Base64

        Args:
            fitness_history: 适应度历史记录
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            save_path: 保存路径（相对于output_dir，可选）
            show_avg: 是否显示移动平均线
            window_size: 移动平均窗口大小

        Returns:
            Base64编码的图片字符串
        """
        if not fitness_history:
            logger.warning("Empty fitness history, cannot generate curve")
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_fitness_curve(ax, fitness_history, title, xlabel, ylabel, window_size, show_avg)

        if save_path:
            _fig_save_to_file(fig, self.output_dir, save_path)

        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64

    def generate_training_dashboard(
        self,
        episode_rewards: List[float],
        policy_losses: List[float],
        value_losses: List[float],
        temperatures: List[float],
        task_id: str
    ) -> str:
        """旧版兼容接口：生成训练仪表板并返回Base64"""
        fig, _ = plt.subplots(2, 2, figsize=(14, 10))
        self._draw_training_dashboard(fig, episode_rewards, policy_losses, value_losses, temperatures,
                                      title=f'Training Dashboard - {task_id}')
        save_path = os.path.join(self.output_dir, f"training_{task_id}.png")
        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception:
            pass
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64

    def generate_genetic_progress(
        self,
        fitness_history: List[float],
        avg_fitness_history: Optional[List[float]] = None,
        task_id: str = ""
    ) -> str:
        """旧版兼容接口：生成遗传算法进度图并返回Base64"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_genetic_progress(ax, fitness_history, avg_fitness_history,
                                    title=f'Genetic Algorithm Progress - {task_id}')
        save_path = os.path.join(self.output_dir, f"genetic_{task_id}.png")
        try:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        except Exception:
            pass
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64

    def generate_comparison_chart(
        self,
        diff_rewards: List[float],
        non_diff_rewards: List[float],
        title: str = "Network Comparison"
    ) -> str:
        """旧版兼容接口：生成对比图并返回Base64"""
        fig, _ = plt.subplots(1, 2, figsize=(12, 5))
        self._draw_comparison_chart(fig, diff_rewards, non_diff_rewards, title)
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return b64

    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """计算移动平均"""
        if len(data) < window:
            return data
        result = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            result.append(avg)
        return result

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """清理output_dir中超过指定时间的旧图片文件，返回清理数量"""
        import time
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        count = 0
        pattern = os.path.join(self.output_dir, "*.png")
        for fp in glob.glob(pattern):
            try:
                if os.path.getmtime(fp) < cutoff:
                    os.remove(fp)
                    count += 1
            except OSError as e:
                logger.warning(f"Failed to remove old plot {fp}: {e}")
        if count:
            logger.info(f"Cleaned up {count} old plot files (> {max_age_hours}h)")
        return count
