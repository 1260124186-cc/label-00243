"""
可视化服务模块
实现训练过程中的适应度曲线生成和实时显示
"""
import os
import io
import base64
from typing import List, Optional, Dict, Any
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


class VisualizationService:
    """
    可视化服务
    生成训练过程中的适应度曲线
    """

    def __init__(self, output_dir: str = "plots"):
        """
        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        logger.info(f"VisualizationService initialized, output_dir={output_dir}")

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
        生成适应度曲线图

        Args:
            fitness_history: 适应度历史记录
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            save_path: 保存路径（可选）
            show_avg: 是否显示移动平均线
            window_size: 移动平均窗口大小

        Returns:
            Base64编码的图片字符串
        """
        if not fitness_history:
            logger.warning("Empty fitness history, cannot generate curve")
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))

        generations = list(range(1, len(fitness_history) + 1))

        # 绘制原始适应度曲线
        ax.plot(generations, fitness_history, 'b-', alpha=0.6, label='Best Fitness', linewidth=1)

        # 绘制移动平均线
        if show_avg and len(fitness_history) >= window_size:
            moving_avg = self._moving_average(fitness_history, window_size)
            avg_x = list(range(window_size, len(fitness_history) + 1))
            ax.plot(avg_x, moving_avg, 'r-', label=f'Moving Avg ({window_size})', linewidth=2)

        # 标记最高点
        max_fitness = max(fitness_history)
        max_gen = fitness_history.index(max_fitness) + 1
        ax.scatter([max_gen], [max_fitness], color='green', s=100, zorder=5, label=f'Max: {max_fitness:.2f}')
        ax.annotate(f'{max_fitness:.2f}', (max_gen, max_fitness),
                   textcoords="offset points", xytext=(0, 10), ha='center')

        # 绘制200分及格线
        ax.axhline(y=200, color='orange', linestyle='--', linewidth=2, label='Target (200)')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # 设置Y轴范围
        y_min = min(min(fitness_history), 0) - 50
        y_max = max(max(fitness_history), 250) + 50
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()

        # 保存到文件
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            fig.savefig(full_path, dpi=150, bbox_inches='tight')
            logger.info(f"Fitness curve saved to {full_path}")

        # 转换为Base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        plt.close(fig)

        return image_base64

    def generate_training_dashboard(
        self,
        episode_rewards: List[float],
        policy_losses: List[float],
        value_losses: List[float],
        temperatures: List[float],
        task_id: str
    ) -> str:
        """
        生成训练仪表板（多子图）

        Args:
            episode_rewards: 每回合奖励
            policy_losses: 策略损失
            value_losses: 价值损失
            temperatures: 温度历史
            task_id: 任务ID

        Returns:
            Base64编码的图片字符串
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 回合奖励曲线
        ax1 = axes[0, 0]
        if episode_rewards:
            episodes = list(range(1, len(episode_rewards) + 1))
            ax1.plot(episodes, episode_rewards, 'b-', alpha=0.5, linewidth=0.5)

            # 移动平均
            if len(episode_rewards) >= 100:
                avg = self._moving_average(episode_rewards, 100)
                ax1.plot(range(100, len(episode_rewards) + 1), avg, 'r-', linewidth=2, label='Avg (100)')

            ax1.axhline(y=200, color='orange', linestyle='--', label='Target (200)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. 策略损失
        ax2 = axes[0, 1]
        if policy_losses:
            ax2.plot(policy_losses, 'g-', linewidth=1)
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.set_title('Policy Loss')
            ax2.grid(True, alpha=0.3)

        # 3. 价值损失
        ax3 = axes[1, 0]
        if value_losses:
            ax3.plot(value_losses, 'm-', linewidth=1)
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.set_title('Value Loss')
            ax3.grid(True, alpha=0.3)

        # 4. 温度退火曲线
        ax4 = axes[1, 1]
        if temperatures:
            ax4.plot(temperatures, 'c-', linewidth=2)
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Temperature')
            ax4.set_title('Temperature Annealing')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Training Dashboard - {task_id}', fontsize=14)
        plt.tight_layout()

        # 保存到文件
        save_path = os.path.join(self.output_dir, f"training_{task_id}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

        # 转换为Base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        plt.close(fig)

        return image_base64

    def generate_genetic_progress(
        self,
        fitness_history: List[float],
        avg_fitness_history: Optional[List[float]] = None,
        task_id: str = ""
    ) -> str:
        """
        生成遗传算法进度图

        Args:
            fitness_history: 最佳适应度历史
            avg_fitness_history: 平均适应度历史（可选）
            task_id: 任务ID

        Returns:
            Base64编码的图片字符串
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        generations = list(range(1, len(fitness_history) + 1))

        # 最佳适应度
        ax.plot(generations, fitness_history, 'b-', linewidth=2, label='Best Fitness')

        # 平均适应度
        if avg_fitness_history and len(avg_fitness_history) == len(fitness_history):
            ax.plot(generations, avg_fitness_history, 'g--', linewidth=1, alpha=0.7, label='Avg Fitness')

        # 及格线
        ax.axhline(y=200, color='orange', linestyle='--', linewidth=2, label='Target (200)')

        # 填充区域
        ax.fill_between(generations, fitness_history, alpha=0.2)

        # 标记最高点
        if fitness_history:
            max_fitness = max(fitness_history)
            max_gen = fitness_history.index(max_fitness) + 1
            ax.scatter([max_gen], [max_fitness], color='red', s=100, zorder=5)
            ax.annotate(f'Best: {max_fitness:.2f}', (max_gen, max_fitness),
                       textcoords="offset points", xytext=(10, 10), ha='left',
                       fontsize=12, fontweight='bold')

        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        ax.set_title(f'Genetic Algorithm Progress - {task_id}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存到文件
        save_path = os.path.join(self.output_dir, f"genetic_{task_id}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

        # 转换为Base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        plt.close(fig)

        return image_base64

    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """计算移动平均"""
        if len(data) < window:
            return data

        result = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            result.append(avg)
        return result

    def generate_comparison_chart(
        self,
        diff_rewards: List[float],
        non_diff_rewards: List[float],
        title: str = "Network Comparison"
    ) -> str:
        """
        生成网络对比图

        Args:
            diff_rewards: 可微网络奖励列表
            non_diff_rewards: 不可微网络奖励列表
            title: 标题

        Returns:
            Base64编码的图片字符串
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 箱线图
        ax1 = axes[0]
        data = [diff_rewards, non_diff_rewards]
        bp = ax1.boxplot(data, tick_labels=['Differentiable', 'Non-Differentiable'])
        ax1.axhline(y=200, color='orange', linestyle='--', label='Target (200)')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 直方图
        ax2 = axes[1]
        ax2.hist(diff_rewards, bins=20, alpha=0.5, label='Differentiable', color='blue')
        ax2.hist(non_diff_rewards, bins=20, alpha=0.5, label='Non-Differentiable', color='green')
        ax2.axvline(x=200, color='orange', linestyle='--', label='Target (200)')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Histogram')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        # 转换为Base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

        plt.close(fig)

        return image_base64
