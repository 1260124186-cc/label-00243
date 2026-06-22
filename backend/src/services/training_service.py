"""
训练服务模块
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import gymnasium as gym
import numpy as np
from loguru import logger

from ..models.ppo_agent import PPOAgent, TrainingResult
from ..schemas.requests import TrainingStartRequest
from ..schemas.responses import TrainingStatusData, TrainingResultData, TrainingHistoryItem
from ..core.exceptions import TrainingException


class TrainingTask:
    """训练任务"""
    
    def __init__(self, task_id: str, config: TrainingStartRequest, config_snapshot: Optional[Any] = None):
        self.task_id = task_id
        self.config = config
        self.config_snapshot = config_snapshot
        self.status = "pending"
        self.current_episode = 0
        self.total_episodes = config.total_episodes
        self.best_reward = float('-inf')
        self.avg_reward_last_100 = 0.0
        self.current_temperature = config.initial_temperature
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[TrainingResult] = None
        self.agent: Optional[PPOAgent] = None
        self._stop_flag = threading.Event()
        self.auto_start_ga = config.auto_start_ga
        self.child_ga_task_id: Optional[str] = None
        
    def should_stop(self) -> bool:
        return self._stop_flag.is_set()
    
    def request_stop(self):
        self._stop_flag.set()
        
    def get_progress(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return (self.current_episode / self.total_episodes) * 100


class TrainingService:
    """
    训练服务
    管理PPO训练任务的生命周期
    """
    
    def __init__(self, max_concurrent_tasks: Optional[int] = None):
        from ..config import settings
        if max_concurrent_tasks is None:
            max_concurrent_tasks = settings.MAX_CONCURRENT_TRAINING_TASKS
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, TrainingTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self._lock = threading.Lock()
        self._genetic_service = None
        logger.info(f"TrainingService initialized with max_concurrent_tasks={max_concurrent_tasks}")
    
    def set_genetic_service(self, genetic_service) -> None:
        """设置遗传算法服务引用，用于自动启动关联GA任务"""
        self._genetic_service = genetic_service
    
    def update_max_concurrent_tasks(self, new_max: int) -> None:
        """
        更新并发任务上限
        
        Args:
            new_max: 新的并发上限
        """
        with self._lock:
            old_max = self.max_concurrent_tasks
            self.max_concurrent_tasks = new_max
            
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(max_workers=new_max)
            
            old_executor.shutdown(wait=False, cancel_futures=False)
            
            logger.info(
                f"TrainingService max_concurrent_tasks updated: {old_max} -> {new_max}"
            )
    
    def start_training(self, request: TrainingStartRequest) -> str:
        """
        启动训练任务
        
        Args:
            request: 训练请求参数
            
        Returns:
            任务ID
        """
        from ..config import create_config_snapshot
        
        task_id = str(uuid.uuid4())
        config_snapshot = create_config_snapshot()
        task = TrainingTask(task_id, request, config_snapshot)
        
        with self._lock:
            self.tasks[task_id] = task
            
        # 提交到线程池执行
        self.executor.submit(self._run_training, task)
        
        logger.info(
            f"Training task {task_id} started with config: {request.model_dump()}, "
            f"snapshot_time: {config_snapshot.snapshot_time}"
        )
        return task_id
    
    def _maybe_auto_start_ga(self, task: TrainingTask, model_path: str) -> None:
        """
        检查是否需要自动启动关联的GA任务
        条件：
        1. auto_start_ga 为 true
        2. 训练状态为 completed
        3. avg_reward_last_100 >= 200
        4. 已设置 genetic_service 引用
        5. 配置中包含 ga_config
        
        Args:
            task: 训练任务
            model_path: 保存的模型路径
        """
        if not task.auto_start_ga:
            return
        
        if task.status != "completed":
            return
        
        if task.avg_reward_last_100 < 200:
            return
        
        if self._genetic_service is None:
            logger.warning(
                f"Cannot auto-start GA for task {task.task_id}: "
                f"genetic_service not set"
            )
            return
        
        if task.config.ga_config is None:
            logger.warning(
                f"Cannot auto-start GA for task {task.task_id}: "
                f"ga_config not provided"
            )
            return
        
        try:
            # 基于用户提供的GA配置创建新的请求，并设置父任务ID和目标权重路径
            from ..schemas.requests import GeneticStartRequest
            
            ga_config = task.config.ga_config
            ga_request = GeneticStartRequest(
                population_size=ga_config.population_size,
                max_generations=ga_config.max_generations,
                mutation_rate=ga_config.mutation_rate,
                crossover_rate=ga_config.crossover_rate,
                elite_size=ga_config.elite_size,
                seed_range_min=ga_config.seed_range_min,
                seed_range_max=ga_config.seed_range_max,
                target_fitness=ga_config.target_fitness,
                evaluation_episodes=ga_config.evaluation_episodes,
                env_name=ga_config.env_name,
                alpha=ga_config.alpha,
                target_weights_path=model_path,
                target_seeds=ga_config.target_seeds,
                parent_task_id=task.task_id,
            )
            
            ga_task_id = self._genetic_service.start_search(ga_request)
            task.child_ga_task_id = ga_task_id
            
            logger.info(
                f"Auto-started GA task {ga_task_id} for PPO task {task.task_id} "
                f"(avg_reward_last_100={task.avg_reward_last_100:.2f})"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to auto-start GA task for PPO task {task.task_id}: {str(e)}"
            )
    
    def _run_training(self, task: TrainingTask) -> None:
        """
        执行训练任务（在线程中运行）
        """
        try:
            task.status = "running"
            task.started_at = datetime.now()
            
            # 创建环境
            env = gym.make(task.config.env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            # 创建PPO智能体（包含target_mode配置）
            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=task.config.learning_rate,
                gamma=task.config.gamma,
                epsilon=task.config.epsilon,
                initial_temperature=task.config.initial_temperature,
                temperature_decay=task.config.temperature_decay,
                min_temperature=task.config.min_temperature,
                regularization_coef=task.config.regularization_coef,
                target_mode=task.config.target_mode,
                target_seeds=task.config.target_seeds,
                target_quantize_bits=task.config.target_quantize_bits,
                weight_copy_interval=task.config.weight_copy_interval,
                harden_on_copy=task.config.harden_on_copy
            )
            
            task.agent = agent
            
            # 定义回调函数更新任务状态
            def training_callback(episode: int, result: TrainingResult):
                if task.should_stop():
                    raise InterruptedError("Training stopped by user")
                    
                task.current_episode = episode + 1
                task.best_reward = result.best_reward
                if len(result.episode_rewards) >= 100:
                    task.avg_reward_last_100 = np.mean(result.episode_rewards[-100:])
                elif result.episode_rewards:
                    task.avg_reward_last_100 = np.mean(result.episode_rewards)
                task.current_temperature = agent.policy_net.temperature
            
            # 执行训练
            result = agent.train(
                env=env,
                total_episodes=task.config.total_episodes,
                max_steps=task.config.max_steps,
                callback=training_callback
            )
            
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            
            # 保存模型
            model_path = f"models/ppo_{task.task_id}.pt"
            import os
            os.makedirs("models", exist_ok=True)
            agent.save(model_path)
            
            logger.info(
                f"Training task {task.task_id} completed. "
                f"Best reward: {task.best_reward:.2f}, "
                f"Avg last 100: {task.avg_reward_last_100:.2f}"
            )
            
            # 检查是否需要自动启动GA任务
            self._maybe_auto_start_ga(task, model_path)
            
        except InterruptedError:
            task.status = "stopped"
            task.completed_at = datetime.now()
            logger.info(f"Training task {task.task_id} stopped by user")
            
        except Exception as e:
            task.status = "failed"
            task.completed_at = datetime.now()
            logger.error(f"Training task {task.task_id} failed: {str(e)}")
            
        finally:
            env.close()
    
    def get_status(self, task_id: str) -> TrainingStatusData:
        """
        获取训练状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            训练状态数据
        """
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise TrainingException(f"Training task {task_id} not found")
            
        return TrainingStatusData(
            task_id=task.task_id,
            status=task.status,
            current_episode=task.current_episode,
            total_episodes=task.total_episodes,
            best_reward=task.best_reward,
            avg_reward_last_100=task.avg_reward_last_100,
            current_temperature=task.current_temperature,
            started_at=task.started_at,
            completed_at=task.completed_at,
            progress=task.get_progress(),
            auto_start_ga=task.auto_start_ga,
            child_ga_task_id=task.child_ga_task_id
        )
    
    def stop_training(self, task_id: str) -> bool:
        """
        停止训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功停止
        """
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise TrainingException(f"Training task {task_id} not found")
            
        if task.status != "running":
            return False
            
        task.request_stop()
        logger.info(f"Stop requested for training task {task_id}")
        return True
    
    def get_history(
        self,
        task_id: str,
        page: int = 1,
        page_size: int = 100
    ) -> tuple[list[TrainingHistoryItem], int]:
        """
        获取训练历史
        
        Args:
            task_id: 任务ID
            page: 页码
            page_size: 每页大小
            
        Returns:
            (历史记录列表, 总数)
        """
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise TrainingException(f"Training task {task_id} not found")
            
        if task.result is None:
            return [], 0
            
        result = task.result
        total = len(result.episode_rewards)
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        
        history = []
        for i in range(start_idx, end_idx):
            item = TrainingHistoryItem(
                episode=i + 1,
                reward=result.episode_rewards[i],
                length=result.episode_lengths[i] if i < len(result.episode_lengths) else 0,
                policy_loss=result.policy_losses[i] if i < len(result.policy_losses) else None,
                value_loss=result.value_losses[i] if i < len(result.value_losses) else None,
                temperature=result.temperatures[i] if i < len(result.temperatures) else None
            )
            history.append(item)
            
        return history, total
    
    def get_result(self, task_id: str) -> TrainingResultData:
        """
        获取训练结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            训练结果数据
        """
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise TrainingException(f"Training task {task_id} not found")
            
        if task.status not in ["completed", "stopped"]:
            raise TrainingException(f"Training task {task_id} is not completed yet")
            
        duration = 0.0
        if task.started_at and task.completed_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            
        return TrainingResultData(
            task_id=task.task_id,
            status=task.status,
            total_episodes=task.current_episode,
            best_reward=task.best_reward,
            avg_reward_last_100=task.avg_reward_last_100,
            episode_rewards=task.result.episode_rewards if task.result else [],
            training_duration_seconds=duration
        )
    
    def list_tasks(self) -> list[TrainingStatusData]:
        """列出所有训练任务"""
        with self._lock:
            tasks = list(self.tasks.values())
            
        return [
            TrainingStatusData(
                task_id=t.task_id,
                status=t.status,
                current_episode=t.current_episode,
                total_episodes=t.total_episodes,
                best_reward=t.best_reward,
                avg_reward_last_100=t.avg_reward_last_100,
                current_temperature=t.current_temperature,
                started_at=t.started_at,
                completed_at=t.completed_at,
                progress=t.get_progress(),
                auto_start_ga=t.auto_start_ga,
                child_ga_task_id=t.child_ga_task_id
            )
            for t in tasks
        ]
    
    def get_active_task_count(self) -> int:
        """获取活跃任务数量"""
        with self._lock:
            return sum(1 for t in self.tasks.values() if t.status == "running")
