"""
遗传算法服务模块
"""
import uuid
from datetime import datetime
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor
import threading
import gymnasium as gym
import numpy as np
import torch
from loguru import logger

from ..models.genetic_algorithm import GeneticAlgorithm, Individual, WeightGenerator
from ..models.network import NonDifferentiableNetwork
from ..schemas.requests import GeneticStartRequest
from ..schemas.responses import GeneticStatusData, IndividualData, GeneticPopulationData
from ..core.exceptions import GeneticAlgorithmException


class GeneticTask:
    """遗传算法任务"""
    
    def __init__(self, task_id: str, config: GeneticStartRequest):
        self.task_id = task_id
        self.config = config
        self.status = "pending"
        self.current_generation = 0
        self.max_generations = config.max_generations
        self.best_fitness = float('-inf')
        self.population_size = config.population_size
        self.elite_archive_size = 0
        self.best_individual: Optional[Individual] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.ga: Optional[GeneticAlgorithm] = None
        self._stop_flag = threading.Event()
        
    def should_stop(self) -> bool:
        return self._stop_flag.is_set()
    
    def request_stop(self):
        self._stop_flag.set()
        
    def get_progress(self) -> float:
        if self.max_generations == 0:
            return 0.0
        return (self.current_generation / self.max_generations) * 100


class GeneticService:
    """
    遗传算法服务
    管理遗传算法搜索任务
    """
    
    def __init__(self, max_concurrent_tasks: int = 2):
        self.tasks: Dict[str, GeneticTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self._lock = threading.Lock()
        logger.info(f"GeneticService initialized with max_concurrent_tasks={max_concurrent_tasks}")
        
    def start_search(self, request: GeneticStartRequest) -> str:
        """
        启动遗传算法搜索
        
        Args:
            request: 搜索请求参数
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        task = GeneticTask(task_id, request)
        
        with self._lock:
            self.tasks[task_id] = task
            
        self.executor.submit(self._run_search, task)
        
        logger.info(f"Genetic search task {task_id} started")
        return task_id
    
    def _run_search(self, task: GeneticTask) -> None:
        """
        执行遗传算法搜索
        """
        env = None
        try:
            task.status = "running"
            task.started_at = datetime.now()
            
            # 创建环境
            env = gym.make(task.config.env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            # 创建遗传算法实例
            ga = GeneticAlgorithm(
                population_size=task.config.population_size,
                mutation_rate=task.config.mutation_rate,
                crossover_rate=task.config.crossover_rate,
                elite_size=task.config.elite_size,
                seed_range=(task.config.seed_range_min, task.config.seed_range_max)
            )
            task.ga = ga
            
            # 创建权重生成器和网络
            weight_generator = WeightGenerator()
            
            # 获取网络权重形状
            network = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
            network_shapes = {
                name: tuple(param.shape)
                for name, param in network.named_parameters()
            }
            
            # 评估函数
            def evaluate_individual(individual: Individual) -> float:
                if task.should_stop():
                    raise InterruptedError("Search stopped by user")
                    
                # 生成权重
                weights = weight_generator.generate_weights_from_individual(
                    individual, network_shapes
                )
                
                # 应用权重到网络
                weight_generator.apply_weights_to_network(network, weights)
                
                # 评估多个回合取平均
                total_reward = 0.0
                for _ in range(task.config.evaluation_episodes):
                    state, _ = env.reset()
                    episode_reward = 0.0
                    
                    while True:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state)
                            logits = network(state_tensor)
                            action = logits.argmax().item()
                            
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        state = next_state
                        
                        if terminated or truncated:
                            break
                            
                    total_reward += episode_reward
                    
                return total_reward / task.config.evaluation_episodes
            
            # 代回调函数
            def generation_callback(gen: int, best_ind: Individual, best_fit: float):
                if task.should_stop():
                    raise InterruptedError("Search stopped by user")
                    
                task.current_generation = gen
                task.best_fitness = best_fit
                task.best_individual = best_ind
                task.elite_archive_size = len(ga.elite_archive)
                
                logger.info(
                    f"Task {task.task_id} - Generation {gen}: "
                    f"Best fitness = {best_fit:.2f}"
                )
            
            # 运行遗传算法
            best = ga.run(
                evaluate_fn=evaluate_individual,
                max_generations=task.config.max_generations,
                target_fitness=task.config.target_fitness,
                generation_callback=generation_callback
            )
            
            task.best_individual = best
            task.best_fitness = best.fitness if best else float('-inf')
            task.status = "completed"
            task.completed_at = datetime.now()
            
            logger.info(
                f"Genetic search task {task.task_id} completed. "
                f"Best fitness: {task.best_fitness:.2f}"
            )
            
        except InterruptedError:
            task.status = "stopped"
            task.completed_at = datetime.now()
            logger.info(f"Genetic search task {task.task_id} stopped by user")
            
        except Exception as e:
            task.status = "failed"
            task.completed_at = datetime.now()
            logger.error(f"Genetic search task {task.task_id} failed: {str(e)}")
            
        finally:
            if env is not None:
                env.close()
    
    def get_status(self, task_id: str) -> GeneticStatusData:
        """获取搜索状态"""
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise GeneticAlgorithmException(f"Genetic task {task_id} not found")
            
        best_ind_data = None
        if task.best_individual:
            best_ind_data = IndividualData(
                seeds=task.best_individual.seeds.tolist(),
                fitness=task.best_individual.fitness,
                generation=task.best_individual.generation
            )
            
        return GeneticStatusData(
            task_id=task.task_id,
            status=task.status,
            current_generation=task.current_generation,
            max_generations=task.max_generations,
            best_fitness=task.best_fitness,
            population_size=task.population_size,
            elite_archive_size=task.elite_archive_size,
            best_individual=best_ind_data,
            progress=task.get_progress()
        )
    
    def stop_search(self, task_id: str) -> bool:
        """停止搜索任务"""
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise GeneticAlgorithmException(f"Genetic task {task_id} not found")
            
        if task.status != "running":
            return False
            
        task.request_stop()
        logger.info(f"Stop requested for genetic task {task_id}")
        return True
    
    def get_best_individual(self, task_id: str) -> IndividualData:
        """获取最佳个体"""
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise GeneticAlgorithmException(f"Genetic task {task_id} not found")
            
        if task.best_individual is None:
            raise GeneticAlgorithmException(f"No best individual found for task {task_id}")
            
        return IndividualData(
            seeds=task.best_individual.seeds.tolist(),
            fitness=task.best_individual.fitness,
            generation=task.best_individual.generation
        )
    
    def get_population(self, task_id: str) -> GeneticPopulationData:
        """获取当前种群"""
        with self._lock:
            task = self.tasks.get(task_id)
            
        if task is None:
            raise GeneticAlgorithmException(f"Genetic task {task_id} not found")
            
        if task.ga is None or not task.ga.population:
            raise GeneticAlgorithmException(f"No population data for task {task_id}")
            
        individuals = [
            IndividualData(
                seeds=ind.seeds.tolist(),
                fitness=ind.fitness,
                generation=ind.generation
            )
            for ind in task.ga.population
        ]
        
        fitnesses = [ind.fitness for ind in task.ga.population if ind.fitness != float('-inf')]
        avg_fitness = float(np.mean(fitnesses)) if fitnesses else 0.0
        
        return GeneticPopulationData(
            generation=task.current_generation,
            individuals=individuals,
            best_fitness=task.best_fitness,
            avg_fitness=avg_fitness
        )
    
    def list_tasks(self) -> List[GeneticStatusData]:
        """列出所有遗传算法任务"""
        with self._lock:
            tasks = list(self.tasks.values())
            
        result = []
        for task in tasks:
            best_ind_data = None
            if task.best_individual:
                best_ind_data = IndividualData(
                    seeds=task.best_individual.seeds.tolist(),
                    fitness=task.best_individual.fitness,
                    generation=task.best_individual.generation
                )
                
            result.append(GeneticStatusData(
                task_id=task.task_id,
                status=task.status,
                current_generation=task.current_generation,
                max_generations=task.max_generations,
                best_fitness=task.best_fitness,
                population_size=task.population_size,
                elite_archive_size=task.elite_archive_size,
                best_individual=best_ind_data,
                progress=task.get_progress()
            ))
            
        return result
    
    def get_active_task_count(self) -> int:
        """获取活跃任务数量"""
        with self._lock:
            return sum(1 for t in self.tasks.values() if t.status == "running")
