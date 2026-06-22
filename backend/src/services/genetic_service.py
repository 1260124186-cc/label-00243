"""
遗传算法服务模块
"""
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp
import gymnasium as gym
import numpy as np
import torch
from loguru import logger

from ..models.genetic_algorithm import GeneticAlgorithm, Individual, WeightGenerator
from ..models.network import NonDifferentiableNetwork
from ..schemas.requests import GeneticStartRequest
from ..schemas.responses import GeneticStatusData, IndividualData, GeneticPopulationData
from ..core.exceptions import GeneticAlgorithmException


def _genetic_worker_evaluate(args: Tuple[Any, ...]) -> float:
    """
    多进程 worker：独立评估单个个体（模块顶层函数，必须可pickle）。

    args 元组:
      (env_name, state_dim, action_dim, evaluation_episodes, seeds_flat_list_24)
    """
    (
        env_name,
        state_dim,
        action_dim,
        evaluation_episodes,
        seeds_list,
    ) = args

    env = None
    try:
        individual = Individual.create_from_list(seeds_list)

        env = gym.make(env_name)
        network = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
        network.eval()
        weight_generator = WeightGenerator()

        network_shapes = {
            name: tuple(param.shape)
            for name, param in network.named_parameters()
        }

        weights = weight_generator.generate_weights_from_individual(
            individual, network_shapes
        )
        weight_generator.apply_weights_to_network(network, weights)

        state_buf = torch.empty(state_dim, dtype=torch.float32)

        total_reward = 0.0
        with torch.inference_mode():
            for _ in range(evaluation_episodes):
                state, _ = env.reset()
                episode_reward = 0.0
                while True:
                    state_buf.copy_(torch.from_numpy(state))
                    logits = network(state_buf)
                    action = logits.argmax().item()

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    state = next_state

                    if terminated or truncated:
                        break
                total_reward += episode_reward

        return total_reward / evaluation_episodes
    finally:
        if env is not None:
            env.close()


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

    def __init__(
        self,
        max_concurrent_tasks: int = 2,
        max_parallel_workers: Optional[int] = None,
        use_parallel_eval: bool = True,
    ):
        """
        Args:
            max_concurrent_tasks: 允许同时运行的遗传算法任务数（线程池大小）
            max_parallel_workers: 单个任务内并行评估个体的进程数（默认=cpu_count()//2）
            use_parallel_eval: 是否启用多进程并行评估（关闭则退化为串行）
        """
        self.tasks: Dict[str, GeneticTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self._lock = threading.Lock()

        self.use_parallel_eval = use_parallel_eval
        if use_parallel_eval:
            if max_parallel_workers is None:
                try:
                    max_parallel_workers = max(1, (mp.cpu_count() or 2) // 2)
                except NotImplementedError:
                    max_parallel_workers = 2
            self.max_parallel_workers = max_parallel_workers
        else:
            self.max_parallel_workers = 1

        logger.info(
            f"GeneticService initialized: max_concurrent_tasks={max_concurrent_tasks}, "
            f"use_parallel_eval={use_parallel_eval}, max_parallel_workers={self.max_parallel_workers}"
        )

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
        执行遗传算法搜索（支持多进程并行评估个体）
        """
        env = None
        process_pool: Optional[ProcessPoolExecutor] = None
        try:
            task.status = "running"
            task.started_at = datetime.now()

            env = gym.make(task.config.env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            ga = GeneticAlgorithm(
                population_size=task.config.population_size,
                mutation_rate=task.config.mutation_rate,
                crossover_rate=task.config.crossover_rate,
                elite_size=task.config.elite_size,
                seed_range=(task.config.seed_range_min, task.config.seed_range_max),
            )
            task.ga = ga

            use_parallel = self.use_parallel_eval and self.max_parallel_workers > 1
            env_name_local = task.config.env_name
            eval_episodes = task.config.evaluation_episodes
            evaluate_fn = None
            parallel_executor = None
            parallel_worker_fn = None
            parallel_args_builder = None

            if use_parallel:
                process_pool = ProcessPoolExecutor(max_workers=self.max_parallel_workers)
                parallel_executor = process_pool
                parallel_worker_fn = _genetic_worker_evaluate

                def _args_builder(ind: Individual) -> Tuple[Any, ...]:
                    return (
                        env_name_local,
                        state_dim,
                        action_dim,
                        eval_episodes,
                        ind.to_list(),
                    )

                parallel_args_builder = _args_builder

                logger.info(
                    f"Task {task.task_id} using parallel evaluation with "
                    f"{self.max_parallel_workers} workers"
                )
            else:
                weight_generator = WeightGenerator()
                network = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
                network_shapes = {
                    name: tuple(param.shape)
                    for name, param in network.named_parameters()
                }
                state_tensor_buf = torch.empty(state_dim, dtype=torch.float32)

                @torch.inference_mode()
                def evaluate_individual(individual: Individual) -> float:
                    if task.should_stop():
                        raise InterruptedError("Search stopped by user")

                    weights = weight_generator.generate_weights_from_individual(
                        individual, network_shapes
                    )
                    weight_generator.apply_weights_to_network(network, weights)
                    network.eval()

                    total_reward = 0.0
                    local_buf = state_tensor_buf
                    net = network
                    for _ in range(eval_episodes):
                        state, _ = env.reset()
                        episode_reward = 0.0
                        while True:
                            local_buf.copy_(torch.from_numpy(state))
                            logits = net(local_buf)
                            action = logits.argmax().item()

                            next_state, reward, terminated, truncated, _ = env.step(action)
                            episode_reward += reward
                            state = next_state
                            if terminated or truncated:
                                break
                        total_reward += episode_reward
                    return total_reward / eval_episodes

                evaluate_fn = evaluate_individual

            def generation_callback(gen: int, best_ind: Optional[Individual], best_fit: float):
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

            best = ga.run(
                evaluate_fn=evaluate_fn,
                max_generations=task.config.max_generations,
                target_fitness=task.config.target_fitness,
                generation_callback=generation_callback,
                parallel_executor=parallel_executor,
                parallel_worker_fn=parallel_worker_fn,
                parallel_args_builder=parallel_args_builder,
            )

            task.best_individual = best
            task.best_fitness = best.fitness if best else float("-inf")
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
            if process_pool is not None:
                process_pool.shutdown(wait=True, cancel_futures=True)
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
