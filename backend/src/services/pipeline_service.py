import uuid
import threading
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from ..models.ppo_agent import PPOAgent, TrainingResult
from ..models.network import DifferentiableNetwork, NonDifferentiableNetwork
from ..models.genetic_algorithm import GeneticAlgorithm, Individual, WeightGenerator
from ..schemas.requests import PipelineStartRequest
from ..schemas.responses import (
    PipelineStatusData,
    PipelineStageData,
    ComparisonReportData,
)
from ..core.exceptions import PipelineException


class PipelineStage:
    PPO_TRAINING = "ppo_training"
    WEIGHT_EXPORT = "weight_export"
    GA_SEARCH = "ga_search"
    COMPARISON_REPORT = "comparison_report"

    ORDER = [PPO_TRAINING, WEIGHT_EXPORT, GA_SEARCH, COMPARISON_REPORT]


class PipelineTask:
    def __init__(self, task_id: str, config: PipelineStartRequest):
        self.task_id = task_id
        self.config = config
        self.status = "pending"
        self.current_stage = PipelineStage.PPO_TRAINING
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self._stop_flag = threading.Event()

        self.stages: Dict[str, Dict[str, Any]] = {
            name: {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "details": None,
            }
            for name in PipelineStage.ORDER
        }

        self.ppo_agent: Optional[PPOAgent] = None
        self.ppo_result: Optional[TrainingResult] = None
        self.target_weight_vector: Optional[torch.Tensor] = None
        self.target_weight_dict: Optional[Dict[str, torch.Tensor]] = None
        self.ga_best_individual: Optional[Individual] = None
        self.ga_best_fitness: float = float("-inf")
        self.comparison_report: Optional[ComparisonReportData] = None

    def should_stop(self) -> bool:
        return self._stop_flag.is_set()

    def request_stop(self):
        self._stop_flag.set()

    def _mark_stage(self, name: str, field: str, value: Any):
        if name in self.stages:
            self.stages[name][field] = value

    def start_stage(self, name: str):
        self.current_stage = name
        self._mark_stage(name, "status", "running")
        self._mark_stage(name, "started_at", datetime.now())

    def complete_stage(self, name: str, details: Optional[Dict] = None):
        self._mark_stage(name, "status", "completed")
        self._mark_stage(name, "completed_at", datetime.now())
        self._mark_stage(name, "details", details)

    def fail_stage(self, name: str, error: str):
        self._mark_stage(name, "status", "failed")
        self._mark_stage(name, "completed_at", datetime.now())
        self._mark_stage(name, "details", {"error": error})

    def get_progress(self) -> float:
        completed = sum(
            1 for s in self.stages.values() if s["status"] == "completed"
        )
        return (completed / len(PipelineStage.ORDER)) * 100.0


def _compute_weight_distance(
    weight_dict_a: Dict[str, torch.Tensor],
    weight_dict_b: Dict[str, torch.Tensor],
) -> float:
    total = 0.0
    for name in weight_dict_a:
        if name in weight_dict_b:
            total += torch.sum(
                (weight_dict_a[name].float() - weight_dict_b[name].float()) ** 2
            ).item()
    return float(np.sqrt(total))


class PipelineService:
    def __init__(self, max_concurrent_tasks: int = 2):
        self.tasks: Dict[str, PipelineTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self._lock = threading.Lock()
        logger.info(
            f"PipelineService initialized with max_concurrent_tasks={max_concurrent_tasks}"
        )

    def start_pipeline(self, request: PipelineStartRequest) -> str:
        task_id = str(uuid.uuid4())
        task = PipelineTask(task_id, request)

        with self._lock:
            self.tasks[task_id] = task

        self.executor.submit(self._run_pipeline, task)
        logger.info(f"Pipeline task {task_id} started")
        return task_id

    def _run_pipeline(self, task: PipelineTask) -> None:
        try:
            task.status = "running"
            task.started_at = datetime.now()

            self._stage_ppo_training(task)
            if task.should_stop():
                raise InterruptedError("Pipeline stopped by user")

            self._stage_weight_export(task)
            if task.should_stop():
                raise InterruptedError("Pipeline stopped by user")

            self._stage_ga_search(task)
            if task.should_stop():
                raise InterruptedError("Pipeline stopped by user")

            self._stage_comparison_report(task)

            task.status = "completed"
            task.completed_at = datetime.now()
            logger.info(f"Pipeline task {task.task_id} completed")

        except InterruptedError:
            task.status = "stopped"
            task.completed_at = datetime.now()
            logger.info(f"Pipeline task {task.task_id} stopped by user")

        except Exception as e:
            task.status = "failed"
            task.completed_at = datetime.now()
            logger.error(f"Pipeline task {task.task_id} failed: {str(e)}")

    def _stage_ppo_training(self, task: PipelineTask) -> None:
        task.start_stage(PipelineStage.PPO_TRAINING)
        try:
            ppo_config = task.config.ppo_config
            env = gym.make(ppo_config.env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=ppo_config.learning_rate,
                gamma=ppo_config.gamma,
                epsilon=ppo_config.epsilon,
                initial_temperature=ppo_config.initial_temperature,
                temperature_decay=ppo_config.temperature_decay,
                min_temperature=ppo_config.min_temperature,
                regularization_coef=ppo_config.regularization_coef,
            )
            task.ppo_agent = agent

            target_network = NonDifferentiableNetwork(
                state_dim=state_dim, action_dim=action_dim
            )
            agent.set_target_network(target_network)

            def training_callback(episode: int, result: TrainingResult):
                if task.should_stop():
                    raise InterruptedError("Pipeline stopped by user")

            result = agent.train(
                env=env,
                total_episodes=ppo_config.total_episodes,
                max_steps=ppo_config.max_steps,
                callback=training_callback,
            )

            task.ppo_result = result
            env.close()

            task.complete_stage(
                PipelineStage.PPO_TRAINING,
                details={
                    "best_reward": result.best_reward,
                    "avg_reward_last_100": float(
                        np.mean(result.episode_rewards[-100:])
                    )
                    if result.episode_rewards
                    else 0.0,
                    "total_episodes": len(result.episode_rewards),
                },
            )
            logger.info(
                f"Pipeline {task.task_id} PPO training done, "
                f"best_reward={result.best_reward:.2f}"
            )

        except InterruptedError:
            raise
        except Exception as e:
            task.fail_stage(PipelineStage.PPO_TRAINING, str(e))
            raise

    def _stage_weight_export(self, task: PipelineTask) -> None:
        task.start_stage(PipelineStage.WEIGHT_EXPORT)
        try:
            if task.ppo_agent is None:
                raise PipelineException("PPO agent not available for weight export")

            target_weight_vector = task.ppo_agent.policy_net.export_weight_vector()
            target_weight_dict = task.ppo_agent.policy_net.export_weight_dict()

            task.target_weight_vector = target_weight_vector
            task.target_weight_dict = target_weight_dict

            weight_norm = float(torch.norm(target_weight_vector).item())
            num_params = target_weight_vector.shape[0]

            task.complete_stage(
                PipelineStage.WEIGHT_EXPORT,
                details={
                    "weight_vector_dim": num_params,
                    "weight_norm": weight_norm,
                },
            )
            logger.info(
                f"Pipeline {task.task_id} weight export done, "
                f"dim={num_params}, norm={weight_norm:.4f}"
            )

        except InterruptedError:
            raise
        except Exception as e:
            task.fail_stage(PipelineStage.WEIGHT_EXPORT, str(e))
            raise

    def _stage_ga_search(self, task: PipelineTask) -> None:
        task.start_stage(PipelineStage.GA_SEARCH)
        try:
            ga_config = task.config.ga_config
            env = gym.make(ga_config.env_name)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            target_weight_dict = task.target_weight_dict
            weight_similarity_coef = task.config.weight_similarity_coef

            ga = GeneticAlgorithm(
                population_size=ga_config.population_size,
                mutation_rate=ga_config.mutation_rate,
                crossover_rate=ga_config.crossover_rate,
                elite_size=ga_config.elite_size,
                seed_range=(ga_config.seed_range_min, ga_config.seed_range_max),
            )

            if task.config.target_seeds is not None:
                seed_individual = Individual.create_from_list(task.config.target_seeds)
                ga.initialize_population()
                ga.population[0] = seed_individual
            else:
                ga.initialize_population()

            weight_generator = WeightGenerator()
            network = NonDifferentiableNetwork(
                state_dim=state_dim, action_dim=action_dim
            )
            network_shapes = {
                name: tuple(param.shape)
                for name, param in network.named_parameters()
            }
            state_tensor_buf = torch.empty(state_dim, dtype=torch.float32)
            eval_episodes = ga_config.evaluation_episodes

            @torch.inference_mode()
            def evaluate_individual(individual: Individual) -> float:
                if task.should_stop():
                    raise InterruptedError("Pipeline stopped by user")

                weights = weight_generator.generate_weights_from_individual(
                    individual, network_shapes
                )
                weight_generator.apply_weights_to_network(network, weights)
                network.eval()

                total_reward = 0.0
                for _ in range(eval_episodes):
                    state, _ = env.reset()
                    episode_reward = 0.0
                    while True:
                        state_tensor_buf.copy_(torch.from_numpy(state))
                        logits = network(state_tensor_buf)
                        action = logits.argmax().item()
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        episode_reward += reward
                        state = next_state
                        if terminated or truncated:
                            break
                    total_reward += episode_reward

                env_reward = total_reward / eval_episodes

                if (
                    target_weight_dict is not None
                    and weight_similarity_coef > 0
                ):
                    ga_weight_dict = {
                        name: param for name, param in weights.items()
                    }
                    distance = _compute_weight_distance(
                        target_weight_dict, ga_weight_dict
                    )
                    target_norm = float(
                        torch.norm(task.target_weight_vector).item()
                    )
                    max_possible_distance = max(target_norm, 1e-8)
                    similarity = max(0.0, 1.0 - distance / max_possible_distance)
                    fitness = env_reward + weight_similarity_coef * similarity * 200.0
                else:
                    fitness = env_reward

                return fitness

            def generation_callback(
                gen: int, best_ind: Optional[Individual], best_fit: float
            ):
                if task.should_stop():
                    raise InterruptedError("Pipeline stopped by user")
                task.ga_best_fitness = best_fit
                task.ga_best_individual = best_ind
                logger.info(
                    f"Pipeline {task.task_id} GA gen {gen}: "
                    f"best_fitness={best_fit:.2f}"
                )

            best = ga.run(
                evaluate_fn=evaluate_individual,
                max_generations=ga_config.max_generations,
                target_fitness=ga_config.target_fitness,
                generation_callback=generation_callback,
            )

            task.ga_best_individual = best
            if best is not None:
                task.ga_best_fitness = best.fitness

            env.close()

            task.complete_stage(
                PipelineStage.GA_SEARCH,
                details={
                    "best_fitness": task.ga_best_fitness,
                    "best_seeds": best.to_list() if best else None,
                    "total_generations": ga.generation,
                },
            )
            logger.info(
                f"Pipeline {task.task_id} GA search done, "
                f"best_fitness={task.ga_best_fitness:.2f}"
            )

        except InterruptedError:
            raise
        except Exception as e:
            task.fail_stage(PipelineStage.GA_SEARCH, str(e))
            raise

    def _stage_comparison_report(self, task: PipelineTask) -> None:
        task.start_stage(PipelineStage.COMPARISON_REPORT)
        try:
            ppo_result = task.ppo_result
            if ppo_result is None:
                raise PipelineException("PPO result not available for comparison")

            ppo_best_reward = ppo_result.best_reward
            ppo_avg_reward = float(np.mean(ppo_result.episode_rewards[-100:])) if ppo_result.episode_rewards else 0.0

            ga_best_fitness = task.ga_best_fitness
            ga_best_seeds = (
                task.ga_best_individual.seeds.tolist()
                if task.ga_best_individual is not None
                else None
            )

            weight_distance = 0.0
            ga_weight_norm = 0.0
            target_weight_norm = 0.0
            similarity_score = 0.0

            if (
                task.target_weight_dict is not None
                and task.ga_best_individual is not None
            ):
                ga_config = task.config.ga_config
                env = gym.make(ga_config.env_name)
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n
                env.close()

                weight_generator = WeightGenerator()
                network = NonDifferentiableNetwork(
                    state_dim=state_dim, action_dim=action_dim
                )
                network_shapes = {
                    name: tuple(param.shape)
                    for name, param in network.named_parameters()
                }
                ga_weights = weight_generator.generate_weights_from_individual(
                    task.ga_best_individual, network_shapes
                )
                ga_weight_dict = {name: param for name, param in ga_weights.items()}

                weight_distance = _compute_weight_distance(
                    task.target_weight_dict, ga_weight_dict
                )

                ga_flat = torch.cat(
                    [w.flatten().float() for w in ga_weights.values()]
                )
                ga_weight_norm = float(torch.norm(ga_flat).item())
                target_weight_norm = float(
                    torch.norm(task.target_weight_vector).item()
                )

                max_possible = max(target_weight_norm, ga_weight_norm, 1e-8)
                similarity_score = max(
                    0.0, 1.0 - weight_distance / max_possible
                )

            performance_gap = ppo_best_reward - ga_best_fitness

            report = ComparisonReportData(
                ppo_best_reward=ppo_best_reward,
                ppo_avg_reward=ppo_avg_reward,
                ga_best_fitness=ga_best_fitness,
                ga_best_seeds=ga_best_seeds,
                weight_distance=weight_distance,
                performance_gap=performance_gap,
                target_weight_norm=target_weight_norm,
                ga_weight_norm=ga_weight_norm,
                similarity_score=similarity_score,
            )
            task.comparison_report = report

            task.complete_stage(
                PipelineStage.COMPARISON_REPORT,
                details=report.model_dump(),
            )
            logger.info(
                f"Pipeline {task.task_id} comparison report done, "
                f"gap={performance_gap:.2f}, similarity={similarity_score:.4f}"
            )

        except InterruptedError:
            raise
        except Exception as e:
            task.fail_stage(PipelineStage.COMPARISON_REPORT, str(e))
            raise

    def get_status(self, task_id: str) -> PipelineStatusData:
        with self._lock:
            task = self.tasks.get(task_id)

        if task is None:
            raise PipelineException(f"Pipeline task {task_id} not found")

        stages = [
            PipelineStageData(
                stage_name=name,
                status=info["status"],
                started_at=info["started_at"],
                completed_at=info["completed_at"],
                details=info["details"],
            )
            for name, info in task.stages.items()
        ]

        return PipelineStatusData(
            task_id=task.task_id,
            status=task.status,
            current_stage=task.current_stage,
            stages=stages,
            started_at=task.started_at,
            completed_at=task.completed_at,
            progress=task.get_progress(),
        )

    def get_report(self, task_id: str) -> ComparisonReportData:
        with self._lock:
            task = self.tasks.get(task_id)

        if task is None:
            raise PipelineException(f"Pipeline task {task_id} not found")

        if task.comparison_report is None:
            raise PipelineException(
                f"Comparison report not available for task {task_id}"
            )

        return task.comparison_report

    def stop_pipeline(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)

        if task is None:
            raise PipelineException(f"Pipeline task {task_id} not found")

        if task.status != "running":
            return False

        task.request_stop()
        logger.info(f"Stop requested for pipeline task {task_id}")
        return True

    def list_tasks(self) -> List[PipelineStatusData]:
        with self._lock:
            tasks = list(self.tasks.values())

        result = []
        for task in tasks:
            stages = [
                PipelineStageData(
                    stage_name=name,
                    status=info["status"],
                    started_at=info["started_at"],
                    completed_at=info["completed_at"],
                    details=info["details"],
                )
                for name, info in task.stages.items()
            ]
            result.append(
                PipelineStatusData(
                    task_id=task.task_id,
                    status=task.status,
                    current_stage=task.current_stage,
                    stages=stages,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                    progress=task.get_progress(),
                )
            )

        return result

    def get_active_task_count(self) -> int:
        with self._lock:
            return sum(1 for t in self.tasks.values() if t.status == "running")
