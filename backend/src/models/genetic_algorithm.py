"""
遗传算法模块
严格按照Prompt规格实现

个体结构: 24个整数种子，排列为4行6列
- 第一列权值：[0.4, 0.3, 0.2, 0.1]
- 每行后三个权值：[0.1, 0.01, 0.001] * 该行第一列权值
- 每行最后两个种子作为随机算法的a和b

权重生成：
- 同余随机整数生成算法
- 每层独立的最大值归一化
- 种子为0时不生成权重，也不参与加权
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from copy import deepcopy
import random
from concurrent.futures import Executor, as_completed
from loguru import logger


@dataclass
class Individual:
    """
    遗传算法个体
    包含24个整数种子，排列为4行6列

    结构:
    Row 0: [seed0, seed1, seed2, seed3, a0, b0]  权重0.4
    Row 1: [seed4, seed5, seed6, seed7, a1, b1]  权重0.3
    Row 2: [seed8, seed9, seed10, seed11, a2, b2] 权重0.2
    Row 3: [seed12, seed13, seed14, seed15, a3, b3] 权重0.1

    每行前4个是种子（第一列+后三个），最后两个是LCG参数a和b
    """
    seeds: np.ndarray  # shape: (4, 6)
    fitness: float = float('-inf')
    env_reward: float = float('-inf')
    weight_similarity: float = float('-inf')
    generation: int = 0

    def __post_init__(self):
        if self.seeds.shape != (4, 6):
            raise ValueError(f"Seeds must have shape (4, 6), got {self.seeds.shape}")

    @classmethod
    def create_random(cls, seed_range: Tuple[int, int] = (0, 10000)) -> 'Individual':
        """创建随机个体"""
        seeds = np.random.randint(seed_range[0], seed_range[1], size=(4, 6))
        return cls(seeds=seeds)

    @classmethod
    def create_from_list(cls, seed_list: List[int]) -> 'Individual':
        """从列表创建个体"""
        if len(seed_list) != 24:
            raise ValueError(f"Expected 24 seeds, got {len(seed_list)}")
        seeds = np.array(seed_list).reshape(4, 6)
        return cls(seeds=seeds)

    @classmethod
    def create_traversal(cls, first_seed: int) -> 'Individual':
        """
        创建遍历个体
        第一行第一个种子为指定值，其他种子全为0
        """
        seeds = np.zeros((4, 6), dtype=np.int64)
        seeds[0, 0] = first_seed
        return cls(seeds=seeds)

    def to_list(self) -> List[int]:
        """转换为列表"""
        return self.seeds.flatten().tolist()

    def copy(self) -> 'Individual':
        """深拷贝"""
        return Individual(
            seeds=self.seeds.copy(),
            fitness=self.fitness,
            env_reward=self.env_reward,
            weight_similarity=self.weight_similarity,
            generation=self.generation
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'seeds': self.seeds.tolist(),
            'fitness': self.fitness,
            'env_reward': self.env_reward,
            'weight_similarity': self.weight_similarity,
            'generation': self.generation
        }

    def reset_fitness(self) -> None:
        """重置适应度相关字段"""
        self.fitness = float('-inf')
        self.env_reward = float('-inf')
        self.weight_similarity = float('-inf')


class WeightGenerator:
    """
    权重生成器
    严格按照Prompt规格实现

    - 同余随机整数生成算法，每行使用最后两个种子作为a和b
    - 每层独立的最大值归一化
    - 种子为0时不生成权重，也不参与加权

    加权融合:
    - 第一列权值：[0.4, 0.3, 0.2, 0.1]
    - 每行后三个权值：[0.1, 0.01, 0.001] * 该行第一列权值
    """

    # 第一列权值（4行）
    FIRST_COL_WEIGHTS = [0.4, 0.3, 0.2, 0.1]

    # 后三列的权值乘数
    NEXT_COL_MULTIPLIERS = [0.1, 0.01, 0.001]

    def __init__(self, modulus: int = 2**31):
        """
        Args:
            modulus: 同余算法的模数
        """
        self.modulus = modulus

    def lcg_random(self, seed: int, a: int, b: int, count: int) -> np.ndarray:
        """
        线性同余生成器 (Linear Congruential Generator) - 优化版本
        X_{n+1} = (a * X_n + b) mod m

        优化策略:
        - 预分配 numpy 数组直接写入（避免 list.append 开销）
        - 局部变量绑定，减少属性查找
        - a=1 时使用 numpy 向量化等差数列
        - a=b=0 时直接返回全零数组

        Args:
            seed: 初始种子
            a: 乘数参数（来自该行最后两个种子之一）
            b: 增量参数（来自该行最后两个种子之一）
            count: 生成数量

        Returns:
            随机整数数组 (np.ndarray, dtype=np.int64)
        """
        if count <= 0:
            return np.array([], dtype=np.int64)
        if a == 0 and b == 0:
            return np.zeros(count, dtype=np.int64)

        m = self.modulus

        if a == 1:
            ks = np.arange(1, count + 1, dtype=np.int64)
            return ((seed % m) + (b % m) * ks) % m

        result = np.empty(count, dtype=np.int64)
        x = seed % m
        a_mod = a % m
        b_mod = b % m
        for i in range(count):
            x = (a_mod * x + b_mod) % m
            result[i] = x
        return result

    def lcg_random_batch(
        self,
        seeds: np.ndarray,
        a_params: np.ndarray,
        b_params: np.ndarray,
        count: int
    ) -> np.ndarray:
        """
        批量 LCG 生成：同时生成 N 条独立的 LCG 序列
        每条序列长度为 count，返回形状 (N, count) 的 int64 数组。

        Args:
            seeds: (N,) 初始种子数组
            a_params: (N,) LCG 参数 a
            b_params: (N,) LCG 参数 b
            count: 每条序列生成数量

        Returns:
            (N, count) 随机整数数组
        """
        N = len(seeds)
        if N == 0 or count <= 0:
            return np.empty((N, count), dtype=np.int64)

        m = self.modulus
        out = np.empty((N, count), dtype=np.int64)

        x = seeds.astype(np.int64) % m
        a_mod = a_params.astype(np.int64) % m
        b_mod = b_params.astype(np.int64) % m

        zero_ab = (a_mod == 0) & (b_mod == 0)
        a_eq_1 = (a_mod == 1) & ~zero_ab

        if zero_ab.any():
            out[zero_ab] = 0

        if a_eq_1.any():
            ks = np.arange(1, count + 1, dtype=np.int64)
            x_sel = x[a_eq_1][:, None]
            b_sel = b_mod[a_eq_1][:, None]
            out[a_eq_1] = (x_sel + b_sel * ks[None, :]) % m

        normal = ~zero_ab & ~a_eq_1
        if normal.any():
            xn = x[normal].copy()
            an = a_mod[normal]
            bn = b_mod[normal]
            out_n = out[normal]
            if count == 1:
                xn = (an * xn + bn) % m
                out_n[:, 0] = xn
            else:
                for i in range(count):
                    xn = (an * xn + bn) % m
                    out_n[:, i] = xn
            out[normal] = out_n

        return out

    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        每层独立的最大值归一化

        Args:
            weights: 原始权重

        Returns:
            归一化后的权重 (范围 [-1, 1])
        """
        max_val = np.abs(weights).max()
        if max_val > 0:
            return weights / max_val
        return weights

    def generate_layer_weights(
        self,
        seed: int,
        a: int,
        b: int,
        shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        为单个层生成权重

        Args:
            seed: 种子
            a: LCG参数a
            b: LCG参数b
            shape: 权重形状

        Returns:
            生成的权重矩阵，如果seed为0则返回None
        """
        if seed == 0:
            return None

        total_size = int(np.prod(shape))
        raw_values = self.lcg_random(seed, a, b, total_size)

        weights = raw_values.astype(np.float64) - self.modulus / 2.0
        weights = weights.reshape(shape)
        weights = self.normalize_weights(weights)

        return weights

    def generate_weights_from_individual(
        self,
        individual: Individual,
        network_shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, torch.Tensor]:
        """
        从个体生成完整的网络权重 - 批量向量化优化版本

        对每层一次性批量生成所有 (4行×4列)=16 个 LCG 序列，
        消除 Python 内部循环开销。

        严格按照Prompt规格：
        - 每行前4个种子（第一列+后三个），最后两个是a和b
        - 第一列权值：[0.4, 0.3, 0.2, 0.1]
        - 每行后三个权值：[0.1, 0.01, 0.001] * 该行第一列权值
        - 种子为0时不生成权重，也不参与加权

        Args:
            individual: 遗传算法个体
            network_shapes: 网络各层权重形状

        Returns:
            网络权重字典
        """
        seeds = individual.seeds  # shape: (4, 6)
        N_COMBOS = 16  # 4 rows × 4 cols

        first_col_w = np.array(self.FIRST_COL_WEIGHTS, dtype=np.float64)
        next_col_mul = np.array(self.NEXT_COL_MULTIPLIERS, dtype=np.float64)

        coefs = np.zeros((4, 4), dtype=np.float64)
        coefs[:, 0] = first_col_w
        coefs[:, 1:] = next_col_mul[None, :] * first_col_w[:, None]
        coefs_flat = coefs.reshape(-1)  # (16,)

        flat_seeds = np.zeros(N_COMBOS, dtype=np.int64)
        flat_a = np.zeros(N_COMBOS, dtype=np.int64)
        flat_b = np.zeros(N_COMBOS, dtype=np.int64)
        for row_idx in range(4):
            row_a = int(seeds[row_idx, 4])
            row_b = int(seeds[row_idx, 5])
            for col_idx in range(4):
                i = row_idx * 4 + col_idx
                flat_seeds[i] = int(seeds[row_idx, col_idx])
                flat_a[i] = row_a
                flat_b[i] = row_b

        zero_mask = (flat_seeds == 0)
        coefs_flat[zero_mask] = 0.0

        result_weights = {}

        for layer_name, shape in network_shapes.items():
            total_size = int(np.prod(shape))
            accumulated_weight = np.zeros(shape, dtype=np.float64)
            total_weight_sum = float(np.sum(coefs_flat))

            if total_weight_sum == 0:
                result_weights[layer_name] = torch.zeros(shape, dtype=torch.float32)
                continue

            active_mask = ~zero_mask
            n_active = int(active_mask.sum())

            if n_active == 0:
                result_weights[layer_name] = torch.zeros(shape, dtype=torch.float32)
                continue

            active_seeds = flat_seeds[active_mask]
            active_a = flat_a[active_mask]
            active_b = flat_b[active_mask]
            active_coefs = coefs_flat[active_mask]

            batch_raw = self.lcg_random_batch(active_seeds, active_a, active_b, total_size)

            batch_float = batch_raw.astype(np.float64) - self.modulus / 2.0
            batch_shaped = batch_float.reshape(n_active, *shape)

            for i in range(n_active):
                w = batch_shaped[i]
                max_val = np.abs(w).max()
                if max_val > 0:
                    w = w / max_val
                accumulated_weight += active_coefs[i] * w

            accumulated_weight /= total_weight_sum

            result_weights[layer_name] = torch.tensor(accumulated_weight, dtype=torch.float32)

        return result_weights

    def apply_weights_to_network(
        self,
        network: nn.Module,
        weights: Dict[str, torch.Tensor]
    ) -> None:
        """
        将生成的权重应用到网络

        Args:
            network: 目标网络
            weights: 权重字典
        """
        state_dict = network.state_dict()

        for name, param in weights.items():
            if name in state_dict:
                if state_dict[name].shape == param.shape:
                    state_dict[name] = param
                else:
                    logger.warning(
                        f"Shape mismatch for {name}: "
                        f"expected {state_dict[name].shape}, got {param.shape}"
                    )

        network.load_state_dict(state_dict)

    def compute_weight_mse(
        self,
        generated_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor]
    ) -> float:
        """
        计算生成权重与目标权重之间的均方误差 (MSE)

        Args:
            generated_weights: 生成的权重字典
            target_weights: 目标权重字典

        Returns:
            MSE 值
        """
        total_mse = 0.0
        total_params = 0

        for name in generated_weights:
            if name not in target_weights:
                continue

            gen_w = generated_weights[name]
            tgt_w = target_weights[name]

            if gen_w.shape != tgt_w.shape:
                logger.warning(f"Shape mismatch for {name}: {gen_w.shape} vs {tgt_w.shape}")
                continue

            mse = torch.mean((gen_w - tgt_w) ** 2).item()
            num_params = gen_w.numel()
            total_mse += mse * num_params
            total_params += num_params

        if total_params == 0:
            return 0.0

        return total_mse / total_params

    def compute_weight_similarity(
        self,
        generated_weights: Dict[str, torch.Tensor],
        target_weights: Dict[str, torch.Tensor]
    ) -> float:
        """
        计算权重相似度 = 1 - MSE(generated_weights, target_weights)

        Args:
            generated_weights: 生成的权重字典
            target_weights: 目标权重字典

        Returns:
            权重相似度，范围 (-inf, 1]
        """
        mse = self.compute_weight_mse(generated_weights, target_weights)
        return 1.0 - mse

    def compute_combined_fitness(
        self,
        env_reward: float,
        weight_similarity: float,
        alpha: float = 0.9
    ) -> float:
        """
        计算综合适应度 = alpha * env_reward + (1 - alpha) * weight_similarity

        Args:
            env_reward: 环境奖励适应度
            weight_similarity: 权重相似度
            alpha: 环境奖励的权重系数

        Returns:
            综合适应度
        """
        return alpha * env_reward + (1 - alpha) * weight_similarity


class GeneticAlgorithm:
    """
    遗传算法实现
    严格按照Prompt规格

    变异：
    - 随机改变一个种子
    - 随机交换相邻行

    交叉：
    - 至少交换一行对应行

    精英保留：
    - 保存获得历史更高适应度的个体及其适应度
    - 有概率回到场上，适应度越大概率越高

    遍历：
    - 生成第一行第一个种子按代数逐渐递增、其他种子全为0的个体
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        seed_range: Tuple[int, int] = (0, 10000),
        elite_return_probability: float = 0.1,
        traversal_enabled: bool = True,
        alpha: float = 0.9,
        target_weights: Optional[Dict[str, torch.Tensor]] = None,
        network_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    ):
        """
        Args:
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_size: 精英保留数量
            seed_range: 种子范围
            elite_return_probability: 精英回归概率基数
            traversal_enabled: 是否启用遍历个体
            alpha: 双目标适应度权重：alpha * env_reward + (1-alpha) * weight_similarity
            target_weights: 目标权重字典，用于计算权重相似度（PPO 训练得到的权重）
            network_shapes: 网络各层权重形状，用于生成个体权重
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed_range = seed_range
        self.elite_return_probability = elite_return_probability
        self.traversal_enabled = traversal_enabled
        self.alpha = alpha
        self.target_weights = target_weights
        self.network_shapes = network_shapes

        self.population: List[Individual] = []
        self.elite_archive: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.best_fitness = float('-inf')
        self.fitness_history: List[float] = []
        self.env_reward_history: List[float] = []
        self.weight_similarity_history: List[float] = []

        # 遍历计数器
        self.traversal_counter = 0

        self.weight_generator = WeightGenerator()

        logger.info(
            f"GeneticAlgorithm initialized: pop_size={population_size}, "
            f"mutation_rate={mutation_rate}, crossover_rate={crossover_rate}, "
            f"alpha={alpha}, target_weights={'enabled' if target_weights else 'disabled'}"
        )

    def set_target_weights(
        self,
        target_weights: Dict[str, torch.Tensor],
        network_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    ) -> None:
        """
        设置目标权重用于权重相似度计算

        Args:
            target_weights: 目标权重字典
            network_shapes: 网络各层权重形状
        """
        self.target_weights = target_weights
        if network_shapes is not None:
            self.network_shapes = network_shapes
        logger.info(f"Target weights set, alpha={self.alpha}")

    def _compute_weight_similarity(self, individual: Individual) -> float:
        """
        计算个体的权重相似度

        Args:
            individual: 遗传算法个体

        Returns:
            权重相似度，若无目标权重则返回 0.0
        """
        if self.target_weights is None or self.network_shapes is None:
            return 0.0

        generated_weights = self.weight_generator.generate_weights_from_individual(
            individual, self.network_shapes
        )
        return self.weight_generator.compute_weight_similarity(
            generated_weights, self.target_weights
        )

    def _evaluate_single(
        self,
        individual: Individual,
        env_reward: float
    ) -> float:
        """
        计算单个个体的完整适应度（双目标）

        Args:
            individual: 遗传算法个体
            env_reward: 环境奖励

        Returns:
            综合适应度
        """
        weight_sim = self._compute_weight_similarity(individual)
        individual.env_reward = env_reward
        individual.weight_similarity = weight_sim

        if self.target_weights is None:
            individual.fitness = env_reward
        else:
            individual.fitness = self.weight_generator.compute_combined_fitness(
                env_reward, weight_sim, self.alpha
            )

        return individual.fitness

    def initialize_population(self) -> None:
        """初始化种群"""
        self.population = []

        for i in range(self.population_size):
            if self.traversal_enabled and i == 0:
                # 第一个个体为遍历个体
                individual = Individual.create_traversal(self.traversal_counter)
            else:
                individual = Individual.create_random(self.seed_range)
            self.population.append(individual)

        self.generation = 0
        logger.info(f"Population initialized with {self.population_size} individuals")

    def mutate(self, individual: Individual) -> Individual:
        """
        变异操作（严格按照Prompt）
        - 随机改变一个种子
        - 随机交换相邻行

        Args:
            individual: 原始个体

        Returns:
            变异后的新个体
        """
        new_individual = individual.copy()
        seeds = new_individual.seeds

        mutation_type = random.random()

        if mutation_type < 0.5:
            # 随机改变一个种子
            row = random.randint(0, 3)
            col = random.randint(0, 5)
            seeds[row, col] = random.randint(self.seed_range[0], self.seed_range[1])
        else:
            # 随机交换相邻行
            row = random.randint(0, 2)  # 0, 1, 2 可以与下一行交换
            seeds[[row, row + 1]] = seeds[[row + 1, row]]

        new_individual.reset_fitness()  # 重置适应度
        return new_individual

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        交叉操作（严格按照Prompt）
        至少交换一行对应行

        Args:
            parent1: 父代1
            parent2: 父代2

        Returns:
            子代个体
        """
        child_seeds = parent1.seeds.copy()

        # 至少交换一行
        num_rows_to_swap = random.randint(1, 2)  # 1-2行
        rows_to_swap = random.sample(range(4), num_rows_to_swap)

        for row in rows_to_swap:
            child_seeds[row] = parent2.seeds[row].copy()

        return Individual(seeds=child_seeds, generation=self.generation + 1)

    def select_parents(self) -> Tuple[Individual, Individual]:
        """
        锦标赛选择

        Returns:
            两个父代个体
        """
        tournament_size = min(5, len(self.population))

        def tournament_select():
            competitors = random.sample(self.population, tournament_size)
            return max(competitors, key=lambda x: x.fitness)

        return tournament_select(), tournament_select()

    def update_elite_archive(self, individual: Individual) -> None:
        """
        更新精英档案
        保存获得历史更高适应度的个体及其适应度

        Args:
            individual: 要考虑加入档案的个体
        """
        if individual.fitness > self.best_fitness:
            self.best_fitness = individual.fitness
            self.best_individual = individual.copy()
            logger.info(f"New best individual found! Fitness: {individual.fitness:.2f}")

        # 检查是否应该加入精英档案
        if len(self.elite_archive) < self.elite_size:
            self.elite_archive.append(individual.copy())
        else:
            # 替换最弱的精英
            min_fitness = min(e.fitness for e in self.elite_archive)
            if individual.fitness > min_fitness:
                weakest_idx = next(
                    i for i, e in enumerate(self.elite_archive)
                    if e.fitness == min_fitness
                )
                self.elite_archive[weakest_idx] = individual.copy()

    def maybe_return_elite(self) -> Optional[Individual]:
        """
        根据概率返回精英个体到种群
        适应度越高，返回概率越大

        Returns:
            返回的精英个体或None
        """
        if not self.elite_archive:
            return None

        # 计算返回概率
        max_fitness = max(e.fitness for e in self.elite_archive)
        min_fitness = min(e.fitness for e in self.elite_archive)
        fitness_range = max_fitness - min_fitness if max_fitness != min_fitness else 1.0

        for elite in self.elite_archive:
            # 归一化适应度到 [0, 1]
            normalized_fitness = (elite.fitness - min_fitness) / fitness_range
            return_prob = self.elite_return_probability * (1 + normalized_fitness)

            if random.random() < return_prob:
                logger.debug(f"Elite returning to population with fitness {elite.fitness:.2f}")
                return elite.copy()

        return None

    def evolve(self) -> List[Individual]:
        """
        进化一代

        Returns:
            新一代种群
        """
        new_population = []

        # 精英保留
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        for i in range(min(self.elite_size, len(sorted_pop))):
            elite = sorted_pop[i].copy()
            elite.generation = self.generation + 1
            new_population.append(elite)
            self.update_elite_archive(elite)

        # 可能返回精英
        returned_elite = self.maybe_return_elite()
        if returned_elite is not None:
            new_population.append(returned_elite)

        # 遍历个体
        if self.traversal_enabled:
            self.traversal_counter += 1
            traversal_individual = Individual.create_traversal(self.traversal_counter)
            new_population.append(traversal_individual)

        # 生成新个体
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()

            # 交叉
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1.copy()
                child.generation = self.generation + 1

            # 变异
            if random.random() < self.mutation_rate:
                child = self.mutate(child)

            new_population.append(child)

        self.population = new_population[:self.population_size]
        self.generation += 1

        return self.population

    def evaluate_population(
        self,
        evaluate_fn: Optional[Callable[["Individual"], float]] = None,
        update_callback: Optional[Callable[[int, int, float], None]] = None,
        parallel_executor: Optional[Executor] = None,
        parallel_worker_fn: Optional[Callable[[Any], float]] = None,
        parallel_args_builder: Optional[Callable[["Individual"], Any]] = None,
    ) -> None:
        """
        评估整个种群 - 支持串行和并行两种模式

        串行模式 (parallel_executor is None):
            使用 evaluate_fn(individual) 逐个评估。

        并行模式 (parallel_executor 存在):
            对每个待评估个体：
              args = parallel_args_builder(individual)
              提交到进程池: executor.submit(parallel_worker_fn, args)
            完成后回填 individual.fitness。

        Args:
            evaluate_fn: 串行评估函数 (Individual) -> float，返回环境奖励
            update_callback: 进度回调函数 (idx, total, fitness) -> None
            parallel_executor: 并行执行器（如 ProcessPoolExecutor）
            parallel_worker_fn: 顶层 worker 函数，用于子进程执行，返回环境奖励
            parallel_args_builder: 为 worker 构造参数的函数
        """
        pop_size = len(self.population)
        pending_indices: List[int] = []
        pending_individuals: List[Individual] = []

        for i, individual in enumerate(self.population):
            if individual.fitness == float("-inf"):
                pending_indices.append(i)
                pending_individuals.append(individual)

        if parallel_executor is not None and pending_individuals:
            if parallel_worker_fn is None or parallel_args_builder is None:
                raise ValueError(
                    "parallel_worker_fn and parallel_args_builder are required "
                    "when parallel_executor is provided"
                )

            fut_to_idx = {}
            for idx_in_pending, (orig_idx, ind) in enumerate(
                zip(pending_indices, pending_individuals)
            ):
                args = parallel_args_builder(ind)
                fut = parallel_executor.submit(parallel_worker_fn, args)
                fut_to_idx[fut] = (orig_idx, idx_in_pending)

            completed_count = 0
            total_pending = len(pending_individuals)
            for fut in as_completed(fut_to_idx):
                orig_idx, _ = fut_to_idx[fut]
                env_reward = fut.result()
                self._evaluate_single(self.population[orig_idx], env_reward)
                completed_count += 1
                if update_callback:
                    update_callback(orig_idx, pop_size, self.population[orig_idx].fitness)
        else:
            for i, individual in enumerate(self.population):
                if individual.fitness == float("-inf"):
                    assert evaluate_fn is not None, "evaluate_fn required in serial mode"
                    env_reward = evaluate_fn(individual)
                    self._evaluate_single(individual, env_reward)

                if update_callback:
                    update_callback(i, pop_size, individual.fitness)

        best_in_gen = max(self.population, key=lambda x: x.fitness)
        self.fitness_history.append(best_in_gen.fitness)
        self.env_reward_history.append(best_in_gen.env_reward)
        self.weight_similarity_history.append(best_in_gen.weight_similarity)

        if self.target_weights is not None:
            logger.info(
                f"Generation {self.generation}: "
                f"Best fitness = {best_in_gen.fitness:.2f} "
                f"(env_reward={best_in_gen.env_reward:.2f}, "
                f"weight_sim={best_in_gen.weight_similarity:.4f}), "
                f"Avg fitness = {np.mean([ind.fitness for ind in self.population]):.2f}"
            )
        else:
            logger.info(
                f"Generation {self.generation}: "
                f"Best fitness = {best_in_gen.fitness:.2f}, "
                f"Avg fitness = {np.mean([ind.fitness for ind in self.population]):.2f}"
            )

    def run(
        self,
        evaluate_fn: Optional[Callable[["Individual"], float]] = None,
        max_generations: int = 100,
        target_fitness: float = 200.0,
        generation_callback: Optional[Callable[[int, Optional["Individual"], float], None]] = None,
        update_callback: Optional[Callable[[int, int, float], None]] = None,
        parallel_executor: Optional[Executor] = None,
        parallel_worker_fn: Optional[Callable[[Any], float]] = None,
        parallel_args_builder: Optional[Callable[["Individual"], Any]] = None,
    ) -> Optional["Individual"]:
        """
        运行遗传算法 - 支持串行/并行评估

        Args:
            evaluate_fn: 串行评估函数 (Individual) -> float
            max_generations: 最大代数
            target_fitness: 目标适应度
            generation_callback: 每代结束回调 (gen, best_ind, best_fit)
            update_callback: 个体评估进度回调 (idx, total, fitness)
            parallel_executor: 并行执行器（ProcessPoolExecutor 等）
            parallel_worker_fn: 顶层 worker 评估函数
            parallel_args_builder: 从 Individual 构造 worker 参数的函数

        Returns:
            最佳个体
        """
        self.initialize_population()

        for gen in range(max_generations):
            self.evaluate_population(
                evaluate_fn=evaluate_fn,
                update_callback=update_callback,
                parallel_executor=parallel_executor,
                parallel_worker_fn=parallel_worker_fn,
                parallel_args_builder=parallel_args_builder,
            )

            if generation_callback:
                generation_callback(self.generation, self.best_individual, self.best_fitness)

            if self.best_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at generation {gen + 1}!")
                break

            self.evolve()

        return self.best_individual

    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'elite_archive_size': len(self.elite_archive),
            'fitness_history': self.fitness_history,
            'env_reward_history': self.env_reward_history,
            'weight_similarity_history': self.weight_similarity_history,
            'alpha': self.alpha,
            'target_weights_enabled': self.target_weights is not None,
            'traversal_counter': self.traversal_counter
        }
