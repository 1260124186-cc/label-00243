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
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy
import random
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
        return Individual(seeds=self.seeds.copy(), fitness=self.fitness, generation=self.generation)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'seeds': self.seeds.tolist(),
            'fitness': self.fitness,
            'generation': self.generation
        }


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
        
    def lcg_random(self, seed: int, a: int, b: int, count: int) -> List[int]:
        """
        线性同余生成器 (Linear Congruential Generator)
        X_{n+1} = (a * X_n + b) mod m
        
        Args:
            seed: 初始种子
            a: 乘数参数（来自该行最后两个种子之一）
            b: 增量参数（来自该行最后两个种子之一）
            count: 生成数量
            
        Returns:
            随机整数列表
        """
        if a == 0 and b == 0:
            return [0] * count
            
        results = []
        x = seed
        for _ in range(count):
            x = (a * x + b) % self.modulus
            results.append(x)
        return results
    
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
        
        # 转换为浮点数（先转为有符号整数范围）
        weights = np.array(raw_values, dtype=np.float64)
        weights = weights - self.modulus / 2  # 中心化到 [-m/2, m/2]
        
        # 重塑并归一化（每层独立归一化）
        weights = weights.reshape(shape)
        weights = self.normalize_weights(weights)
        
        return weights
    
    def generate_weights_from_individual(
        self,
        individual: Individual,
        network_shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, torch.Tensor]:
        """
        从个体生成完整的网络权重
        
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
        result_weights = {}
        
        for layer_name, shape in network_shapes.items():
            accumulated_weight = np.zeros(shape)
            total_weight_sum = 0.0
            
            for row_idx in range(4):  # 4行
                row = seeds[row_idx]
                
                # 最后两个种子作为a和b
                a = int(row[4])
                b = int(row[5])
                
                # 第一列权值
                first_col_weight = self.FIRST_COL_WEIGHTS[row_idx]
                
                # 处理前4个种子
                for col_idx in range(4):
                    seed = int(row[col_idx])
                    if seed == 0:
                        continue  # 种子为0时不生成权重
                    
                    # 计算该位置的权重系数
                    if col_idx == 0:
                        # 第一列
                        weight_coef = first_col_weight
                    else:
                        # 后三列：[0.1, 0.01, 0.001] * 第一列权值
                        weight_coef = self.NEXT_COL_MULTIPLIERS[col_idx - 1] * first_col_weight
                    
                    # 生成权重
                    layer_weights = self.generate_layer_weights(seed, a, b, shape)
                    if layer_weights is not None:
                        accumulated_weight += weight_coef * layer_weights
                        total_weight_sum += weight_coef
            
            # 归一化最终权重
            if total_weight_sum > 0:
                accumulated_weight = accumulated_weight / total_weight_sum
                
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
        traversal_enabled: bool = True
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
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.seed_range = seed_range
        self.elite_return_probability = elite_return_probability
        self.traversal_enabled = traversal_enabled
        
        self.population: List[Individual] = []
        self.elite_archive: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.best_fitness = float('-inf')
        self.fitness_history: List[float] = []
        
        # 遍历计数器
        self.traversal_counter = 0
        
        self.weight_generator = WeightGenerator()
        
        logger.info(
            f"GeneticAlgorithm initialized: pop_size={population_size}, "
            f"mutation_rate={mutation_rate}, crossover_rate={crossover_rate}"
        )
    
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
        
        new_individual.fitness = float('-inf')  # 重置适应度
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
        evaluate_fn: callable,
        update_callback: Optional[callable] = None
    ) -> None:
        """
        评估整个种群
        
        Args:
            evaluate_fn: 评估函数 (Individual) -> float
            update_callback: 进度回调函数
        """
        for i, individual in enumerate(self.population):
            if individual.fitness == float('-inf'):
                individual.fitness = evaluate_fn(individual)
                
            if update_callback:
                update_callback(i, len(self.population), individual.fitness)
        
        # 记录最佳适应度
        best_in_gen = max(self.population, key=lambda x: x.fitness)
        self.fitness_history.append(best_in_gen.fitness)
        
        logger.info(
            f"Generation {self.generation}: "
            f"Best fitness = {best_in_gen.fitness:.2f}, "
            f"Avg fitness = {np.mean([ind.fitness for ind in self.population]):.2f}"
        )
    
    def run(
        self,
        evaluate_fn: callable,
        max_generations: int = 100,
        target_fitness: float = 200.0,
        generation_callback: Optional[callable] = None
    ) -> Individual:
        """
        运行遗传算法
        
        Args:
            evaluate_fn: 评估函数
            max_generations: 最大代数
            target_fitness: 目标适应度
            generation_callback: 每代结束回调
            
        Returns:
            最佳个体
        """
        self.initialize_population()
        
        for gen in range(max_generations):
            self.evaluate_population(evaluate_fn)
            
            if generation_callback:
                generation_callback(self.generation, self.best_individual, self.best_fitness)
            
            # 检查是否达到目标
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
            'traversal_counter': self.traversal_counter
        }
