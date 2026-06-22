"""
API请求模型
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingStartRequest(BaseModel):
    """启动训练请求"""
    total_episodes: int = Field(default=1000, ge=1, le=100000, description="总训练回合数")
    max_steps: int = Field(default=1000, ge=100, le=10000, description="每回合最大步数")
    learning_rate: float = Field(default=3e-4, gt=0, le=1, description="学习率")
    gamma: float = Field(default=0.99, ge=0, le=1, description="折扣因子")
    epsilon: float = Field(default=0.2, ge=0, le=1, description="PPO裁剪参数")
    initial_temperature: float = Field(default=1.0, gt=0, description="初始温度")
    temperature_decay: float = Field(default=0.995, gt=0, le=1, description="温度衰减率")
    min_temperature: float = Field(default=0.01, gt=0, description="最小温度")
    regularization_coef: float = Field(default=0.1, ge=0, description="正则化系数")
    env_name: str = Field(default="LunarLander-v2", description="环境名称")
    target_mode: Literal["random", "frozen_differentiable", "seed_based"] = Field(
        default="random",
        description="目标网络模式：random（随机不可微网络）、frozen_differentiable（可微网络量化后作为目标）、seed_based（由24种子生成不可微网络）"
    )
    target_seeds: Optional[List[int]] = Field(
        default=None,
        description="seed_based模式下的24个整数种子"
    )
    target_quantize_bits: int = Field(
        default=8,
        ge=1,
        le=32,
        description="frozen_differentiable模式下的量化比特数"
    )
    weight_copy_interval: int = Field(
        default=10,
        ge=1,
        description="周期性copy_weights_from的间隔（更新次数）"
    )
    harden_on_copy: bool = Field(
        default=True,
        description="copy_weights_from时是否执行argmax硬化"
    )
    auto_start_ga: bool = Field(
        default=False,
        description="训练完成且avg_reward_last_100 >= 200时，是否自动启动关联的GA任务"
    )
    ga_config: Optional["GeneticStartRequest"] = Field(
        default=None,
        description="自动启动GA任务时使用的配置，auto_start_ga为true时必填"
    )

    @field_validator('target_seeds')
    @classmethod
    def validate_target_seeds(cls, v, info):
        if v is not None:
            if len(v) != 24:
                raise ValueError('target_seeds must contain exactly 24 integers')
        if info.data.get('target_mode') == 'seed_based' and v is None:
            raise ValueError('target_seeds is required when target_mode is seed_based')
        return v

    @model_validator(mode='after')
    def validate_auto_start_ga_config(self):
        if self.auto_start_ga and self.ga_config is None:
            raise ValueError('ga_config is required when auto_start_ga is true')
        return self


class GeneticStartRequest(BaseModel):
    """启动遗传算法请求"""
    population_size: int = Field(default=50, ge=10, le=1000, description="种群大小")
    max_generations: int = Field(default=100, ge=1, le=10000, description="最大代数")
    mutation_rate: float = Field(default=0.1, ge=0, le=1, description="变异率")
    crossover_rate: float = Field(default=0.7, ge=0, le=1, description="交叉率")
    elite_size: int = Field(default=5, ge=1, le=50, description="精英保留数量")
    seed_range_min: int = Field(default=0, ge=0, description="种子最小值")
    seed_range_max: int = Field(default=10000, ge=1, description="种子最大值")
    target_fitness: float = Field(default=200.0, description="目标适应度")
    evaluation_episodes: int = Field(default=5, ge=1, le=100, description="每个个体的评估回合数")
    env_name: str = Field(default="LunarLander-v2", description="环境名称")
    alpha: float = Field(default=0.9, ge=0, le=1, description="双目标适应度权重：alpha * env_reward + (1-alpha) * weight_similarity")
    target_weights_path: Optional[str] = Field(default=None, description="PPO训练好的权重文件路径，用于计算权重相似度")
    target_seeds: Optional[List[int]] = Field(default=None, description="目标网络的24个种子，用于生成目标权重计算相似度")
    parent_task_id: Optional[str] = Field(
        default=None,
        description="父任务ID（通常是PPO训练任务ID），用于关联任务流水线"
    )

    @field_validator('seed_range_max')
    @classmethod
    def validate_seed_range(cls, v, info):
        if 'seed_range_min' in info.data and v <= info.data['seed_range_min']:
            raise ValueError('seed_range_max must be greater than seed_range_min')
        return v

    @field_validator('target_seeds')
    @classmethod
    def validate_target_seeds(cls, v):
        if v is not None and len(v) != 24:
            raise ValueError('target_seeds must contain exactly 24 integers (4 rows x 6 columns)')
        return v


class EvaluationRequest(BaseModel):
    """评估请求"""
    network_type: str = Field(
        default="differentiable",
        description="网络类型: differentiable, non_differentiable"
    )
    num_episodes: int = Field(default=10, ge=1, le=1000, description="评估回合数")
    model_path: Optional[str] = Field(default=None, description="模型路径（可选）")
    env_name: str = Field(default="LunarLander-v2", description="环境名称")
    
    @field_validator('network_type')
    @classmethod
    def validate_network_type(cls, v):
        allowed = ['differentiable', 'non_differentiable']
        if v not in allowed:
            raise ValueError(f'network_type must be one of {allowed}')
        return v


class ComparisonRequest(BaseModel):
    """对比评估请求"""
    num_episodes: int = Field(default=10, ge=1, le=1000, description="评估回合数")
    differentiable_model_path: Optional[str] = Field(default=None, description="可微网络模型路径")
    non_differentiable_seeds: Optional[List[int]] = Field(
        default=None,
        description="不可微网络种子（24个整数，4行6列）"
    )
    env_name: str = Field(default="LunarLander-v2", description="环境名称")
    
    @field_validator('non_differentiable_seeds')
    @classmethod
    def validate_seeds(cls, v):
        if v is not None and len(v) != 24:
            raise ValueError('non_differentiable_seeds must contain exactly 24 integers (4 rows x 6 columns)')
        return v


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    ppo: Optional[Dict[str, Any]] = Field(default=None, description="PPO配置")
    genetic: Optional[Dict[str, Any]] = Field(default=None, description="遗传算法配置")
    environment: Optional[Dict[str, Any]] = Field(default=None, description="环境配置")


class PageRequest(BaseModel):
    """分页请求"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页大小")


class PipelineStartRequest(BaseModel):
    ppo_config: TrainingStartRequest = Field(description="PPO训练配置")
    ga_config: GeneticStartRequest = Field(description="GA搜索配置")
    target_seeds: Optional[List[int]] = Field(
        default=None,
        description="可选目标种子（24个整数），用于GA种群初始化引导"
    )
    weight_similarity_coef: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="权重相似度在GA适应度中的系数"
    )

    @field_validator('target_seeds')
    @classmethod
    def validate_target_seeds(cls, v):
        if v is not None and len(v) != 24:
            raise ValueError('target_seeds must contain exactly 24 integers (4 rows x 6 columns)')
        return v


class VisualizationRequest(BaseModel):
    """可视化请求"""
    task_id: str = Field(description="任务ID")
    chart_type: str = Field(
        default="fitness_curve",
        description="图表类型: fitness_curve, dashboard, progress"
    )
    window_size: int = Field(default=10, ge=1, le=100, description="移动平均窗口大小")
    
    @field_validator('chart_type')
    @classmethod
    def validate_chart_type(cls, v):
        allowed = ['fitness_curve', 'dashboard', 'progress', 'comparison']
        if v not in allowed:
            raise ValueError(f'chart_type must be one of {allowed}')
        return v
