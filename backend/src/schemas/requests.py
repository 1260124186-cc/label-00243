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


class PPOConfigUpdate(BaseModel):
    """PPO配置更新"""
    learning_rate: Optional[float] = Field(default=None, gt=0, le=1, description="学习率")
    gamma: Optional[float] = Field(default=None, ge=0, le=1, description="折扣因子")
    epsilon: Optional[float] = Field(default=None, ge=0, le=1, description="PPO裁剪参数")
    epochs: Optional[int] = Field(default=None, ge=1, le=1000, description="迭代次数")
    batch_size: Optional[int] = Field(default=None, ge=1, le=10000, description="批次大小")
    initial_temperature: Optional[float] = Field(default=None, gt=0, description="初始温度")
    temperature_decay: Optional[float] = Field(default=None, gt=0, le=1, description="温度衰减率")
    min_temperature: Optional[float] = Field(default=None, gt=0, description="最小温度")
    regularization_coef: Optional[float] = Field(default=None, ge=0, description="正则化系数")


class GeneticConfigUpdate(BaseModel):
    """遗传算法配置更新"""
    population_size: Optional[int] = Field(default=None, ge=10, le=1000, description="种群大小")
    mutation_rate: Optional[float] = Field(default=None, ge=0, le=1, description="变异率")
    crossover_rate: Optional[float] = Field(default=None, ge=0, le=1, description="交叉率")
    elite_size: Optional[int] = Field(default=None, ge=1, le=50, description="精英保留数量")
    max_generations: Optional[int] = Field(default=None, ge=1, le=10000, description="最大代数")
    seed_range_min: Optional[int] = Field(default=None, ge=0, description="种子最小值")
    seed_range_max: Optional[int] = Field(default=None, ge=1, description="种子最大值")
    alpha: Optional[float] = Field(default=None, ge=0, le=1, description="双目标适应度权重")


class EnvironmentConfigUpdate(BaseModel):
    """环境配置更新"""
    default_env: Optional[str] = Field(default=None, min_length=1, description="默认环境名称")
    max_steps: Optional[int] = Field(default=None, ge=100, le=10000, description="每回合最大步数")
    total_episodes: Optional[int] = Field(default=None, ge=1, le=100000, description="总训练回合数")
    max_concurrent_training_tasks: Optional[int] = Field(default=None, ge=1, le=100, description="训练任务并发上限")
    max_concurrent_genetic_tasks: Optional[int] = Field(default=None, ge=1, le=100, description="遗传算法任务并发上限")


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    ppo: Optional[PPOConfigUpdate] = Field(default=None, description="PPO配置")
    genetic: Optional[GeneticConfigUpdate] = Field(default=None, description="遗传算法配置")
    environment: Optional[EnvironmentConfigUpdate] = Field(default=None, description="环境配置")

    @model_validator(mode='after')
    def check_at_least_one_config(self):
        if self.ppo is None and self.genetic is None and self.environment is None:
            raise ValueError("At least one configuration category must be provided")
        
        has_updates = False
        if self.ppo is not None:
            ppo_dict = self.ppo.model_dump(exclude_none=True)
            if ppo_dict:
                has_updates = True
        if self.genetic is not None:
            genetic_dict = self.genetic.model_dump(exclude_none=True)
            if genetic_dict:
                has_updates = True
        if self.environment is not None:
            env_dict = self.environment.model_dump(exclude_none=True)
            if env_dict:
                has_updates = True
        
        if not has_updates:
            raise ValueError("At least one configuration value must be provided")
        
        return self


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
    """可视化生成请求"""
    chart_type: Literal["fitness_curve", "dashboard", "progress", "comparison"] = Field(
        description="图表类型: fitness_curve(适应度曲线), dashboard(训练仪表板), progress(遗传算法进度), comparison(对比图)"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="任务ID（与raw_data二选一，优先使用task_id从对应服务拉取数据）"
    )
    task_type: Optional[Literal["training", "genetic"]] = Field(
        default=None,
        description="任务类型：training(训练任务)、genetic(遗传任务)，当task_id存在时可自动推断，无法推断时需显式指定"
    )
    raw_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="原始数据（与task_id二选一）。键说明：\n- fitness_curve/progress: {fitness_history: List[float], avg_fitness_history?: List[float]}\n- dashboard: {episode_rewards: List[float], policy_losses?: List[float], value_losses?: List[float], temperatures?: List[float]}\n- comparison: {diff_rewards: List[float], non_diff_rewards: List[float]}"
    )
    window_size: int = Field(default=10, ge=1, le=500, description="移动平均窗口大小")
    save_to_plots: bool = Field(default=False, description="是否将图片保存到plots/目录")
    format: Literal["base64", "file_url", "both"] = Field(
        default="base64",
        description="输出格式：base64(仅Base64编码)、file_url(仅文件URL)、both(两者都返回)"
    )
    title: Optional[str] = Field(default=None, description="图表标题（可选，不传则自动生成）")
    xlabel: Optional[str] = Field(default=None, description="X轴标签（可选）")
    ylabel: Optional[str] = Field(default=None, description="Y轴标签（可选）")
    show_avg: bool = Field(default=True, description="fitness_curve下是否显示移动平均线")

    @model_validator(mode='after')
    def validate_task_id_or_raw_data(self):
        if self.task_id is None and self.raw_data is None:
            raise ValueError("Either task_id or raw_data must be provided")
        if self.task_id is not None and self.raw_data is not None:
            raise ValueError("Only one of task_id or raw_data should be provided")
        return self

    @model_validator(mode='after')
    def validate_raw_data_required_keys(self):
        if self.raw_data is None:
            return self
        ct = self.chart_type
        keys = list(self.raw_data.keys())
        if ct == "fitness_curve":
            if "fitness_history" not in self.raw_data:
                raise ValueError("raw_data must contain 'fitness_history' for fitness_curve chart")
        elif ct == "progress":
            if "fitness_history" not in self.raw_data:
                raise ValueError("raw_data must contain 'fitness_history' for progress chart")
        elif ct == "dashboard":
            if "episode_rewards" not in self.raw_data:
                raise ValueError("raw_data must contain 'episode_rewards' for dashboard chart")
        elif ct == "comparison":
            if "diff_rewards" not in self.raw_data or "non_diff_rewards" not in self.raw_data:
                raise ValueError("raw_data must contain 'diff_rewards' and 'non_diff_rewards' for comparison chart")
        return self


class VisualizationComparisonQuery(BaseModel):
    """可视化对比查询参数（用于GET /comparison）"""
    differentiable_task_id: Optional[str] = Field(
        default=None,
        description="可微网络（PPO训练）任务ID，与genetic_task_id配对使用"
    )
    genetic_task_id: Optional[str] = Field(
        default=None,
        description="不可微网络（遗传算法）任务ID，与differentiable_task_id配对使用"
    )
    differentiable_model_path: Optional[str] = Field(
        default=None,
        description="可微网络模型文件路径（与differentiable_task_id二选一）"
    )
    genetic_seeds: Optional[List[int]] = Field(
        default=None,
        description="不可微网络的24个种子（与genetic_task_id二选一）"
    )
    diff_rewards: Optional[List[float]] = Field(
        default=None,
        description="可微网络的评估奖励列表（直接传评估结果数据）"
    )
    non_diff_rewards: Optional[List[float]] = Field(
        default=None,
        description="不可微网络的评估奖励列表（直接传评估结果数据）"
    )
    num_episodes: int = Field(default=10, ge=1, le=500, description="从task_id/模型/种子生成评估数据时的评估回合数")
    format: Literal["base64", "file_url", "both"] = Field(
        default="base64",
        description="输出格式：base64(仅Base64编码)、file_url(仅文件URL)、both(两者都返回)"
    )
    save_to_plots: bool = Field(default=False, description="是否将图片保存到plots/目录")
    title: Optional[str] = Field(default=None, description="图表标题（可选）")

    @model_validator(mode='after')
    def validate_data_source(self):
        has_tasks = self.differentiable_task_id is not None and self.genetic_task_id is not None
        has_model_seeds = self.differentiable_model_path is not None and self.genetic_seeds is not None
        has_rewards = self.diff_rewards is not None and self.non_diff_rewards is not None

        sources = sum([has_tasks, has_model_seeds, has_rewards])
        if sources == 0:
            raise ValueError(
                "Must provide one of: "
                "(differentiable_task_id + genetic_task_id), "
                "(differentiable_model_path + genetic_seeds), "
                "or (diff_rewards + non_diff_rewards)"
            )
        if sources > 1:
            raise ValueError("Only one data source combination should be provided")
        return self

    @field_validator('genetic_seeds')
    @classmethod
    def validate_genetic_seeds(cls, v):
        if v is not None and len(v) != 24:
            raise ValueError('genetic_seeds must contain exactly 24 integers (4 rows x 6 columns)')
        return v
