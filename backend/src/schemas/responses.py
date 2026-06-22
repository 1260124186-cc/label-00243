"""
API响应模型
"""
import math
from typing import Generic, TypeVar, Optional, List, Any, Dict
from pydantic import BaseModel, Field, field_serializer
from datetime import datetime

T = TypeVar('T')


def safe_float(value: float) -> Optional[float]:
    """将无穷大/NaN转换为None，以便JSON序列化"""
    if value is None:
        return None
    if math.isinf(value) or math.isnan(value):
        return None
    return value


class BaseResponse(BaseModel, Generic[T]):
    """统一响应格式"""
    code: int = Field(default=200, description="响应状态码")
    message: str = Field(default="success", description="响应消息")
    data: Optional[T] = Field(default=None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    
    @classmethod
    def success(cls, data: T = None, message: str = "success") -> 'BaseResponse[T]':
        return cls(code=200, message=message, data=data)
    
    @classmethod
    def error(cls, code: int = 500, message: str = "error", data: T = None) -> 'BaseResponse[T]':
        return cls(code=code, message=message, data=data)


class PageInfo(BaseModel):
    """分页信息"""
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页大小")
    total: int = Field(description="总数量")
    total_pages: int = Field(description="总页数")


class PageResponse(BaseModel, Generic[T]):
    """分页响应"""
    code: int = Field(default=200)
    message: str = Field(default="success")
    data: List[T] = Field(default_factory=list)
    page_info: PageInfo
    timestamp: datetime = Field(default_factory=datetime.now)


# ==================== 训练相关响应 ====================

class TrainingStatusData(BaseModel):
    """训练状态数据"""
    task_id: str = Field(description="训练任务ID")
    status: str = Field(description="状态: pending, running, completed, failed, stopped")
    current_episode: int = Field(default=0, description="当前回合")
    total_episodes: int = Field(description="总回合数")
    best_reward: float = Field(default=float('-inf'), description="最佳奖励")
    avg_reward_last_100: float = Field(default=0.0, description="最近100回合平均奖励")
    current_temperature: float = Field(default=1.0, description="当前温度")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    progress: float = Field(default=0.0, description="进度百分比")
    
    @field_serializer('best_reward')
    def serialize_best_reward(self, value: float) -> Optional[float]:
        return safe_float(value)


class TrainingHistoryItem(BaseModel):
    """训练历史项"""
    episode: int
    reward: float
    length: int
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    temperature: Optional[float] = None


class TrainingResultData(BaseModel):
    """训练结果数据"""
    task_id: str
    status: str
    total_episodes: int
    best_reward: float
    avg_reward_last_100: float
    episode_rewards: List[float]
    training_duration_seconds: float


# ==================== 遗传算法相关响应 ====================

class IndividualData(BaseModel):
    """
    个体数据
    24个整数种子，排列为4行6列
    """
    seeds: List[List[int]] = Field(description="4x6种子矩阵")
    fitness: float = Field(description="适应度")
    generation: int = Field(description="所属代数")
    
    @field_serializer('fitness')
    def serialize_fitness(self, value: float) -> Optional[float]:
        return safe_float(value)


class GeneticStatusData(BaseModel):
    """遗传算法状态数据"""
    task_id: str = Field(description="任务ID")
    status: str = Field(description="状态")
    current_generation: int = Field(default=0, description="当前代数")
    max_generations: int = Field(description="最大代数")
    best_fitness: float = Field(default=float('-inf'), description="最佳适应度")
    population_size: int = Field(description="种群大小")
    elite_archive_size: int = Field(default=0, description="精英档案大小")
    best_individual: Optional[IndividualData] = Field(default=None, description="最佳个体")
    progress: float = Field(default=0.0, description="进度百分比")
    
    @field_serializer('best_fitness')
    def serialize_best_fitness(self, value: float) -> Optional[float]:
        return safe_float(value)


class GeneticPopulationData(BaseModel):
    """种群数据"""
    generation: int
    individuals: List[IndividualData]
    best_fitness: float
    avg_fitness: float
    
    @field_serializer('best_fitness', 'avg_fitness')
    def serialize_fitness_values(self, value: float) -> Optional[float]:
        return safe_float(value)


# ==================== 评估相关响应 ====================

class EvaluationResultData(BaseModel):
    """评估结果数据"""
    task_id: str
    network_type: str = Field(description="网络类型: differentiable, non_differentiable")
    num_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    passed: bool = Field(description="是否通过200分及格线")


class ComparisonResultData(BaseModel):
    """对比结果数据"""
    differentiable_result: EvaluationResultData
    non_differentiable_result: EvaluationResultData
    weight_difference_norm: float = Field(description="权重差异L2范数")
    performance_gap: float = Field(description="性能差距")


# ==================== 可视化相关响应 ====================

class VisualizationData(BaseModel):
    """可视化数据"""
    task_id: str = Field(description="任务ID")
    image_base64: str = Field(description="Base64编码的图片")
    image_type: str = Field(default="png", description="图片类型")
    generated_at: datetime = Field(default_factory=datetime.now, description="生成时间")


class TrainingDashboardData(BaseModel):
    """训练仪表板数据"""
    task_id: str
    dashboard_image: str = Field(description="仪表板图片Base64")
    fitness_curve_image: str = Field(description="适应度曲线图片Base64")
    current_stats: Dict[str, Any] = Field(description="当前统计数据")


class GeneticProgressData(BaseModel):
    """遗传算法进度数据"""
    task_id: str
    progress_image: str = Field(description="进度图片Base64")
    fitness_history: List[float] = Field(description="适应度历史")
    best_fitness: float
    current_generation: int
    
    @field_serializer('best_fitness')
    def serialize_best_fitness(self, value: float) -> Optional[float]:
        return safe_float(value)


# ==================== 系统相关响应 ====================

class HealthData(BaseModel):
    """健康检查数据"""
    status: str = Field(default="healthy")
    version: str
    uptime_seconds: float
    gpu_available: bool
    active_tasks: int


class ConfigData(BaseModel):
    """配置数据"""
    ppo: Dict[str, Any]
    genetic: Dict[str, Any]
    environment: Dict[str, Any]
