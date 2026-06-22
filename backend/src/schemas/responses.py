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
    auto_start_ga: bool = Field(default=False, description="是否自动启动关联GA任务")
    child_ga_task_id: Optional[str] = Field(default=None, description="自动启动的GA子任务ID")
    
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
    parent_task_id: Optional[str] = Field(default=None, description="父任务ID（PPO训练任务ID）")
    
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
    task_id: str = Field(description="任务ID，原始数据时为'custom'")
    chart_type: str = Field(description="图表类型: fitness_curve, dashboard, progress, comparison")
    image_base64: Optional[str] = Field(default=None, description="Base64编码的图片（format=base64/both时返回）")
    file_url: Optional[str] = Field(default=None, description="图片文件URL（format=file_url/both且save_to_plots=true时返回）")
    file_path: Optional[str] = Field(default=None, description="图片本地绝对路径（save_to_plots=true时返回）")
    image_type: str = Field(default="png", description="图片类型")
    generated_at: datetime = Field(default_factory=datetime.now, description="生成时间")


class VisualizationGenerateData(BaseModel):
    """可视化生成响应数据"""
    task_id: Optional[str] = Field(default=None, description="关联的任务ID（如果通过task_id生成）")
    chart_type: str = Field(description="生成的图表类型")
    format: str = Field(description="输出格式: base64, file_url, both")
    image_base64: Optional[str] = Field(default=None, description="Base64编码的图片（format=base64/both时返回）")
    file_url: Optional[str] = Field(default=None, description="图片文件访问URL（format=file_url/both且保存了文件时返回）")
    file_path: Optional[str] = Field(default=None, description="图片本地文件路径（保存时返回）")
    image_type: str = Field(default="png", description="图片类型")
    width: int = Field(default=0, description="图片宽度（像素）")
    height: int = Field(default=0, description="图片高度（像素）")
    stats: Optional[Dict[str, Any]] = Field(default=None, description="附加统计信息（如适应度统计、数据摘要等）")
    generated_at: datetime = Field(default_factory=datetime.now, description="生成时间")


class VisualizationComparisonData(BaseModel):
    """可视化对比响应数据（返回箱线图+直方图）"""
    format: str = Field(description="输出格式: base64, file_url, both")
    boxplot_base64: Optional[str] = Field(default=None, description="箱线图Base64（format=base64/both时返回）")
    histogram_base64: Optional[str] = Field(default=None, description="直方图Base64（format=base64/both时返回）")
    combined_base64: Optional[str] = Field(default=None, description="组合图Base64（箱线图+直方图并列展示，format=base64/both时返回）")
    boxplot_url: Optional[str] = Field(default=None, description="箱线图文件URL")
    histogram_url: Optional[str] = Field(default=None, description="直方图文件URL")
    combined_url: Optional[str] = Field(default=None, description="组合图文件URL")
    boxplot_path: Optional[str] = Field(default=None, description="箱线图本地路径")
    histogram_path: Optional[str] = Field(default=None, description="直方图本地路径")
    combined_path: Optional[str] = Field(default=None, description="组合图本地路径")
    differentiable_stats: Dict[str, Any] = Field(description="可微网络统计：mean, std, min, max, median, passed")
    non_differentiable_stats: Dict[str, Any] = Field(description="不可微网络统计：mean, std, min, max, median, passed")
    performance_gap: float = Field(description="性能差距：mean(diff) - mean(non_diff)")
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


class ConfigDiffData(BaseModel):
    """配置变更差异数据"""
    before: Dict[str, Any] = Field(description="变更前配置")
    after: Dict[str, Any] = Field(description="变更后配置")
    changed_keys: List[str] = Field(description="变更的键列表")
    timestamp: datetime = Field(default_factory=datetime.now, description="变更时间")


class PipelineStageData(BaseModel):
    stage_name: str = Field(description="阶段名称: ppo_training, weight_export, ga_search, comparison_report")
    status: str = Field(description="阶段状态: pending, running, completed, failed, skipped")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    details: Optional[Dict[str, Any]] = Field(default=None, description="阶段详情")


class PipelineStatusData(BaseModel):
    task_id: str = Field(description="流水线任务ID")
    status: str = Field(description="整体状态: pending, running, completed, failed, stopped")
    current_stage: str = Field(description="当前阶段")
    stages: List[PipelineStageData] = Field(description="各阶段状态")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    progress: float = Field(default=0.0, description="总进度百分比")


class ComparisonReportData(BaseModel):
    ppo_best_reward: float = Field(description="PPO训练最佳奖励")
    ppo_avg_reward: float = Field(description="PPO训练平均奖励")
    ga_best_fitness: float = Field(description="GA搜索最佳适应度")
    ga_best_seeds: Optional[List[List[int]]] = Field(default=None, description="GA最佳个体种子")
    weight_distance: float = Field(description="GA最佳权重与目标权重的L2距离")
    performance_gap: float = Field(description="PPO与GA性能差距")
    target_weight_norm: float = Field(description="目标权重向量范数")
    ga_weight_norm: float = Field(description="GA生成权重向量范数")
    similarity_score: float = Field(description="权重相似度分数(0-1)")
