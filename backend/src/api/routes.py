"""
API路由模块
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from loguru import logger

from ..schemas.requests import (
    TrainingStartRequest,
    GeneticStartRequest,
    EvaluationRequest,
    ComparisonRequest,
    ConfigUpdateRequest,
    PageRequest,
    VisualizationRequest
)
from ..schemas.responses import (
    BaseResponse,
    PageResponse,
    PageInfo,
    TrainingStatusData,
    TrainingResultData,
    TrainingHistoryItem,
    GeneticStatusData,
    GeneticPopulationData,
    IndividualData,
    EvaluationResultData,
    ComparisonResultData,
    HealthData,
    ConfigData,
    VisualizationData,
    TrainingDashboardData,
    GeneticProgressData
)
from ..services.training_service import TrainingService
from ..services.genetic_service import GeneticService
from ..services.evaluation_service import EvaluationService
from ..services.visualization_service import VisualizationService
from ..core.exceptions import BaseAppException

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["API"])

# 服务实例（依赖注入）
_training_service: Optional[TrainingService] = None
_genetic_service: Optional[GeneticService] = None
_evaluation_service: Optional[EvaluationService] = None
_visualization_service: Optional[VisualizationService] = None


def get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service


def get_genetic_service() -> GeneticService:
    global _genetic_service
    if _genetic_service is None:
        _genetic_service = GeneticService()
    return _genetic_service


def get_evaluation_service() -> EvaluationService:
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service


def get_visualization_service() -> VisualizationService:
    global _visualization_service
    if _visualization_service is None:
        _visualization_service = VisualizationService()
    return _visualization_service


# ==================== 训练管理接口 ====================

training_router = APIRouter(prefix="/training", tags=["Training"])


@training_router.post(
    "/start",
    response_model=BaseResponse[str],
    summary="启动PPO训练",
    description="启动一个新的PPO训练任务，返回任务ID"
)
async def start_training(
    request: TrainingStartRequest,
    service: TrainingService = Depends(get_training_service)
):
    """启动PPO训练任务"""
    logger.info(f"Starting training with config: {request.model_dump()}")
    task_id = service.start_training(request)
    return BaseResponse.success(data=task_id, message="Training task started")


@training_router.get(
    "/status/{task_id}",
    response_model=BaseResponse[TrainingStatusData],
    summary="获取训练状态",
    description="根据任务ID获取训练任务的当前状态"
)
async def get_training_status(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """获取训练任务状态"""
    status = service.get_status(task_id)
    return BaseResponse.success(data=status)


@training_router.post(
    "/stop/{task_id}",
    response_model=BaseResponse[bool],
    summary="停止训练",
    description="停止指定的训练任务"
)
async def stop_training(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """停止训练任务"""
    logger.info(f"Stopping training task: {task_id}")
    success = service.stop_training(task_id)
    if success:
        return BaseResponse.success(data=True, message="Training task stopped")
    return BaseResponse.error(code=400, message="Task is not running", data=False)


@training_router.get(
    "/history/{task_id}",
    response_model=PageResponse[TrainingHistoryItem],
    summary="获取训练历史",
    description="分页获取训练历史记录"
)
async def get_training_history(
    task_id: str,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(100, ge=1, le=1000, description="每页大小"),
    service: TrainingService = Depends(get_training_service)
):
    """获取训练历史"""
    history, total = service.get_history(task_id, page, page_size)
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    return PageResponse(
        data=history,
        page_info=PageInfo(
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages
        )
    )


@training_router.get(
    "/result/{task_id}",
    response_model=BaseResponse[TrainingResultData],
    summary="获取训练结果",
    description="获取已完成训练任务的结果"
)
async def get_training_result(
    task_id: str,
    service: TrainingService = Depends(get_training_service)
):
    """获取训练结果"""
    result = service.get_result(task_id)
    return BaseResponse.success(data=result)


@training_router.get(
    "/tasks",
    response_model=BaseResponse[List[TrainingStatusData]],
    summary="列出所有训练任务",
    description="获取所有训练任务的状态列表"
)
async def list_training_tasks(
    service: TrainingService = Depends(get_training_service)
):
    """列出所有训练任务"""
    tasks = service.list_tasks()
    return BaseResponse.success(data=tasks)


# ==================== 遗传算法接口 ====================

genetic_router = APIRouter(prefix="/genetic", tags=["Genetic Algorithm"])


@genetic_router.post(
    "/start",
    response_model=BaseResponse[str],
    summary="启动遗传算法搜索",
    description="启动遗传算法搜索最优种子组合"
)
async def start_genetic_search(
    request: GeneticStartRequest,
    service: GeneticService = Depends(get_genetic_service)
):
    """启动遗传算法搜索"""
    logger.info(f"Starting genetic search with config: {request.model_dump()}")
    task_id = service.start_search(request)
    return BaseResponse.success(data=task_id, message="Genetic search started")


@genetic_router.get(
    "/status/{task_id}",
    response_model=BaseResponse[GeneticStatusData],
    summary="获取搜索状态",
    description="获取遗传算法搜索的当前状态"
)
async def get_genetic_status(
    task_id: str,
    service: GeneticService = Depends(get_genetic_service)
):
    """获取遗传算法状态"""
    status = service.get_status(task_id)
    return BaseResponse.success(data=status)


@genetic_router.post(
    "/stop/{task_id}",
    response_model=BaseResponse[bool],
    summary="停止搜索",
    description="停止遗传算法搜索"
)
async def stop_genetic_search(
    task_id: str,
    service: GeneticService = Depends(get_genetic_service)
):
    """停止遗传算法搜索"""
    logger.info(f"Stopping genetic search task: {task_id}")
    success = service.stop_search(task_id)
    if success:
        return BaseResponse.success(data=True, message="Genetic search stopped")
    return BaseResponse.error(code=400, message="Task is not running", data=False)


@genetic_router.get(
    "/best/{task_id}",
    response_model=BaseResponse[IndividualData],
    summary="获取最佳个体",
    description="获取搜索过程中发现的最佳个体"
)
async def get_best_individual(
    task_id: str,
    service: GeneticService = Depends(get_genetic_service)
):
    """获取最佳个体"""
    best = service.get_best_individual(task_id)
    return BaseResponse.success(data=best)


@genetic_router.get(
    "/population/{task_id}",
    response_model=BaseResponse[GeneticPopulationData],
    summary="获取当前种群",
    description="获取当前代的完整种群数据"
)
async def get_population(
    task_id: str,
    service: GeneticService = Depends(get_genetic_service)
):
    """获取当前种群"""
    population = service.get_population(task_id)
    return BaseResponse.success(data=population)


@genetic_router.get(
    "/tasks",
    response_model=BaseResponse[List[GeneticStatusData]],
    summary="列出所有遗传算法任务",
    description="获取所有遗传算法任务的状态列表"
)
async def list_genetic_tasks(
    service: GeneticService = Depends(get_genetic_service)
):
    """列出所有遗传算法任务"""
    tasks = service.list_tasks()
    return BaseResponse.success(data=tasks)


# ==================== 评估接口 ====================

evaluation_router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@evaluation_router.post(
    "/run",
    response_model=BaseResponse[EvaluationResultData],
    summary="运行评估",
    description="评估指定网络的性能"
)
async def run_evaluation(
    request: EvaluationRequest,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """运行评估"""
    logger.info(f"Running evaluation: {request.model_dump()}")
    result = service.evaluate_network(request)
    return BaseResponse.success(data=result)


@evaluation_router.post(
    "/seeds",
    response_model=BaseResponse[EvaluationResultData],
    summary="使用种子评估",
    description="使用24个种子（4行6列）评估不可微网络"
)
async def evaluate_with_seeds(
    seeds: List[int],
    env_name: str = Query("LunarLander-v2", description="环境名称"),
    num_episodes: int = Query(10, ge=1, le=100, description="评估回合数"),
    service: EvaluationService = Depends(get_evaluation_service)
):
    """使用种子评估"""
    logger.info(f"Evaluating with seeds: {seeds[:4]}...")
    result = service.evaluate_with_seeds(seeds, env_name, num_episodes)
    return BaseResponse.success(data=result)


@evaluation_router.post(
    "/compare",
    response_model=BaseResponse[ComparisonResultData],
    summary="比较网络",
    description="比较可微和不可微网络的性能差异"
)
async def compare_networks(
    request: ComparisonRequest,
    service: EvaluationService = Depends(get_evaluation_service)
):
    """比较网络性能"""
    logger.info(f"Comparing networks: {request.model_dump()}")
    result = service.compare_networks(request)
    return BaseResponse.success(data=result)


# ==================== 可视化接口 ====================

visualization_router = APIRouter(prefix="/visualization", tags=["Visualization"])


@visualization_router.get(
    "/training/{task_id}",
    response_model=BaseResponse[TrainingDashboardData],
    summary="获取训练可视化",
    description="获取训练过程的可视化仪表板，包含适应度曲线"
)
async def get_training_visualization(
    task_id: str,
    training_service: TrainingService = Depends(get_training_service),
    viz_service: VisualizationService = Depends(get_visualization_service)
):
    """获取训练可视化"""
    # 获取训练数据
    status = training_service.get_status(task_id)
    history, _ = training_service.get_history(task_id, 1, 100000)
    
    episode_rewards = [h.reward for h in history]
    policy_losses = [h.policy_loss for h in history if h.policy_loss is not None]
    value_losses = [h.value_loss for h in history if h.value_loss is not None]
    temperatures = [h.temperature for h in history if h.temperature is not None]
    
    # 生成仪表板
    dashboard_image = viz_service.generate_training_dashboard(
        episode_rewards=episode_rewards,
        policy_losses=policy_losses,
        value_losses=value_losses,
        temperatures=temperatures,
        task_id=task_id
    )
    
    # 生成适应度曲线
    fitness_image = viz_service.generate_fitness_curve(
        fitness_history=episode_rewards,
        title=f"Training Rewards - {task_id}",
        xlabel="Episode",
        ylabel="Reward"
    )
    
    result = TrainingDashboardData(
        task_id=task_id,
        dashboard_image=dashboard_image,
        fitness_curve_image=fitness_image,
        current_stats={
            "current_episode": status.current_episode,
            "best_reward": status.best_reward,
            "avg_reward_last_100": status.avg_reward_last_100,
            "current_temperature": status.current_temperature,
            "progress": status.progress
        }
    )
    
    return BaseResponse.success(data=result)


@visualization_router.get(
    "/genetic/{task_id}",
    response_model=BaseResponse[GeneticProgressData],
    summary="获取遗传算法可视化",
    description="获取遗传算法搜索过程的适应度曲线"
)
async def get_genetic_visualization(
    task_id: str,
    genetic_service: GeneticService = Depends(get_genetic_service),
    viz_service: VisualizationService = Depends(get_visualization_service)
):
    """获取遗传算法可视化"""
    status = genetic_service.get_status(task_id)
    
    # 从任务中获取适应度历史
    fitness_history = []
    if status.best_individual:
        # 使用服务内部的fitness_history
        task = genetic_service.tasks.get(task_id)
        if task and task.ga:
            fitness_history = task.ga.fitness_history
    
    # 生成进度图
    progress_image = viz_service.generate_genetic_progress(
        fitness_history=fitness_history,
        task_id=task_id
    )
    
    result = GeneticProgressData(
        task_id=task_id,
        progress_image=progress_image,
        fitness_history=fitness_history,
        best_fitness=status.best_fitness,
        current_generation=status.current_generation
    )
    
    return BaseResponse.success(data=result)


@visualization_router.get(
    "/fitness-curve",
    response_model=BaseResponse[VisualizationData],
    summary="生成适应度曲线",
    description="根据提供的数据生成适应度曲线图"
)
async def generate_fitness_curve(
    data: str = Query(..., description="逗号分隔的适应度值"),
    title: str = Query("Fitness Curve", description="图表标题"),
    viz_service: VisualizationService = Depends(get_visualization_service)
):
    """生成适应度曲线"""
    fitness_history = [float(x.strip()) for x in data.split(",")]
    
    image_base64 = viz_service.generate_fitness_curve(
        fitness_history=fitness_history,
        title=title
    )
    
    result = VisualizationData(
        task_id="custom",
        image_base64=image_base64,
        image_type="png"
    )
    
    return BaseResponse.success(data=result)


# ==================== 系统管理接口 ====================

system_router = APIRouter(prefix="", tags=["System"])


@system_router.get(
    "/health",
    response_model=BaseResponse[HealthData],
    summary="健康检查",
    description="检查服务健康状态"
)
async def health_check(
    training_service: TrainingService = Depends(get_training_service),
    genetic_service: GeneticService = Depends(get_genetic_service)
):
    """健康检查"""
    import torch
    import time
    from ..config import settings
    
    active_tasks = (
        training_service.get_active_task_count() +
        genetic_service.get_active_task_count()
    )
    
    health_data = HealthData(
        status="healthy",
        version=settings.VERSION,
        uptime_seconds=time.time() - settings.START_TIME,
        gpu_available=torch.cuda.is_available(),
        active_tasks=active_tasks
    )
    
    return BaseResponse.success(data=health_data)


@system_router.get(
    "/config",
    response_model=BaseResponse[ConfigData],
    summary="获取配置",
    description="获取系统当前配置"
)
async def get_config():
    """获取系统配置"""
    from ..config import settings
    
    config_data = ConfigData(
        ppo={
            "learning_rate": settings.PPO_LEARNING_RATE,
            "gamma": settings.PPO_GAMMA,
            "epsilon": settings.PPO_EPSILON,
            "initial_temperature": settings.INITIAL_TEMPERATURE,
            "temperature_decay": settings.TEMPERATURE_DECAY,
            "min_temperature": settings.MIN_TEMPERATURE
        },
        genetic={
            "population_size": settings.GA_POPULATION_SIZE,
            "mutation_rate": settings.GA_MUTATION_RATE,
            "crossover_rate": settings.GA_CROSSOVER_RATE,
            "elite_size": settings.GA_ELITE_SIZE
        },
        environment={
            "default_env": settings.DEFAULT_ENV,
            "max_steps": settings.MAX_STEPS
        }
    )
    
    return BaseResponse.success(data=config_data)


# 注册子路由
router.include_router(training_router)
router.include_router(genetic_router)
router.include_router(evaluation_router)
router.include_router(visualization_router)
router.include_router(system_router)
