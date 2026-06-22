"""
API路由模块
"""
import os
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from loguru import logger

from ..schemas.requests import (
    TrainingStartRequest,
    GeneticStartRequest,
    EvaluationRequest,
    ComparisonRequest,
    ConfigUpdateRequest,
    PageRequest,
    VisualizationRequest,
    PipelineStartRequest,
    VisualizationComparisonQuery
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
    ConfigDiffData,
    VisualizationData,
    VisualizationGenerateData,
    VisualizationComparisonData,
    TrainingDashboardData,
    GeneticProgressData,
    PipelineStatusData,
    PipelineStageData,
    ComparisonReportData
)
from ..services.training_service import TrainingService
from ..services.genetic_service import GeneticService
from ..services.evaluation_service import EvaluationService
from ..services.visualization_service import VisualizationService
from ..services.pipeline_service import PipelineService
from ..core.exceptions import BaseAppException

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["API"])

# 服务实例（依赖注入）
_training_service: Optional[TrainingService] = None
_genetic_service: Optional[GeneticService] = None
_evaluation_service: Optional[EvaluationService] = None
_visualization_service: Optional[VisualizationService] = None
_pipeline_service: Optional[PipelineService] = None


def get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
        # 关联遗传算法服务，用于自动启动GA任务
        genetic_service = get_genetic_service()
        _training_service.set_genetic_service(genetic_service)
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


def get_pipeline_service() -> PipelineService:
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service


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
        chart_type="fitness_curve",
        image_base64=image_base64,
        image_type="png"
    )
    
    return BaseResponse.success(data=result)


def _resolve_task_type(
    task_id: str,
    training_service: TrainingService,
    genetic_service: GeneticService,
    explicit_task_type: Optional[str] = None
) -> str:
    """
    解析task_id对应的任务类型
    优先使用显式指定，否则在两个服务中查找
    """
    if explicit_task_type in ("training", "genetic"):
        return explicit_task_type

    with training_service._lock:
        if task_id in training_service.tasks:
            return "training"

    with genetic_service._lock:
        if task_id in genetic_service.tasks:
            return "genetic"

    raise HTTPException(
        status_code=404,
        detail=f"Task {task_id} not found in training or genetic services"
    )


def _collect_data_for_chart(
    request: VisualizationRequest,
    training_service: TrainingService,
    genetic_service: GeneticService
) -> Dict[str, Any]:
    """
    根据请求收集图表所需的数据：
    - 提供raw_data则直接返回
    - 提供task_id则从服务中拉取
    """
    if request.raw_data is not None:
        return request.raw_data

    assert request.task_id is not None
    task_type = _resolve_task_type(request.task_id, training_service, genetic_service, request.task_type)

    data: Dict[str, Any] = {}

    if request.chart_type == "fitness_curve":
        if task_type == "training":
            history, _ = training_service.get_history(request.task_id, 1, 100000)
            data["fitness_history"] = [h.reward for h in history]
        else:
            task = genetic_service.tasks.get(request.task_id)
            fh = task.ga.fitness_history if (task and task.ga) else []
            data["fitness_history"] = list(fh)

    elif request.chart_type == "dashboard":
        if task_type != "training":
            raise HTTPException(
                status_code=400,
                detail="dashboard chart_type requires a training task"
            )
        history, _ = training_service.get_history(request.task_id, 1, 100000)
        data["episode_rewards"] = [h.reward for h in history]
        data["policy_losses"] = [h.policy_loss for h in history if h.policy_loss is not None]
        data["value_losses"] = [h.value_loss for h in history if h.value_loss is not None]
        data["temperatures"] = [h.temperature for h in history if h.temperature is not None]

    elif request.chart_type == "progress":
        if task_type != "genetic":
            raise HTTPException(
                status_code=400,
                detail="progress chart_type requires a genetic task"
            )
        task = genetic_service.tasks.get(request.task_id)
        fh = list(task.ga.fitness_history) if (task and task.ga) else []
        data["fitness_history"] = fh
        if task and task.ga and hasattr(task.ga, "avg_fitness_history"):
            data["avg_fitness_history"] = list(task.ga.avg_fitness_history)

    elif request.chart_type == "comparison":
        raise HTTPException(
            status_code=400,
            detail="comparison chart_type via POST /generate requires raw_data with diff_rewards & non_diff_rewards. "
                   "Use GET /visualization/comparison for task_id-based comparison."
        )

    return data


@visualization_router.post(
    "/generate",
    response_model=BaseResponse[VisualizationGenerateData],
    summary="生成可视化图表",
    description="""根据task_id或raw_data生成指定类型的可视化图表。

支持图表类型：
- **fitness_curve**: 适应度曲线（训练/遗传都支持）
- **dashboard**: 训练仪表板（仅训练任务）
- **progress**: 遗传算法进度（仅遗传任务）
- **comparison**: 对比图（仅raw_data，需提供diff_rewards和non_diff_rewards）

输出格式：
- **base64**: 仅返回Base64编码图片（默认）
- **file_url**: 仅返回文件URL（自动保存到plots/）
- **both**: 同时返回两者
"""
)
async def generate_visualization(
    request: VisualizationRequest,
    training_service: TrainingService = Depends(get_training_service),
    genetic_service: GeneticService = Depends(get_genetic_service),
    viz_service: VisualizationService = Depends(get_visualization_service)
):
    """生成可视化图表"""
    logger.info(
        f"Generating visualization: chart_type={request.chart_type}, "
        f"task_id={request.task_id}, format={request.format}, save={request.save_to_plots}"
    )

    try:
        data = _collect_data_for_chart(request, training_service, genetic_service)
    except HTTPException:
        raise

    try:
        result = viz_service.generate(
            chart_type=request.chart_type,
            data=data,
            task_id=request.task_id,
            window_size=request.window_size,
            save_to_plots=request.save_to_plots,
            fmt=request.format,
            title=request.title,
            xlabel=request.xlabel,
            ylabel=request.ylabel,
            show_avg=request.show_avg
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_data = VisualizationGenerateData(
        task_id=request.task_id,
        chart_type=request.chart_type,
        format=request.format,
        image_base64=result.get("image_base64"),
        file_url=result.get("file_url"),
        file_path=result.get("file_path"),
        image_type="png",
        width=result.get("width", 0),
        height=result.get("height", 0),
        stats=result.get("stats")
    )

    return BaseResponse.success(data=response_data, message=f"{request.chart_type} generated")


async def _collect_comparison_rewards(
    query: VisualizationComparisonQuery,
    training_service: TrainingService,
    genetic_service: GeneticService,
    evaluation_service: EvaluationService
) -> tuple[List[float], List[float]]:
    """
    根据GET /comparison查询参数，提取(diff_rewards, non_diff_rewards)
    三种数据源：
    1. diff_rewards + non_diff_rewards 直接传
    2. differentiable_model_path + genetic_seeds -> 调用评估服务
    3. differentiable_task_id + genetic_task_id -> 从任务中取模型/种子，再调用评估
    """
    # 情况1：直接传rewards
    if query.diff_rewards is not None and query.non_diff_rewards is not None:
        return list(query.diff_rewards), list(query.non_diff_rewards)

    diff_model_path: Optional[str] = None
    genetic_seeds: Optional[List[int]] = None

    # 情况2：模型路径 + 种子
    if query.differentiable_model_path and query.genetic_seeds:
        diff_model_path = query.differentiable_model_path
        genetic_seeds = list(query.genetic_seeds)

    # 情况3：task_id 对
    elif query.differentiable_task_id and query.genetic_task_id:
        # 从训练任务取保存的模型路径
        try:
            tstatus = training_service.get_status(query.differentiable_task_id)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Training task {query.differentiable_task_id} not found: {e}"
            )
        from ..config import settings
        model_candidate = os.path.join(settings.MODEL_SAVE_DIR, f"ppo_{query.differentiable_task_id}.pt")
        if not os.path.exists(model_candidate):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot find model for training task {query.differentiable_task_id}. "
                       f"Expected at {model_candidate}"
            )
        diff_model_path = model_candidate

        # 从遗传任务取最佳个体的种子
        try:
            best = genetic_service.get_best_individual(query.genetic_task_id)
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Genetic task {query.genetic_task_id} not found or no best individual: {e}"
            )
        genetic_seeds = [s for row in best.seeds for s in row]

    if diff_model_path is None or genetic_seeds is None:
        raise HTTPException(status_code=400, detail="Invalid comparison data source")

    # 执行评估获取两组rewards
    diff_req = EvaluationRequest(
        network_type="differentiable",
        num_episodes=query.num_episodes,
        model_path=diff_model_path,
        env_name="LunarLander-v2"
    )
    # 直接使用底层方法获取每回合rewards
    diff_rewards = _evaluate_and_get_rewards(evaluation_service, diff_req, genetic_seeds, mode="diff")
    non_diff_rewards = _evaluate_and_get_rewards(evaluation_service, diff_req, genetic_seeds, mode="non_diff")

    return diff_rewards, non_diff_rewards


def _evaluate_and_get_rewards(
    service: EvaluationService,
    diff_req: EvaluationRequest,
    genetic_seeds: List[int],
    mode: str
) -> List[float]:
    """
    执行评估并获取每回合奖励列表。
    这里复用evaluation_service的实现逻辑，但收集各episode奖励而不是聚合统计。
    """
    import uuid
    import gymnasium as gym
    import torch
    import numpy as np
    from ..models.network import NonDifferentiableNetwork, DifferentiableNetwork
    from ..models.genetic_algorithm import Individual, WeightGenerator

    env_name = diff_req.env_name
    num_episodes = diff_req.num_episodes

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    try:
        if mode == "diff":
            network = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
            checkpoint = torch.load(diff_req.model_path, map_location='cpu')
            if 'policy_state_dict' in checkpoint:
                network.load_state_dict(checkpoint['policy_state_dict'])
            else:
                network.load_state_dict(checkpoint)
        else:
            individual = Individual.create_from_list(genetic_seeds)
            network = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
            wg = WeightGenerator()
            shapes = {name: tuple(p.shape) for name, p in network.named_parameters()}
            weights = wg.generate_weights_from_individual(individual, shapes)
            wg.apply_weights_to_network(network, weights)

        network.eval()
        rewards: List[float] = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            ep = 0.0
            while True:
                with torch.no_grad():
                    logits = network(torch.FloatTensor(state))
                    action = logits.argmax().item()
                next_state, r, term, trunc, _ = env.step(action)
                ep += r
                state = next_state
                if term or trunc:
                    break
            rewards.append(ep)
        return rewards
    finally:
        env.close()


@visualization_router.get(
    "/comparison",
    response_model=BaseResponse[VisualizationComparisonData],
    summary="生成网络对比可视化",
    description="""生成可微网络 vs 不可微网络的对比可视化。

三种数据源（任选其一）：
1. **直接传评估结果**：`diff_rewards` + `non_diff_rewards`（逗号分隔的数值）
2. **模型+种子**：`differentiable_model_path` + `genetic_seeds`（24个整数种子）
3. **任务ID对**：`differentiable_task_id`（PPO训练任务） + `genetic_task_id`（GA任务）

输出包含：
- 独立的**箱线图**（Base64/URL/path）
- 独立的**直方图**（Base64/URL/path）
- **组合图**（箱线+直方图并列）

输出格式通过`format`控制：`base64` | `file_url` | `both`
"""
)
async def get_visualization_comparison(
    differentiable_task_id: Optional[str] = Query(None, description="PPO训练任务ID"),
    genetic_task_id: Optional[str] = Query(None, description="遗传算法任务ID"),
    differentiable_model_path: Optional[str] = Query(None, description="可微网络模型文件路径"),
    genetic_seeds: Optional[str] = Query(None, description="24个整数种子（逗号分隔）"),
    diff_rewards: Optional[str] = Query(None, description="可微网络奖励列表（逗号分隔）"),
    non_diff_rewards: Optional[str] = Query(None, description="不可微网络奖励列表（逗号分隔）"),
    num_episodes: int = Query(10, ge=1, le=500, description="评估回合数"),
    format: str = Query("base64", pattern="^(base64|file_url|both)$", description="输出格式"),
    save_to_plots: bool = Query(False, description="是否保存图片到plots/"),
    title: Optional[str] = Query(None, description="图表标题"),
    training_service: TrainingService = Depends(get_training_service),
    genetic_service: GeneticService = Depends(get_genetic_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
    viz_service: VisualizationService = Depends(get_visualization_service)
):
    """生成对比可视化（箱线图+直方图）"""
    import os

    def _parse_float_list(s: Optional[str]) -> Optional[List[float]]:
        if not s:
            return None
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
        if not s:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    parsed_seeds = _parse_int_list(genetic_seeds)
    parsed_diff_rewards = _parse_float_list(diff_rewards)
    parsed_non_diff_rewards = _parse_float_list(non_diff_rewards)

    try:
        query = VisualizationComparisonQuery(
            differentiable_task_id=differentiable_task_id,
            genetic_task_id=genetic_task_id,
            differentiable_model_path=differentiable_model_path,
            genetic_seeds=parsed_seeds,
            diff_rewards=parsed_diff_rewards,
            non_diff_rewards=parsed_non_diff_rewards,
            num_episodes=num_episodes,
            format=format,  # type: ignore[arg-type]
            save_to_plots=save_to_plots,
            title=title
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    diff_r, non_diff_r = await _collect_comparison_rewards(
        query, training_service, genetic_service, evaluation_service
    )

    if not diff_r or not non_diff_r:
        raise HTTPException(status_code=400, detail="Empty reward lists for comparison")

    logger.info(
        f"Generating comparison visualization: diff_n={len(diff_r)}, "
        f"non_diff_n={len(non_diff_r)}, format={format}, save={save_to_plots}"
    )

    result = viz_service.generate_comparison(
        diff_rewards=diff_r,
        non_diff_rewards=non_diff_r,
        save_to_plots=save_to_plots,
        fmt=query.format,
        title=title
    )

    response_data = VisualizationComparisonData(
        format=query.format,
        boxplot_base64=result.get("boxplot_base64"),
        histogram_base64=result.get("histogram_base64"),
        combined_base64=result.get("combined_base64"),
        boxplot_url=result.get("boxplot_url"),
        histogram_url=result.get("histogram_url"),
        combined_url=result.get("combined_url"),
        boxplot_path=result.get("boxplot_path"),
        histogram_path=result.get("histogram_path"),
        combined_path=result.get("combined_path"),
        differentiable_stats=result["differentiable_stats"],
        non_differentiable_stats=result["non_differentiable_stats"],
        performance_gap=result["performance_gap"],
        image_type="png"
    )

    return BaseResponse.success(data=response_data, message="Comparison visualization generated")


# ==================== 流水线接口 ====================

pipeline_router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@pipeline_router.post(
    "/start",
    response_model=BaseResponse[str],
    summary="启动流水线",
    description="启动PPO+GA流水线：PPO训练→权重导出→GA搜索→对比报告"
)
async def start_pipeline(
    request: PipelineStartRequest,
    service: PipelineService = Depends(get_pipeline_service)
):
    logger.info(f"Starting pipeline with config: {request.model_dump()}")
    task_id = service.start_pipeline(request)
    return BaseResponse.success(data=task_id, message="Pipeline started")


@pipeline_router.get(
    "/status/{task_id}",
    response_model=BaseResponse[PipelineStatusData],
    summary="获取流水线状态",
    description="获取流水线任务的当前状态，包含各阶段进度"
)
async def get_pipeline_status(
    task_id: str,
    service: PipelineService = Depends(get_pipeline_service)
):
    status = service.get_status(task_id)
    return BaseResponse.success(data=status)


@pipeline_router.get(
    "/report/{task_id}",
    response_model=BaseResponse[ComparisonReportData],
    summary="获取对比报告",
    description="获取流水线完成后的对比报告"
)
async def get_pipeline_report(
    task_id: str,
    service: PipelineService = Depends(get_pipeline_service)
):
    report = service.get_report(task_id)
    return BaseResponse.success(data=report)


@pipeline_router.post(
    "/stop/{task_id}",
    response_model=BaseResponse[bool],
    summary="停止流水线",
    description="停止指定的流水线任务"
)
async def stop_pipeline(
    task_id: str,
    service: PipelineService = Depends(get_pipeline_service)
):
    logger.info(f"Stopping pipeline task: {task_id}")
    success = service.stop_pipeline(task_id)
    if success:
        return BaseResponse.success(data=True, message="Pipeline task stopped")
    return BaseResponse.error(code=400, message="Task is not running", data=False)


@pipeline_router.get(
    "/tasks",
    response_model=BaseResponse[List[PipelineStatusData]],
    summary="列出所有流水线任务",
    description="获取所有流水线任务的状态列表"
)
async def list_pipeline_tasks(
    service: PipelineService = Depends(get_pipeline_service)
):
    tasks = service.list_tasks()
    return BaseResponse.success(data=tasks)


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
            "epochs": settings.PPO_EPOCHS,
            "batch_size": settings.PPO_BATCH_SIZE,
            "initial_temperature": settings.INITIAL_TEMPERATURE,
            "temperature_decay": settings.TEMPERATURE_DECAY,
            "min_temperature": settings.MIN_TEMPERATURE,
            "regularization_coef": settings.REGULARIZATION_COEF
        },
        genetic={
            "population_size": settings.GA_POPULATION_SIZE,
            "mutation_rate": settings.GA_MUTATION_RATE,
            "crossover_rate": settings.GA_CROSSOVER_RATE,
            "elite_size": settings.GA_ELITE_SIZE,
            "max_generations": settings.GA_MAX_GENERATIONS,
            "seed_range_min": settings.GA_SEED_RANGE_MIN,
            "seed_range_max": settings.GA_SEED_RANGE_MAX,
            "alpha": settings.GA_ALPHA
        },
        environment={
            "default_env": settings.DEFAULT_ENV,
            "max_steps": settings.MAX_STEPS,
            "total_episodes": settings.TOTAL_EPISODES,
            "max_concurrent_training_tasks": settings.MAX_CONCURRENT_TRAINING_TASKS,
            "max_concurrent_genetic_tasks": settings.MAX_CONCURRENT_GENETIC_TASKS
        }
    )
    
    return BaseResponse.success(data=config_data)


@system_router.put(
    "/config",
    response_model=BaseResponse[ConfigDiffData],
    summary="更新配置",
    description="运行时更新配置，仅影响新启动的任务，运行中任务沿用启动时快照。非法值返回422。"
)
async def update_config(
    request: ConfigUpdateRequest,
    training_service: TrainingService = Depends(get_training_service),
    genetic_service: GeneticService = Depends(get_genetic_service)
):
    """运行时更新配置"""
    from ..config import config_manager
    
    updates_dict: Dict[str, Any] = {}
    
    if request.ppo is not None:
        ppo_updates = request.ppo.model_dump(exclude_none=True)
        if ppo_updates:
            updates_dict["ppo"] = ppo_updates
    
    if request.genetic is not None:
        genetic_updates = request.genetic.model_dump(exclude_none=True)
        if genetic_updates:
            updates_dict["genetic"] = genetic_updates
    
    if request.environment is not None:
        env_updates = request.environment.model_dump(exclude_none=True)
        if env_updates:
            updates_dict["environment"] = env_updates
    
    if not updates_dict:
        raise HTTPException(
            status_code=422,
            detail="No valid configuration updates provided"
        )
    
    try:
        before, after = config_manager.apply_updates(updates_dict)
    except ValueError as e:
        logger.warning(f"Config update validation failed: {e}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    
    changed_keys: List[str] = []
    for category in before:
        for key in before[category]:
            changed_keys.append(f"{category}.{key}")
    
    if "environment" in updates_dict:
        env_updates = updates_dict["environment"]
        if "max_concurrent_training_tasks" in env_updates:
            new_limit = env_updates["max_concurrent_training_tasks"]
            training_service.update_max_concurrent_tasks(new_limit)
            logger.info(f"Training service max concurrent tasks updated to {new_limit}")
        
        if "max_concurrent_genetic_tasks" in env_updates:
            new_limit = env_updates["max_concurrent_genetic_tasks"]
            genetic_service.update_max_concurrent_tasks(new_limit)
            logger.info(f"Genetic service max concurrent tasks updated to {new_limit}")
    
    diff_data = ConfigDiffData(
        before=before,
        after=after,
        changed_keys=changed_keys
    )
    
    logger.info(f"Config hot-reload completed. Changed keys: {changed_keys}")
    
    return BaseResponse.success(
        data=diff_data,
        message="Configuration updated successfully"
    )


# 注册子路由
router.include_router(training_router)
router.include_router(genetic_router)
router.include_router(evaluation_router)
router.include_router(visualization_router)
router.include_router(pipeline_router)
router.include_router(system_router)
