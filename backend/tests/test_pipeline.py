import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.network import DifferentiableNetwork, NonDifferentiableNetwork
from src.models.genetic_algorithm import Individual, WeightGenerator
from src.services.pipeline_service import (
    PipelineService,
    PipelineTask,
    PipelineStage,
    _compute_weight_distance,
)
from src.schemas.requests import PipelineStartRequest, TrainingStartRequest, GeneticStartRequest
from src.schemas.responses import PipelineStatusData, ComparisonReportData
from src.core.exceptions import PipelineException


@pytest.fixture
def diff_network(state_dim, action_dim):
    return DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim, initial_temperature=1.0)


@pytest.fixture
def pipeline_request():
    return PipelineStartRequest(
        ppo_config=TrainingStartRequest(total_episodes=10, max_steps=100, env_name="CartPole-v1"),
        ga_config=GeneticStartRequest(
            max_generations=5,
            population_size=10,
            evaluation_episodes=2,
            env_name="CartPole-v1",
        ),
        weight_similarity_coef=0.1,
    )


@pytest.fixture
def pipeline_request_with_seeds():
    seeds = list(range(1, 25))
    return PipelineStartRequest(
        ppo_config=TrainingStartRequest(total_episodes=10, max_steps=100, env_name="CartPole-v1"),
        ga_config=GeneticStartRequest(
            max_generations=5,
            population_size=10,
            evaluation_episodes=2,
            env_name="CartPole-v1",
        ),
        target_seeds=seeds,
        weight_similarity_coef=0.05,
    )


@pytest.fixture
def pipeline_service():
    return PipelineService(max_concurrent_tasks=1)


def test_pipeline_task_initial_state(pipeline_request):
    task = PipelineTask("pipe-1", pipeline_request)
    assert task.task_id == "pipe-1"
    assert task.status == "pending"
    assert task.current_stage == PipelineStage.PPO_TRAINING
    assert task.ppo_agent is None
    assert task.ppo_result is None
    assert task.target_weight_vector is None
    assert task.target_weight_dict is None
    assert task.ga_best_individual is None
    assert task.comparison_report is None
    assert not task.should_stop()


def test_pipeline_task_request_stop(pipeline_request):
    task = PipelineTask("pipe-2", pipeline_request)
    assert not task.should_stop()
    task.request_stop()
    assert task.should_stop()


def test_pipeline_task_get_progress(pipeline_request):
    task = PipelineTask("pipe-3", pipeline_request)
    assert task.get_progress() == 0.0
    task.start_stage(PipelineStage.PPO_TRAINING)
    assert task.get_progress() == 0.0
    task.complete_stage(PipelineStage.PPO_TRAINING)
    assert task.get_progress() == 25.0
    task.complete_stage(PipelineStage.WEIGHT_EXPORT)
    assert task.get_progress() == 50.0
    task.complete_stage(PipelineStage.GA_SEARCH)
    assert task.get_progress() == 75.0
    task.complete_stage(PipelineStage.COMPARISON_REPORT)
    assert task.get_progress() == 100.0


def test_pipeline_task_stage_lifecycle(pipeline_request):
    task = PipelineTask("pipe-4", pipeline_request)
    task.start_stage(PipelineStage.WEIGHT_EXPORT)
    assert task.current_stage == PipelineStage.WEIGHT_EXPORT
    assert task.stages[PipelineStage.WEIGHT_EXPORT]["status"] == "running"
    assert task.stages[PipelineStage.WEIGHT_EXPORT]["started_at"] is not None

    task.complete_stage(PipelineStage.WEIGHT_EXPORT, details={"dim": 100})
    assert task.stages[PipelineStage.WEIGHT_EXPORT]["status"] == "completed"
    assert task.stages[PipelineStage.WEIGHT_EXPORT]["completed_at"] is not None
    assert task.stages[PipelineStage.WEIGHT_EXPORT]["details"]["dim"] == 100


def test_pipeline_task_fail_stage(pipeline_request):
    task = PipelineTask("pipe-5", pipeline_request)
    task.start_stage(PipelineStage.GA_SEARCH)
    task.fail_stage(PipelineStage.GA_SEARCH, "something broke")
    assert task.stages[PipelineStage.GA_SEARCH]["status"] == "failed"
    assert task.stages[PipelineStage.GA_SEARCH]["details"]["error"] == "something broke"


def test_pipeline_stage_order():
    assert PipelineStage.ORDER == [
        PipelineStage.PPO_TRAINING,
        PipelineStage.WEIGHT_EXPORT,
        PipelineStage.GA_SEARCH,
        PipelineStage.COMPARISON_REPORT,
    ]


def test_pipeline_service_list_tasks_empty(pipeline_service):
    assert pipeline_service.list_tasks() == []


def test_pipeline_service_get_status_not_found_raises(pipeline_service):
    with pytest.raises(PipelineException):
        pipeline_service.get_status("nonexistent-id")


def test_pipeline_service_get_report_not_found_raises(pipeline_service):
    with pytest.raises(PipelineException):
        pipeline_service.get_report("nonexistent-id")


def test_pipeline_service_stop_pipeline_not_found_raises(pipeline_service):
    with pytest.raises(PipelineException):
        pipeline_service.stop_pipeline("nonexistent-id")


def test_pipeline_service_stop_non_running_task(pipeline_service, pipeline_request):
    task = PipelineTask("pipe-6", pipeline_request)
    task.status = "completed"
    with pipeline_service._lock:
        pipeline_service.tasks["pipe-6"] = task
    result = pipeline_service.stop_pipeline("pipe-6")
    assert result is False


def test_pipeline_service_stop_running_task(pipeline_service, pipeline_request):
    task = PipelineTask("pipe-7", pipeline_request)
    task.status = "running"
    with pipeline_service._lock:
        pipeline_service.tasks["pipe-7"] = task
    result = pipeline_service.stop_pipeline("pipe-7")
    assert result is True
    assert task.should_stop()


def test_pipeline_service_get_active_task_count(pipeline_service, pipeline_request):
    task1 = PipelineTask("pipe-a1", pipeline_request)
    task1.status = "running"
    task2 = PipelineTask("pipe-a2", pipeline_request)
    task2.status = "completed"
    with pipeline_service._lock:
        pipeline_service.tasks["pipe-a1"] = task1
        pipeline_service.tasks["pipe-a2"] = task2
    assert pipeline_service.get_active_task_count() == 1


def test_pipeline_service_get_status_returns_correct_data(pipeline_service, pipeline_request):
    task = PipelineTask("pipe-status-1", pipeline_request)
    task.status = "running"
    task.current_stage = PipelineStage.WEIGHT_EXPORT
    task.started_at = datetime.now()
    task.start_stage(PipelineStage.PPO_TRAINING)
    task.complete_stage(PipelineStage.PPO_TRAINING, details={"best_reward": 150.0})
    task.start_stage(PipelineStage.WEIGHT_EXPORT)

    with pipeline_service._lock:
        pipeline_service.tasks["pipe-status-1"] = task

    status = pipeline_service.get_status("pipe-status-1")
    assert isinstance(status, PipelineStatusData)
    assert status.task_id == "pipe-status-1"
    assert status.status == "running"
    assert status.current_stage == PipelineStage.WEIGHT_EXPORT
    assert len(status.stages) == 4
    assert status.stages[0].stage_name == PipelineStage.PPO_TRAINING
    assert status.stages[0].status == "completed"
    assert status.stages[1].stage_name == PipelineStage.WEIGHT_EXPORT
    assert status.stages[1].status == "running"
    assert status.stages[2].stage_name == PipelineStage.GA_SEARCH
    assert status.stages[2].status == "pending"


def test_pipeline_service_get_report_completed(pipeline_service, pipeline_request):
    task = PipelineTask("pipe-report-1", pipeline_request)
    task.status = "completed"
    task.comparison_report = ComparisonReportData(
        ppo_best_reward=250.0,
        ppo_avg_reward=220.0,
        ga_best_fitness=200.0,
        ga_best_seeds=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                       [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
        weight_distance=10.5,
        performance_gap=50.0,
        target_weight_norm=50.0,
        ga_weight_norm=45.0,
        similarity_score=0.79,
    )

    with pipeline_service._lock:
        pipeline_service.tasks["pipe-report-1"] = task

    report = pipeline_service.get_report("pipe-report-1")
    assert isinstance(report, ComparisonReportData)
    assert report.ppo_best_reward == 250.0
    assert report.ga_best_fitness == 200.0
    assert report.performance_gap == 50.0
    assert report.similarity_score == 0.79


def test_pipeline_service_get_report_not_available(pipeline_service, pipeline_request):
    task = PipelineTask("pipe-report-2", pipeline_request)
    task.status = "running"

    with pipeline_service._lock:
        pipeline_service.tasks["pipe-report-2"] = task

    with pytest.raises(PipelineException):
        pipeline_service.get_report("pipe-report-2")


def test_compute_weight_distance_identical(state_dim, action_dim):
    net = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    w1 = net.export_weight_dict()
    w2 = net.export_weight_dict()
    assert _compute_weight_distance(w1, w2) == 0.0


def test_compute_weight_distance_different(state_dim, action_dim):
    net1 = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    net2 = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    w1 = net1.export_weight_dict()
    w2 = net2.export_weight_dict()
    distance = _compute_weight_distance(w1, w2)
    assert distance > 0.0


def test_compute_weight_distance_symmetric(state_dim, action_dim):
    net1 = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    net2 = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    w1 = net1.export_weight_dict()
    w2 = net2.export_weight_dict()
    d1 = _compute_weight_distance(w1, w2)
    d2 = _compute_weight_distance(w2, w1)
    assert abs(d1 - d2) < 1e-6


def test_differentiable_network_export_weight_vector(state_dim, action_dim):
    net = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    vector = net.export_weight_vector()
    assert isinstance(vector, torch.Tensor)
    assert vector.dim() == 1
    total_params = sum(p.numel() for p in net.parameters())
    assert vector.shape[0] == total_params


def test_differentiable_network_export_weight_dict(state_dim, action_dim):
    net = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    weight_dict = net.export_weight_dict()
    assert isinstance(weight_dict, dict)
    for name, param in net.named_parameters():
        assert name in weight_dict
        assert torch.equal(weight_dict[name], param.data)


def test_differentiable_network_export_weight_vector_consistency(state_dim, action_dim):
    net = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
    vector = net.export_weight_vector()
    flat_from_dict = torch.cat([weight.flatten() for weight in net.export_weight_dict().values()])
    assert torch.allclose(vector, flat_from_dict)


def test_pipeline_start_request_defaults():
    req = PipelineStartRequest(
        ppo_config=TrainingStartRequest(),
        ga_config=GeneticStartRequest(),
    )
    assert req.target_seeds is None
    assert req.weight_similarity_coef == 0.1


def test_pipeline_start_request_with_target_seeds():
    seeds = list(range(1, 25))
    req = PipelineStartRequest(
        ppo_config=TrainingStartRequest(),
        ga_config=GeneticStartRequest(),
        target_seeds=seeds,
        weight_similarity_coef=0.2,
    )
    assert req.target_seeds == seeds
    assert req.weight_similarity_coef == 0.2


def test_pipeline_start_request_invalid_target_seeds():
    with pytest.raises(Exception):
        PipelineStartRequest(
            ppo_config=TrainingStartRequest(),
            ga_config=GeneticStartRequest(),
            target_seeds=[1, 2, 3],
        )


def test_pipeline_service_start_pipeline_returns_task_id(pipeline_service):
    request = PipelineStartRequest(
        ppo_config=TrainingStartRequest(total_episodes=5, max_steps=100, env_name="CartPole-v1"),
        ga_config=GeneticStartRequest(
            max_generations=2,
            population_size=10,
            evaluation_episodes=1,
            env_name="CartPole-v1",
        ),
    )
    task_id = pipeline_service.start_pipeline(request)
    assert isinstance(task_id, str)
    assert len(task_id) > 0
    assert task_id in pipeline_service.tasks


from datetime import datetime
