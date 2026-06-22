import base64
import os
from datetime import datetime

import pytest

from src.services.training_service import TrainingService, TrainingTask
from src.services.evaluation_service import EvaluationService
from src.services.genetic_service import GeneticService, GeneticTask
from src.services.visualization_service import VisualizationService
from src.schemas.requests import TrainingStartRequest, GeneticStartRequest
from src.core.exceptions import TrainingException, GeneticAlgorithmException, ModelException


def test_training_task_initial_state():
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("task-1", config)
    assert task.task_id == "task-1"
    assert task.status == "pending"
    assert task.current_episode == 0
    assert task.total_episodes == 500
    assert task.best_reward == float("-inf")
    assert task.avg_reward_last_100 == 0.0
    assert task.started_at is None
    assert task.completed_at is None
    assert task.result is None
    assert not task.should_stop()


def test_training_task_request_stop():
    config = TrainingStartRequest(total_episodes=100)
    task = TrainingTask("task-2", config)
    assert not task.should_stop()
    task.request_stop()
    assert task.should_stop()


def test_training_task_get_progress():
    config = TrainingStartRequest(total_episodes=200)
    task = TrainingTask("task-3", config)
    assert task.get_progress() == 0.0
    task.current_episode = 50
    assert task.get_progress() == pytest.approx(25.0)
    task.current_episode = 200
    assert task.get_progress() == pytest.approx(100.0)


def test_training_task_get_progress_zero_episodes():
    config = TrainingStartRequest(total_episodes=1)
    task = TrainingTask("task-4", config)
    task.total_episodes = 0
    assert task.get_progress() == 0.0


def test_training_service_list_tasks_empty():
    service = TrainingService(max_concurrent_tasks=1)
    assert service.list_tasks() == []


def test_training_service_get_status_not_found_raises():
    service = TrainingService(max_concurrent_tasks=1)
    with pytest.raises(TrainingException):
        service.get_status("nonexistent-id")


def test_training_service_stop_training_not_found_raises():
    service = TrainingService(max_concurrent_tasks=1)
    with pytest.raises(TrainingException):
        service.stop_training("nonexistent-id")


def test_genetic_task_initial_state():
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("ga-1", config)
    assert task.task_id == "ga-1"
    assert task.status == "pending"
    assert task.current_generation == 0
    assert task.max_generations == 100
    assert task.best_fitness == float("-inf")
    assert task.best_individual is None
    assert task.started_at is None
    assert task.completed_at is None
    assert not task.should_stop()


def test_genetic_task_request_stop():
    config = GeneticStartRequest(max_generations=50)
    task = GeneticTask("ga-2", config)
    assert not task.should_stop()
    task.request_stop()
    assert task.should_stop()


def test_genetic_task_get_progress():
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("ga-3", config)
    assert task.get_progress() == 0.0
    task.current_generation = 25
    assert task.get_progress() == pytest.approx(25.0)
    task.current_generation = 100
    assert task.get_progress() == pytest.approx(100.0)


def test_genetic_task_get_progress_zero_generations():
    config = GeneticStartRequest(max_generations=1)
    task = GeneticTask("ga-4", config)
    task.max_generations = 0
    assert task.get_progress() == 0.0


def test_genetic_service_list_tasks_empty():
    service = GeneticService(max_concurrent_tasks=1)
    assert service.list_tasks() == []


def test_genetic_service_get_status_not_found_raises():
    service = GeneticService(max_concurrent_tasks=1)
    with pytest.raises(GeneticAlgorithmException):
        service.get_status("nonexistent-id")


def test_evaluation_service_evaluate_with_seeds_invalid_count_raises():
    service = EvaluationService()
    with pytest.raises(ModelException, match="Expected 24 seeds"):
        service.evaluate_with_seeds(seeds=[1, 2, 3], env_name="LunarLander-v2", num_episodes=1)


def test_evaluation_service_evaluate_with_seeds_empty_raises():
    service = EvaluationService()
    with pytest.raises(ModelException, match="Expected 24 seeds"):
        service.evaluate_with_seeds(seeds=[], env_name="LunarLander-v2", num_episodes=1)


def test_visualization_service_generate_fitness_curve_returns_base64(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    history = [10.0, 50.0, 80.0, 120.0, 160.0, 200.0, 220.0, 210.0, 230.0, 250.0,
               240.0, 260.0, 270.0, 280.0, 290.0]
    result = service.generate_fitness_curve(fitness_history=history)
    assert isinstance(result, str)
    assert len(result) > 0
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def test_visualization_service_generate_fitness_curve_empty_returns_empty(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    result = service.generate_fitness_curve(fitness_history=[])
    assert result == ""


def test_visualization_service_generate_training_dashboard_returns_base64(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    rewards = [float(i) for i in range(150)]
    policy_losses = [0.5 - i * 0.001 for i in range(50)]
    value_losses = [1.0 - i * 0.005 for i in range(50)]
    temperatures = [1.0 * (0.995 ** i) for i in range(50)]
    result = service.generate_training_dashboard(
        episode_rewards=rewards,
        policy_losses=policy_losses,
        value_losses=value_losses,
        temperatures=temperatures,
        task_id="test-dash"
    )
    assert isinstance(result, str)
    assert len(result) > 0
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def test_visualization_service_generate_genetic_progress_returns_base64(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    fitness_history = [50.0, 80.0, 120.0, 160.0, 200.0, 220.0, 240.0, 260.0, 270.0, 280.0]
    avg_fitness = [30.0, 50.0, 80.0, 110.0, 150.0, 170.0, 190.0, 210.0, 220.0, 230.0]
    result = service.generate_genetic_progress(
        fitness_history=fitness_history,
        avg_fitness_history=avg_fitness,
        task_id="test-ga"
    )
    assert isinstance(result, str)
    assert len(result) > 0
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def test_visualization_service_generate_comparison_chart_returns_base64(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    diff_rewards = [200.0 + float(i) for i in range(30)]
    non_diff_rewards = [150.0 + float(i) for i in range(30)]
    result = service.generate_comparison_chart(
        diff_rewards=diff_rewards,
        non_diff_rewards=non_diff_rewards,
        title="Test Comparison"
    )
    assert isinstance(result, str)
    assert len(result) > 0
    decoded = base64.b64decode(result)
    assert decoded[:4] == b"\x89PNG"


def test_visualization_service_moving_average(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = service._moving_average(data, 3)
    assert len(result) == 8
    assert result[0] == pytest.approx(2.0)
    assert result[1] == pytest.approx(3.0)
    assert result[-1] == pytest.approx(9.0)


def test_visualization_service_moving_average_window_exceeds_data(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    data = [1.0, 2.0, 3.0]
    result = service._moving_average(data, 10)
    assert result == data


def test_visualization_service_moving_average_single_element(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    result = service._moving_average([5.0], 1)
    assert result == [5.0]


def test_training_service_start_training_returns_task_id():
    service = TrainingService(max_concurrent_tasks=1)
    request = TrainingStartRequest(
        total_episodes=10,
        max_steps=100,
        env_name="CartPole-v1"
    )
    task_id = service.start_training(request)
    assert isinstance(task_id, str)
    assert len(task_id) > 0
    assert task_id in service.tasks


def test_training_service_get_status_returns_correct_data():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-1", config)
    task.status = "running"
    task.current_episode = 100
    task.best_reward = 150.0
    task.avg_reward_last_100 = 100.0

    with service._lock:
        service.tasks["test-task-1"] = task

    status = service.get_status("test-task-1")
    assert status.task_id == "test-task-1"
    assert status.status == "running"
    assert status.current_episode == 100
    assert status.total_episodes == 500
    assert status.best_reward == 150.0
    assert status.avg_reward_last_100 == 100.0
    assert status.progress == pytest.approx(20.0)


def test_training_service_stop_training_running_task():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-2", config)
    task.status = "running"

    with service._lock:
        service.tasks["test-task-2"] = task

    result = service.stop_training("test-task-2")
    assert result is True
    assert task.should_stop()


def test_training_service_stop_training_completed_task():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-3", config)
    task.status = "completed"

    with service._lock:
        service.tasks["test-task-3"] = task

    result = service.stop_training("test-task-3")
    assert result is False


def test_training_service_get_history_empty_result():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-4", config)

    with service._lock:
        service.tasks["test-task-4"] = task

    history, total = service.get_history("test-task-4")
    assert history == []
    assert total == 0


def test_training_service_get_history_with_result():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-5", config)

    from src.models.ppo_agent import TrainingResult
    task.result = TrainingResult(
        episode_rewards=[100.0, 150.0, 200.0],
        episode_lengths=[50, 75, 100],
        policy_losses=[0.5, 0.3],
        value_losses=[1.0, 0.8],
        temperatures=[1.0, 0.9],
        best_reward=200.0
    )

    with service._lock:
        service.tasks["test-task-5"] = task

    history, total = service.get_history("test-task-5", page=1, page_size=10)
    assert total == 3
    assert len(history) == 3
    assert history[0].episode == 1
    assert history[0].reward == 100.0
    assert history[0].length == 50


def test_training_service_get_history_pagination():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-6", config)

    from src.models.ppo_agent import TrainingResult
    task.result = TrainingResult(
        episode_rewards=[float(i) for i in range(25)],
        episode_lengths=[50] * 25,
    )

    with service._lock:
        service.tasks["test-task-6"] = task

    history, total = service.get_history("test-task-6", page=2, page_size=10)
    assert total == 25
    assert len(history) == 10
    assert history[0].episode == 11
    assert history[0].reward == 10.0


def test_training_service_get_result_completed():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-7", config)
    task.status = "completed"
    task.current_episode = 500
    task.best_reward = 250.0
    task.avg_reward_last_100 = 220.0
    task.started_at = datetime(2024, 1, 1, 10, 0, 0)
    task.completed_at = datetime(2024, 1, 1, 11, 0, 0)

    from src.models.ppo_agent import TrainingResult
    task.result = TrainingResult(
        episode_rewards=[200.0, 250.0, 220.0],
        best_reward=250.0
    )

    with service._lock:
        service.tasks["test-task-7"] = task

    result = service.get_result("test-task-7")
    assert result.task_id == "test-task-7"
    assert result.status == "completed"
    assert result.total_episodes == 500
    assert result.best_reward == 250.0
    assert result.avg_reward_last_100 == 220.0
    assert result.training_duration_seconds == 3600.0


def test_training_service_get_result_not_completed_raises():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-8", config)
    task.status = "running"

    with service._lock:
        service.tasks["test-task-8"] = task

    with pytest.raises(TrainingException):
        service.get_result("test-task-8")


def test_training_service_get_active_task_count():
    service = TrainingService(max_concurrent_tasks=2)

    config = TrainingStartRequest(total_episodes=100)
    task1 = TrainingTask("task-active-1", config)
    task1.status = "running"
    task2 = TrainingTask("task-active-2", config)
    task2.status = "completed"
    task3 = TrainingTask("task-active-3", config)
    task3.status = "running"

    with service._lock:
        service.tasks["task-active-1"] = task1
        service.tasks["task-active-2"] = task2
        service.tasks["task-active-3"] = task3

    assert service.get_active_task_count() == 2


def test_genetic_service_start_search_returns_task_id():
    service = GeneticService(max_concurrent_tasks=1)
    request = GeneticStartRequest(
        max_generations=10,
        population_size=10,
        env_name="CartPole-v1"
    )
    task_id = service.start_search(request)
    assert isinstance(task_id, str)
    assert len(task_id) > 0
    assert task_id in service.tasks


def test_genetic_service_get_status_returns_correct_data():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100, population_size=50)
    task = GeneticTask("test-ga-1", config)
    task.status = "running"
    task.current_generation = 25
    task.best_fitness = 180.0
    task.elite_archive_size = 5

    from src.models.genetic_algorithm import Individual
    import numpy as np
    task.best_individual = Individual.create_random()
    task.best_individual.fitness = 180.0
    task.best_individual.generation = 20

    with service._lock:
        service.tasks["test-ga-1"] = task

    status = service.get_status("test-ga-1")
    assert status.task_id == "test-ga-1"
    assert status.status == "running"
    assert status.current_generation == 25
    assert status.max_generations == 100
    assert status.best_fitness == 180.0
    assert status.population_size == 50
    assert status.elite_archive_size == 5
    assert status.progress == pytest.approx(25.0)
    assert status.best_individual is not None
    assert status.best_individual.fitness == 180.0


def test_genetic_service_stop_search_running_task():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("test-ga-2", config)
    task.status = "running"

    with service._lock:
        service.tasks["test-ga-2"] = task

    result = service.stop_search("test-ga-2")
    assert result is True
    assert task.should_stop()


def test_genetic_service_stop_search_completed_task():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("test-ga-3", config)
    task.status = "completed"

    with service._lock:
        service.tasks["test-ga-3"] = task

    result = service.stop_search("test-ga-3")
    assert result is False


def test_genetic_service_get_best_individual():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("test-ga-4", config)

    from src.models.genetic_algorithm import Individual
    import numpy as np
    seeds = np.array([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24]
    ])
    task.best_individual = Individual(seeds=seeds, fitness=200.0, generation=50)

    with service._lock:
        service.tasks["test-ga-4"] = task

    result = service.get_best_individual("test-ga-4")
    assert result.fitness == 200.0
    assert result.generation == 50
    assert len(result.seeds) == 4
    assert len(result.seeds[0]) == 6


def test_genetic_service_get_best_individual_not_found_raises():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("test-ga-5", config)

    with service._lock:
        service.tasks["test-ga-5"] = task

    with pytest.raises(GeneticAlgorithmException):
        service.get_best_individual("test-ga-5")


def test_genetic_service_get_population():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100, population_size=10)
    task = GeneticTask("test-ga-6", config)
    task.current_generation = 10
    task.best_fitness = 280.0

    from src.models.genetic_algorithm import GeneticAlgorithm, Individual
    ga = GeneticAlgorithm(population_size=10, traversal_enabled=False)
    ga.initialize_population()
    for i, ind in enumerate(ga.population):
        ind.fitness = float(100 + i * 20)

    task.ga = ga

    with service._lock:
        service.tasks["test-ga-6"] = task

    result = service.get_population("test-ga-6")
    assert result.generation == 10
    assert len(result.individuals) == 10
    assert result.best_fitness == pytest.approx(280.0)
    assert result.avg_fitness == pytest.approx(190.0)


def test_genetic_service_get_population_empty_raises():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("test-ga-7", config)

    with service._lock:
        service.tasks["test-ga-7"] = task

    with pytest.raises(GeneticAlgorithmException):
        service.get_population("test-ga-7")


def test_genetic_service_get_active_task_count():
    service = GeneticService(max_concurrent_tasks=2)

    config = GeneticStartRequest(max_generations=100)
    task1 = GeneticTask("ga-active-1", config)
    task1.status = "running"
    task2 = GeneticTask("ga-active-2", config)
    task2.status = "failed"

    with service._lock:
        service.tasks["ga-active-1"] = task1
        service.tasks["ga-active-2"] = task2

    assert service.get_active_task_count() == 1


def test_evaluation_service_passing_score_constant():
    service = EvaluationService()
    assert service.PASSING_SCORE == 200.0


def test_visualization_service_output_dir_creation(tmp_path):
    output_dir = str(tmp_path / "test_plots")
    service = VisualizationService(output_dir=output_dir)
    assert os.path.exists(output_dir)


def test_visualization_service_fitness_curve_with_save_path(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    history = [10.0, 50.0, 100.0, 150.0, 200.0]
    save_path = "test_fitness.png"

    result = service.generate_fitness_curve(
        fitness_history=history,
        save_path=save_path,
        show_avg=False
    )

    assert isinstance(result, str)
    assert len(result) > 0
    full_path = os.path.join(str(tmp_path), save_path)
    assert os.path.exists(full_path)


def test_visualization_service_fitness_curve_mark_max(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    history = [10.0, 50.0, 30.0, 80.0, 60.0]
    result = service.generate_fitness_curve(fitness_history=history)
    assert isinstance(result, str)
    assert len(result) > 0


def test_visualization_service_training_dashboard_empty_data(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    result = service.generate_training_dashboard(
        episode_rewards=[],
        policy_losses=[],
        value_losses=[],
        temperatures=[],
        task_id="empty-dash"
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_visualization_service_genetic_progress_with_avg(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    fitness_history = [50.0, 80.0, 120.0, 150.0, 180.0]
    avg_fitness = [40.0, 60.0, 90.0, 110.0, 140.0]

    result = service.generate_genetic_progress(
        fitness_history=fitness_history,
        avg_fitness_history=avg_fitness,
        task_id="test-ga-progress"
    )

    assert isinstance(result, str)
    assert len(result) > 0


def test_visualization_service_comparison_chart_with_title(tmp_path):
    service = VisualizationService(output_dir=str(tmp_path))
    diff_rewards = [180.0, 200.0, 220.0, 190.0, 210.0]
    non_diff_rewards = [150.0, 170.0, 160.0, 180.0, 175.0]

    result = service.generate_comparison_chart(
        diff_rewards=diff_rewards,
        non_diff_rewards=non_diff_rewards,
        title="Custom Comparison Title"
    )

    assert isinstance(result, str)
    assert len(result) > 0


def test_visualization_service_moving_average_window_1():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = VisualizationService(output_dir=tmp_dir)
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = service._moving_average(data, 1)
        assert result == data


def test_visualization_service_moving_average_equal_length():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = VisualizationService(output_dir=tmp_dir)
        data = [1.0, 2.0, 3.0]
        result = service._moving_average(data, 3)
        assert len(result) == 1
        assert result[0] == pytest.approx(2.0)


def test_training_service_get_result_stopped():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-task-stopped", config)
    task.status = "stopped"
    task.current_episode = 250
    task.best_reward = 180.0

    from src.models.ppo_agent import TrainingResult
    task.result = TrainingResult(
        episode_rewards=[100.0, 150.0, 180.0],
        best_reward=180.0
    )

    with service._lock:
        service.tasks["test-task-stopped"] = task

    result = service.get_result("test-task-stopped")
    assert result.status == "stopped"
    assert result.total_episodes == 250


def test_genetic_service_stop_search_not_found_raises():
    service = GeneticService(max_concurrent_tasks=1)
    with pytest.raises(GeneticAlgorithmException):
        service.stop_search("nonexistent")


def test_training_service_stop_training_not_found_raises():
    service = TrainingService(max_concurrent_tasks=1)
    with pytest.raises(TrainingException):
        service.stop_training("nonexistent")


def test_training_task_auto_start_ga_default():
    config = TrainingStartRequest(total_episodes=100)
    task = TrainingTask("task-auto-1", config)
    assert task.auto_start_ga is False
    assert task.child_ga_task_id is None


def test_training_task_auto_start_ga_enabled():
    from src.schemas.requests import GeneticStartRequest
    ga_config = GeneticStartRequest(max_generations=50, population_size=20)
    config = TrainingStartRequest(
        total_episodes=100,
        auto_start_ga=True,
        ga_config=ga_config
    )
    task = TrainingTask("task-auto-2", config)
    assert task.auto_start_ga is True
    assert task.child_ga_task_id is None
    assert task.config.ga_config is not None
    assert task.config.ga_config.max_generations == 50


def test_training_start_request_auto_start_ga_requires_ga_config():
    from src.schemas.requests import GeneticStartRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="ga_config is required"):
        TrainingStartRequest(total_episodes=100, auto_start_ga=True)

    ga_config = GeneticStartRequest(max_generations=10)
    req = TrainingStartRequest(total_episodes=100, auto_start_ga=True, ga_config=ga_config)
    assert req.auto_start_ga is True
    assert req.ga_config is not None


def test_genetic_task_parent_task_id_default():
    config = GeneticStartRequest(max_generations=100)
    task = GeneticTask("ga-parent-1", config)
    assert task.parent_task_id is None


def test_genetic_task_parent_task_id_set():
    config = GeneticStartRequest(max_generations=100, parent_task_id="ppo-task-123")
    task = GeneticTask("ga-parent-2", config)
    assert task.parent_task_id == "ppo-task-123"


def test_training_service_set_genetic_service():
    training_service = TrainingService(max_concurrent_tasks=1)
    genetic_service = GeneticService(max_concurrent_tasks=1)
    assert training_service._genetic_service is None
    training_service.set_genetic_service(genetic_service)
    assert training_service._genetic_service is genetic_service


def test_training_service_get_status_includes_auto_start_ga():
    from src.schemas.requests import GeneticStartRequest
    ga_config = GeneticStartRequest(max_generations=50)
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500, auto_start_ga=True, ga_config=ga_config)
    task = TrainingTask("test-status-auto", config)
    task.status = "running"
    task.current_episode = 100
    task.avg_reward_last_100 = 150.0

    with service._lock:
        service.tasks["test-status-auto"] = task

    status = service.get_status("test-status-auto")
    assert status.auto_start_ga is True
    assert status.child_ga_task_id is None


def test_training_service_get_status_includes_child_ga_task_id():
    service = TrainingService(max_concurrent_tasks=1)
    config = TrainingStartRequest(total_episodes=500)
    task = TrainingTask("test-status-child", config)
    task.status = "completed"
    task.child_ga_task_id = "ga-child-123"

    with service._lock:
        service.tasks["test-status-child"] = task

    status = service.get_status("test-status-child")
    assert status.child_ga_task_id == "ga-child-123"


def test_genetic_service_get_status_includes_parent_task_id():
    service = GeneticService(max_concurrent_tasks=1)
    config = GeneticStartRequest(max_generations=100, parent_task_id="ppo-parent-456")
    task = GeneticTask("test-ga-parent", config)
    task.status = "running"
    task.current_generation = 10

    with service._lock:
        service.tasks["test-ga-parent"] = task

    status = service.get_status("test-ga-parent")
    assert status.parent_task_id == "ppo-parent-456"


def test_training_service_list_tasks_includes_auto_fields():
    from src.schemas.requests import GeneticStartRequest
    ga_config = GeneticStartRequest(max_generations=50)
    service = TrainingService(max_concurrent_tasks=2)
    config1 = TrainingStartRequest(total_episodes=100, auto_start_ga=True, ga_config=ga_config)
    config2 = TrainingStartRequest(total_episodes=200, auto_start_ga=False)
    task1 = TrainingTask("list-auto-1", config1)
    task2 = TrainingTask("list-auto-2", config2)
    task2.child_ga_task_id = "ga-child-xyz"

    with service._lock:
        service.tasks["list-auto-1"] = task1
        service.tasks["list-auto-2"] = task2

    tasks = service.list_tasks()
    assert len(tasks) == 2

    task_map = {t.task_id: t for t in tasks}
    assert task_map["list-auto-1"].auto_start_ga is True
    assert task_map["list-auto-1"].child_ga_task_id is None
    assert task_map["list-auto-2"].auto_start_ga is False
    assert task_map["list-auto-2"].child_ga_task_id == "ga-child-xyz"


def test_genetic_service_list_tasks_includes_parent_task_id():
    service = GeneticService(max_concurrent_tasks=2)
    config1 = GeneticStartRequest(max_generations=50, parent_task_id="ppo-1")
    config2 = GeneticStartRequest(max_generations=50)
    task1 = GeneticTask("list-ga-1", config1)
    task2 = GeneticTask("list-ga-2", config2)

    with service._lock:
        service.tasks["list-ga-1"] = task1
        service.tasks["list-ga-2"] = task2

    tasks = service.list_tasks()
    assert len(tasks) == 2

    task_map = {t.task_id: t for t in tasks}
    assert task_map["list-ga-1"].parent_task_id == "ppo-1"
    assert task_map["list-ga-2"].parent_task_id is None


def test_maybe_auto_start_ga_no_auto_start_flag():
    training_service = TrainingService(max_concurrent_tasks=1)
    genetic_service = GeneticService(max_concurrent_tasks=1)
    training_service.set_genetic_service(genetic_service)

    config = TrainingStartRequest(total_episodes=100, auto_start_ga=False)
    task = TrainingTask("test-no-auto", config)
    task.status = "completed"
    task.avg_reward_last_100 = 250.0

    training_service._maybe_auto_start_ga(task, "/tmp/fake_model.pt")
    assert task.child_ga_task_id is None


def test_maybe_auto_start_ga_not_completed():
    training_service = TrainingService(max_concurrent_tasks=1)
    genetic_service = GeneticService(max_concurrent_tasks=1)
    training_service.set_genetic_service(genetic_service)

    from src.schemas.requests import GeneticStartRequest
    ga_config = GeneticStartRequest(max_generations=10)
    config = TrainingStartRequest(total_episodes=100, auto_start_ga=True, ga_config=ga_config)
    task = TrainingTask("test-not-completed", config)
    task.status = "running"
    task.avg_reward_last_100 = 250.0

    training_service._maybe_auto_start_ga(task, "/tmp/fake_model.pt")
    assert task.child_ga_task_id is None


def test_maybe_auto_start_ga_reward_below_threshold():
    training_service = TrainingService(max_concurrent_tasks=1)
    genetic_service = GeneticService(max_concurrent_tasks=1)
    training_service.set_genetic_service(genetic_service)

    from src.schemas.requests import GeneticStartRequest
    ga_config = GeneticStartRequest(max_generations=10)
    config = TrainingStartRequest(total_episodes=100, auto_start_ga=True, ga_config=ga_config)
    task = TrainingTask("test-low-reward", config)
    task.status = "completed"
    task.avg_reward_last_100 = 150.0

    training_service._maybe_auto_start_ga(task, "/tmp/fake_model.pt")
    assert task.child_ga_task_id is None


def test_maybe_auto_start_ga_no_genetic_service():
    training_service = TrainingService(max_concurrent_tasks=1)

    from src.schemas.requests import GeneticStartRequest
    ga_config = GeneticStartRequest(max_generations=10)
    config = TrainingStartRequest(total_episodes=100, auto_start_ga=True, ga_config=ga_config)
    task = TrainingTask("test-no-ga-service", config)
    task.status = "completed"
    task.avg_reward_last_100 = 250.0

    training_service._maybe_auto_start_ga(task, "/tmp/fake_model.pt")
    assert task.child_ga_task_id is None


def test_maybe_auto_start_ga_no_ga_config():
    training_service = TrainingService(max_concurrent_tasks=1)
    genetic_service = GeneticService(max_concurrent_tasks=1)
    training_service.set_genetic_service(genetic_service)

    config = TrainingStartRequest(total_episodes=100, auto_start_ga=False)
    task = TrainingTask("test-no-ga-config", config)
    task.status = "completed"
    task.avg_reward_last_100 = 250.0
    task.auto_start_ga = True

    training_service._maybe_auto_start_ga(task, "/tmp/fake_model.pt")
    assert task.child_ga_task_id is None


def test_maybe_auto_start_ga_success():
    training_service = TrainingService(max_concurrent_tasks=1)
    genetic_service = GeneticService(max_concurrent_tasks=1)
    training_service.set_genetic_service(genetic_service)

    from src.schemas.requests import GeneticStartRequest
    ga_config = GeneticStartRequest(
        max_generations=5,
        population_size=10,
        env_name="CartPole-v1"
    )
    config = TrainingStartRequest(
        total_episodes=100,
        auto_start_ga=True,
        ga_config=ga_config,
        env_name="CartPole-v1"
    )
    task = TrainingTask("test-auto-success", config)
    task.status = "completed"
    task.avg_reward_last_100 = 250.0

    training_service._maybe_auto_start_ga(task, "/tmp/fake_model.pt")
    assert task.child_ga_task_id is not None
    assert task.child_ga_task_id in genetic_service.tasks

    ga_task = genetic_service.tasks[task.child_ga_task_id]
    assert ga_task.parent_task_id == "test-auto-success"
    assert ga_task.config.target_weights_path == "/tmp/fake_model.pt"


# ==================== 新增: VisualizationService generate() 方法测试 ====================


def test_visualization_service_generate_fitness_curve_raw_data():
    service = VisualizationService(output_dir="/tmp/test_plots_1")
    result = service.generate(
        chart_type="fitness_curve",
        data={"fitness_history": [50.0, 80.0, 120.0, 150.0, 200.0, 210.0]},
        task_id="test-task-001",
        window_size=3,
        save_to_plots=False,
        fmt="base64",
    )
    assert "image_base64" in result
    assert result["image_base64"] is not None
    assert isinstance(result["image_base64"], str)
    assert len(result["image_base64"]) > 0
    assert "stats" in result
    assert result["stats"]["mean"] == pytest.approx(135.0)
    assert result["stats"]["max"] == 210.0
    assert result["stats"]["count"] == 6
    assert result["stats"]["passed"] is False
    assert result["width"] == 1500
    assert result["height"] == 900
    # format=base64 且 save=false -> 不应有文件相关字段
    assert "file_path" not in result
    assert "file_url" not in result


def test_visualization_service_generate_save_to_plots_file_url_format():
    output = "/tmp/test_plots_2"
    service = VisualizationService(output_dir=output, base_url="/plots")
    result = service.generate(
        chart_type="fitness_curve",
        data={"fitness_history": [100.0, 150.0, 200.0, 250.0]},
        fmt="file_url",
    )
    assert "file_url" in result
    assert result["file_url"].startswith("/plots/")
    assert "file_path" in result
    assert os.path.exists(result["file_path"])
    assert result["file_path"].startswith(os.path.abspath(output))
    # format=file_url -> 不应有base64
    assert "image_base64" not in result
    # 清理
    if os.path.exists(result["file_path"]):
        os.remove(result["file_path"])


def test_visualization_service_generate_both_format():
    output = "/tmp/test_plots_3"
    service = VisualizationService(output_dir=output)
    result = service.generate(
        chart_type="fitness_curve",
        data={"fitness_history": [80.0, 160.0, 240.0]},
        fmt="both",
    )
    assert "image_base64" in result
    assert "file_url" in result
    assert "file_path" in result
    assert os.path.exists(result["file_path"])
    # 清理
    if os.path.exists(result["file_path"]):
        os.remove(result["file_path"])


def test_visualization_service_generate_dashboard_with_all_fields():
    service = VisualizationService(output_dir="/tmp/test_plots_4")
    result = service.generate(
        chart_type="dashboard",
        data={
            "episode_rewards": list(range(120)) + [200.0] * 50,
            "policy_losses": [0.5, 0.4, 0.35, 0.3, 0.28],
            "value_losses": [1.2, 1.0, 0.9, 0.85, 0.8],
            "temperatures": [1.0, 0.95, 0.9, 0.85, 0.8],
        },
        title="Custom Dashboard Title",
    )
    assert result["width"] == 2100
    assert result["height"] == 1500
    assert "image_base64" in result
    assert result["stats"]["has_policy_losses"] is True
    assert result["stats"]["has_value_losses"] is True
    assert result["stats"]["has_temperatures"] is True
    assert result["stats"]["count"] == 170


def test_visualization_service_generate_progress_with_avg():
    service = VisualizationService(output_dir="/tmp/test_plots_5")
    result = service.generate(
        chart_type="progress",
        data={
            "fitness_history": [50.0, 80.0, 120.0, 180.0, 220.0],
            "avg_fitness_history": [45.0, 70.0, 100.0, 150.0, 190.0],
        },
    )
    assert result["stats"]["max"] == 220.0
    assert result["stats"]["count"] == 5
    assert "image_base64" in result


def test_visualization_service_generate_comparison_raw():
    service = VisualizationService(output_dir="/tmp/test_plots_6")
    diff = [205.0, 210.0, 220.0, 195.0, 215.0]
    non_diff = [170.0, 180.0, 165.0, 190.0, 175.0]
    result = service.generate(
        chart_type="comparison",
        data={"diff_rewards": diff, "non_diff_rewards": non_diff},
    )
    assert "stats" in result
    assert result["stats"]["differentiable"]["mean"] == pytest.approx(209.0)
    assert result["stats"]["non_differentiable"]["mean"] == pytest.approx(176.0)
    assert result["stats"]["performance_gap"] == pytest.approx(33.0)
    assert result["stats"]["differentiable"]["passed"] is True
    assert result["stats"]["non_differentiable"]["passed"] is False


def test_visualization_service_generate_empty_data_raises():
    service = VisualizationService()
    with pytest.raises(ValueError):
        service.generate(
            chart_type="fitness_curve",
            data={"fitness_history": []},
        )
    with pytest.raises(ValueError):
        service.generate(
            chart_type="dashboard",
            data={"episode_rewards": []},
        )
    with pytest.raises(ValueError):
        service.generate(
            chart_type="progress",
            data={"fitness_history": []},
        )
    with pytest.raises(ValueError):
        service.generate(
            chart_type="comparison",
            data={"diff_rewards": [], "non_diff_rewards": []},
        )


def test_visualization_service_generate_unknown_chart_type_raises():
    service = VisualizationService()
    with pytest.raises(ValueError):
        service.generate(
            chart_type="unknown_chart",  # type: ignore[arg-type]
            data={},
        )


def test_visualization_service_generate_missing_keys_raises():
    service = VisualizationService()
    with pytest.raises(ValueError):
        service.generate(
            chart_type="fitness_curve",
            data={"wrong_key": [1.0, 2.0]},
        )
    with pytest.raises(ValueError):
        service.generate(
            chart_type="dashboard",
            data={"wrong_key": [1.0, 2.0]},
        )
    with pytest.raises(ValueError):
        service.generate(
            chart_type="progress",
            data={"wrong_key": [1.0, 2.0]},
        )
    with pytest.raises(ValueError):
        service.generate(
            chart_type="comparison",
            data={"diff_rewards": [1.0]},
        )


# ==================== 新增: VisualizationService generate_comparison() 方法测试 ====================


def test_visualization_service_generate_comparison_base64_only():
    service = VisualizationService(output_dir="/tmp/test_plots_7")
    diff = [200.0 + i for i in range(15)]
    non_diff = [150.0 + i for i in range(15)]
    result = service.generate_comparison(
        diff_rewards=diff,
        non_diff_rewards=non_diff,
        fmt="base64",
        save_to_plots=False,
    )
    assert "boxplot_base64" in result
    assert "histogram_base64" in result
    assert "combined_base64" in result
    assert isinstance(result["boxplot_base64"], str)
    assert len(result["boxplot_base64"]) > 0
    assert isinstance(result["histogram_base64"], str)
    assert len(result["histogram_base64"]) > 0
    assert isinstance(result["combined_base64"], str)
    assert len(result["combined_base64"]) > 0
    # format=base64 且不保存 -> 不应有路径
    assert "boxplot_path" not in result
    assert "histogram_path" not in result
    assert "combined_path" not in result
    # 统计
    assert result["differentiable_stats"]["mean"] == pytest.approx(sum(diff) / len(diff))
    assert result["non_differentiable_stats"]["mean"] == pytest.approx(sum(non_diff) / len(non_diff))
    assert result["performance_gap"] == pytest.approx((sum(diff) - sum(non_diff)) / len(diff))


def test_visualization_service_generate_comparison_both_format_and_save():
    output = "/tmp/test_plots_8"
    service = VisualizationService(output_dir=output)
    diff = list(range(10))
    non_diff = list(range(10))
    result = service.generate_comparison(
        diff_rewards=diff,
        non_diff_rewards=non_diff,
        fmt="both",
        save_to_plots=True,
    )
    # Base64 存在
    assert "boxplot_base64" in result
    assert "histogram_base64" in result
    assert "combined_base64" in result
    # 路径存在
    for key in ("boxplot_path", "histogram_path", "combined_path"):
        assert key in result
        assert os.path.exists(result[key])
        assert result[key].startswith(os.path.abspath(output))
    # URL 存在
    for key in ("boxplot_url", "histogram_url", "combined_url"):
        assert key in result
        assert result[key].startswith("/plots/")
    # 清理
    for key in ("boxplot_path", "histogram_path", "combined_path"):
        if os.path.exists(result[key]):
            os.remove(result[key])


def test_visualization_service_generate_comparison_file_url_format():
    output = "/tmp/test_plots_9"
    service = VisualizationService(output_dir=output, base_url="/static/plots")
    result = service.generate_comparison(
        diff_rewards=[10.0, 20.0, 30.0],
        non_diff_rewards=[5.0, 15.0, 25.0],
        fmt="file_url",
    )
    # 自动保存 -> 路径存在
    assert "boxplot_path" in result
    assert os.path.exists(result["boxplot_path"])
    # URL 使用自定义 base_url
    assert result["boxplot_url"].startswith("/static/plots/")
    # format=file_url -> 无 base64
    assert "boxplot_base64" not in result
    assert "histogram_base64" not in result
    assert "combined_base64" not in result
    # 清理
    for key in ("boxplot_path", "histogram_path", "combined_path"):
        if key in result and os.path.exists(result[key]):
            os.remove(result[key])


def test_visualization_service_generate_comparison_custom_title():
    service = VisualizationService()
    result = service.generate_comparison(
        diff_rewards=[100.0, 200.0, 300.0],
        non_diff_rewards=[80.0, 180.0, 280.0],
        title="My Custom Comparison Title",
    )
    # 只要不报错并返回有效结果即通过
    assert result["performance_gap"] == pytest.approx(20.0)
    assert len(result["boxplot_base64"]) > 0


def test_visualization_service_cleanup_old_files(tmp_path):
    import time
    output_dir = str(tmp_path / "plots_cleanup")
    os.makedirs(output_dir, exist_ok=True)
    # 创建一些"老"文件
    old_file = os.path.join(output_dir, "old_plot.png")
    with open(old_file, "w") as f:
        f.write("fake")
    # 修改 mtime 为 48 小时前
    old_time = time.time() - 48 * 3600
    os.utime(old_file, (old_time, old_time))
    # 创建一个"新"文件
    new_file = os.path.join(output_dir, "new_plot.png")
    with open(new_file, "w") as f:
        f.write("fake")
    service = VisualizationService(output_dir=output_dir)
    removed = service.cleanup_old_files(max_age_hours=24)
    assert removed >= 1
    assert not os.path.exists(old_file)
    assert os.path.exists(new_file)
