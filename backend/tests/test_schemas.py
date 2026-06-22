import pytest
import math
from src.schemas.requests import (
    TrainingStartRequest,
    GeneticStartRequest,
    EvaluationRequest,
    ComparisonRequest,
    VisualizationRequest,
)
from src.schemas.responses import BaseResponse, safe_float


class TestTrainingStartRequest:
    def test_default_values(self):
        req = TrainingStartRequest(env_name="CartPole-v1")
        assert req.total_episodes == 1000
        assert req.max_steps == 1000
        assert req.learning_rate == 3e-4
        assert req.gamma == 0.99
        assert req.epsilon == 0.2
        assert req.initial_temperature == 1.0
        assert req.temperature_decay == 0.995
        assert req.min_temperature == 0.01
        assert req.regularization_coef == 0.1

    def test_invalid_learning_rate_zero(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", learning_rate=0)

    def test_invalid_total_episodes(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", total_episodes=0)
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", total_episodes=100001)


class TestGeneticStartRequest:
    def test_default_values(self):
        req = GeneticStartRequest(env_name="CartPole-v1")
        assert req.population_size == 50
        assert req.max_generations == 100
        assert req.mutation_rate == 0.1
        assert req.crossover_rate == 0.7
        assert req.elite_size == 5
        assert req.seed_range_min == 0
        assert req.seed_range_max == 10000
        assert req.target_fitness == 200.0
        assert req.evaluation_episodes == 5

    def test_invalid_seed_range_max_le_min(self):
        with pytest.raises(Exception):
            GeneticStartRequest(
                env_name="CartPole-v1",
                seed_range_min=100,
                seed_range_max=100,
            )
        with pytest.raises(Exception):
            GeneticStartRequest(
                env_name="CartPole-v1",
                seed_range_min=100,
                seed_range_max=50,
            )


class TestEvaluationRequest:
    def test_valid_network_types(self):
        req1 = EvaluationRequest(
            network_type="differentiable", num_episodes=10, env_name="CartPole-v1"
        )
        assert req1.network_type == "differentiable"
        req2 = EvaluationRequest(
            network_type="non_differentiable", num_episodes=10, env_name="CartPole-v1"
        )
        assert req2.network_type == "non_differentiable"

    def test_invalid_network_type(self):
        with pytest.raises(Exception):
            EvaluationRequest(
                network_type="invalid", num_episodes=10, env_name="CartPole-v1"
            )


class TestComparisonRequest:
    def test_valid_seeds_24(self):
        seeds = list(range(24))
        req = ComparisonRequest(
            num_episodes=10, non_differentiable_seeds=seeds, env_name="CartPole-v1"
        )
        assert len(req.non_differentiable_seeds) == 24

    def test_invalid_seeds_count(self):
        with pytest.raises(Exception):
            ComparisonRequest(
                num_episodes=10,
                non_differentiable_seeds=list(range(10)),
                env_name="CartPole-v1",
            )
        with pytest.raises(Exception):
            ComparisonRequest(
                num_episodes=10,
                non_differentiable_seeds=list(range(25)),
                env_name="CartPole-v1",
            )


class TestVisualizationRequest:
    def test_valid_chart_types(self):
        for chart_type in ["fitness_curve", "dashboard", "progress", "comparison"]:
            req = VisualizationRequest(chart_type=chart_type, task_id="test-task")
            assert req.chart_type == chart_type

    def test_invalid_chart_type(self):
        with pytest.raises(Exception):
            VisualizationRequest(chart_type="invalid", task_id="test-task")


class TestSafeFloat:
    def test_inf_returns_none(self):
        assert safe_float(float("inf")) is None
        assert safe_float(float("-inf")) is None

    def test_nan_returns_none(self):
        assert safe_float(float("nan")) is None

    def test_normal_returns_value(self):
        assert safe_float(3.14) == 3.14
        assert safe_float(0.0) == 0.0
        assert safe_float(-1.5) == -1.5

    def test_none_returns_none(self):
        assert safe_float(None) is None


class TestBaseResponse:
    def test_success_factory(self):
        resp = BaseResponse.success(data={"key": "value"}, message="ok")
        assert resp.code == 200
        assert resp.message == "ok"
        assert resp.data == {"key": "value"}
        assert resp.timestamp is not None

    def test_error_factory(self):
        resp = BaseResponse.error(message="fail", code=500)
        assert resp.code == 500
        assert resp.message == "fail"
        assert resp.data is None

    def test_success_factory_default_message(self):
        resp = BaseResponse.success(data="test")
        assert resp.code == 200
        assert resp.message == "success"
        assert resp.data == "test"

    def test_error_factory_defaults(self):
        resp = BaseResponse.error()
        assert resp.code == 500
        assert resp.message == "error"
        assert resp.data is None


class TestTrainingStartRequestValidation:
    def test_invalid_total_episodes_negative(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", total_episodes=-1)

    def test_invalid_max_steps(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", max_steps=50)

    def test_invalid_gamma_negative(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", gamma=-0.1)

    def test_invalid_epsilon_range(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", epsilon=1.5)

    def test_invalid_temperature_negative(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", initial_temperature=-1.0)

    def test_invalid_regularization_coef(self):
        with pytest.raises(Exception):
            TrainingStartRequest(env_name="CartPole-v1", regularization_coef=-0.5)

    def test_model_dump(self):
        req = TrainingStartRequest(
            total_episodes=500,
            max_steps=500,
            env_name="CartPole-v1"
        )
        data = req.model_dump()
        assert data['total_episodes'] == 500
        assert data['max_steps'] == 500
        assert data['env_name'] == "CartPole-v1"


class TestGeneticStartRequestValidation:
    def test_invalid_population_size(self):
        with pytest.raises(Exception):
            GeneticStartRequest(env_name="CartPole-v1", population_size=5)

    def test_invalid_max_generations(self):
        with pytest.raises(Exception):
            GeneticStartRequest(env_name="CartPole-v1", max_generations=0)

    def test_invalid_mutation_rate(self):
        with pytest.raises(Exception):
            GeneticStartRequest(env_name="CartPole-v1", mutation_rate=1.5)

    def test_invalid_crossover_rate(self):
        with pytest.raises(Exception):
            GeneticStartRequest(env_name="CartPole-v1", crossover_rate=-0.1)

    def test_invalid_elite_size(self):
        with pytest.raises(Exception):
            GeneticStartRequest(env_name="CartPole-v1", elite_size=100)

    def test_invalid_seed_range_min(self):
        with pytest.raises(Exception):
            GeneticStartRequest(env_name="CartPole-v1", seed_range_min=-1)

    def test_invalid_evaluation_episodes(self):
        with pytest.raises(Exception):
            GeneticStartRequest(env_name="CartPole-v1", evaluation_episodes=0)


class TestEvaluationRequestValidation:
    def test_invalid_num_episodes(self):
        with pytest.raises(Exception):
            EvaluationRequest(
                network_type="differentiable",
                num_episodes=0,
                env_name="CartPole-v1"
            )

    def test_valid_with_model_path(self):
        req = EvaluationRequest(
            network_type="differentiable",
            num_episodes=10,
            model_path="/tmp/model.pt",
            env_name="CartPole-v1"
        )
        assert req.model_path == "/tmp/model.pt"


class TestComparisonRequestValidation:
    def test_invalid_num_episodes(self):
        with pytest.raises(Exception):
            ComparisonRequest(
                num_episodes=0,
                env_name="CartPole-v1"
            )

    def test_valid_with_both_model_and_seeds(self):
        seeds = list(range(24))
        req = ComparisonRequest(
            num_episodes=10,
            differentiable_model_path="/tmp/diff.pt",
            non_differentiable_seeds=seeds,
            env_name="CartPole-v1"
        )
        assert req.differentiable_model_path == "/tmp/diff.pt"
        assert len(req.non_differentiable_seeds) == 24


class TestVisualizationRequestValidation:
    def test_invalid_window_size(self):
        with pytest.raises(Exception):
            VisualizationRequest(
                task_id="test-task",
                chart_type="fitness_curve",
                window_size=0
            )

    def test_invalid_window_size_too_large(self):
        with pytest.raises(Exception):
            VisualizationRequest(
                task_id="test-task",
                chart_type="fitness_curve",
                window_size=200
            )

    def test_valid_window_size(self):
        req = VisualizationRequest(
            task_id="test-task",
            chart_type="fitness_curve",
            window_size=50
        )
        assert req.window_size == 50


class TestConfigUpdateRequest:
    def test_create_with_ppo_config(self):
        from src.schemas.requests import ConfigUpdateRequest, PPOConfigUpdate
        req = ConfigUpdateRequest(ppo=PPOConfigUpdate(learning_rate=0.001))
        assert req.ppo is not None
        assert req.ppo.learning_rate == 0.001
        assert req.genetic is None

    def test_create_with_all_configs(self):
        from src.schemas.requests import (
            ConfigUpdateRequest, PPOConfigUpdate,
            GeneticConfigUpdate, EnvironmentConfigUpdate
        )
        req = ConfigUpdateRequest(
            ppo=PPOConfigUpdate(learning_rate=0.001),
            genetic=GeneticConfigUpdate(population_size=100),
            environment=EnvironmentConfigUpdate(default_env="CartPole-v1")
        )
        assert req.ppo is not None
        assert req.ppo.learning_rate == 0.001
        assert req.genetic is not None
        assert req.genetic.population_size == 100
        assert req.environment is not None
        assert req.environment.default_env == "CartPole-v1"

    def test_at_least_one_config_required(self):
        from src.schemas.requests import ConfigUpdateRequest
        with pytest.raises(Exception):
            ConfigUpdateRequest()

    def test_empty_configs_not_allowed(self):
        from src.schemas.requests import ConfigUpdateRequest, PPOConfigUpdate
        ppo = PPOConfigUpdate()
        with pytest.raises(Exception):
            ConfigUpdateRequest(ppo=ppo)

    def test_invalid_ppo_learning_rate(self):
        from src.schemas.requests import ConfigUpdateRequest, PPOConfigUpdate
        with pytest.raises(Exception):
            ConfigUpdateRequest(ppo=PPOConfigUpdate(learning_rate=-0.1))
        with pytest.raises(Exception):
            ConfigUpdateRequest(ppo=PPOConfigUpdate(learning_rate=1.5))

    def test_invalid_genetic_population_size(self):
        from src.schemas.requests import ConfigUpdateRequest, GeneticConfigUpdate
        with pytest.raises(Exception):
            ConfigUpdateRequest(genetic=GeneticConfigUpdate(population_size=5))
        with pytest.raises(Exception):
            ConfigUpdateRequest(genetic=GeneticConfigUpdate(population_size=2000))

    def test_invalid_environment_max_steps(self):
        from src.schemas.requests import ConfigUpdateRequest, EnvironmentConfigUpdate
        with pytest.raises(Exception):
            ConfigUpdateRequest(environment=EnvironmentConfigUpdate(max_steps=50))
        with pytest.raises(Exception):
            ConfigUpdateRequest(environment=EnvironmentConfigUpdate(max_steps=20000))

    def test_valid_config_update(self):
        from src.schemas.requests import (
            ConfigUpdateRequest, PPOConfigUpdate,
            GeneticConfigUpdate, EnvironmentConfigUpdate
        )
        req = ConfigUpdateRequest(
            ppo=PPOConfigUpdate(
                learning_rate=0.001,
                epsilon=0.3,
                regularization_coef=0.05
            ),
            genetic=GeneticConfigUpdate(
                mutation_rate=0.15,
                crossover_rate=0.8,
                population_size=100
            ),
            environment=EnvironmentConfigUpdate(
                max_concurrent_training_tasks=3,
                max_concurrent_genetic_tasks=4
            )
        )
        assert req.ppo.learning_rate == 0.001
        assert req.genetic.mutation_rate == 0.15
        assert req.environment.max_concurrent_training_tasks == 3


class TestPageRequest:
    def test_default_values(self):
        from src.schemas.requests import PageRequest
        req = PageRequest()
        assert req.page == 1
        assert req.page_size == 20

    def test_invalid_page(self):
        from src.schemas.requests import PageRequest
        with pytest.raises(Exception):
            PageRequest(page=0)

    def test_invalid_page_size(self):
        from src.schemas.requests import PageRequest
        with pytest.raises(Exception):
            PageRequest(page_size=0)


class TestResponseModels:
    def test_training_status_data_infinite_reward(self):
        from src.schemas.responses import TrainingStatusData
        import math
        data = TrainingStatusData(
            task_id="test",
            status="running",
            total_episodes=100,
            best_reward=float('-inf'),
        )
        d = data.model_dump()
        assert d['best_reward'] is None

    def test_training_status_data_nan_reward(self):
        from src.schemas.responses import TrainingStatusData
        import math
        data = TrainingStatusData(
            task_id="test",
            status="running",
            total_episodes=100,
            best_reward=float('nan'),
        )
        d = data.model_dump()
        assert d['best_reward'] is None

    def test_training_status_data_normal_reward(self):
        from src.schemas.responses import TrainingStatusData
        data = TrainingStatusData(
            task_id="test",
            status="running",
            total_episodes=100,
            best_reward=200.5,
        )
        d = data.model_dump()
        assert d['best_reward'] == 200.5

    def test_training_history_item_optional_fields(self):
        from src.schemas.responses import TrainingHistoryItem
        item = TrainingHistoryItem(
            episode=1,
            reward=100.0,
            length=50,
        )
        assert item.policy_loss is None
        assert item.value_loss is None
        assert item.temperature is None

    def test_training_history_item_with_all_fields(self):
        from src.schemas.responses import TrainingHistoryItem
        item = TrainingHistoryItem(
            episode=1,
            reward=100.0,
            length=50,
            policy_loss=0.5,
            value_loss=1.0,
            temperature=0.9
        )
        assert item.policy_loss == 0.5
        assert item.value_loss == 1.0
        assert item.temperature == 0.9

    def test_individual_data_serialization(self):
        from src.schemas.responses import IndividualData
        import math
        data = IndividualData(
            seeds=[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                   [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
            fitness=250.0,
            generation=10
        )
        d = data.model_dump()
        assert len(d['seeds']) == 4
        assert d['fitness'] == 250.0
        assert d['generation'] == 10

    def test_individual_data_infinite_fitness(self):
        from src.schemas.responses import IndividualData
        import math
        data = IndividualData(
            seeds=[[0]*6 for _ in range(4)],
            fitness=float('inf'),
            generation=10
        )
        d = data.model_dump()
        assert d['fitness'] is None

    def test_evaluation_result_data(self):
        from src.schemas.responses import EvaluationResultData
        data = EvaluationResultData(
            task_id="eval-1",
            network_type="differentiable",
            num_episodes=10,
            mean_reward=220.5,
            std_reward=15.2,
            min_reward=180.0,
            max_reward=250.0,
            passed=True
        )
        assert data.passed is True
        assert data.mean_reward == 220.5
        assert data.network_type == "differentiable"

    def test_comparison_result_data(self):
        from src.schemas.responses import ComparisonResultData, EvaluationResultData
        diff_result = EvaluationResultData(
            task_id="diff-1",
            network_type="differentiable",
            num_episodes=10,
            mean_reward=220.0,
            std_reward=10.0,
            min_reward=200.0,
            max_reward=240.0,
            passed=True
        )
        non_diff_result = EvaluationResultData(
            task_id="non-diff-1",
            network_type="non_differentiable",
            num_episodes=10,
            mean_reward=180.0,
            std_reward=15.0,
            min_reward=150.0,
            max_reward=210.0,
            passed=False
        )
        comparison = ComparisonResultData(
            differentiable_result=diff_result,
            non_differentiable_result=non_diff_result,
            weight_difference_norm=0.5,
            performance_gap=40.0
        )
        assert comparison.performance_gap == 40.0
        assert comparison.weight_difference_norm == 0.5
        assert comparison.differentiable_result.passed is True
        assert comparison.non_differentiable_result.passed is False
