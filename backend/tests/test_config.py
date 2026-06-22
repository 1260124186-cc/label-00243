import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    Settings,
    ConfigSnapshot,
    ConfigManager,
    validate_config_value,
    validate_config_updates,
    CONFIG_VALIDATION_RULES,
    config_manager,
    create_config_snapshot,
)
from src.schemas.requests import (
    ConfigUpdateRequest,
    PPOConfigUpdate,
    GeneticConfigUpdate,
    EnvironmentConfigUpdate,
)


class TestConfigValidation:
    def test_validate_ppo_learning_rate_valid(self):
        valid, error = validate_config_value("ppo.learning_rate", 0.001)
        assert valid is True
        assert error is None

    def test_validate_ppo_learning_rate_invalid_type(self):
        valid, error = validate_config_value("ppo.learning_rate", "not_a_float")
        assert valid is False
        assert "Invalid type" in error

    def test_validate_ppo_learning_rate_invalid_range(self):
        valid, error = validate_config_value("ppo.learning_rate", -0.1)
        assert valid is False
        assert "greater than" in error

        valid, error = validate_config_value("ppo.learning_rate", 1.5)
        assert valid is False
        assert "less than or equal" in error

    def test_validate_ppo_epsilon_valid(self):
        valid, error = validate_config_value("ppo.epsilon", 0.3)
        assert valid is True

    def test_validate_ppo_epsilon_invalid(self):
        valid, error = validate_config_value("ppo.epsilon", -0.1)
        assert valid is False

    def test_validate_ppo_regularization_coef_valid(self):
        valid, error = validate_config_value("ppo.regularization_coef", 0.05)
        assert valid is True

    def test_validate_ppo_regularization_coef_invalid(self):
        valid, error = validate_config_value("ppo.regularization_coef", -0.1)
        assert valid is False

    def test_validate_genetic_population_size_valid(self):
        valid, error = validate_config_value("genetic.population_size", 100)
        assert valid is True

    def test_validate_genetic_population_size_invalid(self):
        valid, error = validate_config_value("genetic.population_size", 5)
        assert valid is False

        valid, error = validate_config_value("genetic.population_size", 2000)
        assert valid is False

    def test_validate_genetic_mutation_rate_valid(self):
        valid, error = validate_config_value("genetic.mutation_rate", 0.15)
        assert valid is True

    def test_validate_genetic_mutation_rate_invalid(self):
        valid, error = validate_config_value("genetic.mutation_rate", 1.5)
        assert valid is False

    def test_validate_genetic_crossover_rate_valid(self):
        valid, error = validate_config_value("genetic.crossover_rate", 0.8)
        assert valid is True

    def test_validate_environment_default_env_valid(self):
        valid, error = validate_config_value("environment.default_env", "CartPole-v1")
        assert valid is True

    def test_validate_environment_default_env_invalid(self):
        valid, error = validate_config_value("environment.default_env", "")
        assert valid is False

    def test_validate_environment_max_concurrent_training_tasks_valid(self):
        valid, error = validate_config_value("environment.max_concurrent_training_tasks", 5)
        assert valid is True

    def test_validate_environment_max_concurrent_training_tasks_invalid(self):
        valid, error = validate_config_value("environment.max_concurrent_training_tasks", 0)
        assert valid is False

    def test_validate_unknown_key(self):
        valid, error = validate_config_value("unknown.key", 123)
        assert valid is False
        assert "Unknown config key" in error

    def test_validate_int_as_float(self):
        valid, error = validate_config_value("ppo.learning_rate", 1)
        assert valid is True

    def test_validate_config_updates_valid(self):
        updates = {
            "ppo": {"learning_rate": 0.001, "epsilon": 0.3},
            "genetic": {"mutation_rate": 0.15}
        }
        valid, errors = validate_config_updates(updates)
        assert valid is True
        assert len(errors) == 0

    def test_validate_config_updates_invalid(self):
        updates = {
            "ppo": {"learning_rate": -0.1},
            "genetic": {"population_size": 5}
        }
        valid, errors = validate_config_updates(updates)
        assert valid is False
        assert len(errors) == 2


class TestConfigSnapshot:
    def test_snapshot_captures_all_values(self):
        settings = Settings()
        snapshot = ConfigSnapshot(settings)
        
        assert snapshot.ppo_learning_rate == settings.PPO_LEARNING_RATE
        assert snapshot.ppo_epsilon == settings.PPO_EPSILON
        assert snapshot.regularization_coef == settings.REGULARIZATION_COEF
        assert snapshot.ga_population_size == settings.GA_POPULATION_SIZE
        assert snapshot.ga_mutation_rate == settings.GA_MUTATION_RATE
        assert snapshot.ga_crossover_rate == settings.GA_CROSSOVER_RATE
        assert snapshot.default_env == settings.DEFAULT_ENV
        assert snapshot.max_concurrent_training_tasks == settings.MAX_CONCURRENT_TRAINING_TASKS
        assert snapshot.max_concurrent_genetic_tasks == settings.MAX_CONCURRENT_GENETIC_TASKS
        assert snapshot.snapshot_time is not None

    def test_snapshot_to_dict(self):
        settings = Settings()
        snapshot = ConfigSnapshot(settings)
        snapshot_dict = snapshot.to_dict()
        
        assert "ppo" in snapshot_dict
        assert "genetic" in snapshot_dict
        assert "environment" in snapshot_dict
        assert "snapshot_time" in snapshot_dict
        assert snapshot_dict["ppo"]["learning_rate"] == settings.PPO_LEARNING_RATE
        assert snapshot_dict["genetic"]["population_size"] == settings.GA_POPULATION_SIZE
        assert snapshot_dict["environment"]["default_env"] == settings.DEFAULT_ENV

    def test_snapshot_is_independent_of_settings_changes(self):
        settings = Settings()
        original_lr = settings.PPO_LEARNING_RATE
        snapshot = ConfigSnapshot(settings)
        
        settings.PPO_LEARNING_RATE = 0.999
        
        assert snapshot.ppo_learning_rate == original_lr
        assert snapshot.ppo_learning_rate != settings.PPO_LEARNING_RATE


class TestConfigManager:
    def test_initialization(self):
        settings = Settings()
        manager = ConfigManager(settings)
        assert manager._settings is settings

    def test_get_with_dotted_key(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        value = manager.get("ppo.learning_rate")
        assert value == settings.PPO_LEARNING_RATE

    def test_get_with_settings_key(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        value = manager.get("PPO_LEARNING_RATE")
        assert value == settings.PPO_LEARNING_RATE

    def test_get_with_default(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        value = manager.get("non_existent_key", "default_value")
        assert value == "default_value"

    def test_snapshot(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        snapshot = manager.snapshot()
        assert isinstance(snapshot, ConfigSnapshot)
        assert snapshot.ppo_learning_rate == settings.PPO_LEARNING_RATE

    def test_apply_updates_ppo(self):
        settings = Settings()
        original_lr = settings.PPO_LEARNING_RATE
        original_epsilon = settings.PPO_EPSILON
        manager = ConfigManager(settings)
        
        updates = {
            "ppo": {
                "learning_rate": 0.001,
                "epsilon": 0.3,
                "regularization_coef": 0.05
            }
        }
        
        before, after = manager.apply_updates(updates)
        
        assert before["ppo"]["learning_rate"] == original_lr
        assert before["ppo"]["epsilon"] == original_epsilon
        assert after["ppo"]["learning_rate"] == 0.001
        assert after["ppo"]["epsilon"] == 0.3
        assert after["ppo"]["regularization_coef"] == 0.05
        assert settings.PPO_LEARNING_RATE == 0.001
        assert settings.PPO_EPSILON == 0.3
        assert settings.REGULARIZATION_COEF == 0.05

    def test_apply_updates_genetic(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        updates = {
            "genetic": {
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "population_size": 100
            }
        }
        
        before, after = manager.apply_updates(updates)
        
        assert after["genetic"]["mutation_rate"] == 0.15
        assert after["genetic"]["crossover_rate"] == 0.8
        assert after["genetic"]["population_size"] == 100
        assert settings.GA_MUTATION_RATE == 0.15
        assert settings.GA_CROSSOVER_RATE == 0.8
        assert settings.GA_POPULATION_SIZE == 100

    def test_apply_updates_environment(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        updates = {
            "environment": {
                "default_env": "CartPole-v1",
                "max_steps": 2000,
                "max_concurrent_training_tasks": 5,
                "max_concurrent_genetic_tasks": 3
            }
        }
        
        before, after = manager.apply_updates(updates)
        
        assert after["environment"]["default_env"] == "CartPole-v1"
        assert after["environment"]["max_steps"] == 2000
        assert after["environment"]["max_concurrent_training_tasks"] == 5
        assert after["environment"]["max_concurrent_genetic_tasks"] == 3
        assert settings.DEFAULT_ENV == "CartPole-v1"
        assert settings.MAX_STEPS == 2000
        assert settings.MAX_CONCURRENT_TRAINING_TASKS == 5
        assert settings.MAX_CONCURRENT_GENETIC_TASKS == 3

    def test_apply_updates_multiple_categories(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        updates = {
            "ppo": {"learning_rate": 0.0005},
            "genetic": {"mutation_rate": 0.2},
            "environment": {"default_env": "CartPole-v1"}
        }
        
        before, after = manager.apply_updates(updates)
        
        assert "ppo" in before
        assert "genetic" in before
        assert "environment" in before
        assert after["ppo"]["learning_rate"] == 0.0005
        assert after["genetic"]["mutation_rate"] == 0.2
        assert after["environment"]["default_env"] == "CartPole-v1"

    def test_apply_updates_invalid_raises_value_error(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        updates = {
            "ppo": {"learning_rate": -0.1}
        }
        
        with pytest.raises(ValueError):
            manager.apply_updates(updates)

    def test_apply_updates_creates_audit_log(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        updates = {
            "ppo": {"learning_rate": 0.001}
        }
        
        before, after = manager.apply_updates(updates)
        
        audit_log = manager.get_audit_log()
        assert len(audit_log) == 1
        assert "timestamp" in audit_log[0]
        assert "changes" in audit_log[0]
        assert audit_log[0]["changes"]["before"]["ppo"]["learning_rate"] == before["ppo"]["learning_rate"]
        assert audit_log[0]["changes"]["after"]["ppo"]["learning_rate"] == 0.001

    def test_audit_log_order(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        manager.apply_updates({"ppo": {"learning_rate": 0.001}})
        manager.apply_updates({"ppo": {"learning_rate": 0.002}})
        
        audit_log = manager.get_audit_log()
        assert len(audit_log) == 2
        assert audit_log[0]["changes"]["after"]["ppo"]["learning_rate"] == 0.002
        assert audit_log[1]["changes"]["after"]["ppo"]["learning_rate"] == 0.001

    def test_audit_log_limit(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        for i in range(150):
            manager.apply_updates({"ppo": {"learning_rate": 0.001 + i * 0.00001}})
        
        audit_log = manager.get_audit_log(limit=50)
        assert len(audit_log) == 50


class TestGlobalConfigManager:
    def test_global_config_manager_exists(self):
        assert config_manager is not None
        assert isinstance(config_manager, ConfigManager)

    def test_create_config_snapshot(self):
        snapshot = create_config_snapshot()
        assert isinstance(snapshot, ConfigSnapshot)


class TestConfigUpdateRequestValidation:
    def test_ppo_config_validation(self):
        with pytest.raises(Exception):
            PPOConfigUpdate(learning_rate=-1)
        
        with pytest.raises(Exception):
            PPOConfigUpdate(epsilon=2)
        
        with pytest.raises(Exception):
            PPOConfigUpdate(regularization_coef=-0.1)

    def test_genetic_config_validation(self):
        with pytest.raises(Exception):
            GeneticConfigUpdate(population_size=5)
        
        with pytest.raises(Exception):
            GeneticConfigUpdate(mutation_rate=2)
        
        with pytest.raises(Exception):
            GeneticConfigUpdate(crossover_rate=-1)

    def test_environment_config_validation(self):
        with pytest.raises(Exception):
            EnvironmentConfigUpdate(max_steps=50)
        
        with pytest.raises(Exception):
            EnvironmentConfigUpdate(max_concurrent_training_tasks=0)
        
        with pytest.raises(Exception):
            EnvironmentConfigUpdate(default_env="")


class TestConfigHotReloadScenario:
    def test_running_tasks_use_snapshot(self):
        settings = Settings()
        manager = ConfigManager(settings)
        
        snapshot_before = manager.snapshot()
        
        manager.apply_updates({
            "ppo": {"learning_rate": 0.999},
            "genetic": {"mutation_rate": 0.999}
        })
        
        assert snapshot_before.ppo_learning_rate != settings.PPO_LEARNING_RATE
        assert snapshot_before.ga_mutation_rate != settings.GA_MUTATION_RATE
        
        snapshot_after = manager.snapshot()
        assert snapshot_after.ppo_learning_rate == 0.999
        assert snapshot_after.ga_mutation_rate == 0.999
