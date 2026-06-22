import pytest
from src.core.exceptions import (
    BaseAppException,
    TrainingException,
    GeneticAlgorithmException,
    ModelException,
    ValidationException,
    ResourceNotFoundException,
    ConfigurationException,
)


class TestBaseAppException:
    def test_message_code_details(self):
        exc = BaseAppException("something went wrong", code=500, details={"key": "val"})
        assert str(exc) == "something went wrong"
        assert exc.code == 500
        assert exc.details == {"key": "val"}

    def test_is_exception(self):
        assert issubclass(BaseAppException, Exception)


class TestTrainingException:
    def test_code_is_5001(self):
        exc = TrainingException("training failed")
        assert exc.code == 5001


class TestGeneticAlgorithmException:
    def test_code_is_5002(self):
        exc = GeneticAlgorithmException("ga failed")
        assert exc.code == 5002


class TestModelException:
    def test_code_is_5003(self):
        exc = ModelException("model failed")
        assert exc.code == 5003


class TestValidationException:
    def test_code_is_4001(self):
        exc = ValidationException("validation failed")
        assert exc.code == 4001


class TestResourceNotFoundException:
    def test_auto_message(self):
        exc = ResourceNotFoundException("Model", "abc123")
        assert "Model" in str(exc)
        assert "abc123" in str(exc)

    def test_code_is_4004(self):
        exc = ResourceNotFoundException("Model", "abc123")
        assert exc.code == 4004


class TestConfigurationException:
    def test_code_is_5004(self):
        exc = ConfigurationException("config failed")
        assert exc.code == 5004


class TestExceptionInheritance:
    @pytest.mark.parametrize(
        "exc_class",
        [
            TrainingException,
            GeneticAlgorithmException,
            ModelException,
            ValidationException,
            ResourceNotFoundException,
            ConfigurationException,
        ],
    )
    def test_inherits_from_base(self, exc_class):
        assert issubclass(exc_class, BaseAppException)


class TestExceptionDetails:
    def test_base_exception_with_details(self):
        details = {"error": "something", "code": 123}
        exc = BaseAppException("test message", code=500, details=details)
        assert exc.details == details
        assert exc.code == 500
        assert str(exc) == "test message"

    def test_training_exception_with_details(self):
        details = {"task_id": "task-123", "episode": 50}
        exc = TrainingException("training failed", details=details)
        assert exc.code == 5001
        assert exc.details == details

    def test_genetic_exception_with_details(self):
        details = {"generation": 10, "population_size": 50}
        exc = GeneticAlgorithmException("ga failed", details=details)
        assert exc.code == 5002
        assert exc.details == details

    def test_model_exception_with_details(self):
        details = {"layer": "layer1", "shape": (8, 24)}
        exc = ModelException("model error", details=details)
        assert exc.code == 5003
        assert exc.details == details

    def test_validation_exception_with_details(self):
        details = {"field": "learning_rate", "value": -0.1}
        exc = ValidationException("validation failed", details=details)
        assert exc.code == 4001
        assert exc.details == details

    def test_resource_not_found_exception_details(self):
        exc = ResourceNotFoundException("Model", "model-123")
        assert exc.details == {"resource": "Model", "identifier": "model-123"}
        assert exc.code == 4004
        assert "Model" in str(exc)
        assert "model-123" in str(exc)

    def test_configuration_exception_with_details(self):
        details = {"config_key": "batch_size", "expected": 32}
        exc = ConfigurationException("invalid config", details=details)
        assert exc.code == 5004
        assert exc.details == details

    def test_exception_default_details_is_none(self):
        exc = BaseAppException("test")
        assert exc.details is None

    def test_exception_str_equals_message(self):
        exc = TrainingException("this is the message")
        assert str(exc) == "this is the message"

    def test_exception_is_raiseable(self):
        with pytest.raises(TrainingException, match="test error"):
            raise TrainingException("test error")

    def test_all_exception_codes_are_unique(self):
        exc1 = TrainingException("test")
        exc2 = GeneticAlgorithmException("test")
        exc3 = ModelException("test")
        exc4 = ValidationException("test")
        exc5 = ResourceNotFoundException("test", "test")
        exc6 = ConfigurationException("test")

        codes = {exc1.code, exc2.code, exc3.code, exc4.code, exc5.code, exc6.code}
        assert len(codes) == 6

    def test_exception_inheritance_chain(self):
        assert issubclass(TrainingException, BaseAppException)
        assert issubclass(TrainingException, Exception)
        assert isinstance(TrainingException("test"), BaseAppException)
        assert isinstance(TrainingException("test"), Exception)


class TestExceptionUsageScenarios:
    def test_training_exception_for_missing_task(self):
        from src.core.exceptions import TrainingException
        exc = TrainingException("Training task 'abc123' not found")
        assert "abc123" in str(exc)
        assert exc.code == 5001

    def test_model_exception_for_invalid_seed_count(self):
        from src.core.exceptions import ModelException
        exc = ModelException("Expected 24 seeds, got 10")
        assert "24" in str(exc)
        assert "10" in str(exc)
        assert exc.code == 5003

    def test_validation_exception_for_invalid_parameter(self):
        from src.core.exceptions import ValidationException
        exc = ValidationException("learning_rate must be positive")
        assert exc.code == 4001

    def test_configuration_exception_for_missing_key(self):
        from src.core.exceptions import ConfigurationException
        exc = ConfigurationException("Required config 'env_name' is missing")
        assert exc.code == 5004
