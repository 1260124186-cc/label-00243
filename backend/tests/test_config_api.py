import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


class TestConfigEndpoint:
    def test_get_config(self):
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "ppo" in data["data"]
        assert "genetic" in data["data"]
        assert "environment" in data["data"]
        assert "learning_rate" in data["data"]["ppo"]
        assert "epsilon" in data["data"]["ppo"]
        assert "regularization_coef" in data["data"]["ppo"]
        assert "population_size" in data["data"]["genetic"]
        assert "mutation_rate" in data["data"]["genetic"]
        assert "crossover_rate" in data["data"]["genetic"]
        assert "default_env" in data["data"]["environment"]
        assert "max_concurrent_training_tasks" in data["data"]["environment"]
        assert "max_concurrent_genetic_tasks" in data["data"]["environment"]

    def test_update_config_ppo(self):
        payload = {
            "ppo": {
                "learning_rate": 0.001,
                "epsilon": 0.3,
                "regularization_coef": 0.05
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert "before" in data["data"]
        assert "after" in data["data"]
        assert "changed_keys" in data["data"]
        assert "ppo.learning_rate" in data["data"]["changed_keys"]
        assert "ppo.epsilon" in data["data"]["changed_keys"]
        assert "ppo.regularization_coef" in data["data"]["changed_keys"]
        assert data["data"]["after"]["ppo"]["learning_rate"] == 0.001
        assert data["data"]["after"]["ppo"]["epsilon"] == 0.3
        assert data["data"]["after"]["ppo"]["regularization_coef"] == 0.05

        get_response = client.get("/api/v1/config")
        get_data = get_response.json()
        assert get_data["data"]["ppo"]["learning_rate"] == 0.001
        assert get_data["data"]["ppo"]["epsilon"] == 0.3
        assert get_data["data"]["ppo"]["regularization_coef"] == 0.05

    def test_update_config_genetic(self):
        payload = {
            "genetic": {
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "population_size": 100
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["after"]["genetic"]["mutation_rate"] == 0.15
        assert data["data"]["after"]["genetic"]["crossover_rate"] == 0.8
        assert data["data"]["after"]["genetic"]["population_size"] == 100

    def test_update_config_environment(self):
        payload = {
            "environment": {
                "default_env": "CartPole-v1",
                "max_steps": 2000,
                "max_concurrent_training_tasks": 5,
                "max_concurrent_genetic_tasks": 3
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["after"]["environment"]["default_env"] == "CartPole-v1"
        assert data["data"]["after"]["environment"]["max_steps"] == 2000
        assert data["data"]["after"]["environment"]["max_concurrent_training_tasks"] == 5
        assert data["data"]["after"]["environment"]["max_concurrent_genetic_tasks"] == 3

    def test_update_config_multiple_categories(self):
        payload = {
            "ppo": {
                "learning_rate": 0.0005
            },
            "genetic": {
                "mutation_rate": 0.2
            },
            "environment": {
                "default_env": "MountainCar-v0"
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "ppo" in data["data"]["before"]
        assert "genetic" in data["data"]["before"]
        assert "environment" in data["data"]["before"]
        assert len(data["data"]["changed_keys"]) == 3

    def test_update_config_invalid_ppo_returns_422(self):
        payload = {
            "ppo": {
                "learning_rate": -0.1
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_invalid_genetic_returns_422(self):
        payload = {
            "genetic": {
                "population_size": 5
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_invalid_environment_returns_422(self):
        payload = {
            "environment": {
                "max_concurrent_training_tasks": 0
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_empty_body_returns_422(self):
        response = client.put("/api/v1/config", json={})
        assert response.status_code == 422

    def test_update_config_all_nulls_returns_422(self):
        payload = {
            "ppo": {},
            "genetic": {},
            "environment": {}
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_mixed_valid_invalid_returns_422(self):
        payload = {
            "ppo": {
                "learning_rate": 0.001
            },
            "genetic": {
                "population_size": 5
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_invalid_type_returns_422(self):
        payload = {
            "ppo": {
                "learning_rate": "not_a_number"
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_unknown_field_returns_422(self):
        payload = {
            "ppo": {
                "unknown_field": 123
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_unknown_category_returns_422(self):
        payload = {
            "unknown_category": {
                "field": 123
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 422

    def test_update_config_diff_response_structure(self):
        payload = {
            "ppo": {
                "learning_rate": 0.0001,
                "epsilon": 0.1
            }
        }
        response = client.put("/api/v1/config", json=payload)
        data = response.json()
        
        assert "before" in data["data"]
        assert "after" in data["data"]
        assert "changed_keys" in data["data"]
        assert "timestamp" in data["data"]
        assert "ppo" in data["data"]["before"]
        assert "ppo" in data["data"]["after"]
        assert "learning_rate" in data["data"]["before"]["ppo"]
        assert "learning_rate" in data["data"]["after"]["ppo"]

    def test_config_update_only_affects_new_tasks(self):
        from src.config import config_manager, create_config_snapshot
        
        snapshot_before = create_config_snapshot()
        
        payload = {
            "ppo": {
                "learning_rate": 0.777
            }
        }
        response = client.put("/api/v1/config", json=payload)
        assert response.status_code == 200
        
        assert snapshot_before.ppo_learning_rate != 0.777
        
        snapshot_after = create_config_snapshot()
        assert snapshot_after.ppo_learning_rate == 0.777

    def test_concurrent_updates(self):
        import threading
        
        def update_config(lr):
            payload = {"ppo": {"learning_rate": lr}}
            client.put("/api/v1/config", json=payload)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=update_config, args=(0.0001 + i * 0.0001,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        response = client.get("/api/v1/config")
        assert response.status_code == 200
