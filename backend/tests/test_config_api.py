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


# ==================== 新增: 可视化 API 端点测试 ====================


class TestVisualizationGenerateEndpoint:
    """测试 POST /api/v1/visualization/generate"""

    def test_generate_fitness_curve_with_raw_data_base64(self):
        payload = {
            "chart_type": "fitness_curve",
            "raw_data": {"fitness_history": [50.0, 100.0, 150.0, 200.0, 210.0]},
            "window_size": 3,
            "format": "base64",
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["chart_type"] == "fitness_curve"
        assert data["data"]["format"] == "base64"
        assert data["data"]["image_base64"] is not None
        assert isinstance(data["data"]["image_base64"], str)
        assert len(data["data"]["image_base64"]) > 0
        assert data["data"]["file_url"] is None
        assert data["data"]["file_path"] is None
        assert data["data"]["width"] > 0
        assert data["data"]["height"] > 0
        assert data["data"]["stats"]["count"] == 5
        assert data["data"]["stats"]["max"] == 210.0

    def test_generate_fitness_curve_with_raw_data_both_format_and_save(self):
        payload = {
            "chart_type": "fitness_curve",
            "raw_data": {"fitness_history": list(range(50, 250, 10))},
            "format": "both",
            "save_to_plots": True,
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["image_base64"] is not None
        assert data["file_url"] is not None
        assert data["file_url"].startswith("/plots/")
        assert data["file_path"] is not None

    def test_generate_dashboard_with_raw_data(self):
        payload = {
            "chart_type": "dashboard",
            "raw_data": {
                "episode_rewards": [float(i) for i in range(150)],
                "policy_losses": [0.5, 0.4, 0.3, 0.2, 0.15],
                "value_losses": [1.2, 1.0, 0.9, 0.8, 0.7],
                "temperatures": [1.0, 0.99, 0.98, 0.97, 0.96],
            },
            "format": "base64",
            "title": "My Dashboard",
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["chart_type"] == "dashboard"
        assert data["image_base64"] is not None
        assert data["width"] == 2100
        assert data["height"] == 1500

    def test_generate_progress_with_raw_data(self):
        payload = {
            "chart_type": "progress",
            "raw_data": {
                "fitness_history": [50.0, 90.0, 140.0, 180.0, 220.0],
                "avg_fitness_history": [40.0, 80.0, 120.0, 160.0, 200.0],
            },
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["chart_type"] == "progress"
        assert data["stats"]["count"] == 5

    def test_generate_comparison_with_raw_data(self):
        payload = {
            "chart_type": "comparison",
            "raw_data": {
                "diff_rewards": [210.0, 220.0, 205.0, 215.0, 200.0],
                "non_diff_rewards": [170.0, 180.0, 165.0, 175.0, 190.0],
            },
            "format": "base64",
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["chart_type"] == "comparison"
        assert data["stats"]["performance_gap"] > 0
        assert data["stats"]["differentiable"]["passed"] is True

    def test_generate_missing_data_source_returns_422(self):
        payload = {
            "chart_type": "fitness_curve",
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        # Pydantic validation error -> 422
        assert response.status_code == 422

    def test_generate_invalid_chart_type_returns_422(self):
        payload = {
            "chart_type": "invalid_chart",
            "raw_data": {"fitness_history": [1.0]},
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 422

    def test_generate_missing_raw_data_keys_returns_422(self):
        payload = {
            "chart_type": "fitness_curve",
            "raw_data": {"wrong_key": [1.0, 2.0]},
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 422

    def test_generate_with_task_id_not_found_returns_404(self):
        payload = {
            "chart_type": "fitness_curve",
            "task_id": "nonexistent-task-id-xyz",
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 404

    def test_generate_file_url_format_saves_file(self):
        payload = {
            "chart_type": "fitness_curve",
            "raw_data": {"fitness_history": [100.0, 200.0, 300.0]},
            "format": "file_url",
        }
        response = client.post("/api/v1/visualization/generate", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["file_url"] is not None
        assert data["file_url"].startswith("/plots/")
        assert data["file_path"] is not None
        # image_base64 应该为 None(format=file_url)
        assert data["image_base64"] is None


class TestVisualizationComparisonEndpoint:
    """测试 GET /api/v1/visualization/comparison"""

    def test_comparison_with_rewards_base64(self):
        params = {
            "diff_rewards": "200,210,220,205,215,225,195,210,200,208",
            "non_diff_rewards": "170,180,165,175,190,160,185,170,178,182",
            "format": "base64",
        }
        response = client.get("/api/v1/visualization/comparison", params=params)
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["format"] == "base64"
        # Base64 返回
        assert data["data"]["boxplot_base64"] is not None
        assert len(data["data"]["boxplot_base64"]) > 0
        assert data["data"]["histogram_base64"] is not None
        assert len(data["data"]["histogram_base64"]) > 0
        assert data["data"]["combined_base64"] is not None
        assert len(data["data"]["combined_base64"]) > 0
        # 路径/URL None
        assert data["data"]["boxplot_path"] is None
        assert data["data"]["boxplot_url"] is None
        # 统计
        diff_stats = data["data"]["differentiable_stats"]
        nd_stats = data["data"]["non_differentiable_stats"]
        assert diff_stats["mean"] > 200
        assert diff_stats["passed"] is True
        assert nd_stats["mean"] < 200
        assert nd_stats["passed"] is False
        assert data["data"]["performance_gap"] == pytest.approx(diff_stats["mean"] - nd_stats["mean"])

    def test_comparison_with_rewards_both_format_and_save(self):
        params = {
            "diff_rewards": "200,210,220,205,215",
            "non_diff_rewards": "170,180,165,175,190",
            "format": "both",
            "save_to_plots": "true",
        }
        response = client.get("/api/v1/visualization/comparison", params=params)
        assert response.status_code == 200
        data = response.json()["data"]
        # Base64
        assert data["boxplot_base64"] is not None
        assert data["histogram_base64"] is not None
        # 路径和URL
        assert data["boxplot_path"] is not None
        assert data["histogram_path"] is not None
        assert data["combined_path"] is not None
        assert data["boxplot_url"] is not None
        assert data["boxplot_url"].startswith("/plots/")
        assert data["combined_url"] is not None

    def test_comparison_with_rewards_file_url_format(self):
        params = {
            "diff_rewards": "200,210,220",
            "non_diff_rewards": "150,160,170",
            "format": "file_url",
            "title": "Comparison Title Test",
        }
        response = client.get("/api/v1/visualization/comparison", params=params)
        assert response.status_code == 200
        data = response.json()["data"]
        # 无 Base64
        assert data["boxplot_base64"] is None
        assert data["histogram_base64"] is None
        assert data["combined_base64"] is None
        # 有路径和URL
        assert data["boxplot_url"] is not None
        assert data["histogram_url"] is not None
        assert data["boxplot_path"] is not None

    def test_comparison_missing_data_source_returns_422(self):
        response = client.get("/api/v1/visualization/comparison")
        # Pydantic 通过 VisualizationComparisonQuery 校验 -> 最终抛出 HTTPException
        assert response.status_code in (400, 422)

    def test_comparison_invalid_format_returns_422(self):
        params = {
            "diff_rewards": "200,210",
            "non_diff_rewards": "150,160",
            "format": "invalid_fmt",
        }
        response = client.get("/api/v1/visualization/comparison", params=params)
        assert response.status_code == 422

    def test_comparison_partial_rewards_returns_422(self):
        params = {
            "diff_rewards": "200,210,220",
        }
        response = client.get("/api/v1/visualization/comparison", params=params)
        assert response.status_code in (400, 422)

    def test_comparison_with_custom_title(self):
        params = {
            "diff_rewards": "200,210,220",
            "non_diff_rewards": "180,190,200",
            "format": "base64",
            "title": "Custom Comparison Title",
        }
        response = client.get("/api/v1/visualization/comparison", params=params)
        assert response.status_code == 200
        # 不报错即可
        data = response.json()["data"]
        assert data["differentiable_stats"]["count"] == 3
