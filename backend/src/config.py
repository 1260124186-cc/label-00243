"""
应用配置模块
"""
import time
import threading
import json
from datetime import datetime
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger


CONFIG_VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    "ppo.learning_rate": {"type": float, "gt": 0, "le": 1, "default": 3e-4},
    "ppo.gamma": {"type": float, "ge": 0, "le": 1, "default": 0.99},
    "ppo.epsilon": {"type": float, "ge": 0, "le": 1, "default": 0.2},
    "ppo.epochs": {"type": int, "ge": 1, "le": 1000, "default": 10},
    "ppo.batch_size": {"type": int, "ge": 1, "le": 10000, "default": 64},
    "ppo.initial_temperature": {"type": float, "gt": 0, "default": 1.0},
    "ppo.temperature_decay": {"type": float, "gt": 0, "le": 1, "default": 0.995},
    "ppo.min_temperature": {"type": float, "gt": 0, "default": 0.01},
    "ppo.regularization_coef": {"type": float, "ge": 0, "default": 0.1},
    "genetic.population_size": {"type": int, "ge": 10, "le": 1000, "default": 50},
    "genetic.mutation_rate": {"type": float, "ge": 0, "le": 1, "default": 0.1},
    "genetic.crossover_rate": {"type": float, "ge": 0, "le": 1, "default": 0.7},
    "genetic.elite_size": {"type": int, "ge": 1, "le": 50, "default": 5},
    "genetic.max_generations": {"type": int, "ge": 1, "le": 10000, "default": 100},
    "genetic.seed_range_min": {"type": int, "ge": 0, "default": 0},
    "genetic.seed_range_max": {"type": int, "ge": 1, "default": 10000},
    "genetic.alpha": {"type": float, "ge": 0, "le": 1, "default": 0.9},
    "environment.default_env": {"type": str, "min_length": 1, "default": "LunarLander-v2"},
    "environment.max_steps": {"type": int, "ge": 100, "le": 10000, "default": 1000},
    "environment.total_episodes": {"type": int, "ge": 1, "le": 100000, "default": 1000},
    "environment.max_concurrent_training_tasks": {"type": int, "ge": 1, "le": 100, "default": 2},
    "environment.max_concurrent_genetic_tasks": {"type": int, "ge": 1, "le": 100, "default": 2},
}


SETTINGS_KEY_MAP: Dict[str, str] = {
    "ppo.learning_rate": "PPO_LEARNING_RATE",
    "ppo.gamma": "PPO_GAMMA",
    "ppo.epsilon": "PPO_EPSILON",
    "ppo.epochs": "PPO_EPOCHS",
    "ppo.batch_size": "PPO_BATCH_SIZE",
    "ppo.initial_temperature": "INITIAL_TEMPERATURE",
    "ppo.temperature_decay": "TEMPERATURE_DECAY",
    "ppo.min_temperature": "MIN_TEMPERATURE",
    "ppo.regularization_coef": "REGULARIZATION_COEF",
    "genetic.population_size": "GA_POPULATION_SIZE",
    "genetic.mutation_rate": "GA_MUTATION_RATE",
    "genetic.crossover_rate": "GA_CROSSOVER_RATE",
    "genetic.elite_size": "GA_ELITE_SIZE",
    "genetic.max_generations": "GA_MAX_GENERATIONS",
    "genetic.seed_range_min": "GA_SEED_RANGE_MIN",
    "genetic.seed_range_max": "GA_SEED_RANGE_MAX",
    "genetic.alpha": "GA_ALPHA",
    "environment.default_env": "DEFAULT_ENV",
    "environment.max_steps": "MAX_STEPS",
    "environment.total_episodes": "TOTAL_EPISODES",
    "environment.max_concurrent_training_tasks": "MAX_CONCURRENT_TRAINING_TASKS",
    "environment.max_concurrent_genetic_tasks": "MAX_CONCURRENT_GENETIC_TASKS",
}


def validate_config_value(key: str, value: Any) -> Tuple[bool, Optional[str]]:
    """
    验证配置值是否符合规则

    Args:
        key: 配置键，格式如 "ppo.learning_rate"
        value: 配置值

    Returns:
        (是否有效, 错误信息)
    """
    if key not in CONFIG_VALIDATION_RULES:
        return False, f"Unknown config key: {key}"

    rules = CONFIG_VALIDATION_RULES[key]
    expected_type = rules["type"]

    if expected_type == float and isinstance(value, int):
        value = float(value)

    if not isinstance(value, expected_type):
        return False, f"Invalid type for {key}: expected {expected_type.__name__}, got {type(value).__name__}"

    if "gt" in rules and value <= rules["gt"]:
        return False, f"Value for {key} must be greater than {rules['gt']}"
    if "ge" in rules and value < rules["ge"]:
        return False, f"Value for {key} must be greater than or equal to {rules['ge']}"
    if "lt" in rules and value >= rules["lt"]:
        return False, f"Value for {key} must be less than {rules['lt']}"
    if "le" in rules and value > rules["le"]:
        return False, f"Value for {key} must be less than or equal to {rules['le']}"
    if "min_length" in rules and len(str(value)) < rules["min_length"]:
        return False, f"Value for {key} must have length at least {rules['min_length']}"

    return True, None


def validate_config_updates(updates: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证配置更新字典

    Args:
        updates: 配置更新字典，格式如 {"ppo": {"learning_rate": 0.001}, ...}

    Returns:
        (是否全部有效, 错误信息列表)
    """
    errors: List[str] = []

    for category, category_values in updates.items():
        if not isinstance(category_values, dict):
            errors.append(f"Category '{category}' must be an object")
            continue

        for sub_key, value in category_values.items():
            full_key = f"{category}.{sub_key}"
            valid, error = validate_config_value(full_key, value)
            if not valid:
                errors.append(error or f"Invalid value for {full_key}")

    return len(errors) == 0, errors


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基本信息
    APP_NAME: str = "Non-Differentiable Network RL System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    START_TIME: float = time.time()
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # PPO 默认配置
    PPO_LEARNING_RATE: float = 3e-4
    PPO_GAMMA: float = 0.99
    PPO_EPSILON: float = 0.2
    PPO_EPOCHS: int = 10
    PPO_BATCH_SIZE: int = 64
    INITIAL_TEMPERATURE: float = 1.0
    TEMPERATURE_DECAY: float = 0.995
    MIN_TEMPERATURE: float = 0.01
    REGULARIZATION_COEF: float = 0.1
    
    # 遗传算法默认配置
    GA_POPULATION_SIZE: int = 50
    GA_MUTATION_RATE: float = 0.1
    GA_CROSSOVER_RATE: float = 0.7
    GA_ELITE_SIZE: int = 5
    GA_MAX_GENERATIONS: int = 100
    GA_SEED_RANGE_MIN: int = 0
    GA_SEED_RANGE_MAX: int = 10000
    GA_ALPHA: float = 0.9
    
    # 环境配置
    DEFAULT_ENV: str = "LunarLander-v2"
    MAX_STEPS: int = 1000
    TOTAL_EPISODES: int = 1000
    
    # 任务配置
    MAX_CONCURRENT_TRAINING_TASKS: int = 2
    MAX_CONCURRENT_GENETIC_TASKS: int = 2
    
    # 模型保存配置
    MODEL_SAVE_DIR: str = "models"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class ConfigSnapshot:
    """配置快照，用于任务启动时的配置固定"""

    def __init__(self, settings: Settings):
        self.ppo_learning_rate = settings.PPO_LEARNING_RATE
        self.ppo_gamma = settings.PPO_GAMMA
        self.ppo_epsilon = settings.PPO_EPSILON
        self.ppo_epochs = settings.PPO_EPOCHS
        self.ppo_batch_size = settings.PPO_BATCH_SIZE
        self.initial_temperature = settings.INITIAL_TEMPERATURE
        self.temperature_decay = settings.TEMPERATURE_DECAY
        self.min_temperature = settings.MIN_TEMPERATURE
        self.regularization_coef = settings.REGULARIZATION_COEF
        self.ga_population_size = settings.GA_POPULATION_SIZE
        self.ga_mutation_rate = settings.GA_MUTATION_RATE
        self.ga_crossover_rate = settings.GA_CROSSOVER_RATE
        self.ga_elite_size = settings.GA_ELITE_SIZE
        self.ga_max_generations = settings.GA_MAX_GENERATIONS
        self.ga_seed_range_min = settings.GA_SEED_RANGE_MIN
        self.ga_seed_range_max = settings.GA_SEED_RANGE_MAX
        self.ga_alpha = settings.GA_ALPHA
        self.default_env = settings.DEFAULT_ENV
        self.max_steps = settings.MAX_STEPS
        self.total_episodes = settings.TOTAL_EPISODES
        self.max_concurrent_training_tasks = settings.MAX_CONCURRENT_TRAINING_TASKS
        self.max_concurrent_genetic_tasks = settings.MAX_CONCURRENT_GENETIC_TASKS
        self.snapshot_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ppo": {
                "learning_rate": self.ppo_learning_rate,
                "gamma": self.ppo_gamma,
                "epsilon": self.ppo_epsilon,
                "epochs": self.ppo_epochs,
                "batch_size": self.ppo_batch_size,
                "initial_temperature": self.initial_temperature,
                "temperature_decay": self.temperature_decay,
                "min_temperature": self.min_temperature,
                "regularization_coef": self.regularization_coef,
            },
            "genetic": {
                "population_size": self.ga_population_size,
                "mutation_rate": self.ga_mutation_rate,
                "crossover_rate": self.ga_crossover_rate,
                "elite_size": self.ga_elite_size,
                "max_generations": self.ga_max_generations,
                "seed_range_min": self.ga_seed_range_min,
                "seed_range_max": self.ga_seed_range_max,
                "alpha": self.ga_alpha,
            },
            "environment": {
                "default_env": self.default_env,
                "max_steps": self.max_steps,
                "total_episodes": self.total_episodes,
                "max_concurrent_training_tasks": self.max_concurrent_training_tasks,
                "max_concurrent_genetic_tasks": self.max_concurrent_genetic_tasks,
            },
            "snapshot_time": self.snapshot_time,
        }


class ConfigManager:
    """
    配置管理器
    支持热更新、配置快照、审计日志
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = threading.RLock()
        self._audit_log: List[Dict[str, Any]] = []
        logger.info("ConfigManager initialized")

    def snapshot(self) -> ConfigSnapshot:
        """
        创建当前配置的快照

        Returns:
            配置快照
        """
        with self._lock:
            return ConfigSnapshot(self._settings)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，格式如 "ppo.learning_rate" 或 Settings 属性名
            default: 默认值

        Returns:
            配置值
        """
        with self._lock:
            if key in SETTINGS_KEY_MAP:
                settings_key = SETTINGS_KEY_MAP[key]
                return getattr(self._settings, settings_key, default)
            return getattr(self._settings, key, default)

    def apply_updates(self, updates: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        应用配置更新

        Args:
            updates: 配置更新字典，格式如 {"ppo": {"learning_rate": 0.001}, ...}

        Returns:
            (变更前值字典, 变更后值字典)

        Raises:
            ValueError: 如果验证失败
        """
        with self._lock:
            valid, errors = validate_config_updates(updates)
            if not valid:
                raise ValueError("; ".join(errors))

            before: Dict[str, Any] = {}
            after: Dict[str, Any] = {}

            for category, category_values in updates.items():
                if category not in before:
                    before[category] = {}
                    after[category] = {}
                for sub_key, value in category_values.items():
                    full_key = f"{category}.{sub_key}"
                    settings_key = SETTINGS_KEY_MAP[full_key]
                    old_value = getattr(self._settings, settings_key)

                    before[category][sub_key] = old_value
                    after[category][sub_key] = value

                    setattr(self._settings, settings_key, value)

                    logger.info(
                        f"Config updated: {full_key} = {old_value} -> {value}"
                    )

            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "changes": {
                    "before": before,
                    "after": after,
                },
            }
            self._audit_log.append(audit_entry)

            logger.info(
                f"Config hot-reload applied. Changes: {json.dumps(audit_entry, default=str)}"
            )

            return before, after

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取审计日志

        Args:
            limit: 返回的最大条目数

        Returns:
            审计日志列表（最新的在前）
        """
        with self._lock:
            return list(reversed(self._audit_log[-limit:]))


# 全局配置实例
settings = Settings()

# 全局配置管理器
config_manager = ConfigManager(settings)

# 便捷函数
def create_config_snapshot() -> ConfigSnapshot:
    """创建配置快照的便捷函数"""
    return config_manager.snapshot()
