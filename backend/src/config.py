"""
应用配置模块
"""
import time
from pydantic_settings import BaseSettings
from typing import Optional


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
    DEFAULT_ENV: str = "LunarLander-v2"  # 注意：gymnasium 0.29.1 只支持 v2，不支持 v3
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


# 全局配置实例
settings = Settings()
