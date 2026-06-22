"""
评估服务模块
"""
import uuid
import torch
import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any
from loguru import logger

from ..models.network import NonDifferentiableNetwork, DifferentiableNetwork
from ..models.ppo_agent import PPOAgent
from ..models.genetic_algorithm import Individual, WeightGenerator
from ..schemas.requests import EvaluationRequest, ComparisonRequest
from ..schemas.responses import EvaluationResultData, ComparisonResultData
from ..core.exceptions import ModelException


class EvaluationService:
    """
    评估服务
    用于评估训练好的模型性能
    """
    
    PASSING_SCORE = 200.0  # 及格线
    
    def __init__(self):
        self.weight_generator = WeightGenerator()
        logger.info("EvaluationService initialized")
        
    def evaluate_network(
        self,
        request: EvaluationRequest,
        agent: Optional[PPOAgent] = None
    ) -> EvaluationResultData:
        """
        评估网络性能
        
        Args:
            request: 评估请求
            agent: PPO智能体（可选，如果提供则使用其网络）
            
        Returns:
            评估结果
        """
        task_id = str(uuid.uuid4())
        
        env = gym.make(request.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        try:
            # 创建或加载网络
            if request.network_type == "differentiable":
                if agent is not None:
                    network = agent.policy_net
                elif request.model_path:
                    network = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
                    checkpoint = torch.load(request.model_path)
                    network.load_state_dict(checkpoint['policy_state_dict'])
                else:
                    raise ModelException("No model provided for differentiable network evaluation")
            else:
                network = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
                if request.model_path:
                    network.load_state_dict(torch.load(request.model_path))
            
            network.eval()
            
            # 评估
            rewards = []
            for episode in range(request.num_episodes):
                state, _ = env.reset()
                episode_reward = 0.0
                
                while True:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state)
                        logits = network(state_tensor)
                        action = logits.argmax().item()
                        
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                    if terminated or truncated:
                        break
                        
                rewards.append(episode_reward)
                
            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))
            min_reward = float(np.min(rewards))
            max_reward = float(np.max(rewards))
            passed = mean_reward >= self.PASSING_SCORE
            
            logger.info(
                f"Evaluation completed: {request.network_type} network, "
                f"mean={mean_reward:.2f}, std={std_reward:.2f}, passed={passed}"
            )
            
            return EvaluationResultData(
                task_id=task_id,
                network_type=request.network_type,
                num_episodes=request.num_episodes,
                mean_reward=mean_reward,
                std_reward=std_reward,
                min_reward=min_reward,
                max_reward=max_reward,
                passed=passed
            )
            
        finally:
            env.close()
    
    def evaluate_with_seeds(
        self,
        seeds: list[int],
        env_name: str = "LunarLander-v2",
        num_episodes: int = 10
    ) -> EvaluationResultData:
        """
        使用种子评估不可微网络
        
        Args:
            seeds: 24个整数种子（4行6列）
            env_name: 环境名称
            num_episodes: 评估回合数
            
        Returns:
            评估结果
        """
        if len(seeds) != 24:
            raise ModelException(f"Expected 24 seeds (4x6), got {len(seeds)}")
            
        task_id = str(uuid.uuid4())
        
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        try:
            # 创建个体和网络
            individual = Individual.create_from_list(seeds)
            network = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
            
            # 获取网络形状并生成权重
            network_shapes = {
                name: tuple(param.shape)
                for name, param in network.named_parameters()
            }
            weights = self.weight_generator.generate_weights_from_individual(
                individual, network_shapes
            )
            self.weight_generator.apply_weights_to_network(network, weights)
            network.eval()
            
            # 评估
            rewards = []
            for _ in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0.0
                
                while True:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state)
                        logits = network(state_tensor)
                        action = logits.argmax().item()
                        
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                    if terminated or truncated:
                        break
                        
                rewards.append(episode_reward)
                
            mean_reward = float(np.mean(rewards))
            
            return EvaluationResultData(
                task_id=task_id,
                network_type="non_differentiable",
                num_episodes=num_episodes,
                mean_reward=mean_reward,
                std_reward=float(np.std(rewards)),
                min_reward=float(np.min(rewards)),
                max_reward=float(np.max(rewards)),
                passed=mean_reward >= self.PASSING_SCORE
            )
            
        finally:
            env.close()
    
    def compare_networks(self, request: ComparisonRequest) -> ComparisonResultData:
        """
        比较可微和不可微网络的性能
        
        Args:
            request: 比较请求
            
        Returns:
            比较结果
        """
        env = gym.make(request.env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        
        # 评估可微网络
        diff_request = EvaluationRequest(
            network_type="differentiable",
            num_episodes=request.num_episodes,
            model_path=request.differentiable_model_path,
            env_name=request.env_name
        )
        diff_result = self.evaluate_network(diff_request)
        
        # 评估不可微网络
        if request.non_differentiable_seeds:
            non_diff_result = self.evaluate_with_seeds(
                seeds=request.non_differentiable_seeds,
                env_name=request.env_name,
                num_episodes=request.num_episodes
            )
        else:
            non_diff_request = EvaluationRequest(
                network_type="non_differentiable",
                num_episodes=request.num_episodes,
                env_name=request.env_name
            )
            non_diff_result = self.evaluate_network(non_diff_request)
        
        # 计算权重差异（如果都有模型的话）
        weight_diff_norm = 0.0
        if request.differentiable_model_path and request.non_differentiable_seeds:
            diff_net = DifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
            checkpoint = torch.load(request.differentiable_model_path)
            diff_net.load_state_dict(checkpoint['policy_state_dict'])
            
            non_diff_net = NonDifferentiableNetwork(state_dim=state_dim, action_dim=action_dim)
            individual = Individual.create_from_list(request.non_differentiable_seeds)
            network_shapes = {
                name: tuple(param.shape)
                for name, param in non_diff_net.named_parameters()
            }
            weights = self.weight_generator.generate_weights_from_individual(
                individual, network_shapes
            )
            self.weight_generator.apply_weights_to_network(non_diff_net, weights)
            
            # 计算L2范数差异
            total_diff = 0.0
            for (_, p1), (_, p2) in zip(
                diff_net.named_parameters(),
                non_diff_net.named_parameters()
            ):
                total_diff += torch.sum((p1 - p2) ** 2).item()
            weight_diff_norm = float(np.sqrt(total_diff))
        
        performance_gap = diff_result.mean_reward - non_diff_result.mean_reward
        
        logger.info(
            f"Comparison completed: diff={diff_result.mean_reward:.2f}, "
            f"non_diff={non_diff_result.mean_reward:.2f}, gap={performance_gap:.2f}"
        )
        
        return ComparisonResultData(
            differentiable_result=diff_result,
            non_differentiable_result=non_diff_result,
            weight_difference_norm=weight_diff_norm,
            performance_gap=performance_gap
        )
