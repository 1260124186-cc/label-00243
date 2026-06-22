"""
PPO (Proximal Policy Optimization) 智能体模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from .network import DifferentiableNetwork, NonDifferentiableNetwork


@dataclass
class Trajectory:
    """轨迹数据"""
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    def clear(self):
        """清空轨迹"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


@dataclass
class TrainingResult:
    """训练结果"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    best_reward: float = float('-inf')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'temperatures': self.temperatures,
            'best_reward': self.best_reward,
            'avg_reward_last_100': float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0
        }


class ValueNetwork(nn.Module):
    """价值网络（Critic）"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PPOAgent:
    """
    PPO 智能体
    
    使用可微网络作为策略网络，通过温度退火逐渐逼近不可微版本
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        initial_temperature: float = 1.0,
        temperature_decay: float = 0.995,
        min_temperature: float = 0.01,
        regularization_coef: float = 0.1,
        device: str = "auto"
    ):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            ppo_epochs: 每次更新的epoch数
            batch_size: 批次大小
            initial_temperature: 初始温度
            temperature_decay: 温度衰减率
            min_temperature: 最小温度
            regularization_coef: 正则化系数
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        self.regularization_coef = regularization_coef
        
        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # 策略网络（可微版本）
        self.policy_net = DifferentiableNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            initial_temperature=initial_temperature
        ).to(self.device)
        
        # 价值网络
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_net.parameters(), 'lr': learning_rate},
            {'params': self.value_net.parameters(), 'lr': learning_rate}
        ])
        
        # 目标网络（不可微版本，用于正则化）
        self.target_network: Optional[nn.Module] = None
        
        # 状态归一化
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.state_count = 0
        
        logger.info(
            f"PPOAgent initialized: state_dim={state_dim}, action_dim={action_dim}, "
            f"lr={learning_rate}, gamma={gamma}, epsilon={epsilon}"
        )
        
    def normalize_state(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        归一化状态（支持任意维度输入）
        
        Args:
            state: 原始状态
            update_stats: 是否更新统计信息
            
        Returns:
            归一化后的状态
        """
        if update_stats:
            self.state_count += 1
            delta = state - self.state_mean
            self.state_mean += delta / self.state_count
            self.state_std = np.sqrt(
                ((self.state_count - 1) * self.state_std ** 2 + delta * (state - self.state_mean)) 
                / self.state_count
            )
            self.state_std = np.maximum(self.state_std, 1e-6)
            
        return (state - self.state_mean) / self.state_std
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否确定性选择
            
        Returns:
            (动作, log概率, 状态价值)
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
            
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = probs.argmax().item()
        else:
            action = dist.sample().item()
            
        log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        
        return action, log_prob, value.item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计 (GAE)
        
        Args:
            rewards: 奖励列表
            values: 价值列表
            dones: 终止标志列表
            next_value: 下一状态的价值
            
        Returns:
            (优势, 回报)
        """
        advantages = []
        gae = 0
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)
        
        return advantages, returns
    
    def update(self, trajectory: Trajectory, next_value: float) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            trajectory: 轨迹数据
            next_value: 最后状态的价值估计
            
        Returns:
            损失字典
        """
        # 转换数据
        states = torch.FloatTensor(np.array(trajectory.states)).to(self.device)
        actions = torch.LongTensor(trajectory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(trajectory.log_probs).to(self.device)
        
        # 计算GAE
        advantages, returns = self.compute_gae(
            trajectory.rewards,
            trajectory.values,
            trajectory.dones,
            next_value
        )
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_reg_loss = 0
        update_count = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                logits = self.policy_net(batch_states)
                values = self.value_net(batch_states)
                
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO裁剪
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values, batch_returns)
                
                # 正则化损失（逼近目标网络）
                reg_loss = torch.tensor(0.0).to(self.device)
                if self.target_network is not None:
                    reg_loss = self.policy_net.get_regularization_loss(self.target_network)
                
                # 总损失
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    - self.entropy_coef * entropy
                    + self.regularization_coef * reg_loss
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_reg_loss += reg_loss.item()
                update_count += 1
        
        # 温度退火
        new_temp = self.policy_net.anneal_temperature(
            self.temperature_decay,
            self.min_temperature
        )
        
        return {
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count,
            'reg_loss': total_reg_loss / update_count,
            'temperature': new_temp
        }
    
    def set_target_network(self, target_network: nn.Module) -> None:
        """设置目标网络（不可微版本）用于正则化"""
        self.target_network = target_network.to(self.device)
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def train(
        self,
        env,
        total_episodes: int = 1000,
        max_steps: int = 1000,
        update_interval: int = 2048,
        log_interval: int = 10,
        callback: Optional[callable] = None
    ) -> TrainingResult:
        """
        训练智能体
        
        Args:
            env: Gymnasium 环境
            total_episodes: 总训练回合数
            max_steps: 每回合最大步数
            update_interval: 更新间隔（步数）
            log_interval: 日志间隔（回合数）
            callback: 回调函数
            
        Returns:
            训练结果
        """
        result = TrainingResult()
        trajectory = Trajectory()
        total_steps = 0
        
        for episode in range(total_episodes):
            state, _ = env.reset()
            state = self.normalize_state(state)
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                action, log_prob, value = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_state = self.normalize_state(next_state)
                
                trajectory.states.append(state)
                trajectory.actions.append(action)
                trajectory.rewards.append(reward)
                trajectory.log_probs.append(log_prob)
                trajectory.values.append(value)
                trajectory.dones.append(done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # 更新策略
                if total_steps % update_interval == 0:
                    with torch.no_grad():
                        next_value = self.value_net(
                            torch.FloatTensor(state).to(self.device)
                        ).item()
                    
                    losses = self.update(trajectory, next_value)
                    result.policy_losses.append(losses['policy_loss'])
                    result.value_losses.append(losses['value_loss'])
                    result.temperatures.append(losses['temperature'])
                    trajectory.clear()
                
                if done:
                    break
            
            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(episode_length)
            
            if episode_reward > result.best_reward:
                result.best_reward = episode_reward
            
            # 日志
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(result.episode_rewards[-log_interval:])
                logger.info(
                    f"Episode {episode + 1}/{total_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Best: {result.best_reward:.2f} | "
                    f"Temp: {self.policy_net.temperature:.4f}"
                )
            
            # 回调
            if callback is not None:
                callback(episode, result)
                
            # 检查是否达到目标（200分及格线）
            if len(result.episode_rewards) >= 100:
                avg_last_100 = np.mean(result.episode_rewards[-100:])
                if avg_last_100 >= 200:
                    logger.info(f"Target reached! Average reward over last 100 episodes: {avg_last_100:.2f}")
                    break
        
        return result
    
    def evaluate(
        self,
        env,
        num_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        评估智能体
        
        Args:
            env: 评估环境
            num_episodes: 评估回合数
            render: 是否渲染
            
        Returns:
            评估结果
        """
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            state = self.normalize_state(state, update_stats=False)
            episode_reward = 0
            
            while True:
                action, _, _ = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = self.normalize_state(next_state, update_stats=False)
                episode_reward += reward
                
                if done:
                    break
                    
            rewards.append(episode_reward)
            
        return {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards))
        }
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'state_count': self.state_count,
        }, path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.state_count = checkpoint['state_count']
        logger.info(f"Model loaded from {path}")
