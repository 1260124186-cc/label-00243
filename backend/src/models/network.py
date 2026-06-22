"""
神经网络模型模块
严格按照Prompt规格实现不可微网络和可微网络

不可微网络架构:
- 输入: input_dim维状态向量（如LunarLander的8维）
- 第一层:
  - linear(input_dim, input_dim*3) -> reshape为 input_dim*3 = q1
  - linear(input_dim, input_dim*3) -> reshape为 input_dim*3 = k1
  - linear(input_dim, input_dim*3) -> reshape为 input_dim*3 = q2
  - linear(input_dim, 3*3) -> reshape为 3*3 = k2
  - argmax(q1*k1.T) -> input_dim*1 = idx
  - q2*k2.T -> input_dim*3 = v
  - v[idx] -> input_dim*3 -> 1*(input_dim*3) = r (flatten)
- 第二层: 同结构，输入input_dim*3维，输出6维
- 输出层: linear(6, action_dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger


class NonDifferentiableAttentionLayer(nn.Module):
    """
    不可微注意力选择层
    严格按照Prompt规格实现v[idx]操作
    
    实现:
    - q1 = linear(x).reshape(input_dim, 3)
    - k1 = linear(x).reshape(input_dim, 3)
    - q2 = linear(x).reshape(input_dim, 3)
    - k2 = linear(x).reshape(3, 3)
    - attention = q1 @ k1.T  -> (input_dim, input_dim)
    - idx = argmax(attention, dim=-1)  -> (input_dim,)
    - v = q2 @ k2.T  -> (input_dim, 3)
    - r[i] = v[idx[i]]  -> (input_dim, 3)
    - output = r.flatten()  -> (input_dim * 3,)
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度（应该是input_dim * 3对于第一层）
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 计算中间维度
        # 对于第一层：input_dim=8, output_dim=24 (8*3)
        # 对于第二层：input_dim=24, output_dim=6
        self.num_rows = input_dim  # q1, k1, q2的行数
        self.num_cols = 3  # 固定为3列
        
        # q1, k1, q2: linear(input_dim, input_dim*3) -> reshape(input_dim, 3)
        self.linear_q1 = nn.Linear(input_dim, input_dim * 3)
        self.linear_k1 = nn.Linear(input_dim, input_dim * 3)
        self.linear_q2 = nn.Linear(input_dim, input_dim * 3)
        
        # k2: linear(input_dim, 9) -> reshape(3, 3)
        self.linear_k2 = nn.Linear(input_dim, 9)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim] 或 [input_dim]
            
        Returns:
            输出张量 [batch_size, input_dim*3] 或 [input_dim*3]
        """
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
            
        batch_size = x.shape[0]
        
        # 计算 q1, k1: (batch, input_dim, 3)
        q1 = self.linear_q1(x).view(batch_size, self.input_dim, 3)
        k1 = self.linear_k1(x).view(batch_size, self.input_dim, 3)
        
        # 计算 q2: (batch, input_dim, 3)
        q2 = self.linear_q2(x).view(batch_size, self.input_dim, 3)
        
        # 计算 k2: (batch, 3, 3)
        k2 = self.linear_k2(x).view(batch_size, 3, 3)
        
        # attention = q1 @ k1.T -> (batch, input_dim, input_dim)
        attention = torch.bmm(q1, k1.transpose(1, 2))
        
        # idx = argmax(attention, dim=-1) -> (batch, input_dim)
        idx = attention.argmax(dim=-1)
        
        # v = q2 @ k2.T -> (batch, input_dim, 3)
        v = torch.bmm(q2, k2.transpose(1, 2))
        
        # r[i] = v[idx[i]] -> 使用gather实现
        # idx: (batch, input_dim) -> (batch, input_dim, 3)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)
        # r: (batch, input_dim, 3)
        r = torch.gather(v, 1, idx_expanded)
        
        # flatten: (batch, input_dim * 3)
        output = r.view(batch_size, -1)
        
        if squeeze_output:
            output = output.squeeze(0)
            
        return output


class DifferentiableAttentionLayer(nn.Module):
    """
    可微注意力选择层
    使用softmax替代argmax实现可微性
    通过温度参数控制逼近程度
    """
    
    def __init__(self, input_dim: int, output_dim: int, temperature: float = 1.0):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            temperature: softmax温度参数
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.temperature = temperature
        
        self.num_rows = input_dim
        self.num_cols = 3
        
        self.linear_q1 = nn.Linear(input_dim, input_dim * 3)
        self.linear_k1 = nn.Linear(input_dim, input_dim * 3)
        self.linear_q2 = nn.Linear(input_dim, input_dim * 3)
        self.linear_k2 = nn.Linear(input_dim, 9)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（可微版本）"""
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
            
        batch_size = x.shape[0]
        
        q1 = self.linear_q1(x).view(batch_size, self.input_dim, 3)
        k1 = self.linear_k1(x).view(batch_size, self.input_dim, 3)
        q2 = self.linear_q2(x).view(batch_size, self.input_dim, 3)
        k2 = self.linear_k2(x).view(batch_size, 3, 3)
        
        # attention scores
        attention = torch.bmm(q1, k1.transpose(1, 2))
        
        # 可微版本：使用softmax替代argmax
        weights = F.softmax(attention / self.temperature, dim=-1)
        
        # v = q2 @ k2.T
        v = torch.bmm(q2, k2.transpose(1, 2))
        
        # 软选择：r = weights @ v
        r = torch.bmm(weights, v)
        
        output = r.view(batch_size, -1)
        
        if squeeze_output:
            output = output.squeeze(0)
            
        return output
    
    def set_temperature(self, temperature: float) -> None:
        """设置温度参数"""
        self.temperature = max(temperature, 1e-6)


class NonDifferentiableNetwork(nn.Module):
    """
    不可微神经网络
    严格按照Prompt规格实现
    
    架构:
    - 输入: state_dim维状态向量
    - 第一层: NonDifferentiableAttentionLayer (state_dim -> state_dim*3)
    - 第二层: NonDifferentiableAttentionLayer (state_dim*3 -> 6)
    - 输出层: Linear (6 -> action_dim)
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4
    ):
        """
        Args:
            state_dim: 状态空间维度（LunarLander-v2 为 8）
            action_dim: 动作空间维度（LunarLander-v2 为 4）
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 第一层: state_dim -> state_dim * 3
        self.layer1 = NonDifferentiableAttentionLayer(
            input_dim=state_dim,
            output_dim=state_dim * 3
        )
        
        # 中间维度
        hidden_dim = state_dim * 3  # 例如8*3=24
        
        # 第二层: hidden_dim -> 6
        self.layer2 = NonDifferentiableAttentionLayer(
            input_dim=hidden_dim,
            output_dim=6
        )
        
        # 第二层实际输出: hidden_dim * 3，需要一个projection到6
        # 根据Prompt，第二层输出为6维
        self.layer2_proj = nn.Linear(hidden_dim * 3, 6)
        
        # 输出层: 6 -> action_dim
        self.output_layer = nn.Linear(6, action_dim)
        
        logger.info(
            f"NonDifferentiableNetwork initialized: "
            f"state_dim={state_dim}, action_dim={action_dim}"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态输入 [batch_size, state_dim] 或 [state_dim]
            
        Returns:
            动作logits [batch_size, action_dim] 或 [action_dim]
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_proj(x)
        x = self.output_layer(x)
        return x


class DifferentiableNetwork(nn.Module):
    """
    可微神经网络（不可微网络的可微版本）
    
    通过使用 softmax 替代 argmax 实现可微性
    温度参数控制逼近程度：温度越低越接近不可微版本
    """
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        initial_temperature: float = 1.0
    ):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            initial_temperature: 初始温度参数
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.temperature = initial_temperature
        
        # 第一层（可微版本）
        self.layer1 = DifferentiableAttentionLayer(
            input_dim=state_dim,
            output_dim=state_dim * 3,
            temperature=initial_temperature
        )
        
        hidden_dim = state_dim * 3
        
        # 第二层（可微版本）
        self.layer2 = DifferentiableAttentionLayer(
            input_dim=hidden_dim,
            output_dim=6,
            temperature=initial_temperature
        )
        
        self.layer2_proj = nn.Linear(hidden_dim * 3, 6)
        
        # 输出层
        self.output_layer = nn.Linear(6, action_dim)
        
        logger.info(
            f"DifferentiableNetwork initialized: "
            f"state_dim={state_dim}, action_dim={action_dim}, "
            f"temperature={initial_temperature}"
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2_proj(x)
        x = self.output_layer(x)
        return x
    
    def set_temperature(self, temperature: float) -> None:
        """设置温度参数"""
        self.temperature = max(temperature, 1e-6)
        self.layer1.set_temperature(self.temperature)
        self.layer2.set_temperature(self.temperature)
        
    def anneal_temperature(self, decay_rate: float = 0.995, min_temperature: float = 0.01) -> float:
        """温度退火"""
        new_temp = max(self.temperature * decay_rate, min_temperature)
        self.set_temperature(new_temp)
        return new_temp
    
    def copy_weights_from(self, source_network: nn.Module) -> None:
        """从另一个网络复制权重"""
        self.load_state_dict(source_network.state_dict())
        
    def get_regularization_loss(self, target_network: nn.Module) -> torch.Tensor:
        """计算与目标网络的正则化损失"""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for (name1, param1), (name2, param2) in zip(
            self.named_parameters(),
            target_network.named_parameters()
        ):
            loss += F.mse_loss(param1, param2.detach())
            
        return loss
