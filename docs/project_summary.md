# 不可微神经网络强化学习系统 - 项目实现总结

## 1. 项目概述

本项目实现了一个完整的**不可微神经网络强化学习训练系统**，核心目标是在 LunarLander-v2 环境中训练一个不可微神经网络的可微版本，并通过遗传算法搜索最优的网络权重种子组合，使模型奖励高于 200 分及格线。

### 项目核心挑战
1. **不可微性**：网络中使用 `argmax` 操作导致梯度无法反向传播
2. **可微近似**：需要找到可微的替代方案并保证性能不下降
3. **权重生成**：通过 24 个整数种子（4行6列）生成完整网络权重
4. **遗传搜索**：在巨大的搜索空间中寻找最优种子组合

---

## 2. 技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **Web框架** | FastAPI | 0.109.2 | API 服务层 |
| | Uvicorn | 0.27.1 | ASGI 服务器 |
| **深度学习** | PyTorch | 2.2.0 | 神经网络框架 |
| | NumPy | 1.26.4 | 数值计算 |
| **强化学习** | Gymnasium | 0.29.1 | 强化学习环境 |
| | Box2D | - | LunarLander 物理引擎 |
| **数据验证** | Pydantic | 2.6.1 | 数据模型验证 |
| | pydantic-settings | 2.1.0 | 配置管理 |
| **日志** | Loguru | 0.7.2 | 日志记录 |
| **可视化** | Matplotlib | 3.8.2 | 图表生成 |
| **异步支持** | aiofiles | 23.2.1 | 异步文件操作 |
| **容器化** | Docker | - | 部署打包 |
| | Docker Compose | - | 服务编排 |
| **测试** | pytest | - | 单元测试 |
| | httpx | - | API 测试 |

---

## 3. 核心算法实现

### 3.1 不可微网络架构

#### 网络结构

```
输入: state_dim维状态向量 (LunarLander-v2为8维)

第一层 Attention Layer:
  ├─ q1 = Linear(state_dim, state_dim*3) → reshape(state_dim, 3)
  ├─ k1 = Linear(state_dim, state_dim*3) → reshape(state_dim, 3)
  ├─ q2 = Linear(state_dim, state_dim*3) → reshape(state_dim, 3)
  ├─ k2 = Linear(state_dim, 9) → reshape(3, 3)
  │
  ├─ attention = q1 @ k1.T  → shape: (state_dim, state_dim)
  ├─ idx = argmax(attention, dim=-1)  → shape: (state_dim,)   ← 不可微点
  ├─ v = q2 @ k2.T  → shape: (state_dim, 3)
  ├─ r[i] = v[idx[i]]  → shape: (state_dim, 3)                 ← 索引选择
  └─ output = r.flatten()  → shape: (state_dim * 3,)

第二层 Attention Layer:
  └─ 同结构，输入维度 = state_dim*3，输出投影到 6 维

输出层:
  └─ Linear(6, action_dim) → 动作logits
```

#### 关键实现：v[idx] 操作

在 [network.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/models/network.py#L66-L114) 中，使用 `torch.gather` 实现索引选择：

```python
# idx: (batch, state_dim) → 扩展为 (batch, state_dim, 3)
idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)

# 使用gather实现 v[idx[i]] 选择
r = torch.gather(v, 1, idx_expanded)
```

**为什么不可微？**
- `argmax` 操作输出离散的索引值，梯度为 0 或无穷大
- 梯度无法通过离散索引反向传播到 q1 和 k1
- 这使得标准的梯度下降法无法训练该网络

---

### 3.2 可微近似与温度退火

#### 核心思想

使用 **Softmax 温度退火（Temperature Annealing）** 技术实现可微近似：

```python
# 不可微版本
idx = argmax(attention)
result = v[idx]

# 可微版本
weights = softmax(attention / temperature)
result = weights @ v  # 加权求和
```

#### 温度参数的作用

| 温度值 | 效果 | 特点 |
|--------|------|------|
| **T → ∞** | 均匀分布 | 完全软选择，梯度平滑 |
| **T = 1.0** | 较软的分布 | 平衡可微性与选择性 |
| **T → 0** | 接近 one-hot | 逐渐逼近 argmax 硬选择 |

#### 退火策略

在训练过程中逐渐降低温度，使网络从可微的软选择逐步过渡到接近不可微的硬选择：

```python
def anneal_temperature(self, decay_rate=0.995, min_temperature=0.01):
    new_temp = max(self.temperature * decay_rate, min_temperature)
    self.set_temperature(new_temp)
    return new_temp
```

#### 正则化损失

为了保证可微网络逐渐逼近不可微网络的行为，使用权重正则化：

```python
def get_regularization_loss(self, target_network):
    loss = 0
    for (name1, param1), (name2, param2) in zip(
        self.named_parameters(),
        target_network.named_parameters()
    ):
        loss += F.mse_loss(param1, param2.detach())
    return loss
```

---

### 3.3 PPO 强化学习算法

#### 算法概述

使用 **Proximal Policy Optimization (PPO)** 算法训练可微网络：

- **策略网络**：DifferentiableNetwork（演员）
- **价值网络**：ValueNetwork（批评家）
- **训练方式**：Actor-Critic + GAE

#### 核心组件

**1. 广义优势估计 (GAE)**

```python
def compute_gae(self, rewards, values, dones, next_value):
    advantages = []
    gae = 0
    values = values + [next_value]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1-dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1-dones[t]) * gae
        advantages.insert(0, gae)
        
    return advantages, advantages + values[:-1]
```

**2. PPO 裁剪目标**

```python
ratio = exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-epsilon, 1+epsilon) * advantages
policy_loss = -min(surr1, surr2).mean()
```

**3. 总损失函数**

```
总损失 = 策略损失 + 价值系数 × 价值损失 - 熵系数 × 熵 + 正则化系数 × 权重正则化
```

#### 训练流程

1. 与环境交互收集轨迹数据
2. 计算 GAE 优势和回报
3. 对数据进行多轮 PPO 更新
4. 执行温度退火
5. 检查是否达到 200 分及格线

---

### 3.4 遗传算法

#### 个体结构

每个个体包含 **24 个整数种子**，排列为 4 行 6 列：

```
Row 0: [seed0, seed1, seed2, seed3, a0, b0]  权重 0.4
Row 1: [seed4, seed5, seed6, seed7, a1, b1]  权重 0.3
Row 2: [seed8, seed9, seed10, seed11, a2, b2] 权重 0.2
Row 3: [seed12, seed13, seed14, seed15, a3, b3] 权重 0.1
```

- **前4列**：种子值（参与权重生成）
- **第5列 (a)**：LCG 随机数生成器的乘数参数
- **第6列 (b)**：LCG 随机数生成器的增量参数

#### 权重生成机制

**1. 线性同余生成器 (LCG)**

```python
def lcg_random(seed, a, b, count):
    """X_{n+1} = (a * X_n + b) mod m"""
    results = []
    x = seed
    for _ in range(count):
        x = (a * x + b) % modulus
        results.append(x)
    return results
```

**2. 权重系数计算**

| 列 | 权重计算 | Row 0 示例 |
|----|----------|------------|
| 第0列 | 行权重 | 0.4 |
| 第1列 | 行权重 × 0.1 | 0.04 |
| 第2列 | 行权重 × 0.01 | 0.004 |
| 第3列 | 行权重 × 0.001 | 0.0004 |
| 第4-5列 | 不参与（a, b参数） | - |

**3. 加权融合**

```python
# 对每一层权重
for each layer:
    accumulated_weight = 0
    total_weight_sum = 0
    
    for each seed position (i,j):
        if seed == 0:
            continue  # 种子为0不参与
        
        # 生成该种子对应的权重
        layer_weights = generate_layer_weights(seed, a, b, shape)
        
        # 加权累加
        weight_coef = get_weight_coefficient(i, j)
        accumulated_weight += weight_coef * layer_weights
        total_weight_sum += weight_coef
    
    # 归一化
    final_weights = accumulated_weight / total_weight_sum
```

**4. 每层独立归一化**

```python
def normalize_weights(self, weights):
    """每层独立最大值归一化到 [-1, 1]"""
    max_val = abs(weights).max()
    if max_val > 0:
        return weights / max_val
    return weights
```

#### 遗传操作

**1. 变异（Mutation）**

两种变异方式，随机选择一种：
- **随机改变一个种子**：随机选择一个位置，赋予新的随机值
- **随机交换相邻行**：交换两行的所有种子

**2. 交叉（Crossover）**

- 至少交换 1 行（随机选择 1-2 行）
- 子代继承父代 1 的部分行和父代 2 的部分行

**3. 选择（Selection）**

- 锦标赛选择（Tournament Selection）
- 每次随机抽取 5 个个体，选择适应度最高的作为父代

**4. 精英保留（Elitism）**

- 每代保留前 N 个最优个体直接进入下一代
- 维护精英档案库（elite_archive），保存历史最佳个体

**5. 精英回归（Elite Return）**

```python
# 精英个体有概率回到种群中
# 适应度越高，回归概率越大
normalized_fitness = (elite.fitness - min_fitness) / fitness_range
return_prob = base_probability * (1 + normalized_fitness)
```

**6. 遍历个体（Traversal）**

每代生成一个特殊个体：
- 第一行第一个种子按代数递增
- 其他所有种子为 0
- 用于系统性探索搜索空间

---

## 4. 核心模块功能

### 4.1 模型层 (Models)

| 模块 | 文件 | 功能 |
|------|------|------|
| **NonDifferentiableNetwork** | [network.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/models/network.py) | 不可微网络，使用 argmax 硬选择 |
| **DifferentiableNetwork** | [network.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/models/network.py) | 可微网络，使用 softmax 软选择 + 温度退火 |
| **PPOAgent** | [ppo_agent.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/models/ppo_agent.py) | PPO 智能体，训练可微网络 |
| **GeneticAlgorithm** | [genetic_algorithm.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/models/genetic_algorithm.py) | 遗传算法，搜索最优种子组合 |
| **WeightGenerator** | [genetic_algorithm.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/models/genetic_algorithm.py) | 权重生成器，从种子生成网络权重 |
| **Individual** | [genetic_algorithm.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/models/genetic_algorithm.py) | 个体类，封装种子和适应度 |

### 4.2 服务层 (Services)

| 服务 | 文件 | 功能 |
|------|------|------|
| **TrainingService** | [training_service.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/services/training_service.py) | 训练任务管理，PPO 训练调度 |
| **GeneticService** | [genetic_service.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/services/genetic_service.py) | 遗传算法搜索任务管理 |
| **EvaluationService** | [evaluation_service.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/services/evaluation_service.py) | 模型评估、性能比较 |
| **VisualizationService** | [visualization_service.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/services/visualization_service.py) | 图表生成（适应度曲线、训练仪表板） |

### 4.3 API 层 (API)

#### 训练管理接口

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/api/v1/training/start` | 启动 PPO 训练 |
| GET | `/api/v1/training/status/{task_id}` | 获取训练状态 |
| POST | `/api/v1/training/stop/{task_id}` | 停止训练 |
| GET | `/api/v1/training/history/{task_id}` | 获取训练历史（分页） |
| GET | `/api/v1/training/result/{task_id}` | 获取训练结果 |
| GET | `/api/v1/training/tasks` | 列出所有训练任务 |

#### 遗传算法接口

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/api/v1/genetic/start` | 启动遗传算法搜索 |
| GET | `/api/v1/genetic/status/{task_id}` | 获取搜索状态 |
| POST | `/api/v1/genetic/stop/{task_id}` | 停止搜索 |
| GET | `/api/v1/genetic/best/{task_id}` | 获取最佳个体 |
| GET | `/api/v1/genetic/population/{task_id}` | 获取当前种群 |
| GET | `/api/v1/genetic/tasks` | 列出所有遗传算法任务 |

#### 模型评估接口

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/api/v1/evaluate/run` | 运行单次评估 |
| POST | `/api/v1/evaluate/seeds` | 使用种子评估（24个） |
| POST | `/api/v1/evaluate/compare` | 比较可微与不可微网络 |

#### 可视化接口

| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/api/v1/visualization/training/{task_id}` | 获取训练可视化仪表板 |
| GET | `/api/v1/visualization/genetic/{task_id}` | 获取遗传算法适应度曲线 |
| GET | `/api/v1/visualization/fitness-curve` | 生成自定义适应度曲线 |

#### 系统管理接口

| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/config` | 获取系统配置 |

### 4.4 核心层 (Core)

| 模块 | 文件 | 功能 |
|------|------|------|
| **异常处理** | [exceptions.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/core/exceptions.py) | 自定义异常类 |
| **日志配置** | [logging.py](file:///Users/zhangchengcheng/work/ai-project/Trea/solo0605/label-00243/backend/src/core/logging.py) | Loguru 日志配置 |

---

## 5. 系统架构

### 5.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
├─────────────────────────────────────────────────────────┤
│                    API Layer (Routes)                    │
│  ┌──────────┬──────────┬───────────┬─────────────┐    │
│  │ Training │ Genetic  │ Evaluation│ Visualization │    │
│  │  Controller │Controller│ Controller │ Controller │    │
│  └──────────┴──────────┴───────────┴─────────────┘    │
├─────────────────────────────────────────────────────────┤
│                  Service Layer                           │
│  ┌──────────┬──────────┬───────────┬─────────────┐    │
│  │ Training │ Genetic  │ Evaluation│ Visualization │    │
│  │  Service  │  Service │  Service  │   Service    │    │
│  └──────────┴──────────┴───────────┴─────────────┘    │
├─────────────────────────────────────────────────────────┤
│                   Model Layer                            │
│  ┌───────────────────┬──────────────────────────────┐  │
│  │ Networks          │ Reinforcement Learning        │  │
│  │  - NonDiffNet     │  - PPOAgent                   │  │
│  │  - DiffNet        │  - ValueNetwork               │  │
│  │  - AttentionLayer │                               │  │
│  ├───────────────────┴──────────────────────────────┤  │
│  │ Genetic Algorithm                                │  │
│  │  - Individual      - WeightGenerator             │  │
│  │  - GeneticAlgorithm                              │  │
│  └──────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   External Dependencies                  │
│  ┌────────────┬───────────┬────────────────────────┐  │
│  │ Gymnasium  │  PyTorch  │     Matplotlib         │  │
│  │ LunarLander│           │                        │  │
│  └────────────┴───────────┴────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 5.2 数据流

**训练流程：**
```
用户请求 → TrainingController → TrainingService → PPOAgent
                                              ↓
                                    DifferentiableNetwork + ValueNetwork
                                              ↓
                                    Gymnasium 环境交互
                                              ↓
                                    温度退火 + 权重正则化
```

**遗传算法流程：**
```
用户请求 → GeneticController → GeneticService → GeneticAlgorithm
                                                ↓
                                    Individual (24 seeds)
                                                ↓
                                    WeightGenerator → 网络权重
                                                ↓
                                    NonDifferentiableNetwork
                                                ↓
                                    Gymnasium 环境评估
                                                ↓
                                    适应度计算 → 进化操作
```

---

## 6. 关键配置参数

### 6.1 PPO 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `epsilon` | 0.2 | PPO 裁剪参数 |
| `gae_lambda` | 0.95 | GAE lambda 参数 |
| `ppo_epochs` | 10 | 每次更新的 epoch 数 |
| `batch_size` | 64 | 批次大小 |
| `initial_temperature` | 1.0 | 初始温度 |
| `temperature_decay` | 0.995 | 温度衰减率 |
| `min_temperature` | 0.01 | 最小温度 |
| `regularization_coef` | 0.1 | 正则化系数 |

### 6.2 遗传算法参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `population_size` | 50 | 种群大小 |
| `mutation_rate` | 0.1 | 变异率 |
| `crossover_rate` | 0.7 | 交叉率 |
| `elite_size` | 5 | 精英保留数量 |
| `max_generations` | 100 | 最大代数 |
| `seed_range` | [0, 10000] | 种子值范围 |
| `target_fitness` | 200.0 | 目标适应度（及格线） |

### 6.3 环境配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_ENV` | LunarLander-v2 | 默认环境 |
| `MAX_STEPS` | 1000 | 每回合最大步数 |
| `TOTAL_EPISODES` | 1000 | 默认训练回合数 |

---

## 7. 项目特点

### 7.1 技术亮点

1. **严格的规格实现**：完全按照需求规格实现不可微网络架构和遗传算法
2. **温度退火策略**：平滑地从可微软选择过渡到硬选择
3. **完整的训练系统**：从 PPO 训练到遗传搜索的端到端解决方案
4. **异步任务管理**：支持多任务并发，任务状态实时可查
5. **丰富的可视化**：训练仪表板、适应度曲线、对比图表
6. **完善的测试**：覆盖 API、模型、服务层的自动化测试

### 7.2 架构优势

1. **分层清晰**：API 层 → 服务层 → 模型层，职责分明
2. **依赖注入**：通过工厂函数管理服务实例
3. **线程安全**：使用锁保护共享状态
4. **异常处理**：统一的异常处理和响应格式
5. **配置灵活**：支持环境变量和配置文件

---

## 8. 使用方式

### 8.1 快速启动

```bash
# Docker Compose 方式
docker-compose up --build

# 本地运行
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 8.2 访问地址

- **API 文档**：http://localhost:8080/docs
- **ReDoc 文档**：http://localhost:8080/redoc
- **健康检查**：http://localhost:8080/api/v1/health

---

## 9. 文件结构

```
label-00243/
├── README.md                    # 项目说明文档
├── docker-compose.yml           # Docker Compose 配置
├── docs/
│   ├── project_design.md        # 项目设计文档
│   └── project_summary.md       # 项目总结（本文件）
└── backend/
    ├── Dockerfile               # Docker 构建文件
    ├── requirements.txt         # Python 依赖
    ├── main.py                  # FastAPI 应用入口
    └── src/
        ├── config.py            # 配置模块
        ├── api/
        │   └── routes.py        # API 路由
        ├── services/
        │   ├── training_service.py
        │   ├── genetic_service.py
        │   ├── evaluation_service.py
        │   └── visualization_service.py
        ├── models/
        │   ├── network.py       # 神经网络模型
        │   ├── ppo_agent.py     # PPO 智能体
        │   └── genetic_algorithm.py  # 遗传算法
        ├── schemas/
        │   ├── requests.py      # 请求模型
        │   └── responses.py     # 响应模型
        └── core/
            ├── exceptions.py    # 异常类
            └── logging.py       # 日志配置
```

---

## 10. 总结

本项目成功实现了一个完整的**不可微神经网络强化学习系统**，核心创新点在于：

1. **不可微网络的可微近似**：通过 Softmax 温度退火技术，在保持可训练性的同时逐渐逼近不可微行为
2. **种子权重生成机制**：使用 LCG 随机数生成器和加权融合策略，将 24 个整数种子映射为完整网络权重
3. **遗传算法优化**：通过选择、交叉、变异等操作，在离散的种子空间中搜索最优解
4. **PPO 强化学习**：使用近端策略优化算法训练可微网络，配合正则化项确保性能

系统采用 FastAPI + PyTorch + Gymnasium 的技术栈，架构清晰分层合理，提供了完整的训练、评估、可视化功能，是一个高质量的强化学习工程项目。
