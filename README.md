# 不可微神经网络强化学习系统

## How to Run

### 使用 Docker Compose (推荐)

```bash
# 构建并启动服务
docker-compose up --build

# 后台运行
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 本地运行

#### macOS 系统前置依赖

在 macOS 上运行前，需要先安装以下依赖：

```bash
# 安装 Homebrew（如果尚未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装 swig（用于编译 box2d）
brew install swig

# Apple Silicon (M1/M2/M3) 用户可能还需要：
brew install cmake
```

#### 创建虚拟环境并安装依赖

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境 (Windows)
.\venv\Scripts\activate

# 激活虚拟环境 (Linux/macOS)
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行服务
python3 main.py
```

### 访问服务

- API 文档: http://localhost:8080/docs
- ReDoc 文档: http://localhost:8080/redoc
- 健康检查: http://localhost:8080/api/v1/health

## 自动化测试

### 运行测试

项目包含完整的自动化测试套件，覆盖API接口、神经网络、遗传算法和可视化服务。

**重要：运行测试前，请确保已安装所有依赖并激活虚拟环境！**

#### Windows 系统

```bash
# 进入后端目录
cd backend

# 确保虚拟环境已创建并激活
python -m venv venv
.\venv\Scripts\activate

# 安装所有依赖（包括测试依赖）
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx pytest-cov

# 方式一：使用测试脚本运行
python run_tests.py

# 方式二：直接使用pytest运行
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_api.py -v        # API接口测试
python -m pytest tests/test_network.py -v    # 神经网络测试
python -m pytest tests/test_genetic.py -v    # 遗传算法测试
python -m pytest tests/test_visualization.py -v  # 可视化测试

# 运行带覆盖率报告的测试
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# 生成HTML覆盖率报告
python -m pytest tests/ -v --cov=src --cov-report=html
```

#### macOS / Linux 系统

```bash
# 进入后端目录
cd backend

# 确保虚拟环境已创建并激活
python3 -m venv venv
source venv/bin/activate

# 安装所有依赖（包括测试依赖）
pip3 install -r requirements.txt
pip3 install pytest pytest-asyncio httpx pytest-cov

# 方式一：使用测试脚本运行
python3 run_tests.py

# 方式二：直接使用pytest运行
python3 -m pytest tests/ -v

# 运行特定测试文件
python3 -m pytest tests/test_api.py -v        # API接口测试
python3 -m pytest tests/test_network.py -v    # 神经网络测试
python3 -m pytest tests/test_genetic.py -v    # 遗传算法测试
python3 -m pytest tests/test_visualization.py -v  # 可视化测试

# 运行带覆盖率报告的测试
python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

# 生成HTML覆盖率报告
python3 -m pytest tests/ -v --cov=src --cov-report=html
```

### 测试结构

| 测试文件 | 测试内容 | 测试数量 |
|----------|----------|----------|
| test_api.py | API接口、响应格式、分页、OpenAPI文档 | 22个测试 |
| test_network.py | 不可微/可微网络、梯度流动、温度退火 | 15个测试 |
| test_genetic.py | 个体结构、权重生成、遗传操作、进化流程 | 18个测试 |
| test_visualization.py | 适应度曲线、训练仪表板、进度图 | 12个测试 |

### 测试覆盖范围

- **API层测试**：健康检查、配置获取、训练管理、遗传算法管理、评估、可视化接口
- **模型层测试**：不可微注意力层、可微注意力层、完整网络前向传播、梯度计算
- **服务层测试**：权重生成器、遗传算法操作（变异、交叉、选择）、可视化图表生成
- **集成测试**：完整的训练流程、遗传搜索流程、可视化流程

### 常见问题排查

#### 测试时出现 `ModuleNotFoundError: No module named 'fastapi'` 或 `No module named 'loguru'`

**原因**：依赖未安装或未在正确的虚拟环境中运行

**解决方法**：
```bash
cd backend

# 1. 创建并激活虚拟环境（如果还没有）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
.\venv\Scripts\activate  # Windows

# 2. 安装所有依赖
pip install -r requirements.txt

# 3. 安装测试依赖
pip install pytest pytest-asyncio httpx pytest-cov

# 4. 重新运行测试
python3 run_tests.py
```

#### 测试时出现 `gymnasium.error.VersionNotFound: Environment version 'v3' for environment 'LunarLander'`

**原因**：gymnasium 0.29.1 版本不支持 `LunarLander-v3`，只支持 `LunarLander-v2`

**解决方法**：
- 代码中已统一使用 `LunarLander-v2`
- 如果测试仍然失败，请确保使用最新代码（已修复环境版本问题）
- 或者手动将环境名称改为 `LunarLander-v2`：
  ```bash
  # 在 .env 文件中设置
  DEFAULT_ENV=LunarLander-v2
  ```

### 测试注意事项

1. **必须安装所有依赖**：运行测试前，请确保已执行 `pip install -r requirements.txt`，否则会出现 `ModuleNotFoundError`（如 `fastapi`、`loguru` 等模块未找到）

2. **虚拟环境**：强烈建议在虚拟环境中运行测试，避免与系统 Python 环境冲突

3. **环境版本**：本项目使用 `LunarLander-v2`（gymnasium 0.29.1 不支持 v3）。如果遇到版本错误，请确保代码和配置中使用的是 `LunarLander-v2`

4. **环境依赖**：如果需要完整运行所有测试，可以尝试安装 Box2D：

   **Windows 系统**：
   ```bash
   pip install swig
   pip install "gymnasium[box2d]"
   ```

   **macOS 系统**：
   ```bash
   # 先安装系统依赖
   brew install swig cmake
   
   # 然后安装 Python 包
   pip install "gymnasium[box2d]"
   ```

   **Linux 系统**：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install swig cmake
   
   # 然后安装 Python 包
   pip install "gymnasium[box2d]"
   ```

   如果Box2D未安装，相关测试会自动跳过，不影响其他测试的运行。

2. **测试隔离**：每个测试用例独立运行，不依赖其他测试的状态。

3. **预期结果**：正常情况下应有71+测试通过，1个测试跳过（Box2D依赖）。

4. **macOS Apple Silicon 注意事项**：
   - 如果使用 M1/M2/M3 芯片，PyTorch 已原生支持 Apple Silicon
   - 确保使用 Python 3.9 或更高版本
   - 如果遇到依赖问题，尝试使用 `conda` 或 `miniforge` 创建环境

## Services

| 服务 | 端口 | 描述 |
|------|------|------|
| Backend API | 8080 | 主后端服务，提供训练、遗传算法、评估、可视化等接口 |

### API 端点概览

#### 训练管理
- `POST /api/v1/training/start` - 启动PPO训练
- `GET /api/v1/training/status/{task_id}` - 获取训练状态
- `POST /api/v1/training/stop/{task_id}` - 停止训练
- `GET /api/v1/training/history/{task_id}` - 获取训练历史
- `GET /api/v1/training/tasks` - 列出所有训练任务

#### 遗传算法
- `POST /api/v1/genetic/start` - 启动遗传算法搜索
- `GET /api/v1/genetic/status/{task_id}` - 获取搜索状态
- `GET /api/v1/genetic/best/{task_id}` - 获取最佳个体
- `GET /api/v1/genetic/population/{task_id}` - 获取当前种群

#### 模型评估
- `POST /api/v1/evaluate/run` - 运行评估
- `POST /api/v1/evaluate/seeds` - 使用种子评估
- `POST /api/v1/evaluate/compare` - 比较网络性能

#### 可视化
- `GET /api/v1/visualization/training/{task_id}` - 获取训练可视化仪表板
- `GET /api/v1/visualization/genetic/{task_id}` - 获取遗传算法适应度曲线
- `GET /api/v1/visualization/fitness-curve` - 生成自定义适应度曲线

#### 系统管理
- `GET /api/v1/health` - 健康检查
- `GET /api/v1/config` - 获取配置

## 测试账号

本系统为纯后端API服务，无需登录认证。直接通过API文档页面测试即可。

### 快速测试示例

```bash
# 启动PPO训练
curl -X POST "http://localhost:8080/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{"total_episodes": 100, "env_name": "LunarLander-v3"}'

# 启动遗传算法搜索
curl -X POST "http://localhost:8080/api/v1/genetic/start" \
  -H "Content-Type: application/json" \
  -d '{"population_size": 30, "max_generations": 50}'

# 健康检查
curl "http://localhost:8080/api/v1/health"

# 获取训练可视化
curl "http://localhost:8080/api/v1/visualization/training/{task_id}"
```

## 题目内容

在LunarLander-v2环境中使用PPO算法训练一个以下不可微网络的可微版本并使用正则化逐渐接近不可微版本保证其reward高于200分及格线： 
### 不可微网络架构 
- 输入 : 6维状态向量（或其他维度，如LunarLander的8维） 
- 第一层 : 
- linear(6,18) -> 6*3=q1 
- linear(6,18) -> 6*3=k1 
- linear(6,18) -> 6*3=q2 
- linear(6,9) -> 3*3=k2 
- argmax(q1*k1.T) -> 6*1=idx 
- q2*k2.T -> 6*3=v 
- v[idx] -> 6*3 -> 1*18=r 
v[idx]实现 ： 
- 正确实现 r[i] = v[idx[i]] ，即r的第i行是v的第idx[i]行 
- v是6x3矩阵，idx是6x1向量 
- 输出r是6x3矩阵，然后flatten为18维向量 
- 第二层 : 
- 结构与第一层相同，输入为18维，输出为6维 
- 输出层 : 
- linear(6, action_dim) 
### 然后基于以下网络权重生成方法，用遗传算法寻找最符合上步求得的不可微版本网络权重的个体： 
- 个体 : 24个整数种子 
- 随机数生成 : 同余随机整数生成算法，每行使用最后两个种子作为随机算法的a和b 
- 权重生成 : 
- 每个种子生成一套完整权重 
- 整数到浮点数转换：使用每层独立的最大值归一化 
- 种子为0时不生成权重，也不参与加权 
- 加权融合 : 
- 第一列权值： [0.4, 0.3, 0.2, 0.1] 
- 每行后三个权值： [0.1, 0.01, 0.001]*该行第一列权值 
- 变异： 
- 随机改变一个种子 
- 随机交换相邻行 
- 交叉： 
- 至少交换一行对应行 
- 精英保留： 
- 保存获得历史更高适应度的个体及其适应度 
- 有概率回到场上，适应度越大概率越高 
- 遍历： 
- 生成第一行第一个种子按代数逐渐递增、其他种子全为0的的个体 
### 环境支持 
- 支持任意维度的输入状态，输入状态归一化 
- 可扩展到其他Gymnasium环境 
###  特点 
- 灵活架构 : 支持不同输入维度 
- 高效训练 : 遗传算法优化，支持多进程扩展 
- 可配置参数 : 支持调整种群大小、变异率、交叉率等 
- 可视化 : 训练过程中生成适应度曲线并实时显示

---

## 项目介绍

### 系统概述

本项目实现了一个完整的不可微神经网络强化学习训练系统，核心功能包括：

1. **不可微网络架构实现**：严格按照Prompt规格，基于注意力机制的索引选择层，使用 `argmax` 实现不可微的硬选择
2. **可微近似版本**：使用 `softmax` 温度退火技术，通过降低温度逐渐逼近不可微版本
3. **PPO训练**：使用 Proximal Policy Optimization 算法训练可微网络
4. **遗传算法搜索**：搜索最优的24个整数种子组合（4行6列），用于生成网络权重
5. **可视化**：训练过程中实时生成适应度曲线

### 核心算法实现

#### 不可微网络架构（严格按Prompt）

```
输入: state_dim维向量 (如LunarLander的8维)

第一层:
  q1 = linear(state_dim, state_dim*3).reshape(state_dim, 3)
  k1 = linear(state_dim, state_dim*3).reshape(state_dim, 3)
  q2 = linear(state_dim, state_dim*3).reshape(state_dim, 3)
  k2 = linear(state_dim, 9).reshape(3, 3)
  
  attention = q1 @ k1.T  -> (state_dim, state_dim)
  idx = argmax(attention, dim=-1)  -> (state_dim,)
  v = q2 @ k2.T  -> (state_dim, 3)
  r[i] = v[idx[i]]  -> (state_dim, 3)
  output = r.flatten()  -> (state_dim * 3,)

第二层: 同结构
输出层: linear(6, action_dim)
```

#### v[idx] 的实现

```python
# v: [batch, state_dim, 3] - 值矩阵
# idx: [batch, state_dim] - argmax索引
# 实现 r[i] = v[idx[i]]

idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)
result = torch.gather(v, 1, idx_expanded)
```

#### 可微近似 (Softmax Temperature Annealing)

```python
# 使用softmax进行可微近似
weights = F.softmax(attention / temperature, dim=-1)
result = torch.bmm(weights, v)

# 温度退火：逐渐降低temperature使其接近argmax
temperature = max(temperature * decay_rate, min_temperature)
```

#### 遗传算法个体结构（4行6列）

```
Row 0: [seed0, seed1, seed2, seed3, a0, b0]  权重0.4
Row 1: [seed4, seed5, seed6, seed7, a1, b1]  权重0.3
Row 2: [seed8, seed9, seed10, seed11, a2, b2] 权重0.2
Row 3: [seed12, seed13, seed14, seed15, a3, b3] 权重0.1

- 每行前4个是种子（第一列+后三个）
- 最后两个是LCG随机数生成的参数a和b
- 第一列权值：[0.4, 0.3, 0.2, 0.1]
- 后三列权值：[0.1, 0.01, 0.001] * 该行第一列权值
```

### 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
├─────────────────────────────────────────────────────────┤
│  API Layer                                               │
│  ├── TrainingController (PPO训练管理)                    │
│  ├── GeneticController (遗传算法管理)                    │
│  ├── EvaluationController (模型评估)                     │
│  ├── VisualizationController (可视化)                    │
│  └── SystemController (系统管理)                         │
├─────────────────────────────────────────────────────────┤
│  Service Layer                                           │
│  ├── TrainingService (训练任务调度)                      │
│  ├── GeneticService (遗传搜索调度)                       │
│  ├── EvaluationService (评估执行)                        │
│  └── VisualizationService (图表生成)                     │
├─────────────────────────────────────────────────────────┤
│  Model Layer                                             │
│  ├── NonDifferentiableNetwork (不可微网络)               │
│  ├── DifferentiableNetwork (可微网络)                    │
│  ├── PPOAgent (PPO智能体)                                │
│  ├── GeneticAlgorithm (遗传算法)                         │
│  └── WeightGenerator (权重生成器)                        │
├─────────────────────────────────────────────────────────┤
│  External                                                │
│  ├── Gymnasium (LunarLander-v2环境)                      │
│  ├── PyTorch (深度学习框架)                              │
│  └── Matplotlib (可视化)                                 │
└─────────────────────────────────────────────────────────┘
```

### 项目结构

```
label-00243/
├── README.md                    # 项目说明文档
├── docker-compose.yml           # Docker Compose配置
├── .gitignore                   # Git忽略文件
├── docs/
│   └── project_design.md        # 项目设计文档
└── backend/
    ├── Dockerfile               # Docker构建文件（多平台支持AMD64/ARM64）
    ├── requirements.txt         # Python依赖
    ├── .env.example             # 环境变量示例
    ├── main.py                  # FastAPI应用入口
    ├── pytest.ini               # 测试配置
    ├── src/
    │   ├── __init__.py
    │   ├── config.py            # 配置模块
    │   ├── api/
    │   │   ├── __init__.py
    │   │   └── routes.py        # API路由
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── training_service.py
    │   │   ├── genetic_service.py
    │   │   ├── evaluation_service.py
    │   │   └── visualization_service.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── network.py
    │   │   ├── ppo_agent.py
    │   │   └── genetic_algorithm.py
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   ├── requests.py
    │   │   └── responses.py
    │   └── core/
    │       ├── __init__.py
    │       ├── exceptions.py
    │       └── logging.py
    └── tests/
        ├── __init__.py
        ├── test_network.py
        ├── test_genetic.py
        ├── test_api.py
        └── test_visualization.py
```

### 配置参数

#### PPO参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 3e-4 | 学习率 |
| gamma | 0.99 | 折扣因子 |
| epsilon | 0.2 | PPO裁剪参数 |
| initial_temperature | 1.0 | 初始温度 |
| temperature_decay | 0.995 | 温度衰减率 |
| min_temperature | 0.01 | 最小温度 |

#### 遗传算法参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| population_size | 50 | 种群大小 |
| mutation_rate | 0.1 | 变异率 |
| crossover_rate | 0.7 | 交叉率 |
| elite_size | 5 | 精英保留数 |
| max_generations | 100 | 最大代数 |

### 许可证

MIT License
