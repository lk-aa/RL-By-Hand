# 强化学习网格世界项目

这是一个用Python实现的强化学习网格世界环境及求解算法的学习项目。该项目提供了一个可配置的网格世界环境，以及多种强化学习算法来求解最优策略，帮助学习者理解强化学习的基本概念和算法实现。

## 项目结构

项目包含以下主要文件：

- **grid_env.py**: 定义了网格世界环境，基于OpenAI Gymnasium框架构建
- **render.py**: 实现了网格世界的可视化功能
- **solver.py**: 包含了多种强化学习算法的实现

## 核心功能

### 1. 网格世界环境 (GridWorldEnv)

`GridWorldEnv`类提供了一个可配置的网格世界环境，主要特点包括：

- 支持自定义网格大小
- 可设置起点、目标点和障碍物位置
- 提供5种动作：停留、上、右、下、左
- 可配置的奖励机制
- 基于Gymnasium框架，兼容标准的强化学习接口

### 2. 可视化模块 (Render)

`Render`类负责网格世界的可视化，包括：
- 绘制网格世界地图
- 显示障碍物和目标点
- 展示智能体的移动轨迹
- 支持生成视频记录

### 3. 求解器 (Solve)

`Solve`类实现了多种强化学习算法，包括：
- 策略评估 (Policy Evaluation)
- 策略改进 (Policy Improvement)
- 策略迭代 (Policy Iteration)
- 价值迭代 (Value Iteration)

## 安装依赖

项目依赖以下Python库：

- numpy
- matplotlib
- gymnasium

## 使用示例

### 创建网格世界环境

```python
from grid_env import GridWorldEnv

# 创建一个5x5的网格世界
size = 5
start = [0, 0]  # 起点位置
target = [4, 4]  # 目标点位置
forbidden = [[1, 2], [2, 2], [3, 2]]  # 障碍物位置
render_mode = "human"  # 渲染模式

# 初始化环境
env = GridWorldEnv(size, start, target, forbidden, render_mode)
```

### 使用求解器求解最优策略

```python
from solver import Solve

# 创建求解器实例
solver = Solve(env, gamma=0.9)  # gamma为折扣因子

# 使用策略迭代算法求解最优策略
solver.policy_iteration()

# 或者使用价值迭代算法
solver.value_iteration()

# 可视化学习过程或结果
solver.show_policy()  
solver.show_state_value(solver.state_value, y_offset=0.25)
solver.env.render()
```

## 强化学习算法说明

### 策略迭代 (Policy Iteration)

策略迭代算法由两个主要步骤组成：策略评估和策略改进，交替执行直到收敛到最优策略。

### 价值迭代 (Value Iteration)

价值迭代算法直接求解最优价值函数，然后从中提取最优策略。

## 扩展与修改

1. **修改网格世界配置**：可以调整网格大小、起点、目标点和障碍物位置来创建不同的环境。

2. **调整奖励机制**：可以自定义奖励列表，分别设置普通移动、达到目标、撞到障碍物和撞到墙的奖励值。

3. **实现新算法**：可以在`solver.py`文件中扩展`Solve`类，实现更多的强化学习算法。

