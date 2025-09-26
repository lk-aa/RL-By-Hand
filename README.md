# 强化学习网格世界项目

这是一个用Python实现的强化学习网格世界环境及求解算法的学习项目。该项目提供了一个可配置的网格世界环境，以及多种强化学习算法来求解最优策略，帮助学习者理解强化学习的基本概念和算法实现。

## 项目结构

项目包含以下主要文件：

- **grid_env.py**：定义了网格世界环境，基于OpenAI Gymnasium框架构建
- **render.py**：实现了网格世界的可视化功能
- **solver.py**：包含了多种强化学习算法的实现

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
- 策略迭代算法 (Policy Iteration Algorithm)
    - 策略评估 (Policy Evaluation)
    - 策略改进 (Policy Improvement)
- 价值迭代算法 (Value Iteration Algorithm)
    - 策略改进 (Policy Improvement)

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

### 算法：策略迭代算法 (Policy Iteration Algorithm)

策略迭代算法由两个主要步骤组成：策略评估和策略改进，交替执行直到收敛到最优策略。

<details>
<summary>点击查看策略迭代算法</summary>

#### 算法类型
动态规划算法 (Dynamic Programming Algorithm)，用于解决马尔可夫决策过程 (Markov Decision Process, MDP)

#### 算法目标
- **求解贝尔曼最优方程 (Bellman Optimality Equation)**
    - 找到最优状态值函数 $v^*$ (Optimal State-Value Function)
    - 找到最优策略 $\pi^*$ (Optimal Policy)
    - 解决序列决策问题中的长期累积奖励最大化问题

- **数学表达**：
  策略迭代不直接求解贝尔曼最优方程，而是通过迭代策略来逼近最优策略。每次迭代包括两个步骤：
  1. 策略评估：计算当前策略的状态值函数。
  2. 策略改进：根据当前值函数改进策略。

#### 算法原理
- **数学基础**：基于策略评估和策略改进定理。
  - **策略评估**：对于固定策略 $\pi$，通过迭代求解贝尔曼方程得到该策略的状态值函数 $v_{\pi}$。
  - **策略改进**：根据 $v_{\pi}$，通过选择每个状态下的最优动作来改进策略。
- **核心思想**：通过交替执行策略评估和策略改进，逐步提升策略的质量，直至策略不再改变。
- **收敛性保证**：由于策略改进定理，每次迭代都会产生一个严格更好的策略（除非已经最优）。由于策略数量有限，算法会在有限步内收敛。
- **策略评估步骤**：通过迭代贝尔曼期望方程来求解当前策略的值函数。
- **策略改进步骤**：利用当前值函数，对每个状态选择贪婪动作，形成新策略。

#### 输入参数
- **状态空间 (State Space)**：所有可能状态的集合 $S$
- **动作空间 (Action Space)**： 在状态 $s$ 下可用的动作集合 $A(s)$
- **状态转移概率 (State Transition Probabilities)**：从状态 $s$ 执行动作 $a$ 后转移到状态 $s'$ 的概率 $p(s'|s,a)$ 
- **奖励概率 (Reward Probabilities)**：在状态 $s$ 执行动作 $a$ 获得奖励 $r$ 的概率 $p(r|s,a)$
- **折扣因子 (Discount Factor)**：未来奖励的折扣系数 $\gamma \in [0, 1]$
- **策略评估收敛阈值 (Policy Evaluation Convergence Threshold)**：用于判断策略评估步骤中值函数收敛的标准 $\epsilon$
- **初始策略 (Initial Policy)**：策略迭代的起始策略 $\pi_0$
- **最大迭代次数** $K_{max}$（防止无限循环的保险措施）

#### 初始化阶段
- **设置迭代计数器**：表示为 $k = 0$
- **初始化策略**：表示为 $\pi_0$，可以是任意策略，通常选择随机策略或贪婪策略（如果有先验知识）
- **设置收敛标志**：表示为 converged = False

#### 算法流程

<div style="background-color：#f0f0f0; padding：10px; border-radius：5px;">

<div style="background:rgba(179, 190, 197, 0.94); padding：10px; border-radius：5px; margin：5px 0;"> 
<strong> 主迭代循环开始 </strong>
</div>

1. **策略评估 (Policy Evaluation)**：
   - 目标：计算当前策略 $\pi_k$ 的值函数 $v_{\pi_k}$
   - 初始化：设置 $v^{(0)}_{\pi_k}$ 为任意值（例如全零）
   - 迭代：使用贝尔曼期望方程进行迭代，直到值函数收敛（变化小于阈值 $\epsilon$）
     - 对于每个状态 $s \in S$：
$`
v^{(j+1)}_{\pi_k}(s) = \sum_a \pi_k(a|s) \left[ \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v^{(j)}_{\pi_k}(s') \right]
`$
   - 输出：收敛的值函数 $v_{\pi_k}$

2. **策略改进 (Policy Improvement)**：
   - 目标：根据当前值函数 $v_{\pi_k}$ 改进策略
   - 对于每个状态 $s \in S$：
     - 对于每个动作 $a \in A(s)$，计算动作值函数： $q_{\pi_k}(s,a) = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a) v_{\pi_k}(s')$
     - 选择贪婪动作： $a_k^*(s) = \arg\max_{a \in A(s)} q_{\pi_k}(s,a)$
     - 更新策略： $\pi_{k+1}(a|s) = 1$ 如果 $a = a_k^*(s)$，否则为0（确定性策略）

3. **策略收敛检查 (Policy Convergence Check)**：
   - 如果对于所有状态 $s$， $\pi_{k+1}(\cdot | s) = \pi_k(\cdot | s)$ （即策略不再改变），则设置 converged = True
   - 否则，迭代计数器递增： $k \leftarrow k + 1$

<div style="background:rgba(179, 190, 197, 0.94); padding：10px; border-radius：5px; margin：5px 0;"> 
<strong>主迭代循环结束</strong>
</div>
</div>

#### 终止与输出
- **收敛条件**：当策略不再改变（即 $\pi_{k+1} = \pi_k$）或 $k \geq K_{max}$ 时算法终止
- **输出结果**：
  - **最优值函数 (Optimal Value Function)**： $v^* = v_{\pi_k}$
  - **最优策略 (Optimal Policy)**： $\pi^* = \pi_k$
  - **实际迭代次数**： $k$
- **算法保证**：
  - 由于策略改进定理，每次迭代策略都会改进，直到达到最优策略。
  - 最终得到的策略是最优策略，值函数是最优值函数。

#### 算法复杂度分析
- **时间复杂度 (Time Complexity)**： 
  - 每次策略评估： $O(|S|^2 \times |A|)$ 每次迭代，策略评估需要多次迭代（记作 $J$），所以一次策略评估步骤为 $O(J \times |S|^2 \times |A|)$
  - 策略改进： $O(|S|^2 \times |A|)$
  - 总复杂度： $O(K \times (J \times |S|^2 \times |A| + |S|^2 \times |A|))$，其中 $K$ 是策略迭代次数， $J$ 是策略评估的迭代次数。
  - 策略迭代次数 $K$ 通常很少，因为策略会快速收敛。

- **空间复杂度 (Space Complexity)**： 
  - $O(|S| \times |A|)$ 存储转移概率和奖励函数
  - $O(|S|)$ 存储值函数
  - $O(|S| \times |A|)$ 存储策略（对于确定性策略，可以只存储每个状态的动作，即 $O(|S|)$）

- **收敛速率 (Convergence Rate)**： 
  - 策略迭代通常以线性速率收敛，但由于策略空间有限，实际迭代次数很少。

#### 关键性质
- **单调改进 (Monotonic Improvement)**：每次策略改进都会产生一个更好的策略，即 $v_{\pi_{k+1}} \geq v_{\pi_k}$（逐点成立）
- **有限收敛 (Finite Convergence)**：由于策略数量有限，算法在有限步内收敛。
- **最优性条件 (Optimality Condition)**：收敛时满足贝尔曼最优方程。

#### 优缺点分析
**优点**：
- 收敛速度快（通常比值迭代快）
- 策略通常会在值函数收敛之前就稳定下来
- 理论保证收敛到最优解

**缺点**：
- 每次迭代都需要完整的策略评估，计算成本可能高
- 对于大规模问题，策略评估步骤可能很慢
- 需要完整的环境模型

#### 应用场景
- 马尔可夫决策过程 (Markov Decision Processes, MDPs)
- 强化学习规划问题 (Reinforcement Learning Planning)
- 机器人路径规划 (Robot Path Planning)
- 资源分配优化 (Resource Allocation Optimization)
- 任何具有明确模型的序列决策问题

#### 算法伪代码

```python
算法 4.2：策略迭代算法 (Policy Iteration Algorithm)
输入：S, A, P, R, γ, ε, π₀, K_max
输出：v*, π*, k
1： k ← 0
2： π₀ ← 初始策略
3： repeat
4：     # 策略评估
5：     v ← 任意初始值函数（如全零）
6：     repeat
7：         Δ ← 0
8：         for each s ∈ S do
9：             v_old ← v(s)
10：            v_new ← 0
11：            for each a ∈ A(s) do
12：                q ← 0
13：                for each s′ ∈ S do
14：                    q ← q + P(s′|s,a) × [R(s,a,s′) + γ × v(s′)]
15：                end for
16：                v_new ← v_new + πₖ(a|s) × q
17：            end for
18：            v(s) ← v_new
19：            Δ ← max(Δ, |v_old - v_new|)
20：        end for
21：     until Δ < ε
22：     v_πₖ ← v   # 当前策略的值函数
23:
24：     # 策略改进
25：     πₖ₊₁ ← 空策略
26：     for each s ∈ S do
27：         best_a ← null
28：         max_q ← -∞
29：         for each a ∈ A(s) do
30：             q ← 0
31：             for each s′ ∈ S do
32：                 q ← q + P(s′|s,a) × [R(s,a,s′) + γ × v_πₖ(s′)]
33：             end for
34：             if q > max_q then
35：                 max_q ← q
36：                 best_a ← a
37：             end if
38：         end for
39：         πₖ₊₁(s) ← best_a  # 确定性策略，即πₖ₊₁(a|s)=1当a=best_a，否则0
40：     end for
41:
42：     # 检查策略是否稳定
43：     if πₖ₊₁ == πₖ then
44：         converged ← True
45：     else
46：         k ← k + 1
47：         πₖ ← πₖ₊₁
48：     end if
49： until converged or k ≥ K_max
50： return (v_πₖ, πₖ, k)
```

注意：在策略评估中，我们使用了迭代法求解贝尔曼期望方程。实际上，对于小型问题，也可以直接解线性方程组，但迭代法更通用。

</details>

### 算法：值迭代算法 (Value Iteration Algorithm)

价值迭代算法直接求解最优价值函数，然后从中提取最优策略。

<details>
<summary>点击查看值迭代算法</summary>

#### 算法类型
动态规划算法 (Dynamic Programming Algorithm)，用于解决马尔可夫决策过程 (Markov Decision Process, MDP)

#### 算法目标
- **求解贝尔曼最优方程 (Bellman Optimality Equation)**
    - 找到最优状态值函数 $v^*$ (Optimal State-Value Function)
    - 找到最优策略 $\pi^*$ (Optimal Policy)
    - 解决序列决策问题中的长期累积奖励最大化问题

- **数学表达**：
$`v^*(s) = \max\limits_{a \in A} \left[ \sum\limits_{r} p(r\|s,a)r + \gamma \sum\limits_{s'} p(s'\|s,a)v^*(s') \right]`$

#### 算法原理
- **数学基础**：基于贝尔曼最优方程 (Bellman Optimality Equation)：
$`v^*(s) = \max_{\pi\in\Pi} \sum_{a \in A(s)}\pi_k(a|s)\left[ \sum\limits_{r} p(r\|s,a)r + \gamma \sum\limits_{s'} p(s'\|s,a)v^*(s') \right]`$

- **核心思想**：通过迭代方式逐步改进值函数估计，直至收敛到最优值函数。

- **收敛性保证**：贝尔曼最优算子是一个压缩映射 (Contraction Mapping)，满足巴拿赫不动点定理 (Banach Fixed-Point Theorem)，确保算法必然收敛。

- **策略改进定理**：每次迭代都会产生不劣于前一次迭代的策略。

- **备份操作** (Backup Operation)：每个状态的值通过考虑所有可能动作的期望回报来更新。

- **异步收敛** (Asynchronous Convergence)：即使值函数更新顺序任意，算法仍能保证收敛。

#### 输入参数
- **状态空间 (State Space)**：所有可能状态的集合 $S$
- **动作空间 (Action Space)**：在状态 $s$ 下可用的动作集合 $A(s)$
- **状态转移概率 (State Transition Probabilities)**：从状态 $s$ 执行动作 $a$ 后转移到状态 $s'$ 的概率 $p(s'\|s,a)$
- **奖励概率 (Reward Probabilities)**：在状态 $s$ 执行动作 $a$ 获得奖励 $r$ 的概率 $p(r\|s,a)$
- **折扣因子 (Discount Factor)**：未来奖励的折扣系数 $\gamma \in [0, 1]$，$`\gamma=0`$ 表示只考虑即时奖励，$`\gamma=1`$ 表示平等对待所有未来奖励
- **收敛阈值 (Convergence Threshold)**：值函数收敛的判断标准 $\epsilon > 0$，通常取较小的正数（如 $10^{-6}$）
- **初始值函数估计 (Initial Value Function Estimate)**：对每个状态 $s \in S$ 的初始价值估计 $v_0(s)$，可以设为0或随机值
- **最大迭代次数** $K_{max}$（防止无限循环的保险措施）

#### 初始化阶段
- **设置迭代计数器**：表示为 $k = 0$
- **初始化值函数**：
  $v_0(s)$ 对所有状态 $s \in S$
  - 常见初始化方法：全零初始化、随机初始化、基于启发式的初始化
- **初始化策略**：表示为 $\pi_0$ 
  - 可以是任意策略或基于初始值函数的贪婪策略
  - 初始策略对最终结果无影响，但可能影响收敛速度
- **设置收敛标志**：表示为 converged = False

#### 算法流程
<div style="background-color：#f0f0f0; padding：10px; border-radius：5px;">

<div style="background:rgba(179, 190, 197, 0.94); padding：10px; border-radius：5px; margin：5px 0;"> 
<strong> 主迭代循环开始 </strong>
</div>

1. **收敛判断 (Convergence Check)**：
   当 $`|v_k-v_{k-1}|_\infty>\epsilon`$ 且 $`k<K_{max}`$ 时继续迭代
   - 使用无穷范数确保所有状态的值函数变化都小于阈值

2. **状态遍历 (State Iteration)**：对每个状态 $s \in S$ 执行以下操作：  
   **注释**：状态遍历顺序不影响收敛性，但可能影响收敛速度
   
   - **动作评估 (Action Evaluation)**：对每个动作 $a \in A(s)$ 计算：
     - **期望即时奖励**：
$`\mathbb{E}[r\|s,a] = \sum_r p(r\|s,a) \cdot r`$
       - 计算在当前状态执行特定动作的期望即时奖励
     - **期望未来价值**：
$`\mathbb{E}[v_k(s')\|s,a] = \sum_{s'} p(s'\|s,a) \cdot v_k(s')`$
       - 计算在当前状态执行特定动作后的期望未来累积奖励
     - **Q值计算 (Q-value Calculation)**：
$`q_k(s,a) = \mathbb{E}[r\|s,a] + \gamma \cdot \mathbb{E}[v_k(s')\|s,a]`$
       - 综合即时奖励和未来价值的全面评估
   
   - **最优动作选择 (Optimal Action Selection)**：
     - 找到使Q值最大化的动作：
$`a_k^*(s) = \arg\max_{a \in A(s)} q_k(s,a)`$
     - **平局处理策略**：如果多个动作产生相同的最大值：
       - 随机选择一个
       - 选择索引最小的动作
       - 基于额外启发式规则选择
   
   - **策略更新 (Policy Update)**：
     - 为状态 $s$ 设置确定性策略：
$`\pi_{k+1}(a\|s) = \begin{cases} 1 & \text{若 } a = a_k^*(s) \\ 0 & \text{否则} \end{cases}`$
     - 策略是确定性的，每个状态对应一个最优动作
   
   - **值函数更新 (Value Function Update)**：
     - 使用最大Q值更新状态值：
$`v_{k+1}(s) = \max_{a \in A(s)} q_k(s,a) = q_k(s, a_k^*(s))`$
     - 这相当于执行一次贝尔曼最优算子

3. **全局收敛检查 (Global Convergence Check)**：
   - 计算值函数最大变化量：
$`\Delta = \max_{s \in S} \|v_{k+1}(s) - v_k(s)\|`$
   - 如果 $\Delta<\epsilon$，则设置 converged = True
   - 迭代计数器递增： $k\leftarrow{k + 1}$

<div style="background:rgba(179, 190, 197, 0.94); padding：10px; border-radius：5px; margin：5px 0;"> 
<strong>主迭代循环结束</strong>
</div>
</div>

#### 终止与输出
- **收敛条件**：当 $\Delta < \epsilon$ 或 $k \geq K_{max}$ 时算法终止
- **输出结果**：
  - **最优值函数 (Optimal Value Function)**： $v^* = v_k$
  - **最优策略 (Optimal Policy)**： $\pi^* = \pi_k$
  - **实际迭代次数**： $k$
- **算法保证**：
  - $v^*$ 满足贝尔曼最优方程
  - $\pi^*$ 是相对于初始状态分布的最优策略
  - 对于充分小的 $\epsilon$，得到的策略是 $\epsilon$-最优的
  - 误差界限： 满足 $|v_k - v^*|_\infty \leq \frac{\gamma^k}{1-\gamma} |v_1 - v_0|_\infty$
- **验证方法**：可以通过策略评估验证所得策略的性能

#### 算法复杂度分析
- **时间复杂度 (Time Complexity)**： 
  - 每次迭代： $O(\|S\|^2 \times \|A\|)$
  - 总复杂度： $O(K \times \|S\|^2 \times \|A\|)$，其中 $K$ 是迭代次数
  - 迭代次数 $K$ 取决于 $\gamma$ 和 $\epsilon$，通常为 $O\left(\frac{\log(1/\epsilon)}{1-\gamma}\right)$
  - 迭代次数上界： $K = \left\lceil \frac{\log(\epsilon(1-\gamma)) - \log(\|v_1 - v_0\|_\infty)}{\log(\gamma)} \right\rceil$

- **空间复杂度 (Space Complexity)**： 
  - $O(\|S\| \times \|A\|)$ 存储转移概率和奖励函数
  - $O(\|S\|)$ 存储值函数
  - $O(\|S\|)$ 存储策略

- **收敛速率 (Convergence Rate)**： 
  - 线性收敛： 满足 $|v_{k+1} - v^*|_\infty \leq \gamma |v_k - v^*|_\infty$
  - 误差界限： 满足 $|v_k - v^*|_\infty \leq \frac{\gamma^k}{1-\gamma} |v_1 - v_0|_\infty$

#### 关键性质
- **单调改进 (Monotonic Improvement)**： $v_{k+1}(s) \geq v_k(s)$ 对所有 $s \in S$
- **压缩映射 (Contraction Mapping)**：贝尔曼最优算子是模为 $\gamma$ 的压缩映射
- **最优性条件 (Optimality Condition)**：收敛时满足贝尔曼最优方程
- **策略收敛 (Policy Convergence)**：最优策略可能在值函数收敛之前就已稳定
- **异步收敛 (Asynchronous Convergence)**：支持异步更新，但同步更新保证收敛
- **无需策略评估**：与策略迭代不同，值迭代不需要完整的策略评估步骤

#### 优缺点分析
**优点**：
- 理论保证收敛到最优解
- 适用于各种MDP问题
- 算法简单直观，易于实现
- 内存效率高：相比策略迭代，通常需要更少的内存

**缺点**：
- 对于大规模状态空间，计算成本高（维度灾难 Curse of Dimensionality）
- 需要完整的环境模型（转移概率和奖励函数）
- 收敛速度可能较慢，特别是当 $\gamma$ 接近1时
- 同步更新：基本版本需要扫描所有状态，可能效率不高

#### 应用场景
- 马尔可夫决策过程 (Markov Decision Processes, MDPs)
- 强化学习规划问题 (Reinforcement Learning Planning)
- 机器人路径规划 (Robot Path Planning)
- 资源分配优化 (Resource Allocation Optimization)
- 任何具有明确模型的序列决策问题

#### 算法伪代码

```python
算法 4.1：值迭代算法 (Value Iteration Algorithm)
输入：S, A, P, R, γ, ε, v₀, K_max
输出：v*, π*, k
1： k ← 0
2： for each s ∈ S do v₀(s) ← 初始值
3： π₀ ← 任意初始策略
4： repeat
5：     Δ ← 0
6：     for each s ∈ S do
7：         v_old ← vₖ(s)
8：         max_q ← -∞
9：         best_a ← null
10：        for each a ∈ A(s) do
11：            q ← 0
12：            for each s′ ∈ S do
13：                q ← q + P(s′|s,a) × [R(s,a,s′) + γ × vₖ(s′)]
14：            end for
15：            if q > max_q then
16：                max_q ← q
17：                best_a ← a
18：            end if
19：        end for
20：        vₖ₊₁(s) ← max_q
21：        πₖ₊₁(s) ← best_a  # 确定性策略
22：        Δ ← max(Δ, |vₖ₊₁(s) - v_old|)
23：    end for
24：    k ← k + 1
25： until Δ < ε or k ≥ K_max
26： return (vₖ, πₖ, k)
```

</details>

## 扩展与修改

1. **修改网格世界配置**：可以调整网格大小、起点、目标点和障碍物位置来创建不同的环境。

2. **调整奖励机制**：可以自定义奖励列表，分别设置普通移动、达到目标、撞到障碍物和撞到墙的奖励值。

3. **实现新算法**：可以在`solver.py`文件中扩展`Solve`类，实现更多的强化学习算法。
