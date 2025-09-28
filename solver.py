import random
import time
from typing import Union, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

import sys
import pathlib
from typing import Tuple, Optional, Union

sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from grid_env import GridWorldEnv


class Solver:
    """
    强化学习算法求解器类
    
    提供多种强化学习算法实现，用于求解网格世界环境中的最优策略，包括：
        - 值迭代算法（Value Iteration）
        - 策略迭代算法（Policy Iteration）
        - 策略评估算法（Policy Evaluation）
        - 策略改进算法（Policy Improvement）
    """
    def __init__(self, env: GridWorldEnv, gamma: float = 0.9):
        """
        Solver类的构造函数，用于初始化求解器参数

        Args:
            env: 网格世界环境对象，包含状态空间、动作空间等信息
            gamma: 折扣因子，默认为0.9，用于权衡即时奖励和未来奖励的重要性
        """
        # 初始化环境相关参数
        self.gamma = gamma
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        
        # 初始化值函数和策略
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        # self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

    def random_greedy_policy(self) -> np.ndarray:
        """
        生成随机贪心策略（Random Greedy Policy）
        
        为每个状态随机选择一个动作，构建确定性策略（deterministic policy）。
        对于每个状态 s ∈ S，随机选择一个动作 a ∈ A(s)，并将该动作的概率设为1，
        其他动作的概率设为0。
        
        策略形式：
        π(a|s) = 1 如果 a 是随机选中的动作，否则 π(a|s) = 0
        
        用途：
            - 策略迭代算法的初始策略
            - 值迭代算法中的策略初始化
            - 其他需要随机确定性策略的场景
        
        特点：
            - 确定性策略：每个状态只对应一个确定的动作
            - 随机性：初始动作选择是随机的，增加探索性
            - 有效性：满足概率分布约束 Σ_a π(a|s) = 1
        
        Returns: 
            policy: 随机贪心策略矩阵，形状为(state_space_size, action_space_size)
                    其中每行对应一个状态的动作概率分布
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            # 对于每个状态，随机选择一个动作并将其概率设为1
            action = np.random.choice(range(self.action_space_size))
            policy[state_index, action] = 1
        return policy

    def policy_evaluation(
            self, 
            policy: np.ndarray, 
            tolerance: float = 0.001, 
            max_iterations: int = 1000
        ) -> np.ndarray:
        """
        策略评估算法（Policy Evaluation Algorithm）
        
        通过迭代求解贝尔曼期望方程（Bellman Expectation Equation），计算给定策略下的状态值函数。
        
        算法流程（对应伪代码中的策略评估部分）：
            1. 初始化：任意初始猜测 v_{π_k}^{(0)}（这里初始化为全1向量）
            2. 当未收敛且未达到最大迭代次数时循环：
                a. 检查收敛条件：‖v^{(j+1)} - v^{(j)}‖ > tolerance
                b. 对于每个状态 s ∈ S，计算：
                    v_{π_k}^{(j+1)}(s) = Σ_a π_k(a|s) [Σ_r p(r|s,a)·r + γ·Σ_{s'} p(s'|s,a)·v_{π_k}^{(j)}(s')]
                c. 更新状态值函数
        
        具体实现：
            - 使用迭代策略评估（Iterative Policy Evaluation）方法
            - 对于每个状态，计算所有动作的期望值，按策略概率加权平均
            - 每个状态值的更新都基于当前的状态值估计（bootstrapping）
        
        终止条件（满足任一即可）：
            - 收敛条件：前后两次迭代的状态值差异的L1范数小于tolerance
            - 最大迭代次数：达到预设的steps次数
        
        Args:
            policy: 需要评估的策略 π_k，形状为(state_space_size, action_space_size)
            tolerance: 收敛阈值，当前后state_value的L1范数小于此值时认为已收敛
            max_iterations: 最大迭代次数，若在策略迭代中使用较小的值则变为截断策略迭代（truncated policy iteration）

        Returns:
            state_value - 收敛后的状态值函数 v_π，表示在给定策略下的期望累积奖励
        """
        # 初始化状态值函数
        state_value_current = np.ones(self.state_space_size)
        state_value_previous = np.zeros(self.state_space_size)
        
        # 迭代直到收敛或达到最大迭代次数
        iterations = 0
        while np.linalg.norm(state_value_current - state_value_previous, ord=1) > tolerance and iterations < max_iterations:
            state_value_previous = state_value_current.copy()
            
            for state in range(self.state_space_size):
                value = 0
                # 计算该状态下所有动作的期望价值加权和
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qvalue(
                        state=state, 
                        action=action, 
                        state_value=state_value_current.copy()
                    )  # bootstrapping
                state_value_current[state] = value
            
            iterations += 1
        
        return state_value_current

    def policy_improvement(
            self, 
            state_value: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略迭代算法中的策略改进算法（Policy Improvement in Policy Iteration Algorithm）

        基于当前状态值函数，生成一个更好的贪心策略，并更新状态值函数。
        
        这是普通策略改进的变种，用于值迭代算法，也可以供策略迭代算法使用，
        但在策略迭代中通常不需要接收第二个返回值（state_value_k）。
        
        算法流程（对应伪代码中的迭代步骤）：
            1. 对于每个状态 s ∈ S，遍历所有可能的动作 a ∈ A(s)
            2. 计算 q-value: q_k(s, a) = Σ p(r|s,a)·r + γ·Σ p(s'|s,a)·v_k(s')
            3. 找到最优动作 a_k*(s) = argmax_a q_k(s, a)
            4. 更新策略：π_{k+1}(a|s) = 1 if a = a_k*(s), 0 otherwise
            5. 更新状态值：v_{k+1}(s) = max_a q_k(s, a)
        
        具体实现：
            - 更新 q-value：qvalue[state,action] = 期望奖励 + 折扣后的下一状态值
            - 找到每个状态的最优动作 action* = argmax(qvalue[state,action])
            - 更新策略：将 action* 的概率设为 1，其他动作的概率设为 0（贪心策略）
        
        Args:
            state_value: 当前策略对应的状态值函数 v_k
        
        Returns:
            improved_policy (np.ndarray): 改进后的策略 π_{k+1}，形状为(state_space_size, action_space_size)
            state_value_k (np.ndarray): 下一步的状态值 v_{k+1} = max_a q_k(s,a)，形状为(state_space_size,)
        """
        # 初始化新策略
        improved_policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        state_value_k = state_value.copy()
        
        for state in range(self.state_space_size):
            # 计算该状态下所有动作的Q值
            qvalue_list = []
            for action in range(self.action_space_size):
                qvalue_list.append(self.calculate_qvalue(state, action, state_value.copy()))
            
            # 找到最大Q值对应的动作索引
            action_star = qvalue_list.index(max(qvalue_list))
            
            # 更新策略：贪婪策略，将最大Q值动作的概率设为1
            improved_policy[state, action_star] = 1
            
            # 更新状态值函数
            state_value_k[state] = max(qvalue_list)
        
        return improved_policy, state_value_k

    def calculate_qvalue(
            self, 
            state: int, 
            action: int, 
            state_value: np.ndarray
        ) -> float:
        """
        计算状态-动作值函数 q(s, a)（Q-value Calculation）
        
        根据贝尔曼方程计算给定状态和动作的q值：
            q_k(s, a) = Σ_r p(r|s,a)·r + γ·Σ_{s'} p(s'|s,a)·v_k(s')
        
        算法流程：
            1. 奖励期望部分：遍历所有可能的奖励值，计算期望即时奖励
                Σ p(r|s,a)·r = Σ_{i=0}^{reward_space_size-1} reward_list[i] * Rsa[state,action,i]
            2. 状态转移部分：遍历所有可能的下一状态，计算折扣后的期望未来奖励
                γ·Σ p(s'|s,a)·v_k(s') = γ·Σ_{next_state} Psa[state,action,next_state] * state_value[next_state]
            3. 将两部分相加得到完整的q值
        
        Args:
            state (int): 当前状态 s ∈ S
            action (int): 当前动作 a ∈ A(s)
            state_value (np.ndarray): 状态值函数 v_k，包含每个状态的价值估计
        
        Returns:
            q_value (float) - 状态动作对 (s, a) 的q值，表示在状态s执行动作a的期望累积奖励
        
        Notes:
            - self.env.Rsa[state, action, i] 表示在状态s执行动作a获得奖励reward_list[i]的概率 p(r_i|s,a)
            - self.env.Psa[state, action, next_state] 表示状态转移概率 p(s'|s,a)
            - self.gamma 是折扣因子 γ ∈ [0,1]
            - self.reward_list 包含所有可能的奖励值
        """
        qvalue = 0
        
        # 计算期望即时奖励：Σ p(r|s,a)·r
        for i in range(self.reward_space_size):
            qvalue += self.reward_list[i] * self.env.Rsa[state, action, i]
        
        # 计算折扣后的期望未来奖励：γ·Σ p(s'|s,a)·v_k(s')
        for next_state in range(self.state_space_size):
            qvalue += self.gamma * self.env.Psa[state, action, next_state] * state_value[next_state]
        
        return qvalue

    def value_iteration(
            self, 
            tolerance: float = 0.001, 
            max_iterations: int = 1000
        ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        值迭代算法（Value Iteration Algorithm）
        
        通过迭代求解贝尔曼最优方程（Bellman Optimality Equation），找到最优状态值函数和最优策略。
        
        算法流程（对应值迭代伪代码）：
            1. 初始化：state_value_k ← 初始状态值（这里初始化为全1向量）
            2. 当未收敛且未达到最大迭代次数时循环：
                a. 检查收敛条件：‖v_k - v_{k-1}‖ > tolerance
                b. 递减迭代计数器
                c. 更新当前状态值：v_{k-1} ← v_k
                d. 执行策略改进：π_{k+1}, v_{k+1} = policy_improvement(v_k)
                    - 计算q-value: q_k(s,a) = Σ p(r|s,a)·r + γ·Σ p(s'|s,a)·v_k(s')
                    - 找到最优动作: a_k*(s) = argmax_a q_k(s,a)
                    - 更新策略: π_{k+1}(a|s) = 1 if a = a_k*(s), 0 otherwise
                    - 更新状态值: v_{k+1}(s) = max_a q_k(s,a)
        
        终止条件（满足任一即可）：
            - 收敛条件：前后两次迭代的状态值差异的L1范数小于tolerance
            - 最大迭代次数：达到预设的steps次数
        
        Args:
            tolerance (float, optional): 收敛阈值，当前后state_value的L1范数小于此值时认为已收敛。默认值为0.001。
            max_iterations (int, optional): 最大迭代次数，建议设置较大的值以确保充分收敛。默认值为1000。
        
        Returns:
            policy (np.ndarray): 最优策略 π*，形状为 (state_space_size, action_space_size)
            state_value (np.ndarray): 最优状态值函数 v*，形状为 (state_space_size,)
            remaining_iterations (int): 剩余迭代次数（如为0表示达到最大迭代次数，如为正数表示提前收敛）
        
        Notes:
            - 使用L1范数（ord=1）衡量状态值的变化量
            - 每次迭代都会更新self.policy和self.state_value属性
            - 算法保证单调收敛到最优解（对于折扣因子γ<1的有限MDP）
        """
        state_value_previous = np.zeros(self.state_space_size)
        self.state_value = np.ones(self.state_space_size)
        
        # 迭代直到收敛或达到最大迭代次数
        remaining_iterations = max_iterations
        while np.linalg.norm(self.state_value - state_value_previous, ord=1) > tolerance and remaining_iterations > 0:
            remaining_iterations -= 1
            state_value_previous = self.state_value.copy()
            
            # 执行策略改进步骤，同时更新策略和状态值函数
            self.policy, self.state_value = self.policy_improvement(self.state_value)
        
        return self.policy, self.state_value, remaining_iterations

    def policy_iteration(
            self, 
            tolerance: float = 0.001, 
            max_iterations: int = 100
        ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        策略迭代算法（Policy Iteration Algorithm）
        
        通过交替执行策略评估和策略改进，逐步优化策略直至收敛到最优策略。
        
        算法流程（对应伪代码中的策略迭代）：
            1. 初始化：任意初始策略 π_0（这里使用随机贪心策略）
            2. 当策略未收敛且未达到最大迭代次数时循环：
                a. 策略评估（Policy Evaluation）：计算当前策略 π_k 的状态值函数 v_{π_k}
                    - 初始化：任意初始猜测 v_{π_k}^{(0)}
                    - 迭代更新：对于每个状态 s ∈ S，计算
                        v_{π_k}^{(j+1)}(s) = Σ_a π_k(a|s) [Σ_r p(r|s,a)·r + γ·Σ_{s'} p(s'|s,a)·v_{π_k}^{(j)}(s')]
                    - 收敛条件：‖v_{π_k}^{(j+1)} - v_{π_k}^{(j)}‖ < tolerance 或达到最大迭代次数
                b. 策略改进（Policy Improvement）：基于 v_{π_k} 改进策略得到 π_{k+1}
                    - 计算 q-value: q_{π_k}(s,a) = Σ_r p(r|s,a)·r + γ·Σ_{s'} p(s'|s,a)·v_{π_k}(s')
                    - 找到最优动作: a_k*(s) = argmax_a q_{π_k}(s,a)
                    - 更新策略: π_{k+1}(a|s) = 1 if a = a_k*(s), 0 otherwise
                c. 检查策略是否收敛：‖π_{k+1} - π_k‖ < tolerance
                    - 如果收敛，返回当前迭代次数
                    - 如果未收敛，继续下一次迭代
        
        终止条件（满足任一即可）：
            - 收敛条件：前后两次迭代的策略差异的L1范数小于tolerance
            - 最大迭代次数：达到预设的steps次数
        
        Args:
            tolerance (float, optional): 收敛阈值，迭代前后policy的L1范数小于此值时认为已收敛。默认值为0.001。
            max_iterations (int, optional): 最大迭代次数，设置较小时退化为截断策略迭代（truncated policy iteration）。默认值为100。
        
        Returns:
            policy (np.ndarray): 最优策略 π*，形状为 (state_space_size, action_space_size)
            state_value (np.ndarray): 最优状态值函数 v*，形状为 (state_space_size,)
            remaining_iterations (int): 剩余迭代次数（如为0表示达到最大迭代次数，如为正数表示提前收敛）
        
        Notes:
            - 算法保证每次迭代都会改进策略或保持最优（策略改进定理）
            - 对于有限MDP，算法在有限步内收敛到最优策略
            - 每次迭代更新self.policy和self.state_value属性
        """
        policy_previous = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.policy = self.random_greedy_policy()
        
        # 迭代直到收敛或达到最大迭代次数
        remaining_iterations = max_iterations
        while np.linalg.norm(self.policy - policy_previous, ord=1) > tolerance and remaining_iterations > 0:
            remaining_iterations -= 1
            policy_previous = self.policy.copy()
            
            # 执行策略评估：计算当前策略下的状态值函数
            self.state_value = self.policy_evaluation(self.policy, tolerance, max_iterations)
            
            # 执行策略改进：基于状态值函数生成更好的策略
            self.policy, _ = self.policy_improvement(self.state_value.copy())
        
        return self.policy, self.state_value, remaining_iterations

    def show_policy(
            self, 
            policy: Optional[np.ndarray] = None,
            render_mode: str = 'show'
        ) -> None:
        """
        可视化当前策略
        
        Args:
            policy: 要可视化的策略，如果为None则使用当前策略
            render_mode: 渲染模式，'show'表示直接显示，'save'表示保存为图片
        """
        self.env.render_.visualize_policy(
            policy=policy if policy is not None else self.policy,
            action_to_direction=self.env.action_to_direction
        )
        
        if render_mode == 'show':
            self.env.render_.show_frame(t=10, close_after=False)

    def show_state_value(
            self, 
            state_value: Optional[np.ndarray] = None,
            y_offset: float = 0.2, 
            render_mode: str = 'show'
        ) -> None:
        """
        可视化状态值函数
        
        Args:
            state_value: 需要可视化的状态值函数，如果为None则使用当前状态值函数
            y_offset: 文本在y方向上的偏移量
            render_mode: 渲染模式，'show'表示直接显示，'save'表示保存为图片
        """
        self.env.render_.visualize_state_values(
            state_values=state_value if state_value is not None else self.state_value,
            y_offset=y_offset
        )
        
        if render_mode == 'show':
            self.env.render_.show_frame(t=10, close_after=False)

    def obtain_episode(
            self, 
            policy: np.ndarray = None, 
            start_state: int = None,
            start_action: int = None,
            length: int = 1000
        ) -> Tuple[float, List[int], List[int]]:
        """
        运行一个episode，根据给定策略执行动作并收集轨迹
        
        Args:
            policy: 要执行的策略，如果为None则使用当前策略
            start_state: 初始状态，如果为None则随机选择
            start_action: 初始动作，如果为None则随机选择
            length: 轨迹长度
            
        Returns:
            episode: 一个episode的轨迹，每个元素为一个字典，包含状态、动作、奖励、下一个状态和下一个动作
        """
        # 重置环境
        observation, _ = self.env.reset({'start': start_state})
        # self.env.agent_location = self.env.state2pos(start_state)
        episode = []
        next_action = start_action
        next_state = start_state
        while length > 0:
            length -= 1
            state = next_state
            action = next_action
            observation, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.env.pos2state(self.env.agent_location)
            next_action = np.random.choice(
                np.arange(len(policy[next_state])),
                p=policy[next_state]
            )
            episode.append(
                {
                    "state": state, 
                    "action": action, 
                    "reward": reward, 
                    "next_state": next_state,
                    "next_action": next_action
                }
            )
        return episode

    def mc_basic(
            self, 
            length: int = 30, 
            epochs: int = 10
        ):
        """
        蒙特卡洛基本算法，用于估计状态-动作值函数
        每次迭代时，对每个 state-action 对进行采样，计算从该 state-action 对开始的轨迹的总奖励，
        并将该奖励作为该 state-action 对的估计值。
        最后，根据每个 state-action 对的估计值，选择最优动作。

        Args:
            length: 每一个 state-action 对的长度
            epochs: 迭代次数

        Returns:
            policy: 最优策略
        """
        self.policy = self.random_greedy_policy()
        for epoch in range(epochs):
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(self.policy, state, action, length)
                    g = 0
                    for step in range(len(episode) - 1, -1, -1):
                        g = episode[step]['reward'] + self.gama * g
                    self.qvalue[state][action] = g
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                self.policy[state] = np.zeros(shape=self.action_space_size)
                self.policy[state, action_star] = 1
            print(epoch)
        return self.policy

    def mc_exploring_starts(
            self, 
            length: int = 30,
            visit_type: str = 'first'   
        ):
        """
        蒙特卡洛探索开始算法，用于估计状态-动作值函数
        每次迭代时，对每个 state-action 对进行采样，计算从该 state-action 对开始的轨迹的总奖励，
        并将该奖励作为该 state-action 对的估计值。
        最后，根据每个 state-action 对的估计值，选择最优动作。

        Args:
            length: 每一个 state-action 对的长度
            visit_type: 访问类型，'first'表示仅计算第一次访问，'every'表示计算每一次访问

        Returns:
            policy: 最优策略
        """
        self.policy = self.random_greedy_policy()
        policy = self.mean_policy.copy()
        qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        # returns = [[[0 for row in range(1)] for col in range(5)] for block in range(25)]
        returns = np.zeros(shape=(self.state_space_size, self.action_space_size))
        nums = np.zeros(shape=(self.state_space_size, self.action_space_size))
        while np.linalg.norm(policy - self.policy, ord=1) > 0.001:
            policy = self.policy.copy()
            for state in range(self.state_space_size):
                for action in range(self.action_space_size):
                    episode = self.obtain_episode(
                        policy=self.policy, 
                        start_state=state, 
                        start_action=action,
                        length=length
                    )
                    visit_list = []
                    g = 0
                    for step in range(len(episode) - 1, -1, -1):
                        reward = episode[step]['reward']
                        state = episode[step]['state']
                        action = episode[step]['action']
                        g = self.gama * g + reward
                        # first visit
                        if visit_type == 'first' and [state, action] in visit_list:
                            continue
                        
                        visit_list.append([state, action])
                        nums[state, action] += 1
                        returns[state, action] += g
                        qvalue[state, action] = returns[state, action] / nums[state, action]
                        qvalue_star = qvalue[state].max()
                        action_star = qvalue[state].tolist().index(qvalue_star)
                        self.policy[state] = np.zeros(shape=self.action_space_size).copy()
                        self.policy[state, action_star] = 1
            print(np.linalg.norm(policy - self.policy, ord=1))

    def mc_epsilon_greedy(
            self, 
            length: int = 1000, 
            epsilon: float = 0.1, 
            tolerance: float = 1
        ):
        norm_list = []

        qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.policy = self.random_greedy_policy()
        returns = np.zeros(shape=(self.state_space_size, self.action_space_size))
        nums = np.zeros(shape=(self.state_space_size, self.action_space_size))
        while True:
            if epsilon >= 0.01:
                epsilon -= 0.01
                print(epsilon)
            length = 20 + epsilon * length
            if len(norm_list) >= 3:
                if norm_list[-1] < tolerance and norm_list[-2] < tolerance and norm_list[-3] < tolerance:
                    break
            qvalue = self.qvalue.copy()
            state = random.choice(range(self.state_space_size))
            action = random.choice(range(self.action_space_size))
            episode = self.obtain_episode(
                policy=self.policy, 
                start_state=state, 
                start_action=action,
                length=length
            )
            g = 0
            for step in range(len(episode) - 1, -1, -1):
                reward = episode[step]['reward']
                state = episode[step]['state']
                action = episode[step]['action']
                g = self.gama * g + reward
                # every visit
                nums[state, action] += 1
                returns[state, action] += g
                self.qvalue[state, action] = returns[state, action] / nums[state, action]
                qvalue_star = self.qvalue[state].max()
                action_star = self.qvalue[state].tolist().index(qvalue_star)
                for a in range(self.action_space_size):
                    if a == action_star:
                        self.policy[state, a] = 1 - (
                                self.action_space_size - 1) / self.action_space_size * epsilon
                    else:
                        self.policy[state, a] = 1 / self.action_space_size * epsilon
            print(np.linalg.norm(self.qvalue - qvalue, ord=1))
            norm_list.append(np.linalg.norm(self.qvalue - qvalue, ord=1))

    def evaluate_policy(
            self, 
            policy: Optional[np.ndarray] = None, 
            num_episodes: int = 100
        ) -> float:
        """
        通过运行多个episode评估策略性能
        
        Args:
            policy: 要评估的策略，如果为None则使用当前策略
            num_episodes: 要运行的episode数量
            
        Returns:
            average_reward: 平均总奖励
        """
        total_rewards = 0
        for _ in range(num_episodes):
            reward, _, _ = self.run_episode(policy)
            total_rewards += reward
        
        return total_rewards / num_episodes

    def compare_algorithms(
            self, 
            num_runs: int = 10
        ) -> None:
        """
        比较值迭代和策略迭代算法的性能
        
        Args:
            num_runs: 每种算法运行的次数
        """
        value_iteration_times = []
        policy_iteration_times = []
        
        for _ in range(num_runs):
            # 测试值迭代算法
            solver_vi = Solver(self.env, gamma=self.gamma)
            start_time = time.time()
            solver_vi.value_iteration()
            end_time = time.time()
            value_iteration_times.append(end_time - start_time)
            
            # 测试策略迭代算法
            solver_pi = Solver(self.env, gamma=self.gamma)
            start_time = time.time()
            solver_pi.policy_iteration()
            end_time = time.time()
            policy_iteration_times.append(end_time - start_time)
        
        # 打印比较结果
        print(f"值迭代算法平均运行时间: {np.mean(value_iteration_times):.4f} 秒")
        print(f"策略迭代算法平均运行时间: {np.mean(policy_iteration_times):.4f} 秒")
        
        # 可视化比较结果
        plt.figure(figsize=(10, 6))
        plt.bar(['值迭代', '策略迭代'], [np.mean(value_iteration_times), np.mean(policy_iteration_times)])
        plt.ylabel('平均运行时间 (秒)')
        plt.title('算法性能比较')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('algorithm_comparison.png')
        plt.close()


if __name__ == "__main__":
    # 创建网格世界环境
    env = GridWorldEnv(
        size=5, 
        start=[0, 0],   # 在规划类算法中，起点位置无影响
        target=[2, 3],
        forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [1, 4], [3, 3]],
        render_mode='',
        reward_list=[0, 1, -10, -1]  # [普通移动，达到目标，撞到障碍物，撞墙]  [normal, target, forbidden, boundary/wall]
    )

    # 创建求解器并运行值迭代算法
    solver = Solver(env, gamma=0.9)
    start_time = time.time()
    policy, state_value, remaining_iterations = solver.value_iteration()
    # policy, state_value, remaining_iterations = solver.policy_iteration()

    end_time = time.time()
    cost_time = end_time - start_time
    print(f"值迭代算法耗时: {round(cost_time, 4)} 秒, 剩余迭代次数: {remaining_iterations}")
    print(policy)
    print(state_value)

    # 可视化策略和状态值
    solver.show_policy(policy=solver.policy, render_mode='show')
    solver.show_state_value(state_value=solver.state_value, y_offset=0.2, render_mode='show')

    # 显示最终结果
    # env.render()
