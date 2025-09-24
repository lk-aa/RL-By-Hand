import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter  # 导入SummaryWriter

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from grid_env import GridWorldEnv


class Solve:
    def __init__(self, env: GridWorldEnv, gamma: int = 0.9):
        self.gamma = gamma
        self.env = env
        self.action_space_size = env.action_space_size
        self.state_space_size = env.size ** 2
        self.reward_space_size, self.reward_list = len(self.env.reward_list), self.env.reward_list
        self.state_value = np.zeros(shape=self.state_space_size)
        self.qvalue = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.mean_policy = np.ones(shape=(self.state_space_size, self.action_space_size)) / self.action_space_size
        self.policy = self.mean_policy.copy()
        # self.writer = SummaryWriter("logs")  # 实例化SummaryWriter对象

    def random_greed_policy(self):
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
        
        :return: policy - 随机贪心策略矩阵，形状为(state_space_size, action_space_size)
                其中每行对应一个状态的动作概率分布
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        for state_index in range(self.state_space_size):
            action = np.random.choice(range(self.action_space_size))
            policy[state_index, action] = 1
        return policy

    def policy_evaluation(self, policy, tolerance=0.001, steps=10):
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
        
        :param policy: 需要评估的策略 π_k，形状为(state_space_size, action_space_size)
        :param tolerance: 收敛阈值，当前后state_value的L1范数小于此值时认为已收敛
        :param steps: 最大迭代次数，若在策略迭代中使用较小的值则变为截断策略迭代（truncated policy iteration）
        :return: state_value - 收敛后的状态值函数 v_π，表示在给定策略下的期望累积奖励
        """
        state_value_current = np.ones(self.state_space_size)
        state_value_previous = np.zeros(self.state_space_size)
        while np.linalg.norm(state_value_current - state_value_previous, ord=1) > tolerance:
            state_value_previous = state_value_current.copy()
            for state in range(self.state_space_size):
                value = 0
                for action in range(self.action_space_size):
                    value += policy[state, action] * self.calculate_qvalue(state_value=state_value_current.copy(),
                                                                        state=state,
                                                                        action=action)  # bootstrapping
                state_value_current[state] = value
        return state_value_current

    def policy_improvement(self, state_value):
        """
        值迭代算法中的策略改进步骤（Policy Improvement in Value Iteration Algorithm）
        
        这是普通策略改进的变种，用于值迭代算法。该函数也可以供策略迭代算法使用，
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
        
        :param state_value: 当前策略对应的状态值函数 v_k
        :return: improved_policy (改进后的策略 π_{k+1}), 
                 state_value_k (下一步的状态值 v_{k+1} = max_a q_k(s,a))
        """
        policy = np.zeros(shape=(self.state_space_size, self.action_space_size))
        state_value_k = state_value.copy()
        for state in range(self.state_space_size):
            qvalue_list = []
            for action in range(self.action_space_size):
                qvalue_list.append(self.calculate_qvalue(state, action, state_value.copy()))
            # Maximum action value
            action_star = qvalue_list.index(max(qvalue_list))
            # Policy update - Greedy
            policy[state, action_star] = 1
            # State value update
            state_value_k[state] = max(qvalue_list)

        return policy, state_value_k

    def calculate_qvalue(self, state, action, state_value):
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
        
        参数说明：
        :param state: 当前状态 s ∈ S
        :param action: 当前动作 a ∈ A(s)
        :param state_value: 状态值函数 v_k，包含每个状态的价值估计
        
        返回值：
        :return: q_value - 状态动作对 (s, a) 的q值，表示在状态s执行动作a的期望累积奖励
        
        注意：
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

    def value_iteration(self, tolerance=0.001, steps=100):
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
        
        :param tolerance: 收敛阈值，当前后state_value的L1范数小于此值时认为已收敛
        :param steps: 最大迭代次数，建议设置较大的值以确保充分收敛
        :return: 剩余迭代次数（如为0表示达到最大迭代次数，如为正数表示提前收敛）
        
        注意：
        - 使用L1范数（ord=1）衡量状态值的变化量
        - 每次迭代都会更新self.policy和self.state_value属性
        - 算法保证单调收敛到最优解（对于折扣因子γ<1的有限MDP）
        """
        state_value_previous = np.zeros(self.state_space_size)
        self.state_value = np.ones(self.state_space_size)
        while np.linalg.norm(self.state_value - state_value_previous, ord=1) > tolerance and steps > 0:
            steps -= 1
            state_value_previous = self.state_value.copy()
            self.policy, self.state_value = self.policy_improvement(self.state_value)
        return steps

    def policy_iteration(self, tolerance=0.001, steps=100):
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
            - 收敛条件：‖v^{(j+1)} - v^{(j)}‖ < tolerance 或达到最大迭代次数
        b. 策略改进（Policy Improvement）：基于 v_{π_k} 改进策略得到 π_{k+1}
            - 计算 q-value: q_{π_k}(s,a) = Σ_r p(r|s,a)·r + γ·Σ_{s'} p(s'|s,a)·v_{π_k}(s')
            - 找到最优动作: a_k*(s) = argmax_a q_{π_k}(s,a)
            - 更新策略: π_{k+1}(a|s) = 1 if a = a_k*(s), 0 otherwise
        c. 检查策略是否收敛：‖π_{k+1} - π_k‖ < tolerance
        
        终止条件（满足任一即可）：
        - 收敛条件：前后两次迭代的策略差异的L1范数小于tolerance
        - 最大迭代次数：达到预设的steps次数
        
        :param tolerance: 收敛阈值，迭代前后policy的L1范数小于此值时认为已收敛
        :param steps: 最大迭代次数，设置较小时退化为截断策略迭代（truncated policy iteration）
        :return: 剩余迭代次数（如为0表示达到最大迭代次数，如为正数表示提前收敛）
        
        注意：
        - 算法保证每次迭代都会改进策略或保持最优（策略改进定理）
        - 对于有限MDP，算法在有限步内收敛到最优策略
        - 每次迭代更新self.policy和self.state_value属性
        """
        policy_previous = np.zeros(shape=(self.state_space_size, self.action_space_size))
        self.policy = self.random_greed_policy()
        while np.linalg.norm(self.policy - policy_previous, ord=1) > tolerance and steps > 0:
            steps -= 1
            policy_previous = self.policy.copy()
            self.state_value = self.policy_evaluation(self.policy, tolerance, steps)
            self.policy, _ = self.policy_improvement(self.state_value.copy())
        return steps

    def show_policy(self):
        for state in range(self.state_space_size):
            for action in range(self.action_space_size):
                policy = self.policy[state, action]
                self.env.render_.draw_action(
                    pos=self.env.state2pos(state),
                    toward=policy * 0.4 * self.env.action_to_direction[action],
                    radius=policy * 0.1
                )
        # self.env.render_.show_frame(t=100)

    def show_state_value(self, state_value, y_offset=0.2):
        for state in range(self.state_space_size):
            self.env.render_.write_word(
                pos=self.env.state2pos(state), 
                word=str(round(state_value[state], 1)),
                y_offset=y_offset,
                size_discount=0.7
            )
        # self.env.render_.show_frame(t=100)


if __name__ == "__main__":
    env = GridWorldEnv(
        size=5, 
        start=[0, 0],   # 在规划类算法中，起点位置无影响
        target=[2, 3],
        forbidden=[[1, 1], [2, 1], [2, 2], [1, 3], [1, 4], [3, 3]],
        render_mode='',
        reward_list=[0, 1, -10, -1]  # [普通移动，达到目标，撞到障碍物，撞墙]  [normal, target, forbidden, boundary/wall]
    )

    solver = Solve(env, gamma=0.9)
    start_time = time.time()
    solver.value_iteration()
    end_time = time.time()
    cost_time = end_time - start_time
    print("cost_time:{}".format(round(cost_time, 2)))

    solver.show_policy()  
    solver.show_state_value(solver.state_value, y_offset=0.25)

    solver.env.render()
