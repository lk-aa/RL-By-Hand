import time
from typing import Optional, Union, List, Tuple
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType
np.random.seed(1)
import render


def arr_in_list(array, _list):
    """
    检查一个numpy数组是否在另一个列表中
    
    Args:
        array: 要检查的numpy数组
        _list: 包含numpy数组的列表
        
    Returns:
        bool: 如果数组在列表中则返回True，否则返回False
    """
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


class GridWorldEnv(gym.Env):
    """
    强化学习网格世界环境实现，基于gymnasium接口
    
    该环境创建一个正方形网格，包含一个智能体、一个目标点和若干障碍物。
    智能体可以在网格中移动，到达目标点获得奖励，撞到障碍物或墙则受到惩罚。
    """
    def __init__(
            self, 
            size: int, 
            start: Union[list, tuple, np.ndarray],
            target: Union[list, tuple, np.ndarray], 
            forbidden: Union[list, tuple, np.ndarray],
            render_mode: Optional[str] = None,
            reward_list: Optional[List[float]] = None,
            max_steps: int = 100000
        ):
        """
        GridWorldEnv 的构造函数
        
        Args:
            size: 网格世界的边长，必须为正整数
            start: 智能体起始位置，格式为[x, y]
            target: 目标点位置，格式为[x, y]
            forbidden: 不可通行区域列表，格式为[[x1,y1], [x2,y2], ...]
            render_mode: 渲染模式，'video'表示保存视频，None表示不渲染
            reward_list: 奖励列表，分别表示[普通移动奖励, 到达目标奖励, 撞到障碍物惩罚, 撞到墙惩罚]
            max_steps: 每个回合的最大步数限制，超过后会终止回合
        """
        # 参数验证
        if size <= 0 or not isinstance(size, int):
            raise ValueError("网格大小必须为正整数")
            
        # 确保坐标在网格范围内
        def validate_position(pos, name):
            if (pos[0] < 0 or pos[0] >= size or pos[1] < 0 or pos[1] >= size):
                raise ValueError(f"{name}位置必须在网格范围内(0-{size-1})")
            return np.array(pos, dtype=int)
        
        # 初始化变量
        self.time_steps = 0
        self.size = size  # 网格世界的边长
        self.render_mode = render_mode  # 渲染模式
        self.max_steps = max_steps  # 最大步数限制

        # 验证并初始化起点、目标点和障碍物位置
        self.agent_location = validate_position(start, "起始点")
        self.target_location = validate_position(target, "目标点")
        
        # 验证并初始化障碍物位置
        self.forbidden_location = []
        for fob in forbidden:
            fob_pos = validate_position(fob, "障碍物")
            # 确保障碍物不在起点或目标点上
            if np.array_equal(fob_pos, self.agent_location) or np.array_equal(fob_pos, self.target_location):
                raise ValueError("障碍物不能位于起点或目标点上")
            self.forbidden_location.append(fob_pos)

        # 初始化渲染器
        self.render_ = render.Render(target=target, forbidden=forbidden, size=size)

        # 初始化动作空间：5个离散动作（停留、上、右、下、左）
        self.action_space = gym.spaces.Discrete(5)
        self.action_space_size = self.action_space.n

        # 初始化观测空间：字典类型，包含智能体、目标点和障碍物位置信息
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),    # [x, y]坐标
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y]坐标
                "barrier": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y]坐标
            }
        )

        # 奖励配置，分别表示：普通移动、到达目标、撞到障碍物、撞到墙
        self.reward_list = reward_list if reward_list is not None else [0, 1, -10, -10]

        # 动作到位置偏移量的映射，使代码更具可读性
        self.action_to_direction = {
            0: np.array([0, 0]),     # 停留不动
            1: np.array([0, 1]),     # 向上移动（y轴正方向）
            2: np.array([1, 0]),     # 向右移动（x轴正方向）
            3: np.array([0, -1]),    # 向下移动（y轴负方向）
            4: np.array([-1, 0]),    # 向左移动（x轴负方向）
        }

        # 状态转移概率矩阵P(s'|s,a)和奖励概率矩阵R(s,a)
        self.Rsa = None  # 状态-动作对的奖励概率
        self.Psa = None  # 状态-动作对的转移概率

        # 初始化状态转移和奖励概率矩阵
        self.psa_rsa_init()

    def reset(
            self, 
            seed: Optional[int] = None, 
            options: Optional[dict] = None
        ) -> Tuple[ObsType, dict]:
        """
        重置环境到初始状态，开始新的回合
        
        Args:
            seed: 随机种子，用于可重复的实验
            options: 额外配置选项，可以指定新的起点位置 {"start": [x, y]}
        
        Returns:
            tuple: (observation, info) 初始观测和信息
        """
        # 重要：必须首先调用父类的reset方法来正确设置随机数生成器
        super().reset(seed=seed)
        
        # 重置智能体位置，如果提供了新起点则使用新起点
        if options is not None and "start" in options:
            # 验证新起点的有效性
            start_pos = np.array(options['start'])
            if (start_pos[0] < 0 or start_pos[0] >= self.size or 
                start_pos[1] < 0 or start_pos[1] >= self.size):
                raise ValueError(f"新起点必须在网格范围内(0-{self.size-1})")
            self.agent_location = start_pos
        else:
            # 使用默认起点
            self.agent_location = np.array([0, 0])
            
        # 重置步数计数
        self.time_steps = 0
        
        # 获取当前观测和信息
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        在环境中执行一步操作
        
        Args:
            action: 要执行的动作（0-4分别对应停留、上、右、下、左）
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: 新的观测状态
                - reward: 执行动作获得的奖励
                - terminated: 回合是否终止（到达目标或失败）
                - truncated: 回合是否被截断（达到最大步数）
                - info: 额外信息
        """
        # 检查动作的有效性
        if action < 0 or action >= self.action_space_size:
            raise ValueError(f"动作必须在0-{self.action_space_size-1}范围内")
            
        # 获取当前状态和执行动作后的奖励
        current_state = self.pos2state(self.agent_location)
        reward_index = self.Rsa[current_state, action].tolist().index(1)
        reward = self.reward_list[reward_index]

        # 将离散动作映射到移动方向
        direction = self.action_to_direction[action]

        # 更新渲染器中的智能体位置
        new_pos = self.agent_location + direction
        self.render_.upgrade_agent(self.agent_location, direction, new_pos)

        # 更新智能体位置，确保它保持在网格边界内
        self.agent_location = np.clip(new_pos, 0, self.size - 1)

        # 增加步数计数
        self.time_steps += 1

        # 检查是否到达目标
        terminated = np.array_equal(self.agent_location, self.target_location)

        # 检查是否超过最大步数限制
        truncated = self.time_steps >= self.max_steps

        # 获取新的观测和信息
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        """
        渲染当前环境状态
        
        Returns:
            None: 当前实现中返回None
        """
        if self.render_mode == "video":
            # 保存当前帧为视频
            self.render_.save_video('image/' + str(time.time()))
        # 显示当前帧
        self.render_.show_frame(100)
        return None

    def _get_obs(self) -> ObsType:
        """
        获取当前环境的观测
        
        Returns:
            ObsType: 包含智能体、目标点和障碍物位置的观测字典
        """
        return {
            "agent": self.agent_location, 
            "target": self.target_location, 
            "barrier": self.forbidden_location
        }

    def _get_info(self) -> dict:
        """
        获取当前环境的额外信息
        
        Returns:
            dict: 包含当前步数等信息的字典
        """
        return {"time_steps": self.time_steps}

    def state2pos(self, state: int) -> np.ndarray:
        """
        将一维状态索引转换为二维网格位置坐标
        
        Args:
            state: 一维状态索引
        
        Returns:
            np.ndarray: 二维位置坐标 [x, y]
        """
        return np.array((state // self.size, state % self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        """
        将二维网格位置坐标转换为一维状态索引
        
        Args:
            pos: 二维位置坐标 [x, y]
        
        Returns:
            int: 一维状态索引
        """
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        """
        初始化网格世界的状态转移概率矩阵(Psa)和奖励概率矩阵(Rsa)
        
        状态转移概率P(s'|s,a)表示在状态s执行动作a后转移到状态s'的概率
        奖励概率R(s,a)表示在状态s执行动作a后获得特定奖励的概率
        
        注：根据赵老师的解释，奖励r依赖于状态s和动作a，而不是下一个状态s'
        但实际应用中，r与s'的关系被蕴含到了条件概率p(r|s,a)中
        """
        state_size = self.size ** 2
        
        # 初始化状态转移概率矩阵: shape=(状态数, 动作数, 状态数)
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        
        # 初始化奖励概率矩阵: shape=(状态数, 动作数, 奖励类型数)
        self.Rsa = np.zeros(shape=(state_size, self.action_space_size, len(self.reward_list)), dtype=float)
        
        # 遍历所有可能的状态和动作，计算转移概率和奖励概率
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                # 获取当前状态对应的位置
                pos = self.state2pos(state_index)
                # 计算执行动作后的新位置
                next_pos = pos + self.action_to_direction[action_index]
                
                # 检查是否撞墙
                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:
                    # 撞墙，位置保持不变
                    self.Psa[state_index, action_index, state_index] = 1
                    # 撞到墙，给予惩罚（索引3）
                    self.Rsa[state_index, action_index, 3] = 1
                else:
                    # 计算新位置对应的状态索引
                    next_state_index = self.pos2state(next_pos)
                    # 设置状态转移概率
                    self.Psa[state_index, action_index, next_state_index] = 1
                    
                    # 根据新位置计算奖励类型
                    if np.array_equal(next_pos, self.target_location):
                        # 到达目标点，给予正奖励（索引1）
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location):
                        # 撞到障碍物，给予惩罚（索引2）
                        self.Rsa[state_index, action_index, 2] = 1
                    else:
                        # 普通移动，给予基础奖励（索引0）
                        self.Rsa[state_index, action_index, 0] = 1

    def close(self):
        """
        关闭环境，执行必要的清理操作
        """
        pass

    def get_state_space_info(self):
        """
        获取状态空间的信息，包括总状态数、有效状态数等
        
        Returns:
            dict: 包含状态空间信息的字典
        """
        total_states = self.size ** 2
        obstacle_states = [self.pos2state(obs) for obs in self.forbidden_location]
        start_state = self.pos2state(self.agent_location)
        target_state = self.pos2state(self.target_location)
        
        return {
            "total_states": total_states,
            "obstacle_states": obstacle_states,
            "start_state": start_state,
            "target_state": target_state,
            "valid_states": total_states - len(obstacle_states)
        }


if __name__ == "__main__":
    # 示例：创建一个5x5的网格世界环境
    grid = GridWorldEnv(
        size=5, 
        start=[0, 0],
        target=[1, 2], 
        forbidden=[[2, 2]],
        render_mode='',
        max_steps=50
    )
    # 渲染环境
    grid.render()
    # 打印状态空间信息
    print("状态空间信息:", grid.get_state_space_info())
    # 测试重置环境
    obs, info = grid.reset()
    print("初始观测:", obs)
    # 测试执行一步动作（向右移动）
    obs, reward, terminated, truncated, info = grid.step(2)
    print("执行动作后的观测:", obs)
    print("获得的奖励:", reward)
