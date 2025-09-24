import time
from typing import Optional, Union, List, Tuple
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame, ActType, ObsType
np.random.seed(1)
import render


def arr_in_list(array, _list):
    for element in _list:
        if np.array_equal(element, array):
            return True
    return False


class GridWorldEnv(gym.Env):
    def __init__(
            self, 
            size: int, 
            start: Union[list, tuple, np.ndarray],
            target: Union[list, tuple, np.ndarray], 
            forbidden: Union[list, tuple, np.ndarray],
            render_mode: str,
            reward_list: Optional[List[float]] = None
        ):
        """
        GridWorldEnv 的构造函数
        :param size: grid_world 的边长
        :param start: 起点的pos
        :param target: 目标点的pos
        :param forbidden: 不可通行区域, 二维数组或者嵌套列表, 如 [[1,2],[2,2]]
        :param render_mode: 渲染模式, video表示保存视频
        :param reward_list: 奖励列表，分别表示 普通移动，达到目标，撞到障碍物，撞到墙
        """
        # 初始化变量
        self.time_steps = 0
        self.size = size  # grid_world的边长
        self.render_mode = render_mode  # 渲染模式

         # 初始化渲染器
        self.render_ = render.Render(target=target, forbidden=forbidden, size=size)

        # 初始化起点
        self._agent_location = start  # np.array([0, 0])

        # 初始化障碍物位置
        self.forbidden_location = []
        for fob in forbidden:
            self.forbidden_location.append(np.array(fob))

        # 初始化目标点
        self.target_location = np.array(target)

        # 初始化动作空间
        # Define what actions are available (5 directions)
        self.action_space = gym.spaces.Discrete(5)
        self.action_space_size = gym.spaces.Discrete(5).n

        # 初始化观测空间
        # Define what the agent can observe. Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),    # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "barrier": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # 奖励列表，分别表示 普通移动，达到目标，撞到障碍物，撞到墙
        self.reward_list = reward_list if reward_list is not None else [0, 1, -10, -10]

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        # action to pos偏移量 的一个map
        self.action_to_direction = {
            0: np.array([0, 0]),     # Stay in place
            1: np.array([0, 1]),     # Move up (positive y)
            2: np.array([1, 0]),     # Move right (positive x)
            3: np.array([0, -1]),    # Move down (negative y)
            4: np.array([-1, 0]),    # Move left (negative x)
        }

        # Rsa表示 在 指定 state 选取指点 action 得到reward的概率
        self.Rsa = None

        # Psa表示 在 指定 state 选取指点 action 跳到下一个state的概率
        self.Psa = None

        self.psa_rsa_init()

    def reset(
            self, 
            seed: Optional[int] = None, 
            options: Optional[dict] = None
        ) -> Tuple[ObsType, dict]:
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self._agent_location = np.array([0, 0])
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-4 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        reward = self.reward_list[self.Rsa[self.pos2state(self._agent_location), action].tolist().index(1)]

        # Map the discrete action (0-4) to a movement direction
        direction = self.action_to_direction[action]

        self.render_.upgrade_agent(self._agent_location, direction, self._agent_location + direction)

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge        
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self.target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        if self.render_mode == "video":
            self.render_.save_video('image/' + str(time.time()))
        self.render_.show_frame(100)
        return None

    def _get_obs(self) -> ObsType:
        """Get the current observation.

        Returns:
            ObsType: Current observation of the environment. Includes agent, target, and barrier positions.
        """
        return {
            "agent": self._agent_location, 
            "target": self.target_location, 
            "barrier": self.forbidden_location
        }

    def _get_info(self) -> dict:
        return {"time_steps": self.time_steps}

    def state2pos(self, state: int) -> np.ndarray:
        """transform state to position

        Args:
            state (int): 1D state

        Returns:
            np.ndarray: 2D position
        """
        return np.array((state // self.size, state % self.size))

    def pos2state(self, pos: np.ndarray) -> int:
        """transform position to state

        Args:
            pos: 2D position

        Returns:
            int: 1D state
        """
        return pos[0] * self.size + pos[1]

    def psa_rsa_init(self):
        """
        初始化网格世界的 psa 和 rsa
        赵老师在b站评论区回答过 关于 rsa设计的问题
        原问题是；
        B友：老师您好，在spinning up 7.2.5里有写到
        Reward depends on the current state of the world, the action just taken, and the next state of the world.
        但您提到Rewad depends on the state and action, but not the next state.不知道reward 和 next state的关系是怎样的？

        答案如下：
        赵老师：这是一个很细小、但是很好的问题，说明你思考了。也许其他人也会有这样的疑问，我来详细解答一下。
        1）从贝尔曼公式和数学的角度来说，r是由p(r|s,a)决定的，所以从数学的角度r依赖于s,a，而不依赖于下一个状态s’。这是很简明的。
        2）举例，如果在target state刚好旁边是墙，agent试图撞墙又弹回来target state，这时候不应该给正r，而应该是给负r，因为r依赖于a而不是下一个状态。
        3）但是r是否和s’无关呢？实际是有关系的，否则为什么每次进到target state要得到正r呢？不过，这也可以等价理解成是在之前那个状态采取了好的动作才得到了正r。
        总结：r确实和s’有关，但是这种关系被设计蕴含到了条件概率p(r|s,a)中去。
        故而这里的rsa蕴含了next_state的信息
        :return:
        """
        state_size = self.size ** 2
        # s,a -> s'
        self.Psa = np.zeros(shape=(state_size, self.action_space_size, state_size), dtype=float)
        # s,a -> r
        self.Rsa = np.zeros(shape=(state_size, self.action_space_size, len(self.reward_list)), dtype=float)
        for state_index in range(state_size):
            for action_index in range(self.action_space_size):
                pos = self.state2pos(state_index)
                next_pos = pos + self.action_to_direction[action_index]
                if next_pos[0] < 0 or next_pos[1] < 0 or next_pos[0] > self.size - 1 or next_pos[1] > self.size - 1:
                    self.Psa[state_index, action_index, state_index] = 1    # 撞墙，位置不变
                    self.Rsa[state_index, action_index, 3] = 1    # 撞墙，惩罚-10

                else:
                    self.Psa[state_index, action_index, self.pos2state(next_pos)] = 1    # 正常移动
                    # 计算奖励
                    if np.array_equal(next_pos, self.target_location):   # 到达目标
                        self.Rsa[state_index, action_index, 1] = 1
                    elif arr_in_list(next_pos, self.forbidden_location):  # 撞到障碍物
                        self.Rsa[state_index, action_index, 2] = 1
                    else:
                        self.Rsa[state_index, action_index, 0] = 1    # 普通移动，奖励0

    def close(self):
        pass


if __name__ == "__main__":
    grid = GridWorldEnv(size=5, target=[1, 2], forbidden=[[2, 2]], render_mode='')
    grid.render()
