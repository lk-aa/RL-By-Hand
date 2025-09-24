from typing import Union

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)  # 固定随机种子，保证可复现性

class Render:
    def __init__(
            self, 
            target: Union[list, tuple, np.ndarray], 
            forbidden: Union[list, tuple, np.ndarray],
            size: int = 5
        ):
        """
        Render 类的构造函数，用于初始化网格世界的可视化环境。

        :param target: 目标点的位置
        :param forbidden: 障碍物区域位置
        :param size: 网格世界的size 默认为 5x5
        """
        # 初始化
        self.agent = None    # 智能体的可视化对象
        self.target = target   # 目标点坐标
        self.forbidden = forbidden   # 障碍物坐标列表
        self.size = size   # 网格世界的边长
        # 创建画布和坐标轴
        self.fig = plt.figure(figsize=(10, 10), dpi=self.size * 20)
        self.ax = plt.gca()
        self.ax.xaxis.set_ticks_position('top')   # x轴刻度显示在顶部
        self.ax.invert_yaxis()   # y轴反转，符合常见的网格世界习惯，使得原点在左上角
        self.ax.xaxis.set_ticks(range(0, size + 1))
        self.ax.yaxis.set_ticks(range(0, size + 1))
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                            labeltop=False)
        # 绘制网格世界的state index 以及grid边框的标号
        # index = 0
        for y in range(size):
            self.write_word(pos=(-0.6, y), word=str(y + 1), size_discount=0.8)
            self.write_word(pos=(y, -0.6), word=str(y + 1), size_discount=0.8)
            # for x in range(size):
            #     self.write_word(pos=(x, y), word="s" + str(index), size_discount=0.65)
            #     index += 1
        # 填充障碍物和目标格子
        for pos in self.forbidden:
            self.fill_block(pos=pos)
        self.fill_block(pos=self.target, color='darkturquoise')
        self.trajectory = []   # 保存智能体轨迹
        # 初始化智能体的箭头（初始位置在画布外）
        self.agent = patches.Arrow(-10, -10, 0.4, 0, color='red', width=0.5)
        self.ax.add_patch(self.agent)

    def fill_block(
            self, 
            pos: Union[list, tuple, np.ndarray], 
            color: str = '#EDB120', 
            width=1.0,
            height=1.0
        ) -> patches.RegularPolygon:
        """
        对指定pos的网格填充颜色

        :param pos: 需要填充的网格的左上坐标（坐标原点位于左上角）
        :param color: 填充的颜色，默认黄色（障碍物forbidden），目标target格子用蓝色
        :param width: 填充方块宽度
        :param height: 填充方块高度
        :return: Rectangle对象
        """
        return self.ax.add_patch(
            patches.Rectangle(
                (pos[0], pos[1]),
                width=1.0,
                height=1.0,
                facecolor=color,
                fill=True,
                alpha=0.90,
            )
        )

    def draw_random_line(self, pos1: Union[list, tuple, np.ndarray], pos2: Union[list, tuple, np.ndarray]) -> None:
        """
        在pos1 和pos2之间生成一条带有随机扰动的线条，用于可视化轨迹。

        :param pos1: 起点所在位置的坐标
        :param pos2: 终点所在位置的坐标
        :return: None
        """
        offset1 = np.random.uniform(low=-0.05, high=0.05, size=1)
        offset2 = np.random.uniform(low=-0.05, high=0.05, size=1)
        x = [pos1[0] + 0.5, pos2[0] + 0.5]
        y = [pos1[1] + 0.5, pos2[1] + 0.5]
        if pos1[0] == pos2[0]:
            x = [x[0] + offset1, x[1] + offset2]
        else:
            y = [y[0] + offset1, y[1] + offset2]
        self.ax.plot(x, y, color='g', scalex=False, scaley=False)

    def draw_circle(self, pos: Union[list, tuple, np.ndarray], radius: float,
                    color: str = 'green', fill: bool = True) -> patches.CirclePolygon:
        """
        在指定网格内画一个圆（如表示智能体静止）。

        :param pos: 圆心所在格子的左上角坐标（坐标原点位于左上角）
        :param radius: 圆的半径
        :param color: 圆的颜色
        :param fill: 是否填充圆内部
        :return: CirclePolygon对象
        """
        return self.ax.add_patch(
            patches.Circle(
                (pos[0] + 0.5, pos[1] + 0.5),
                radius=radius,
                facecolor=color,
                edgecolor='green',
                linewidth=2,
                fill=fill)
            )

    def draw_action(self, pos: Union[list, tuple, np.ndarray], toward: Union[list, tuple, np.ndarray],
                    color: str = 'green', radius: float = 0.10) -> None:
        """
        可视化某个格子上的动作（箭头或圆）。

        :param pos: 当前格子的左上角坐标（坐标原点位于左上角）
        :param toward: 动作方向向量（0向量则画圆），(a,b)分别表示箭头在x方向和y方向的分量
        :param color: 箭头颜色，默认为绿色
        :param radius: 圆的半径
        :return:None
        """
        if not np.array_equal(np.array(toward), np.array([0, 0])):
            self.ax.add_patch(
                patches.Arrow(pos[0] + 0.5, pos[1] + 0.5, dx=toward[0],
                              dy=toward[1], color=color, width=0.05 + 0.05 * np.linalg.norm(np.array(toward) / 0.5),
                              linewidth=0.5))
        else:
            self.draw_circle(pos=tuple(pos), color='white', radius=radius, fill=False)

    def write_word(self, pos: Union[list, np.ndarray, tuple], word: str, color: str = 'black', y_offset: float = 0,
                   size_discount: float = 1.0) -> None:
        """
        在网格上指定位置写字（如坐标编号）。

        :param pos: 格子的左下角坐标
        :param word: 要写的内容
        :param color: 字体颜色
        :param y_offset: y方向偏移，即字在y方向上关于网格中心的偏移
        :param size_discount: 字体缩放因子，即字体大小 (0-1)
        :return: None
        """
        self.ax.text(pos[0] + 0.5, pos[1] + 0.5 + y_offset, word, size=size_discount * (30 - 2 * self.size), ha='center',
                 va='center', color=color)

    def upgrade_agent(self, pos: Union[list, np.ndarray, tuple], action,
                      next_pos: Union[list, np.ndarray, tuple], ) -> None:
        """
        更新智能体轨迹，即记录智能体从pos位置采取action动作到达next_pos位置的轨迹

        :param pos: 当前状态state位置
        :param action: 当前采取的动作action
        :param next_pos: 当前pos的下一步位置
        :return: None
        """
        self.trajectory.append([tuple(pos), action, tuple(next_pos)])

    def show_frame(self, t: float = 0.2) -> None:
        """
        显示当前画布，暂停t秒后关闭。

        :param t: 显示时长（秒）
        :return: None
        """
        self.fig.show()
        plt.pause(t)
        plt.close(self.fig)

    def save_frame(self, name: str) -> None:
        """
        保存当前帧为图片。
    
        :param name: 文件名（不含扩展名）
        :return: None
        """
        self.fig.savefig(name + ".jpg")

    def save_video(self, name: str) -> None:
        """
        保存智能体轨迹为视频。
        如果指定了起点，想要将agent从起点到终点的轨迹show出来的话，可以使用这个函数保存视频

        :param name: 视频文件名（不含扩展名）
        :return:None
        """
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init(), frames=len(self.trajectory),
                                       interval=25, repeat=False)
        anim.save(name + '.mp4')

    # 以下两个方法为动画服务
    # init 和 animate 都是服务于animation.FuncAnimation
    # 具体用法参考matplotlib官网
    def init(self):
        # 初始化动画帧（可根据需要实现）
        pass

    def animate(self, i):
        """
        动画每一帧的绘制逻辑。

        :param i: 当前帧索引
        """
        print(i,len(self.trajectory))
        location = self.trajectory[i][0]
        action = self.trajectory[i][1]
        next_location = self.trajectory[i][2]
        next_location = np.clip(next_location, -0.4, self.size - 0.6)
        self.agent.remove()
        if action[0] + action[1] != 0:
            self.agent = patches.Arrow(x=location[0] + 0.5, y=location[1] + 0.5,
                                       dx=action[0] / 2, dy=action[1] / 2,
                                       color='b',
                                       width=0.5)
        else:
            self.agent = patches.Circle(xy=(location[0] + 0.5, location[1] + 0.5),
                                        radius=0.15, fill=True, color='b',
                                        )
        self.ax.add_patch(self.agent)

        self.draw_random_line(pos1=location, pos2=next_location)

    def draw_episode(self):
        """
        绘制整个episode的轨迹（所有步的连线）。
        """
        for i in range(len(self.trajectory)):
            location = self.trajectory[i][0]
            next_location = self.trajectory[i][2]
            self.draw_random_line(pos1=location, pos2=next_location)

    def add_subplot_to_fig(self, fig, x, y, subplot_position, xlabel, ylabel, title=''):
        """
        在给定的位置上添加一个子图到当前的图中，并在子图中调用plot函数，设置x,y label和title。

        :param fig: matplotlib的Figure对象
        :param x: x轴数据
        :param y: y轴数据
        :param subplot_position: 子图的位置，格式为 (row, column, index)
        :param xlabel: x轴标签
        :param ylabel: y轴标签
        :param title: 子图标题
        """
        # 在指定位置添加子图
        ax = fig.add_subplot(subplot_position)
        # 调用plot函数绘制图形
        ax.plot(x, y)
        # 设置x,y label和title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

# 测试代码示例
if __name__ == '__main__':
    render = Render(target=[4, 4], forbidden=[np.array([1, 2]), np.array([2, 2])], size=5)
    render.draw_action(pos=[3, 3], toward=(0, 0.4))
    # render.save_frame('test1')

    for num in range(10):
        render.draw_random_line(pos1=[1, 1], pos2=[1, 2])

    action_to_direction = {
        0: np.array([0, 0]),     # Stay in place
        1: np.array([0, 1]),     # Move up (positive y)
        2: np.array([1, 0]),     # Move right (positive x)
        3: np.array([0, -1]),    # Move down (negative y)
        4: np.array([-1, 0]),    # Move left (negative x)
    }
    uniform_policy = np.random.random(size=(25, 5))
    uniform_policy = np.ones(shape=(25, 5))
    for state in range(25):
        for action in range(5):
            policy = uniform_policy[state, action]
            render.draw_action(pos=[state // 5, state % 5], toward=policy * 0.4 * action_to_direction[action],
                               radius=0.03 + 0.07 * policy)
    for a in range(5):
        render.trajectory.append((a, a))
    render.show_frame(t=50)
