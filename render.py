from typing import Union, List, Tuple, Optional

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)  # 固定随机种子，保证可复现性


class Render:
    """
    网格世界可视化工具类
    
    负责网格世界环境的可视化展示，包括绘制网格、智能体位置、障碍物、目标点、智能体轨迹等
    支持静态图像保存和动态视频生成功能
    """
    def __init__(
            self, 
            target: Union[list, tuple, np.ndarray], 
            forbidden: Union[list, tuple, np.ndarray],
            size: int = 5
        ):
        """
        Render 类的构造函数，用于初始化网格世界的可视化环境。

        Args:
            target: 目标点的位置，格式为[x, y]
            forbidden: 障碍物区域位置列表，格式为[[x1,y1], [x2,y2], ...]
            size: 网格世界的边长，默认为5（5x5网格）
        """
        # 初始化变量
        self.agent = None    # 智能体的可视化对象
        self.target = np.array(target)   # 目标点坐标
        # 确保障碍物列表中的元素都是numpy数组
        self.forbidden = [np.array(fob) for fob in forbidden]   # 障碍物坐标列表
        self.size = size   # 网格世界的边长
        
        # 初始化画布相关属性为None，延迟创建
        self.fig = None
        self.ax = None

        # 初始化画布
        self.reset_canvas()
        # print(self.ax)
        
        # 初始化智能体轨迹记录
        self.trajectory = []   # 保存智能体轨迹
        
        # 确保中文显示正常
        plt.rcParams["font.family"] = ["SimHei"]

    def create_canvas(self, figsize: Tuple[int, int] = None) -> None:
        """
        创建画布和坐标轴，允许自定义画布大小
        
        Args:
            figsize: 画布尺寸，格式为(width, height)，默认为None（自动计算）
        """
        # 如果画布已存在，先关闭
        if self.fig is not None:
            plt.close(self.fig)
        
        # 创建新画布和坐标轴
        if figsize is None:
            figsize = (10, 10)
        self.fig = plt.figure(figsize=figsize, dpi=self.size * 20)
        self.ax = plt.gca()
        
        # 重新初始化智能体箭头（初始位置在画布外）
        if self.agent is not None:
            self.agent.remove()
        self.agent = patches.Arrow(-10, -10, 0.4, 0, color='red', width=0.5)
        
    def init_grid(self) -> None:
        """
        初始化网格世界的基础结构，包括坐标轴设置、网格线和标签
        注意：需要先调用create_canvas()创建画布
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        # 设置坐标轴样式
        self.ax.xaxis.set_ticks_position('top')   # x轴刻度显示在顶部
        self.ax.invert_yaxis()   # y轴反转，符合常见的网格世界习惯，使得原点在左上角
        self.ax.xaxis.set_ticks(range(0, self.size + 1))
        self.ax.yaxis.set_ticks(range(0, self.size + 1))
        self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        self.ax.tick_params(
            bottom=False, left=False, right=False, top=False, 
            labelbottom=False, labelleft=False, labeltop=False
        )
        # index = 0
        # 绘制网格世界的坐标标签
        for y in range(self.size):
            self.write_word(pos=(-0.6, y), word=str(y + 1), size_discount=0.8)
            self.write_word(pos=(y, -0.6), word=str(y + 1), size_discount=0.8)
            # for x in range(size):
            #     self.write_word(pos=(x, y), word="s" + str(index), size_discount=0.65)
            #     index += 1
        
        # 添加智能体箭头到画布
        self.ax.add_patch(self.agent)

    def reset_canvas(self, clear_trajectory: bool = True, figsize: Tuple[int, int] = None) -> None:
        """
        重置画布，可选择是否清除轨迹记录
        
        Args:
            clear_trajectory: 是否清除轨迹记录，默认为True
            figsize: 新画布的尺寸，默认为None（保持原尺寸）
        """
        # 创建新画布
        self.create_canvas(figsize)
        
        # 初始化网格
        self.init_grid()
        
        # 填充障碍物和目标格子
        for pos in self.forbidden:
            self.fill_block(pos=pos)
        self.fill_block(pos=self.target, color='darkturquoise')
        
        # 清除轨迹记录
        if clear_trajectory:
            self.trajectory = []

    def fill_block(
            self, 
            pos: Union[list, tuple, np.ndarray], 
            color: str = '#EDB120', 
            width: float = 1.0,
            height: float = 1.0
        ) -> patches.Rectangle:
        """
        对指定位置的网格填充颜色

        Args:
            pos: 需要填充的网格的左上坐标（坐标原点位于左上角）
            color: 填充的颜色，默认黄色（障碍物forbidden），目标target格子用蓝色
            width: 填充方块宽度
            height: 填充方块高度
            
        Returns:
            patches.Rectangle: 创建的矩形补丁对象
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        return self.ax.add_patch(
            patches.Rectangle(
                (pos[0], pos[1]),
                width=width,
                height=height,
                facecolor=color,
                fill=True,
                alpha=0.90,
            )
        )

    def draw_random_line(self, pos1: Union[list, tuple, np.ndarray], pos2: Union[list, tuple, np.ndarray]) -> None:
        """
        在pos1 和pos2之间生成一条带有随机扰动的线条，用于可视化轨迹。

        Args:
            pos1: 起点所在位置的坐标
            pos2: 终点所在位置的坐标
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        # 生成小的随机偏移，使轨迹线看起来不完全重叠
        offset1 = np.random.uniform(low=-0.05, high=0.05, size=1)
        offset2 = np.random.uniform(low=-0.05, high=0.05, size=1)
        
        # 计算线条的起点和终点（网格中心点）
        x = [pos1[0] + 0.5, pos2[0] + 0.5]
        y = [pos1[1] + 0.5, pos2[1] + 0.5]
        
        # 根据移动方向添加偏移
        if pos1[0] == pos2[0]:
            # 垂直移动，在x方向添加偏移
            x = [x[0] + offset1, x[1] + offset2]
        else:
            # 水平移动，在y方向添加偏移
            y = [y[0] + offset1, y[1] + offset2]
        
        # 绘制轨迹线
        self.ax.plot(x, y, color='g', scalex=False, scaley=False)

    def draw_circle(self, pos: Union[list, tuple, np.ndarray], radius: float, 
                    color: str = 'green', fill: bool = True) -> patches.Circle:
        """
        在指定网格内画一个圆（如表示智能体静止）。

        Args:
            pos: 圆心所在格子的左上角坐标（坐标原点位于左上角）
            radius: 圆的半径
            color: 圆的颜色
            fill: 是否填充圆内部
            
        Returns:
            patches.Circle: 创建的圆形补丁对象
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        return self.ax.add_patch(
            patches.Circle(
                (pos[0] + 0.5, pos[1] + 0.5),
                radius=radius,
                facecolor=color,
                edgecolor='green',
                linewidth=2,
                fill=fill
            )
        )

    def draw_action(self, pos: Union[list, tuple, np.ndarray], toward: Union[list, tuple, np.ndarray],
                    color: str = 'green', radius: float = 0.10) -> None:
        """
        可视化某个格子上的动作（箭头或圆）。

        Args:
            pos: 当前格子的左上角坐标（坐标原点位于左上角）
            toward: 动作方向向量（0向量则画圆），(a,b)分别表示箭头在x方向和y方向的分量
            color: 箭头颜色，默认为绿色
            radius: 圆的半径（当动作是静止时使用）
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        if not np.array_equal(np.array(toward), np.array([0, 0])):
            # 非静止动作，绘制箭头
            self.ax.add_patch(
                patches.Arrow(
                    pos[0] + 0.5, pos[1] + 0.5, 
                    dx=toward[0], dy=toward[1], 
                    color=color, 
                    width=0.05 + 0.05 * np.linalg.norm(np.array(toward) / 0.5),
                    linewidth=0.5
                )
            )
        else:
            # 静止动作，绘制圆圈
            self.draw_circle(pos=tuple(pos), color='white', radius=radius, fill=False)

    def write_word(self, pos: Union[list, np.ndarray, tuple], word: str, color: str = 'black', 
                   y_offset: float = 0, size_discount: float = 1.0) -> None:
        """
        在网格上指定位置写字（如坐标编号、状态值等）。

        Args:
            pos: 格子的左下角坐标
            word: 要写的内容
            color: 字体颜色
            y_offset: y方向偏移，即字在y方向上关于网格中心的偏移
            size_discount: 字体缩放因子，即字体大小 (0-1)
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        # 根据网格大小自动调整字体大小
        font_size = size_discount * (30 - 2 * self.size)
        self.ax.text(
            pos[0] + 0.5, pos[1] + 0.5 + y_offset, 
            word, size=font_size, ha='center', va='center', color=color
        )

    def upgrade_agent(self, pos: Union[list, np.ndarray, tuple], action: Union[list, np.ndarray, tuple],
                      next_pos: Union[list, np.ndarray, tuple]) -> None:
        """
        更新智能体轨迹，即记录智能体从pos位置采取action动作到达next_pos位置的轨迹

        Args:
            pos: 当前状态state位置
            action: 当前采取的动作action（方向向量）
            next_pos: 当前pos的下一步位置
        """
        # 记录轨迹信息
        self.trajectory.append([tuple(pos), action, tuple(next_pos)])

    def show_frame(self, t: float = 0.2, close_after: bool = True) -> None:
        """
        显示当前画布

        Args:
            t: 显示时长（秒）
            close_after: 显示后是否关闭画布，默认为True
        """
        if self.fig is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        self.fig.show()
        plt.pause(t)
        if close_after:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def save_frame(self, name: str, dpi: int = 300) -> None:
        """
        保存当前帧为图片。
    
        Args:
            name: 文件名（不含扩展名）
            dpi: 图片分辨率，默认为300
        """
        if self.fig is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        self.fig.savefig(f"{name}.jpg", dpi=dpi, bbox_inches='tight')

    def save_video(self, name: str, fps: int = 4) -> None:
        """
        保存智能体轨迹为视频。
        如果指定了起点，想要将agent从起点到终点的轨迹展示出来的话，可以使用这个函数保存视频

        Args:
            name: 视频文件名（不含扩展名）
            fps: 视频帧率，默认为4帧/秒
        """
        # 确保轨迹不为空
        if not self.trajectory:
            print("警告：没有轨迹数据，无法生成视频")
            return
            
        # 如果画布不存在，创建一个
        if self.fig is None:
            self.create_canvas()
            self.init_grid()
            # 填充障碍物和目标格子
            for pos in self.forbidden:
                self.fill_block(pos=pos)
            self.fill_block(pos=self.target, color='darkturquoise')
            
        # 创建动画
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate, 
            init_func=self.init, 
            frames=len(self.trajectory),
            interval=1000//fps,  # 每帧间隔毫秒数
            repeat=False
        )
        
        # 保存视频
        try:
            anim.save(f"{name}.mp4")
            print(f"视频已保存为: {name}.mp4")
        except Exception as e:
            print(f"保存视频失败: {str(e)}")
        
        # 保存视频后关闭画布
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def init(self):
        """
        初始化动画帧（为FuncAnimation准备）
        
        Returns:
            list: 包含需要更新的matplotlib艺术家对象的列表
        """
        # 重置智能体显示
        if self.agent:
            self.agent.remove()
        self.agent = patches.Arrow(-10, -10, 0.4, 0, color='red', width=0.5)
        self.ax.add_patch(self.agent)
        return [self.agent]

    def animate(self, i: int):
        """
        动画每一帧的绘制逻辑。

        Args:
            i: 当前帧索引
            
        Returns:
            list: 包含更新后的matplotlib艺术家对象的列表
        """
        # 确保索引在有效范围内
        if i >= len(self.trajectory):
            return [self.agent]
            
        # 获取当前帧的位置和动作信息
        location = self.trajectory[i][0]
        action = self.trajectory[i][1]
        next_location = self.trajectory[i][2]
        
        # 确保next_location在网格范围内
        next_location = np.clip(next_location, -0.4, self.size - 0.6)
        
        # 更新智能体显示
        self.agent.remove()
        if np.array_equal(np.array(action), np.array([0, 0])):
            # 静止动作，绘制圆圈
            self.agent = patches.Circle(
                xy=(location[0] + 0.5, location[1] + 0.5),
                radius=0.15, 
                fill=True, 
                color='b'
            )
        else:
            # 移动动作，绘制箭头
            self.agent = patches.Arrow(
                x=location[0] + 0.5, 
                y=location[1] + 0.5,
                dx=action[0] / 2, 
                dy=action[1] / 2,
                color='b',
                width=0.5
            )
        self.ax.add_patch(self.agent)
        
        # 绘制轨迹线
        self.draw_random_line(pos1=location, pos2=next_location)
        
        return [self.agent]

    def draw_episode(self):
        """
        绘制整个episode的轨迹（所有步的连线）。
        """
        for i in range(len(self.trajectory)):
            location = self.trajectory[i][0]
            next_location = self.trajectory[i][2]
            self.draw_random_line(pos1=location, pos2=next_location)

    def add_subplot_to_fig(self, fig: plt.Figure, x: np.ndarray, y: np.ndarray, 
                          subplot_position: Tuple[int, int, int], xlabel: str, 
                          ylabel: str, title: str = '') -> plt.Axes:
        """
        在给定的位置上添加一个子图到当前的图中，并在子图中调用plot函数，设置x,y label和title。

        Args:
            fig: matplotlib的Figure对象
            x: x轴数据
            y: y轴数据
            subplot_position: 子图的位置，格式为 (row, column, index)
            xlabel: x轴标签
            ylabel: y轴标签
            title: 子图标题
            
        Returns:
            plt.Axes: 创建的子图坐标轴对象
        """
        # 在指定位置添加子图
        ax = fig.add_subplot(*subplot_position)
        # 调用plot函数绘制图形
        ax.plot(x, y)
        # 设置x,y label和title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        return ax
        
    def visualize_state_values(self, state_values: np.ndarray, y_offset: float = 0.2) -> None:
        """
        可视化每个状态的值函数
        
        Args:
            state_values: 包含每个状态值的一维数组
            y_offset: 文本在y方向上的偏移量
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        for state in range(self.size * self.size):
            x = state // self.size
            y = state % self.size
            # 显示状态值，保留两位小数
            value_text = f"{state_values[state]:.1f}"
            self.write_word(pos=(x, y), word=value_text, color='black', y_offset=y_offset, size_discount=0.7)
            
    def visualize_policy(self, policy: np.ndarray, action_to_direction: dict) -> None:
        """
        可视化策略（每个状态下的动作概率分布）
        
        Args:
            policy: 策略矩阵，shape=(状态数, 动作数)
            action_to_direction: 动作到方向向量的映射
        """
        if self.ax is None:
            raise ValueError("请先调用create_canvas()创建画布")
        
        for state in range(self.size * self.size):
            x = state // self.size
            y = state % self.size
                
            # 绘制每个可能动作的概率箭头
            for action in range(len(action_to_direction)):
                prob = policy[state, action]
                if prob > 0.0:  # 只显示概率大于0.0的动作
                    direction = action_to_direction[action] * 0.4 * prob
                    self.draw_action(
                        pos=[x, y], 
                        toward=direction,
                        color='green',   # 'purple', 'red'
                        radius=0.03 + 0.07 * prob
                    )


# 测试代码示例
if __name__ == '__main__':
    # 创建一个5x5的网格世界渲染器
    render = Render(
        target=[4, 4], 
        forbidden=[np.array([1, 2]), np.array([2, 2])], 
        size=5
    )
    
    # 创建画布并初始化网格
    render.create_canvas()
    render.init_grid()
    
    # 填充障碍物和目标格子
    for pos in render.forbidden:
        render.fill_block(pos=pos)
    render.fill_block(pos=render.target, color='darkturquoise')
    
    # 测试绘制动作
    render.draw_action(pos=[3, 3], toward=(0, 0.4))
    
    # 测试绘制多条轨迹线
    for num in range(10):
        render.draw_random_line(pos1=[1, 1], pos2=[1, 2])
    
    # 定义动作到方向的映射
    action_to_direction = {
        0: np.array([0, 0]),     # 停留不动
        1: np.array([0, 1]),     # 向上移动（y轴正方向）
        2: np.array([1, 0]),     # 向右移动（x轴正方向）
        3: np.array([0, -1]),    # 向下移动（y轴负方向）
        4: np.array([-1, 0]),    # 向左移动（x轴负方向）
    }
    
    # 创建一个均匀随机策略并可视化
    uniform_policy = np.ones(shape=(25, 5)) / 5  # 均匀分布
    render.visualize_policy(uniform_policy, action_to_direction)
    
    # 测试绘制状态值
    state_values = np.random.rand(25)  # 随机生成状态值
    render.visualize_state_values(state_values)
    
    # 显示结果但不关闭画布
    render.show_frame(t=2, close_after=False)
    
    # 重新绘制一些内容到同一画布
    render.write_word(pos=(2, 2), word="测试", color='red', size_discount=1.2)
    
    # 再次显示并关闭
    render.show_frame(t=3)
    
    # 重置画布并绘制不同内容
    print("重置画布并绘制不同内容...")
    render.reset_canvas()
    render.visualize_policy(uniform_policy, action_to_direction)
    render.show_frame(t=3)
