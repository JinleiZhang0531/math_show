from manim import *
from manim.utils.rate_functions import ease_in_out_sine

class TestScene(Scene):
    def construct(self):
        self.play(Write(Text("Hello Manim!")))
        self.wait(1)

# 创建一个简单的场景类
class HelloManim(Scene):
    def construct(self):
        # 创建文字对象，使用SimHei字体以显示中文
        text = Text("你好，Manim！")

        # 使用Write动画显示文字
        self.play(Write(text))

        # 等待2秒
        self.wait(2)

class BasicShapes(Scene):
    def construct(self):
        # 1. 创建一个圆形
        circle = Circle(
            radius=1.0,           # 半径
            color=BLUE,          # 颜色
            stroke_width=2.0     # 线条宽度
        )

        # 添加圆形到场景
        self.play(Create(circle))
        self.wait(1)

        # 2. 创建一个方形
        square = Square(
            side_length=2.0,     # 边长
            color=RED,           # 颜色
            fill_opacity=0.0     # 填充透明度
        )

        # 将方形移动到圆形右边
        square.next_to(circle, RIGHT)

        # 添加方形到场景
        self.play(Create(square))
        self.wait(1)


class ColorDemo(Scene):
    def construct(self):
        # 1. 使用内置颜色
        text1 = Text("内置颜色", color=RED, font="SimHei")
        self.play(Write(text1))
        self.wait(1)

        # 2. 使用RGB颜色
        text2 = Text(
            "RGB颜色",
            color=rgb_to_color([0.5, 0.8, 0.2]),
            font="SimHei"
        ).next_to(text1, DOWN)
        self.play(Write(text2))
        self.wait(1)

        # 3. 使用HEX颜色
        text3 = Text(
            "HEX颜色",
            color="#FF6B6B",
            font="SimHei"
        ).next_to(text2, DOWN)
        self.play(Write(text3))
        self.wait(1)

class SimpleAnimations(Scene):
    def construct(self):
        # 1. 创建一个圆形
        circle = Circle(color=BLUE)

        # 渐现动画
        self.play(FadeIn(circle))
        self.wait(1)

        # 移动动画
        self.play(circle.animate.shift(RIGHT * 2))
        self.wait(1)

        # 缩放动画
        self.play(circle.animate.scale(2))
        self.wait(1)

        # 改变颜色动画
        self.play(circle.animate.set_color(RED))
        self.wait(1)

from manim import *  # 确保导入完整

class CoordinateSystemDemo(Scene):
    def construct(self):
        # 1. 创建坐标轴（兼容Manim Community版，修复axis_config参数）
        axes = Axes(
            x_range=[-6, 6, 1],  # 步长改为2，避免刻度重叠
            y_range=[-6, 6, 1],
            x_length=8,  # 场景宽度14，留余量设12
            y_length=8,  # 场景高度8，稍超但可显示（Manim会自适应）
            axis_config={
                "include_numbers": True,
                "include_tip": True,
                "color": RED,
                # 可选：调整刻度大小
                "tick_size": 0.1,  # 刻度线长度（默认0.1）
            },
            tips=True,
        )

        # 坐标轴标签（兼容新版：用Tex更稳定，避免Text的字体问题）
        labels = axes.get_axis_labels(
            x_label=Tex("x"),
            y_label=Tex("y")
        )

        # 显示坐标轴（用FadeIn替代Create，适配Axes的渲染逻辑）
        self.play(FadeIn(axes), Write(labels))
        self.wait(1)

        # 2. 绘制点（修复中文显示+坐标稳定性）
        point = Dot(axes.coords_to_point(1, 1), color=RED)  # 显式颜色，易识别
        # 方案1：用Tex（不支持font参数）
        point_label = Tex("(1,1)").next_to(point, UR, buff=0.1)
        # 方案2：若需要中文字体支持，用Text
        # point_label = Text("(1,1)", font="SimHei", color=BLACK).next_to(point, UR, buff=0.1)

        # 播放动画（用FadeIn替代Create，Dot更适配）
        self.play(FadeIn(point), Write(point_label))
        self.wait(2)  # 延长暂停时间，避免画面一闪而过


class AnimationGroups(Scene):
    def construct(self):
        # 1. 创建多个形状
        circle = Circle(color=BLUE)
        square = Square(color=RED)
        triangle = Triangle(color=GREEN)

        # 将形状排列
        shapes = VGroup(circle, square, triangle).arrange(RIGHT, buff=1)

        # 同时显示所有形状
        self.play(
            *[Create(shape) for shape in shapes]
        )
        self.wait(1)

        # 依次改变颜色
        self.play(
            circle.animate.set_color(YELLOW),
            square.animate.set_color(PURPLE),
            triangle.animate.set_color(ORANGE),
            lag_ratio=0.5  # 动画之间的延迟
        )
        self.wait(1)


class CustomAnimation(Scene):
    def construct(self):
        # 1. 创建正方形（初始状态无缩放、无旋转）
        square = Square(color=BLUE, side_length=2)  # 固定边长，避免默认尺寸太小导致视觉错觉
        self.add(square)  # 先显示初始正方形

        # 2. 自定义纯旋转动画（无任何缩放）
        class OnlyRotate(Animation):
            def __init__(self, mobject, **kwargs):
                # 初始化时记录物体的初始状态（关键：确保旋转基于初始状态）
                super().__init__(mobject, **kwargs)

            def interpolate_mobject(self, alpha):
                # alpha从0→1，旋转角度从0→PI（180度）
                # 核心：先重置物体到初始状态，再执行旋转（避免帧间累加）
                self.mobject.become(self.starting_mobject.copy())
                self.mobject.rotate(alpha * PI)  # 仅旋转，无缩放

        # 3. 执行纯旋转动画（放慢速度，便于观察）
        self.play(OnlyRotate(square), run_time=10, rate_func=ease_in_out_sine)
        self.wait(2)


class PathAnimation(Scene):
    def construct(self):
        # 1. 创建一个圆形路径
        circle = Circle(radius=2, color=GRAY)
        dot = Dot(color=RED)

        # 将点放在圆上
        dot.move_to(circle.point_at_angle(0))

        # 显示圆形和点
        self.play(Create(circle), FadeIn(dot))

        # 让点沿着圆移动
        self.play(
            MoveAlongPath(dot, circle),
            run_time=3  # 动画持续3秒
        )
        self.wait(1)