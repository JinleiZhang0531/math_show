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


class TransformExample(Scene):
    def construct(self):
        # 1. 创建起始形状
        circle = Circle(color=BLUE)

        # 2. 创建目标形状
        square = Square(color=RED)

        # 显示圆形
        self.play(Create(circle))
        self.wait(1)

        # 将圆形变换为方形
        self.play(Transform(circle, square))
        self.wait(1)

        # 创建文字
        text1 = Text("你好", font="SimHei")
        text2 = Text("世界", font="SimHei")

        # 文字变换
        self.play(Write(text1))
        self.wait(1)
        self.play(ReplacementTransform(text1, text2))
        self.wait(1)

class CameraExample(Scene):
    def construct(self):
        circle = Circle(radius=1)
        square = Square()
        triangle = Triangle()
        shapes = VGroup(circle, square, triangle).arrange(RIGHT, buff=2)

        self.play(Create(shapes), run_time=1.5)
        self.wait(1)

        # 保存初始位置（在缩放前保存）
        initial_center = shapes.get_center()
        circle_center = circle.get_center()
        square_center = square.get_center()

        # 平滑缩放+移动到圆形
        # 策略：先移动使circle到屏幕中心，然后以中心缩放2倍
        # 移动距离 = ORIGIN - circle_center
        self.play(
            shapes.animate.shift(ORIGIN - circle_center).scale(2, about_point=ORIGIN),
            run_time=2  # 2秒完成动画，更平滑
        )
        self.wait(1)

        # 平滑移动到方形
        # 计算方形相对于圆形的偏移，缩放后需要移动的距离
        # 缩放后方形位置 = ORIGIN + (square_center - circle_center) * 2
        # 要让方形到ORIGIN，需要移动: ORIGIN - [ORIGIN + (square_center - circle_center) * 2]
        square_offset = (square_center - circle_center) * 2
        self.play(
            shapes.animate.shift(-square_offset),
            run_time=1.5
        )
        self.wait(1)

        # 平滑恢复初始视角
        # 先缩小到0.5倍（以当前中心ORIGIN），然后移动到初始位置
        # 缩小后中心还在ORIGIN，所以直接移动到initial_center即可
        self.play(
            shapes.animate.scale(0.5, about_point=ORIGIN).shift(initial_center),
            run_time=2
        )
        self.wait(1)

class FunctionPlotting(Scene):
    def construct(self):
        # 1. 创建坐标系
        axes = Axes(
            x_range=[-3, 3],
            y_range=[-2, 2],
            axis_config={"include_numbers": True}
        )

        # 添加标签
        labels = axes.get_axis_labels(
            x_label="x", y_label="f(x)"
        )

        # 显示坐标系
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # 2. 绘制函数 f(x) = x^2
        parabola = axes.plot(
            lambda x: x**2,
            color=BLUE,
            x_range=[-2, 2]
        )

        # 添加函数表达式
        # 将标签放在坐标系的右上角空白区域，确保完全显示
        func_label = MathTex("f(x)=x^2").scale(0.8).to_corner(UR, buff=0.5)

        # 显示函数图像
        self.play(Create(parabola), Write(func_label))
        self.wait(3)


class ParametricCurve(Scene):
    def construct(self):
        # 1. 创建坐标系
        axes = Axes(
            x_range=[-4, 4],
            y_range=[-4, 4]
        )

        # 2. 创建参数曲线（心形线）
        # 使用axes.plot_parametric_curve方法，它专门用于在坐标系中绘制参数曲线
        heart = axes.plot_parametric_curve(
            lambda t: np.array([
                16 * np.sin(t)**3,
                13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
            ]) / 16,
            t_range=[0, TAU],
            color=RED
        )

        # 显示坐标系和曲线
        self.play(Create(axes))
        self.play(Create(heart))
        self.wait(1)


class DynamicUpdating(Scene):
    def construct(self):
        # 1. 创建一个点和一个圆
        dot = Dot(RIGHT * 2)
        circle = Circle(radius=2)

        # 2. 创建一条连接圆心和点的线
        line = Line(ORIGIN, dot.get_center())
        line.add_updater(
            lambda l: l.become(
                Line(ORIGIN, dot.get_center())
            )
        )

        # 显示所有元素
        self.play(Create(circle), Create(dot), Create(line))

        # 移动点，线会自动更新
        self.play(
            Rotating(
                dot,
                about_point=ORIGIN,
                angle=TAU,
                run_time=4
            )
        )
        self.wait(1)

class SimplePhysics(Scene):
    def construct(self):
        # 1. 创建一个小球
        ball = Circle(radius=0.2, fill_color=BLUE, fill_opacity=1)
        ball.move_to(UP * 3)

        # 2. 创建地面
        ground = Line(LEFT * 4, RIGHT * 4, color=WHITE)

        # 显示物体
        self.play(Create(ground), Create(ball))

        # 3. 模拟自由落体
        def get_y(t):
            g = 9.8  # 重力加速度
            y0 = 3   # 初始高度
            v0 = 0   # 初始速度
            return y0 - 0.5 * g * t**2

        fall_time = 0

        def update_ball(m, dt):
            nonlocal fall_time
            fall_time += dt  # 累计下落时间
            y = get_y(fall_time)
            # 触地停止
            if y <= 0.2:
                y = 0.2
                m.remove_updater(update_ball)
            m.move_to([m.get_center()[0], y, 0])

        ball.add_updater(update_ball)

        # 运行1秒
        self.wait()
        self.play(ball.animate.move_to(ground.get_center() + UP * 0.2))
        self.wait(1)

class VectorField(Scene):
    def construct(self):
        # 1. 创建坐标系
        plane = NumberPlane(
            x_range=[-5, 5],
            y_range=[-5, 5]
        )

        # 2. 创建向量场函数
        def field_func(pos):
            x, y = pos[:2]
            return np.array([
                -y,  # x分量
                x,   # y分量
                0    # z分量
            ]) / 2

        # 3. 创建向量场
        vector_field = ArrowVectorField(
            field_func,
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            length_func=lambda x: 0.5
        )

        # 显示坐标系和向量场
        self.play(Create(plane))
        self.play(Create(vector_field))
        self.wait(1)

        # 4. 添加一个跟随场的粒子
        dot = Dot(RIGHT * 2, color=RED)

        # 让粒子沿着场移动
        self.play(Create(dot))
        self.play(
            MoveAlongPath(
                dot,
                ParametricFunction(
                    lambda t: np.array([
                        2 * np.cos(t),
                        2 * np.sin(t),
                        0
                    ]),
                    t_range=[0, TAU]
                )
            ),
            run_time=4
        )
        self.wait(1)


class BasicLaTeX(Scene):
    def construct(self):
        # 1. 简单的数学公式
        formula1 = MathTex(r"e^{i\pi} + 1 = 0")

        # 显示公式
        self.play(Write(formula1))
        self.wait(1)

        # 2. 多行公式
        formula2 = MathTex(
            r"\frac{d}{dx} e^x &= e^x \\",
            r"\frac{d}{dx} \ln(x) &= \frac{1}{x}"
        )
        formula2.next_to(formula1, DOWN, buff=1)

        # 显示多行公式
        self.play(Write(formula2))
        self.wait(1)

class FormulaTransformation(Scene):
    def construct(self):
        # 1. 创建初始公式
        formula1 = MathTex(
            r"(a + b)^2"
        )

        # 2. 创建展开步骤
        formula2 = MathTex(
            r"a^2 + 2ab + b^2"
        )

        # 显示初始公式
        self.play(Write(formula1))
        self.wait(1)

        # 变换到展开形式
        self.play(
            TransformMatchingTex(
                formula1,
                formula2
            )
        )
        self.wait(1)

class StepByStepDerivation(Scene):
    def construct(self):
        # 1. 创建推导步骤
        steps = VGroup(
            MathTex(r"\int x^2 dx"),
            MathTex(r"= \frac{x^3}{3}"),
            MathTex(r"= \frac{x^3}{3} + C")
        ).arrange(DOWN, buff=0.5)

        # 2. 添加推导箭头
        arrows = VGroup(*[
            Arrow(
                steps[i].get_bottom(),
                steps[i+1].get_top(),
                buff=0.1
            )
            for i in range(len(steps)-1)
        ])

        # 3. 逐步显示推导过程
        for i, step in enumerate(steps):
            self.play(Write(step))
            if i < len(arrows):
                self.play(Create(arrows[i]))
            self.wait(1)

class TextStylesAndEffects(Scene):
    def construct(self):
        # 1. 基本样式
        text1 = Text(
            "不同的文字样式",
            font="SimHei",
            color=BLUE,
            weight=BOLD
        )

        # 2. 渐变色文字
        text2 = Text(
            "渐变色文字效果",
            font="SimHei",
            gradient=[RED, YELLOW, GREEN]
        ).next_to(text1, DOWN)

        # 3. 描边文字
        text3 = Text(
            "描边文字效果",
            font="SimHei",
            stroke_width=5,
            stroke_color=YELLOW,
            fill_color=RED
        ).next_to(text2, DOWN)

        # 显示所有文本
        self.play(Write(text1))
        self.play(Write(text2))
        self.play(Write(text3))
        self.wait(1)

class DynamicText(Scene):
    def construct(self):
        # 1. 创建打字机效果
        text = Text(
            "这是一个打字机效果",
            font="SimHei"
        )

        # 逐字显示
        for i in range(len(text)):
            self.play(
                Create(text[i]),
                run_time=0.2
            )
        self.wait(1)

        # 2. 创建波浪效果
        wave_text = Text(
            "波浪效果文字",
            font="SimHei"
        ).next_to(text, DOWN)

        def wave_effect(mob, dt):
            for i, char in enumerate(mob):
                char.shift(
                    UP * np.sin(
                        self.time - i/3
                    ) * 0.02
                )

        # 显示文字并添加波浪效果
        self.play(Write(wave_text))
        wave_text.add_updater(wave_effect)
        self.wait(3)

class CombineFormulaAndText(Scene):
    def construct(self):
        # 1. 创建标题
        title = Text(
            "二次函数公式",
            font="SimHei",
            color=BLUE
        ).to_edge(UP)

        # 2. 创建公式
        formula = MathTex(
            r"f(x) = ax^2 + bx + c"
        ).next_to(title, DOWN)

        # 3. 创建说明文本
        description = VGroup(
            Text("其中：", font="SimHei"),
            Text("a 是二次项系数", font="SimHei"),
            Text("b 是一次项系数", font="SimHei"),
            Text("c 是常数项", font="SimHei")
        ).arrange(DOWN, aligned_edge=LEFT).next_to(formula, DOWN)

        # 显示所有元素
        self.play(Write(title))
        self.play(Write(formula))
        self.play(Write(description))
        self.wait(1)

        # 4. 添加动画效果
        # 依次高亮各个系数
        for i, term in enumerate(['a', 'b', 'c']):
            self.play(
                formula.get_part_by_tex(term).animate.set_color(YELLOW),
                description[i+1].animate.set_color(YELLOW),
                run_time=0.5
            )
            self.wait(0.5)
            self.play(
                formula.get_part_by_tex(term).animate.set_color(WHITE),
                description[i+1].animate.set_color(WHITE),
                run_time=0.5
            )

class Basic3DObjects(ThreeDScene):
    def construct(self):
        # 设置相机角度
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # 1. 创建一个立方体
        cube = Cube(
            side_length=2,
            fill_opacity=0.8,
            stroke_width=2
        )

        # 显示立方体
        self.play(Create(cube))
        self.wait(1)

        # 旋转立方体
        self.play(
            Rotate(
                cube,
                angle=2*PI,
                axis=RIGHT,
                run_time=3
            )
        )
        self.wait(1)

class ThreeDCoordinates(ThreeDScene):
    def construct(self):
        # 设置相机角度
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # 1. 创建三维坐标系
        axes = ThreeDAxes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            z_range=[-3, 3]
        )

        # 添加坐标轴标签
        labels = VGroup(
            Text("x", font="SimHei").next_to(axes.x_axis, RIGHT),
            Text("y", font="SimHei").next_to(axes.y_axis, UP),
            Text("z", font="SimHei").next_to(axes.z_axis, OUT)
        )

        # 显示坐标系和标签
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # 2. 在坐标系中添加一个点
        point = Dot3D(
            point=axes.coords_to_point(2, 1, 1),
            color=RED
        )

        # 显示点
        self.play(Create(point))
        self.wait(1)


class ParametricSurface(ThreeDScene):
    def construct(self):
        # 设置相机角度
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # 1. 创建一个球面
        sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(30, 30),
            checkerboard_colors=[BLUE_D, BLUE_E]
        )

        # 显示球面
        self.play(Create(sphere))
        self.wait(1)

        # 旋转相机查看不同角度
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

class ThreeDTransformations(ThreeDScene):
    def construct(self):
        # 设置相机角度
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # 1. 创建一个棱柱
        prism = Prism(dimensions=[2, 1, 1])

        # 显示棱柱
        self.play(Create(prism))
        self.wait(1)

        # 2. 进行一系列变换
        # 缩放
        self.play(
            prism.animate.scale(1.5)
        )
        self.wait(1)

        # 旋转
        self.play(
            Rotate(
                prism,
                angle=PI/2,
                axis=UP
            )
        )
        self.wait(1)

        # 移动
        self.play(
            prism.animate.shift(RIGHT * 2)
        )
        self.wait(1)

# from manim import Text3D

class ThreeDText(ThreeDScene):
    def construct(self):
        # 1. 创建3D文字
        text3d = Text3D(
            "Manim 3D",
            font="SimHei",  # 黑体，解决中文显示问题
            depth=0.5  # 3D文字的厚度（深度）
        )
        text3d.scale(2)  # 放大文字（默认太小）

        # 2. 调整3D视角（必须！否则默认视角看不到3D效果）
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # 3. 显示文字 + 旋转视角（展示3D效果）
        self.play(Create(text3d))
        self.play(text3d.animate.rotate(PI / 2, axis=UP), run_time=3)
        self.wait()

class Complex3DScene(ThreeDScene):
    def construct(self):
        # 设置相机角度
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)

        # 1. 创建坐标系
        axes = ThreeDAxes()

        # 2. 创建一个球体
        sphere = Surface(
            lambda u, v: np.array([
                np.cos(u) * np.cos(v),
                np.cos(u) * np.sin(v),
                np.sin(u)
            ]),
            u_range=[-PI/2, PI/2],
            v_range=[0, TAU],
            resolution=(15, 15)
        )

        # 3. 创建一些点
        points = [
            Dot3D(point=axes.coords_to_point(*pos), color=RED)
            for pos in [(1,1,1), (-1,-1,1), (1,-1,-1)]
        ]

        # 4. 创建连接线
        lines = VGroup(*[
            Line3D(
                start=points[i].get_center(),
                end=points[i-1].get_center(),
                color=YELLOW
            )
            for i in range(len(points))
        ])

        # 5. 添加标签
        labels = VGroup(*[
            Text(f"P{i+1}", font="SimHei").next_to(point, UP)
            for i, point in enumerate(points)
        ])

        # 逐步构建场景
        self.play(Create(axes))
        self.play(Create(sphere))

        for point in points:
            self.play(Create(point))

        self.play(Create(lines))
        self.play(Write(labels))

        # 旋转场景
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()