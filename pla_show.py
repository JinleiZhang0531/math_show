from manim import *
import numpy as np
import math  # 添加 math 模块


class PolynomialVsBezier(Scene):
    def construct(self):
        # ===================== 1. 初始化坐标系 =====================
        # 创建坐标系，设置合适的范围以展示两种曲线
        axes = Axes(
            x_range=[-2, 4, 1],
            y_range=[-3, 5, 1],
            x_length=8,
            y_length=8,
            axis_config={"color": GRAY},
            tips=False
        )
        labels = axes.get_axis_labels(x_label="x/t", y_label="y")

        # 添加网格辅助观察
        # 在 Manim Community 版本中，get_grid 方法可能不支持自定义参数
        # 尝试使用默认参数，如果仍有问题可以移除网格
        try:
            grid = axes.get_grid()
            grid.set_stroke(color=LIGHT_GRAY, opacity=0.3)
        except:
            # 如果 get_grid 方法不可用，创建空网格组
            grid = VGroup()

        # ===================== 2. 定义并绘制5次多项式曲线 =====================
        # 定义5次多项式: y = 0.1x⁵ - 0.5x⁴ + 0.3x³ + 0.8x² - 1.2x + 0.5
        def five_degree_poly(x):
            return 0.1 * x ** 5 - 0.5 * x ** 4 + 0.3 * x ** 3 + 0.8 * x ** 2 - 1.2 * x + 0.5

        # 绘制多项式曲线（移除不支持的 label 参数）
        poly_curve = axes.plot(
            five_degree_poly,
            x_range=[-1.5, 3.5],
            color=BLUE,
            stroke_width=4
        )

        # 添加多项式标签（使用Text而不是get_graph_label，因为包含中文）
        # 在曲线上选择一个点来放置标签
        label_point = axes.coords_to_point(3, five_degree_poly(3))
        poly_label = VGroup(
            Text("5次多项式", font_size=20, color=BLUE),
            MathTex("y=0.1x^5-0.5x^4+0.3x^3+0.8x^2-1.2x+0.5", font_size=16, color=BLUE)
        ).arrange(DOWN, buff=0.1).next_to(label_point, UR, buff=0.3)

        # ===================== 3. 定义并绘制5次贝塞尔曲线 =====================
        # 定义6个控制点（5次贝塞尔需要6个控制点）
        control_points = [
            np.array([-1, 0, 0]),  # P0
            np.array([0, 4, 0]),  # P1
            np.array([1, -2, 0]),  # P2
            np.array([2, 3, 0]),  # P3
            np.array([3, -1, 0]),  # P4
            np.array([4, 2, 0])  # P5
        ]

        # 将控制点转换为坐标系中的点
        control_points_on_axes = [axes.c2p(*p[:2]) for p in control_points]

        # 绘制控制点（带编号）
        control_dots = VGroup()
        control_labels = VGroup()
        for i, point in enumerate(control_points_on_axes):
            dot = Dot(point, color=ORANGE, radius=0.08)
            label = Text(f"P{i}", font_size=14).next_to(dot, RIGHT + UP, buff=0.1)
            control_dots.add(dot)
            control_labels.add(label)

        # 绘制控制点连线（凸包）
        convex_hull = Polygon(*control_points_on_axes, color=ORANGE, stroke_width=2, stroke_opacity=0.5, fill_opacity=0)

        # 定义5次贝塞尔曲线的参数方程
        def bezier_5(t, points):
            n = 5  # 5次贝塞尔
            result = np.array([0.0, 0.0, 0.0])
            for i in range(n + 1):
                # 伯恩斯坦基函数: C(5,i) * t^i * (1-t)^(5-i)
                # 使用 math.comb 替代 np.math.comb
                bernstein = math.comb(5, i) * (t ** i) * ((1 - t) ** (5 - i))
                result += bernstein * points[i]
            return result

        # 绘制贝塞尔曲线
        bezier_curve = VMobject(color=GREEN, stroke_width=4)
        bezier_points = [axes.c2p(*bezier_5(t, control_points)[:2]) for t in np.linspace(0, 1, 200)]
        bezier_curve.set_points_as_corners(bezier_points)

        # 添加贝塞尔曲线标签
        bezier_label = Text("5次贝塞尔曲线", font_size=20, color=GREEN).next_to(axes.c2p(2.5, 4), UP)

        # ===================== 4. 动画展示 =====================
        # 先显示坐标系
        self.play(Create(axes), Write(labels), Create(grid), run_time=1)

        # 绘制多项式曲线（移除不支持的 label 参数）
        self.play(Create(poly_curve), Write(poly_label), run_time=2)
        self.wait(1)

        # 绘制贝塞尔曲线相关元素
        self.play(Create(convex_hull), run_time=1)
        self.play(Create(control_dots), Write(control_labels), run_time=1)
        self.play(Create(bezier_curve), Write(bezier_label), run_time=2)

        # 停留展示
        self.wait(3)


class LaneChangeTrajectory(Scene):
    """
    自动驾驶车辆换道轨迹对比：
    5次多项式 vs 5次贝塞尔曲线
    """
    def construct(self):
        # ===================== 1. 创建道路场景 =====================
        # 坐标系：x轴表示纵向距离（前进方向），y轴表示横向位移（换道距离）
        axes = Axes(
            x_range=[0, 100, 10],  # 纵向距离 0-100米
            y_range=[-2, 6, 1],   # 横向位移：-2到6米（假设车道宽度3.5米）
            x_length=10,
            y_length=6,
            axis_config={"color": GRAY, "include_numbers": True},
            tips=False
        )
        
        # 添加轴标签
        labels = axes.get_axis_labels(
            x_label=Text("纵向距离 (m)", font_size=20),
            y_label=Text("横向位移 (m)", font_size=20)
        )
        
        # 绘制车道线
        lane_width = 3.5  # 车道宽度3.5米
        lane1_line = axes.plot(lambda x: 0, x_range=[0, 100], color=YELLOW, stroke_width=2)
        lane2_line = axes.plot(lambda x: lane_width, x_range=[0, 100], color=YELLOW, stroke_width=2)
        lane3_line = axes.plot(lambda x: lane_width * 2, x_range=[0, 100], color=YELLOW, stroke_width=2)
        
        # 车道标签
        lane1_label = Text("车道1", font_size=16, color=YELLOW).next_to(axes.c2p(5, lane_width/2), LEFT)
        lane2_label = Text("车道2", font_size=16, color=YELLOW).next_to(axes.c2p(5, lane_width*1.5), LEFT)
        
        # ===================== 2. 5次多项式换道轨迹 =====================
        # 边界条件：
        # 起始点 (0, 0): 位置y=0, 速度dy/dx=0, 加速度d²y/dx²=0
        # 终点 (100, 3.5): 位置y=3.5, 速度dy/dx=0, 加速度d²y/dx²=0
        # 5次多项式: y = a0 + a1*x + a2*x² + a3*x³ + a4*x⁴ + a5*x⁵
        
        def poly_lane_change(x):
            # 归一化到 [0, 1]
            t = x / 100.0
            # 使用5次多项式实现平滑换道
            # 满足边界条件：t=0时y=0, dy/dt=0, d²y/dt²=0
            #           t=1时y=1, dy/dt=0, d²y/dt²=0
            # 解：y = 6t⁵ - 15t⁴ + 10t³
            y_normalized = 6 * t**5 - 15 * t**4 + 10 * t**3
            return y_normalized * lane_width  # 缩放到实际车道宽度
        
        poly_trajectory = axes.plot(
            poly_lane_change,
            x_range=[0, 100],
            color=BLUE,
            stroke_width=4
        )
        
        # ===================== 3. 5次贝塞尔曲线换道轨迹 =====================
        # 6个控制点的设置策略（针对换道场景）：
        # P0: 起始位置 (0, 0) - 当前车道中心
        # P1: 起始方向控制点 - 保持与P0相同高度，稍微前移，确保起始速度为0
        # P2: 加速阶段控制点 - 稍微向下，控制加速度
        # P3: 中间控制点 - 在换道中点附近
        # P4: 减速阶段控制点 - 接近目标车道
        # P5: 终点位置 (100, 3.5) - 目标车道中心
        
        control_points_bezier = [
            np.array([0, 0, 0]),           # P0: 起始点
            np.array([5, 0, 0]),          # P1: 起始方向（保持水平，确保初始速度为0）
            np.array([20, -0.2, 0]),      # P2: 轻微下压，控制初始加速度
            np.array([50, lane_width/2, 0]),  # P3: 换道中点
            np.array([80, lane_width + 0.2, 0]),  # P4: 接近目标车道
            np.array([100, lane_width, 0])  # P5: 终点（目标车道中心）
        ]
        
        # 贝塞尔曲线函数
        def bezier_5(t, points):
            n = 5
            result = np.array([0.0, 0.0, 0.0])
            for i in range(n + 1):
                bernstein = math.comb(5, i) * (t ** i) * ((1 - t) ** (5 - i))
                result += bernstein * points[i]
            return result
        
        # 绘制贝塞尔曲线轨迹
        bezier_trajectory = VMobject(color=GREEN, stroke_width=4)
        bezier_points_list = [axes.c2p(*bezier_5(t, control_points_bezier)[:2]) 
                              for t in np.linspace(0, 1, 200)]
        bezier_trajectory.set_points_as_corners(bezier_points_list)
        
        # 绘制控制点
        control_dots = VGroup()
        control_labels = VGroup()
        for i, point in enumerate(control_points_bezier):
            dot_pos = axes.c2p(*point[:2])
            dot = Dot(dot_pos, color=ORANGE, radius=0.1)
            label = Text(f"P{i}", font_size=12, color=ORANGE).next_to(dot, UR, buff=0.05)
            control_dots.add(dot)
            control_labels.add(label)
        
        # ===================== 4. 计算曲率对比 =====================
        # 曲率越小，方向盘转动越平顺
        
        # 多项式曲率计算（简化版，展示关键点）
        def poly_curvature_sample(x):
            """计算多项式在x处的近似曲率"""
            h = 0.1
            y1 = poly_lane_change(x - h)
            y0 = poly_lane_change(x)
            y2 = poly_lane_change(x + h)
            # 曲率 ≈ |d²y/dx²| / (1 + (dy/dx)²)^(3/2)
            d2y = (y2 - 2*y0 + y1) / (h**2)
            dy = (y2 - y1) / (2*h)
            curvature = abs(d2y) / ((1 + dy**2)**1.5)
            return curvature
        
        # 贝塞尔曲线曲率计算
        def bezier_curvature_sample(t):
            """计算贝塞尔曲线在t处的曲率"""
            h = 0.01
            p1 = bezier_5(t - h, control_points_bezier)
            p0 = bezier_5(t, control_points_bezier)
            p2 = bezier_5(t + h, control_points_bezier)
            # 一阶导数
            dp = (p2 - p1) / (2*h)
            # 二阶导数
            d2p = (p2 - 2*p0 + p1) / (h**2)
            # 曲率
            cross = np.cross(dp[:2], d2p[:2])
            curvature = abs(cross) / (np.linalg.norm(dp[:2])**3)
            return curvature
        
        # ===================== 5. 动画展示 =====================
        # 显示道路场景
        self.play(
            Create(axes), 
            Write(labels),
            Create(lane1_line),
            Create(lane2_line),
            Create(lane3_line),
            Write(lane1_label),
            Write(lane2_label),
            run_time=2
        )
        self.wait(1)
        
        # 显示5次多项式轨迹
        poly_title = Text("5次多项式换道轨迹", font_size=24, color=BLUE).to_edge(UP)
        self.play(Write(poly_title))
        self.play(Create(poly_trajectory), run_time=2)
        
        # 添加说明
        poly_note = VGroup(
            Text("优点：", font_size=16, color=BLUE),
            Text("• 数学形式简单", font_size=14, color=WHITE),
            Text("• 易于约束边界条件", font_size=14, color=WHITE),
            Text("缺点：", font_size=16, color=BLUE),
            Text("• 曲率变化可能不均匀", font_size=14, color=WHITE),
            Text("• 中间段可能过度弯曲", font_size=14, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(DR, buff=0.5)
        
        self.play(Write(poly_note), run_time=2)
        self.wait(2)
        
        # 清除多项式相关
        self.play(
            FadeOut(poly_trajectory),
            FadeOut(poly_title),
            FadeOut(poly_note)
        )
        
        # 显示5次贝塞尔曲线轨迹
        bezier_title = Text("5次贝塞尔曲线换道轨迹", font_size=24, color=GREEN).to_edge(UP)
        self.play(Write(bezier_title))
        
        # 先显示控制点
        self.play(Create(control_dots), Write(control_labels), run_time=1.5)
        self.wait(0.5)
        
        # 显示贝塞尔曲线
        self.play(Create(bezier_trajectory), run_time=2)
        
        # 添加说明
        bezier_note = VGroup(
            Text("优点：", font_size=16, color=GREEN),
            Text("• 曲率变化更平滑", font_size=14, color=WHITE),
            Text("• 方向盘转动更自然", font_size=14, color=WHITE),
            Text("• 通过控制点直观调整", font_size=14, color=WHITE),
            Text("控制点设置策略：", font_size=16, color=GREEN),
            Text("P0,P1: 起始位置和方向", font_size=14, color=WHITE),
            Text("P2: 控制初始加速度", font_size=14, color=WHITE),
            Text("P3: 换道中点", font_size=14, color=WHITE),
            Text("P4,P5: 结束位置和方向", font_size=14, color=WHITE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).to_corner(DR, buff=0.3).scale(0.8)
        
        self.play(Write(bezier_note), run_time=2)
        self.wait(3)
        
        # 对比总结
        self.play(
            FadeOut(bezier_title),
            FadeOut(bezier_note),
            FadeOut(control_dots),
            FadeOut(control_labels)
        )
        
        summary = VGroup(
            Text("换道轨迹对比总结", font_size=28, color=YELLOW),
            Text("", font_size=10),  # 空行
            Text("5次多项式：", font_size=20, color=BLUE),
            Text("适合简单场景，计算快速", font_size=16, color=WHITE),
            Text("", font_size=10),
            Text("5次贝塞尔曲线：", font_size=20, color=GREEN),
            Text("更适合实际应用，方向盘更平顺", font_size=16, color=WHITE),
            Text("推荐使用贝塞尔曲线", font_size=18, color=YELLOW)
        ).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        
        self.play(Write(summary), run_time=3)
        self.wait(3)


class TrajectoryComparisonWithCurvature(Scene):
    """
    同时显示5次多项式和5次贝塞尔曲线换道轨迹，并对比曲率曲线
    包含P2、P3、P4控制点的约束和经验推荐
    """
    def construct(self):
        # ===================== 1. 创建道路场景 =====================
        # 调整轨迹坐标系位置，曲率图放在上方
        axes = Axes(
            x_range=[0, 100, 10],
            y_range=[-2, 6, 1],
            x_length=9,
            y_length=4.5,
            axis_config={"color": GRAY, "include_numbers": True, "font_size": 16},
            tips=False
        ).shift(DOWN * 1.5)  # 向下移动，为曲率图留空间
        
        labels = axes.get_axis_labels(
            x_label=Text("纵向距离 (m)", font_size=18),
            y_label=Text("横向位移 (m)", font_size=18)
        )
        
        # 绘制车道线
        lane_width = 3.5
        lane1_line = axes.plot(lambda x: 0, x_range=[0, 100], color=YELLOW, stroke_width=2)
        lane2_line = axes.plot(lambda x: lane_width, x_range=[0, 100], color=YELLOW, stroke_width=2)
        lane3_line = axes.plot(lambda x: lane_width * 2, x_range=[0, 100], color=YELLOW, stroke_width=2)
        
        # ===================== 2. 5次多项式轨迹 =====================
        def poly_lane_change(x):
            t = x / 100.0
            y_normalized = 6 * t**5 - 15 * t**4 + 10 * t**3
            return y_normalized * lane_width
        
        poly_trajectory = axes.plot(
            poly_lane_change,
            x_range=[0, 100],
            color=BLUE,
            stroke_width=4
        )
        
        # ===================== 3. 5次贝塞尔曲线轨迹 =====================
        # 优化的控制点设置：确保起点和终点曲率为0（更平顺）
        # 关键约束：
        # 1. P0, P1, P2 共线（水平）→ 确保起点曲率为0
        # 2. P3, P4, P5 共线（水平）→ 确保终点曲率为0
        # 3. P2到P3之间平滑过渡，控制换道过程
        # 优化控制点：确保起点终点曲率接近0，同时整体曲率更小
        # 策略：P0-P1共线确保起点曲率为0，P2提前开始平滑转向（避免突然转向）
        #      P4-P5共线确保终点曲率为0，P3-P4之间平滑过渡
        control_points_bezier = [
            np.array([0, 0, 0]),           # P0: 起始点（当前车道中心）
            np.array([12, 0, 0]),          # P1: 与P0共线，确保起点曲率为0
            np.array([28, 0.3, 0]),        # P2: 提前开始平滑转向，避免突然转向
            np.array([50, lane_width/2 + 0.1, 0]),  # P3: 换道中点（稍微上移，使曲线更平滑）
            np.array([72, lane_width - 0.3, 0]),  # P4: 提前平缓，与P5接近共线
            np.array([100, lane_width, 0])  # P5: 终点，与P4接近共线（终点曲率接近0）
        ]
        
        def bezier_5(t, points):
            n = 5
            result = np.array([0.0, 0.0, 0.0])
            for i in range(n + 1):
                bernstein = math.comb(5, i) * (t ** i) * ((1 - t) ** (5 - i))
                result += bernstein * points[i]
            return result
        
        bezier_trajectory = VMobject(color=GREEN, stroke_width=4)
        bezier_points_list = [axes.c2p(*bezier_5(t, control_points_bezier)[:2]) 
                              for t in np.linspace(0, 1, 200)]
        bezier_trajectory.set_points_as_corners(bezier_points_list)
        
        # ===================== 3.5. 3次多项式轨迹 =====================
        # 边界条件：
        # 起点 (0, 0): 位置y=0, PSI=0 (dy/dx=0)
        # 终点 (100, 3.5): 位置y=3.5, PSI=0 (dy/dx=0)
        # 3次多项式: y = a0 + a1*x + a2*x² + a3*x³
        # 由边界条件解得: y = 3.5 * (3*(x/100)² - 2*(x/100)³)
        def poly3_lane_change(x):
            t = x / 100.0
            # 3次多项式，满足起点终点PSI=0
            y_normalized = 3 * t**2 - 2 * t**3
            return y_normalized * lane_width
        
        poly3_trajectory = axes.plot(
            poly3_lane_change,
            x_range=[0, 100],
            color=RED,
            stroke_width=4
        )
        
        # ===================== 4. 计算曲率 =====================
        def calculate_poly_curvature(x):
            """计算5次多项式轨迹的曲率（使用解析导数）"""
            t = x / 100.0
            # 5次多项式: y = 3.5 * (6*t^5 - 15*t^4 + 10*t^3)
            # 一阶导数: dy/dx = dy/dt * dt/dx = dy/dt / 100
            # dy/dt = 3.5 * (30*t^4 - 60*t^3 + 30*t^2) = 3.5 * 30 * t^2 * (t^2 - 2*t + 1)
            #      = 105 * t^2 * (1-t)^2
            dy_dt = 105 * t**2 * (1-t)**2
            dy_dx = dy_dt / 100.0
            
            # 二阶导数: d²y/dx² = d(dy/dt)/dt * (dt/dx)² = d²y/dt² / 100²
            # d²y/dt² = 3.5 * (120*t^3 - 180*t^2 + 60*t) = 3.5 * 60 * t * (2*t^2 - 3*t + 1)
            #         = 210 * t * (2*t^2 - 3*t + 1)
            d2y_dt2 = 210 * t * (2*t**2 - 3*t + 1)
            d2y_dx2 = d2y_dt2 / (100.0**2)
            
            # 曲率公式: κ = |d²y/dx²| / (1 + (dy/dx)²)^(3/2)
            curvature = abs(d2y_dx2) / ((1 + dy_dx**2)**1.5)
            return curvature
        
        def bezier_derivative_1(t, points):
            """计算5次贝塞尔曲线的一阶导数（解析解）"""
            n = 5
            result = np.array([0.0, 0.0, 0.0])
            for i in range(n):
                # 一阶导数的伯恩斯坦基函数
                bernstein = n * math.comb(n-1, i) * (t ** i) * ((1 - t) ** (n-1-i))
                result += bernstein * (points[i+1] - points[i])
            return result
        
        def bezier_derivative_2(t, points):
            """计算5次贝塞尔曲线的二阶导数（解析解）"""
            n = 5
            result = np.array([0.0, 0.0, 0.0])
            for i in range(n-1):
                # 二阶导数的伯恩斯坦基函数
                bernstein = n * (n-1) * math.comb(n-2, i) * (t ** i) * ((1 - t) ** (n-2-i))
                result += bernstein * (points[i+2] - 2*points[i+1] + points[i])
            return result
        
        def calculate_poly3_curvature(x):
            """计算3次多项式轨迹的曲率（使用解析导数）"""
            t = x / 100.0
            # 3次多项式: y = 3.5 * (3*t^2 - 2*t^3)
            # 一阶导数: dy/dx = dy/dt * dt/dx = dy/dt / 100
            # dy/dt = 3.5 * (6*t - 6*t^2) = 21 * t * (1-t)
            dy_dt = 21 * t * (1 - t)
            dy_dx = dy_dt / 100.0
            
            # 二阶导数: d²y/dx² = d²y/dt² / 100²
            # d²y/dt² = 3.5 * (6 - 12*t) = 21 * (1 - 2*t)
            d2y_dt2 = 21 * (1 - 2*t)
            d2y_dx2 = d2y_dt2 / (100.0**2)
            
            # 曲率公式: κ = |d²y/dx²| / (1 + (dy/dx)²)^(3/2)
            curvature = abs(d2y_dx2) / ((1 + dy_dx**2)**1.5)
            return curvature
        
        def calculate_bezier_curvature(t):
            """计算5次贝塞尔曲线轨迹的曲率（使用解析导数）"""
            if t < 0:
                t = 0
            if t > 1:
                t = 1
            # 计算一阶和二阶导数
            dp = bezier_derivative_1(t, control_points_bezier)
            d2p = bezier_derivative_2(t, control_points_bezier)
            # 曲率公式: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
            x_prime = dp[0]
            y_prime = dp[1]
            x_double_prime = d2p[0]
            y_double_prime = d2p[1]
            # 计算分母
            speed_squared = x_prime**2 + y_prime**2
            if speed_squared < 1e-10:
                return 0
            # 计算分子：叉积的绝对值
            cross_product = abs(x_prime * y_double_prime - y_prime * x_double_prime)
            # 曲率
            curvature = cross_product / (speed_squared ** 1.5)
            return curvature
        
        # 生成曲率数据点 - 覆盖完整的0-100m范围
        # 5次多项式：直接按x采样
        x_samples = np.linspace(0, 100, 400)  # 覆盖完整0-100m范围，增加采样密度
        poly_curvatures = []
        poly_x_valid = []
        for x in x_samples:
            try:
                curv = calculate_poly_curvature(x)
                poly_curvatures.append(curv)
                poly_x_valid.append(x)
            except:
                pass
        
        # 3次多项式：直接按x采样
        poly3_curvatures = []
        poly3_x_valid = []
        for x in x_samples:
            try:
                curv = calculate_poly3_curvature(x)
                poly3_curvatures.append(curv)
                poly3_x_valid.append(x)
            except:
                pass
        
        # 5次贝塞尔曲线：按t采样，然后转换为x坐标
        t_samples = np.linspace(0, 1, 400)  # 覆盖完整t范围[0,1]
        bezier_curvatures = []
        bezier_x_samples = []
        for t in t_samples:
            try:
                curv = calculate_bezier_curvature(t)
                point = bezier_5(t, control_points_bezier)
                x_coord = point[0]
                if 0 <= x_coord <= 100:  # 只保留在有效范围内的点
                    bezier_curvatures.append(curv)
                    bezier_x_samples.append(x_coord)
            except:
                pass
        
        # 找到实际的最大和最小曲率值（包含所有三种曲线）
        max_poly_curvature = max(poly_curvatures) if poly_curvatures else 0
        max_poly3_curvature = max(poly3_curvatures) if poly3_curvatures else 0
        max_bezier_curvature = max(bezier_curvatures) if bezier_curvatures else 0
        max_curvature = max(max_poly_curvature, max_poly3_curvature, max_bezier_curvature)
        min_curvature = min(min(poly_curvatures) if poly_curvatures else 0,
                           min(poly3_curvatures) if poly3_curvatures else 0,
                           min(bezier_curvatures) if bezier_curvatures else 0)
        
        # 计算合适的y轴范围（不对曲率大小进行限制）
        # 将曲率值转换为显示单位（×10⁻³）
        curvature_display_max = max_curvature * 1000  # 转换为×10⁻³单位
        curvature_display_min = min_curvature * 1000
        
        # 设置y轴范围，自动适应实际曲率值，留出一些边距
        curvature_range = curvature_display_max - curvature_display_min
        if curvature_range < 1e-10:
            y_max_rounded = max(0.1, curvature_display_max * 1.2) if curvature_display_max > 0 else 0.1
            y_min_rounded = 0
        else:
            y_max_rounded = curvature_display_max * 1.1  # 留出10%的上边距
            y_min_rounded = max(0, curvature_display_min - curvature_range * 0.1)  # 留出10%的下边距
        
        # 设置y轴刻度，自动调整，并确保刻度值精确到小数点后两位
        y_tick_step = max(0.01, (y_max_rounded - y_min_rounded) / 8)  # 大约8个刻度
        # 将步长四舍五入到合适的值（如0.01, 0.02, 0.05, 0.1等）
        if y_tick_step < 0.05:
            y_tick_step = round(y_tick_step * 100) / 100  # 精确到0.01
        elif y_tick_step < 0.1:
            y_tick_step = round(y_tick_step * 20) / 20  # 精确到0.05
        else:
            y_tick_step = round(y_tick_step * 10) / 10  # 精确到0.1
        
        y_max_rounded = np.ceil(y_max_rounded / y_tick_step) * y_tick_step
        y_min_rounded = np.floor(y_min_rounded / y_tick_step) * y_tick_step
        
        # 创建曲率坐标系，配置y轴数字格式为两位小数
        curvature_axes = Axes(
            x_range=[0, 100, 10],
            y_range=[y_min_rounded, y_max_rounded, y_tick_step],
            x_length=9,
            y_length=3.5,
            axis_config={"color": GRAY, "include_numbers": True, "font_size": 16},
            y_axis_config={
                "include_numbers": True,
                "font_size": 16,
                "decimal_number_config": {
                    "num_decimal_places": 2
                }
            },
            tips=False
        ).shift(UP * 2.5)  # 调整位置到上方，避免与轨迹图重叠
        
        # 手动格式化y轴数字为两位小数
        # 获取y轴上的所有数字并替换为两位小数格式
        try:
            y_axis = curvature_axes.y_axis
            if hasattr(y_axis, 'numbers') and len(y_axis.numbers) > 0:
                new_numbers = VGroup()
                for number in y_axis.numbers:
                    # 尝试从数字对象中提取数值
                    try:
                        # 尝试多种方式获取数值
                        if hasattr(number, 'number'):
                            value = number.number
                        elif hasattr(number, 'get_value'):
                            value = number.get_value()
                        else:
                            # 尝试从文本中解析
                            text = str(number)
                            # 移除可能的LaTeX标记
                            text = text.replace('$', '').strip()
                            value = float(text)
                        
                        # 创建新的格式化数字
                        formatted_text = f"{value:.2f}"
                        new_number = MathTex(formatted_text, font_size=16, color=GRAY)
                        new_number.move_to(number.get_center())
                        new_numbers.add(new_number)
                    except:
                        # 如果解析失败，尝试直接格式化文本
                        try:
                            text = str(number)
                            # 尝试提取数字
                            import re
                            match = re.search(r'[-+]?\d*\.?\d+', text)
                            if match:
                                value = float(match.group())
                                formatted_text = f"{value:.2f}"
                                new_number = MathTex(formatted_text, font_size=16, color=GRAY)
                                new_number.move_to(number.get_center())
                                new_numbers.add(new_number)
                            else:
                                new_numbers.add(number)
                        except:
                            # 如果都失败，保留原数字
                            new_numbers.add(number)
                # 替换y轴的数字
                y_axis.numbers = new_numbers
        except Exception as e:
            # 如果格式化失败，使用默认格式
            pass
        
        curvature_labels = curvature_axes.get_axis_labels(
            x_label=Text("纵向距离 (m)", font_size=18),
            y_label=Text("曲率 (×10⁻³)", font_size=18)
        )
        
        # 绘制曲率曲线 - 将曲率值转换为×10⁻³单位显示（不进行缩放限制）
        poly_curvature_curve = VMobject(color=BLUE, stroke_width=3)
        poly_curvature_points = [curvature_axes.c2p(x, c * 1000) 
                                 for x, c in zip(poly_x_valid, poly_curvatures)
                                 if 0 <= x <= 100]  # 确保x在有效范围内
        if len(poly_curvature_points) > 0:
            poly_curvature_curve.set_points_as_corners(poly_curvature_points)
        
        bezier_curvature_curve = VMobject(color=GREEN, stroke_width=3)
        bezier_curvature_points = [curvature_axes.c2p(x, c * 1000) 
                                   for x, c in zip(bezier_x_samples, bezier_curvatures)
                                   if 0 <= x <= 100]  # 确保x在有效范围内
        if len(bezier_curvature_points) > 0:
            bezier_curvature_curve.set_points_as_corners(bezier_curvature_points)
        
        poly3_curvature_curve = VMobject(color=RED, stroke_width=3)
        poly3_curvature_points = [curvature_axes.c2p(x, c * 1000) 
                                   for x, c in zip(poly3_x_valid, poly3_curvatures)
                                   if 0 <= x <= 100]  # 确保x在有效范围内
        if len(poly3_curvature_points) > 0:
            poly3_curvature_curve.set_points_as_corners(poly3_curvature_points)
        
        # ===================== 6. 动画展示 =====================
        # 显示标题
        title = Text("换道轨迹与曲率对比", font_size=24, color=YELLOW).to_edge(UP, buff=0.1)
        self.play(Write(title), run_time=1)
        
        # 先显示曲率坐标系（在上方）
        self.play(
            Create(curvature_axes),
            Write(curvature_labels),
            run_time=1.5
        )
        self.wait(0.5)
        
        # 显示曲率曲线
        # 添加曲率曲线图例（根据实际y轴范围动态调整位置）
        label_y_pos = y_max_rounded * 0.85  # 标签位置在y轴的85%处
        poly_curvature_label = Text("5次多项式曲率", font_size=14, color=BLUE).next_to(
            curvature_axes.c2p(15, label_y_pos), RIGHT, buff=0.2
        )
        bezier_curvature_label = Text("5次贝塞尔曲率", font_size=14, color=GREEN).next_to(
            curvature_axes.c2p(15, label_y_pos - y_max_rounded * 0.1), RIGHT, buff=0.2
        )
        poly3_curvature_label = Text("3次多项式曲率", font_size=14, color=RED).next_to(
            curvature_axes.c2p(15, label_y_pos - y_max_rounded * 0.2), RIGHT, buff=0.2
        )
        
        self.play(
            Create(poly_curvature_curve),
            Create(bezier_curvature_curve),
            Create(poly3_curvature_curve),
            Write(poly_curvature_label),
            Write(bezier_curvature_label),
            Write(poly3_curvature_label),
            run_time=2
        )
        self.wait(0.5)
        
        # 显示道路场景（在下方）
        self.play(
            Create(axes),
            Write(labels),
            Create(lane1_line),
            Create(lane2_line),
            Create(lane3_line),
            run_time=2
        )
        self.wait(0.5)
        
        # 同时显示三条轨迹
        # 调整标签位置，避免遮挡
        poly_label = Text("5次多项式", font_size=16, color=BLUE).next_to(axes.c2p(85, 5.5), UR, buff=0.2)
        bezier_label = Text("5次贝塞尔曲线", font_size=16, color=GREEN).next_to(axes.c2p(85, 4.8), UR, buff=0.2)
        poly3_label = Text("3次多项式", font_size=16, color=RED).next_to(axes.c2p(85, 4.1), UR, buff=0.2)
        
        self.play(
            Create(poly_trajectory),
            Create(bezier_trajectory),
            Create(poly3_trajectory),
            Write(poly_label),
            Write(bezier_label),
            Write(poly3_label),
            run_time=2
        )
        self.wait(1)
        
        # ===================== 7. 显示P2、P3、P4的约束说明 =====================
        # 调整说明位置，避免遮挡曲率图
        constraint_note = VGroup(
            Text("控制点优化设置", font_size=16, color=YELLOW),
            Text("", font_size=6),
            Text("P0-P1: 共线（y=0）", font_size=12, color=GREEN),
            Text("  起点曲率≈0", font_size=11, color=GRAY),
            Text("", font_size=4),
            Text("P2: 提前转向", font_size=12, color=GREEN),
            Text("  x≈25~30m, y≈0.2~0.3m", font_size=11, color=GRAY),
            Text("  避免突然转向", font_size=11, color=GRAY),
            Text("", font_size=4),
            Text("P3: 换道中点", font_size=12, color=GREEN),
            Text("  x≈50m, y≈1.8~1.9m", font_size=11, color=GRAY),
            Text("", font_size=4),
            Text("P4-P5: 接近共线", font_size=12, color=GREEN),
            Text("  终点曲率≈0", font_size=11, color=GRAY)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).scale(0.65).to_corner(UR, buff=0.2)
        
        self.play(Write(constraint_note), run_time=3)
        self.wait(2)
        
        # 高亮显示关键控制点
        control_dots = VGroup()
        control_lines = VGroup()
        
        # 起点：P0, P1（共线）
        for i in [0, 1]:
            dot_pos = axes.c2p(*control_points_bezier[i][:2])
            dot = Dot(dot_pos, color=YELLOW, radius=0.12)
            label = Text(f"P{i}", font_size=12, color=YELLOW, weight=BOLD).next_to(dot, UP, buff=0.08)
            control_dots.add(VGroup(dot, label))
        
        # 关键控制点：P2（提前转向）
        dot_pos = axes.c2p(*control_points_bezier[2][:2])
        dot = Dot(dot_pos, color=RED, radius=0.15)
        label = Text("P2", font_size=14, color=RED, weight=BOLD).next_to(dot, UR, buff=0.1)
        control_dots.add(VGroup(dot, label))
        
        # 换道中点：P3
        dot_pos = axes.c2p(*control_points_bezier[3][:2])
        dot = Dot(dot_pos, color=RED, radius=0.15)
        label = Text("P3", font_size=14, color=RED, weight=BOLD).next_to(dot, UR, buff=0.1)
        control_dots.add(VGroup(dot, label))
        
        # 关键控制点：P4（提前平缓）
        dot_pos = axes.c2p(*control_points_bezier[4][:2])
        dot = Dot(dot_pos, color=RED, radius=0.15)
        label = Text("P4", font_size=14, color=RED, weight=BOLD).next_to(dot, DOWN, buff=0.1)
        control_dots.add(VGroup(dot, label))
        
        # 终点：P5
        dot_pos = axes.c2p(*control_points_bezier[5][:2])
        dot = Dot(dot_pos, color=YELLOW, radius=0.12)
        label = Text("P5", font_size=12, color=YELLOW, weight=BOLD).next_to(dot, DOWN, buff=0.08)
        control_dots.add(VGroup(dot, label))
        
        # 绘制共线辅助线
        line_start = Line(
            axes.c2p(*control_points_bezier[0][:2]),
            axes.c2p(*control_points_bezier[1][:2]),
            color=YELLOW, stroke_width=1, stroke_opacity=0.3
        )
        line_end = Line(
            axes.c2p(*control_points_bezier[4][:2]),
            axes.c2p(*control_points_bezier[5][:2]),
            color=YELLOW, stroke_width=1, stroke_opacity=0.3
        )
        control_lines.add(line_start, line_end)
        
        self.play(
            Create(control_lines),
            *[Create(dot) for dot in control_dots],
            run_time=2
        )
        self.wait(2)
        
        # 曲率分析说明 - 放在曲率图右侧，避免遮挡
        # 计算实际曲率统计（使用原始值，不缩放）
        poly_avg = np.mean(poly_curvatures) if poly_curvatures else 0
        bezier_avg = np.mean(bezier_curvatures) if bezier_curvatures else 0
        poly3_avg = np.mean(poly3_curvatures) if poly3_curvatures else 0
        poly_max = max_poly_curvature
        bezier_max = max_bezier_curvature
        poly3_max = max_poly3_curvature
        
        curvature_legend = VGroup(
            Text("曲率对比分析:", font_size=16, color=YELLOW),
            Text("", font_size=6),
            Text("蓝色: 5次多项式", font_size=12, color=BLUE),
            Text(f"  最大: {poly_max*1000:.3f}×10⁻³", font_size=11, color=GRAY),
            Text(f"  平均: {poly_avg*1000:.3f}×10⁻³", font_size=11, color=GRAY),
            Text("", font_size=4),
            Text("绿色: 5次贝塞尔", font_size=12, color=GREEN),
            Text(f"  最大: {bezier_max*1000:.3f}×10⁻³", font_size=11, color=GREEN),
            Text(f"  平均: {bezier_avg*1000:.3f}×10⁻³", font_size=11, color=GREEN),
            Text("", font_size=4),
            Text("红色: 3次多项式", font_size=12, color=RED),
            Text(f"  最大: {poly3_max*1000:.3f}×10⁻³", font_size=11, color=RED),
            Text(f"  平均: {poly3_avg*1000:.3f}×10⁻³", font_size=11, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).scale(0.65).next_to(curvature_axes, RIGHT, buff=0.5)
        
        self.play(Write(curvature_legend), run_time=1.5)
        self.wait(2)
        
        # 最终总结 - 保留曲率图，只移除说明文字
        self.play(
            FadeOut(constraint_note),
            FadeOut(curvature_legend),
            FadeOut(control_dots),
            FadeOut(control_lines),
            FadeOut(poly_curvature_label),
            FadeOut(bezier_curvature_label),
            FadeOut(poly3_curvature_label)
        )
        
        summary = VGroup(
            Text("总结", font_size=24, color=YELLOW),
            Text("", font_size=8),
            Text("推荐使用5次贝塞尔曲线", font_size=18, color=GREEN),
            Text("• 曲率变化更平滑", font_size=14, color=WHITE),
            Text("• 方向盘转动更自然", font_size=14, color=WHITE),
            Text("• 通过控制点P2、P3、P4可精确调整", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.15).next_to(curvature_axes, DOWN, buff=0.5)
        
        self.play(Write(summary), run_time=2)
        self.wait(3)
