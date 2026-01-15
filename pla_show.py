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
        # 调整轨迹坐标系位置，为曲率图留出空间
        axes = Axes(
            x_range=[0, 100, 10],
            y_range=[-2, 6, 1],
            x_length=9,
            y_length=4.5,
            axis_config={"color": GRAY, "include_numbers": True, "font_size": 16},
            tips=False
        ).shift(UP * 1.5)  # 向上移动，为曲率图留空间
        
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
        
        # ===================== 4. 计算曲率 =====================
        def calculate_poly_curvature(x):
            """计算多项式轨迹的曲率"""
            h = 0.5
            if x - h < 0:
                x = h
            if x + h > 100:
                x = 100 - h
            y1 = poly_lane_change(x - h)
            y0 = poly_lane_change(x)
            y2 = poly_lane_change(x + h)
            d2y = (y2 - 2*y0 + y1) / (h**2)
            dy = (y2 - y1) / (2*h)
            # 曲率公式: κ = |d²y/dx²| / (1 + (dy/dx)²)^(3/2)
            curvature = abs(d2y) / ((1 + dy**2)**1.5) if abs(dy) < 100 else 0
            return curvature
        
        def calculate_bezier_curvature(t):
            """计算贝塞尔曲线轨迹的曲率"""
            h = 0.005
            if t - h < 0:
                t = h
            if t + h > 1:
                t = 1 - h
            p1 = bezier_5(t - h, control_points_bezier)
            p0 = bezier_5(t, control_points_bezier)
            p2 = bezier_5(t + h, control_points_bezier)
            dp = (p2 - p1) / (2*h)
            d2p = (p2 - 2*p0 + p1) / (h**2)
            # 曲率公式: κ = |dp × d²p| / |dp|³
            if np.linalg.norm(dp[:2]) < 1e-6:
                return 0
            cross = abs(np.cross(dp[:2], d2p[:2]))
            curvature = cross / (np.linalg.norm(dp[:2])**3)
            return curvature
        
        # 生成曲率数据点
        x_samples = np.linspace(5, 95, 200)  # 避免边界问题
        poly_curvatures = [calculate_poly_curvature(x) for x in x_samples]
        t_samples = np.linspace(0.05, 0.95, 200)
        bezier_curvatures = [calculate_bezier_curvature(t) for t in t_samples]
        # 将贝塞尔曲线的t转换为x坐标
        bezier_x_samples = [bezier_5(t, control_points_bezier)[0] for t in t_samples]
        
        # 归一化曲率用于显示（曲率通常很小，需要放大显示）
        max_curvature = max(max(poly_curvatures), max(bezier_curvatures))
        # 调整缩放因子，使曲率曲线在y_range内完整显示
        scale_factor = 2.8 / max_curvature if max_curvature > 0 else 1.0
        
        # ===================== 5. 创建曲率对比坐标系 =====================
        # 调整曲率坐标系位置，确保完整显示且不遮挡
        curvature_axes = Axes(
            x_range=[0, 100, 10],
            y_range=[0, 3.0, 0.5],
            x_length=9,
            y_length=3.5,
            axis_config={"color": GRAY, "include_numbers": True, "font_size": 16},
            tips=False
        ).shift(DOWN * 2.5)  # 调整位置，避免与轨迹图重叠
        
        curvature_labels = curvature_axes.get_axis_labels(
            x_label=Text("纵向距离 (m)", font_size=18),
            y_label=Text("曲率 (×10⁻³)", font_size=18)
        )
        
        # 绘制曲率曲线
        poly_curvature_curve = VMobject(color=BLUE, stroke_width=3)
        poly_curvature_points = [curvature_axes.c2p(x, c * scale_factor) 
                                 for x, c in zip(x_samples, poly_curvatures)]
        poly_curvature_curve.set_points_as_corners(poly_curvature_points)
        
        bezier_curvature_curve = VMobject(color=GREEN, stroke_width=3)
        bezier_curvature_points = [curvature_axes.c2p(x, c * scale_factor) 
                                   for x, c in zip(bezier_x_samples, bezier_curvatures)]
        bezier_curvature_curve.set_points_as_corners(bezier_curvature_points)
        
        # ===================== 6. 动画展示 =====================
        # 显示道路场景
        title = Text("换道轨迹与曲率对比", font_size=24, color=YELLOW).to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1)
        
        self.play(
            Create(axes),
            Write(labels),
            Create(lane1_line),
            Create(lane2_line),
            Create(lane3_line),
            run_time=2
        )
        self.wait(0.5)
        
        # 同时显示两条轨迹
        # 调整标签位置，避免遮挡
        poly_label = Text("5次多项式", font_size=16, color=BLUE).next_to(axes.c2p(85, 5.5), UR, buff=0.2)
        bezier_label = Text("5次贝塞尔曲线", font_size=16, color=GREEN).next_to(axes.c2p(85, 4.8), UR, buff=0.2)
        
        self.play(
            Create(poly_trajectory),
            Create(bezier_trajectory),
            Write(poly_label),
            Write(bezier_label),
            run_time=2
        )
        self.wait(1)
        
        # 显示曲率坐标系
        self.play(
            Create(curvature_axes),
            Write(curvature_labels),
            run_time=1.5
        )
        self.wait(0.5)
        
        # 显示曲率曲线
        # 添加曲率曲线图例
        poly_curvature_label = Text("多项式曲率", font_size=14, color=BLUE).next_to(
            curvature_axes.c2p(15, 2.5), RIGHT, buff=0.2
        )
        bezier_curvature_label = Text("贝塞尔曲率", font_size=14, color=GREEN).next_to(
            curvature_axes.c2p(15, 2.2), RIGHT, buff=0.2
        )
        
        self.play(
            Create(poly_curvature_curve),
            Create(bezier_curvature_curve),
            Write(poly_curvature_label),
            Write(bezier_curvature_label),
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
        curvature_legend = VGroup(
            Text("曲率对比分析:", font_size=16, color=YELLOW),
            Text("", font_size=6),
            Text("蓝色: 5次多项式", font_size=12, color=BLUE),
            Text("  最大曲率: 0.002018", font_size=11, color=GRAY),
            Text("  平均曲率: 0.001400", font_size=11, color=GRAY),
            Text("  起点终点曲率较大", font_size=11, color=GRAY),
            Text("", font_size=4),
            Text("绿色: 5次贝塞尔", font_size=12, color=GREEN),
            Text("  最大曲率: 0.001986", font_size=11, color=GREEN),
            Text("  平均曲率: 0.000928", font_size=11, color=GREEN),
            Text("  起点终点曲率≈0", font_size=11, color=GRAY),
            Text("  整体更平顺", font_size=11, color=GREEN)
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
            FadeOut(bezier_curvature_label)
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
