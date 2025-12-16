import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 数据
x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5])
y = np.array([1.2, 0.8, 1.5, 5.3, 13.1, 25.2, 42.5, 65.8, 95.3, 132.1, 
              176.2, 228.5, 289.8, 361.2, 443.7, 538.2, 645.8, 767.5, 904.2, 1057.0])

# 定义三次多项式函数
def cubic_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# 拟合三次多项式
popt, _ = curve_fit(cubic_func, x, y)
a, b, c, d = popt

# 生成拟合曲线的x点（更密集）
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = cubic_func(x_fit, a, b, c, d)

# 设置中文字体（解决中文显示问题）
# 全局字体配置：优先使用支持数学符号的字体
plt.rcParams.update({
    'font.family': ['DejaVu Sans', 'SimHei'],  # 先尝试DejaVu Sans（支持符号），再用SimHei（支持中文）
    'axes.unicode_minus': False,               # 解决负号显示
    'mathtext.fontset': 'dejavusans',          # 数学符号用DejaVu Sans渲染
})
# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='原始数据点', s=50)
plt.plot(x_fit, y_fit, color='blue', linewidth=2, label=f'拟合曲线: y = {a:.2f}x³ + {b:.2f}x² + {c:.2f}x + {d:.2f}')

# 设置图表样式
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('三次多项式拟合结果', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 显示图表
plt.show()

# 输出拟合系数
print(f"拟合系数：")
print(f"a = {a:.4f}")
print(f"b = {b:.4f}")
print(f"c = {c:.4f}")
print(f"d = {d:.4f}")