import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m1 = 5.0  # 上方小球质量
m2 = 7.0  # 下方小球质量
k = 20.0  # 弹簧常数
c = 2.0  # 阻尼系数
F_s1 = 50.0  # 上方小球的安全带力
F_s2 = 50.0  # 下方小球的安全带力
g = 9.81  # 重力加速度
theta1 = np.radians(195)  # 上方小球安全带力的角度（指向左下角）
theta2 = np.radians(225)  # 下方小球安全带力的角度（指向左下角）

# 初始条件
x1_0, y1_0 = 0.0, 1.0  # 上方小球初始位置
x2_0, y2_0 = 0.0, 0.5  # 下方小球初始位置（竖直对齐）
L0 = y1_0 - y2_0  # 弹簧的自然长度，确保初始无拉伸

v1x_0, v1y_0 = 0.3, 0.0  # 调整上方小球的 x 方向初速度，防止过度偏移
v2x_0, v2y_0 = 0.3, 0.0  # 使两个小球的初始运动更对称

def equations(t, y):
    x1, v1x, y1, v1y, x2, v2x, y2, v2y = y

    dx1_dt = v1x
    dy1_dt = v1y
    dx2_dt = v2x
    dy2_dt = v2y

    dx = x1 - x2
    dy = y1 - y2
    distance = np.sqrt(dx ** 2 + dy ** 2)

    F_spring = -k * (distance - L0)

    if distance != 0:
        ux, uy = dx / distance, dy / distance
    else:
        ux, uy = 0, 0  # 避免除零错误

    # 计算弹簧力的 x, y 分量
    F_spring_x = F_spring * ux
    F_spring_y = F_spring * uy

    # 计算阻尼力（基于相对速度）
    dvx = v1x - v2x
    dvy = v1y - v2y
    F_damping_x = -c * dvx
    F_damping_y = -c * dvy

    # 计算安全带力的 x, y 分量（指向左下角）
    F_s1x = F_s1 * np.cos(theta1)
    F_s1y = F_s1 * np.sin(theta1)  # 修正 y 分量方向（向下）

    F_s2x = F_s2 * np.cos(theta2)
    F_s2y = F_s2 * np.sin(theta2)  # 修正 y 分量方向（向下）

    # 计算加速度（牛顿第二定律）
    dv1x_dt = (F_spring_x + F_damping_x + F_s1x) / m1  # 安全带影响 x 方向
    dv1y_dt = (F_spring_y + F_damping_y + F_s1y) / m1 - g  # 受弹簧、阻尼、安全带和重力影响
    dv2x_dt = (-F_spring_x - F_damping_x + F_s2x) / m2  # 受弹簧、阻尼、安全带影响
    dv2y_dt = (-F_spring_y - F_damping_y + F_s2y) / m2 - g  # 受弹簧、阻尼、安全带和重力影响

    return [dx1_dt, dv1x_dt, dy1_dt, dv1y_dt, dx2_dt, dv2x_dt, dy2_dt, dv2y_dt]

# 求解微分方程
T = 0.07  # 模拟时间（秒）
t_span = (0, T)
t_eval = np.linspace(0, T, 1000)  # 细化步长
init_conditions = [x1_0, v1x_0, y1_0, v1y_0, x2_0, v2x_0, y2_0, v2y_0]

solution = solve_ivp(equations, t_span, init_conditions, t_eval=t_eval)

# 提取结果
t = solution.t
x1, y1 = solution.y[0], solution.y[2]
x2, y2 = solution.y[4], solution.y[6]

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, 'b-', label="Upper Ball (m1) Trajectory")
plt.plot(x2, y2, 'r-', label="Lower Ball (m2) Trajectory")
plt.xlim(left=0)
plt.ylim(0, None)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Motion Trajectory of Two Balls (Spring-Damper System with Angled Seatbelts)")
plt.legend()
plt.grid()
plt.show()



