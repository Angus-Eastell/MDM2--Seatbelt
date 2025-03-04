import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# =============== 1. Parameter Settings ===============

def sigmoid(x, k, x0):
    """Sigmoid function used for smooth transition of seatbelt force with displacement"""
    return 1 / (1 + np.exp(-k * (x - x0)))


# Physical parameters
m1 = 20.0       # Mass of the upper ball (chest)
m2 = 30.0       # Mass of the lower ball (abdomen)
k = 1500.0        # Spring constant (connecting chest and abdomen)
c = 500.0        # Spring damping coefficient (using original formulation)
g = 9.81        # Gravitational acceleration
L0 = 0.5        # Natural spring length
y_floor = 0.0   # Floor height
restitution = 0.9  # Energy loss factor upon collision (1 = perfect bounce, <1 = energy loss)
# Adding parameters for the head mass
m3 = 5.0  # Mass of the head
L1 = 0.15  # Natural length of spring between head and chest
k1 = 10000.0  # Spring constant between head and chest
c1 = 300.0  # Damping coefficient between head and chest


# Seatbelt parameters (for nonlinear seatbelt force model)
F_max = 1000.0    # Maximum seatbelt force
k_s = 250.0        # Sigmoid slope
L_slack1 = 0.1    # Slack length for the chest seatbelt (no force when displacement is within slack)
L_slack2 = 0.1    # Slack length for the abdomen seatbelt
c_s = 50.0        # Seatbelt damping coefficient

# Seatbelt anchor points (fixed points); adjust to match desired seatbelt direction
x_anchor1, y_anchor1 = 0.01, 0.51  # Anchor for chest belt
x_anchor2, y_anchor2 = 0.01, 0.01   # Anchor for abdomen belt

# Impact (impulse) parameters (Gaussian pulse)
A = 800.0         # Peak impulse force
b = 500.0         # Width control for Gaussian pulse
t_impulse = 0.04  # Time at which impulse is centered (sec)
ratio_impulse = 0.3  # Ratio of impulse force's y-component to x-component

# Initial conditions (position & velocity)
x1_0, y1_0 = 0.02, 0.51  # Initial position for chest
x2_0, y2_0 = 0.02, 0.01  # Initial position for abdomen
v1x_0, v1y_0 = 0.0, 0.0  # Initial velocity for chest
v2x_0, v2y_0 = 0.0, 0.0  # Initial velocity for abdomen
# Initial conditions for head
x3_0, y3_0 = 0.02, 1.2  # Position
v3x_0, v3y_0 = 0.0, 0.0  # Velocity

# =============== 2. Define the Differential Equations ===============
def equations(t, Y):
    """
    Y = [x1, v1x, y1, v1y, x2, v2x, y2, v2y]
    Returns dY/dt.
    """
    x1, v1x, y1, v1y, x2, v2x, y2, v2y, x3, v3x, y3, v3y = Y

    # Position derivatives
    dx1_dt = v1x
    dy1_dt = v1y
    dx2_dt = v2x
    dy2_dt = v2y
    dx3_dt = v3x
    dy3_dt = v3y

    # ---------- A. Spring Force (between chest and abdomen) ----------
    dx = x1 - x2
    dy = y1 - y2
    dist = np.sqrt(dx**2 + dy**2)
    if dist == 0:
        dist = 1e-9
    F_spring = -k * (dist - L0)
    ux, uy = dx/dist, dy/dist
    # Damping based on relative velocity (using original formulation)
    rel_vx = v1x - v2x
    rel_vy = v1y - v2y
    v_rel = rel_vx*ux + rel_vy*uy
    F_damp_x = -c * v_rel * ux
    F_damp_y = -c * v_rel * uy
    F_spring_x = F_spring * ux
    F_spring_y = F_spring * uy

    # Spring & damping forces (Head-Chest)
    dxh = x3 - x1
    dyh = y3 - y1
    disth = np.sqrt(dxh ** 2 + dyh ** 2)
    if disth == 0:
        disth = 1e-9
    F_spring_h = -k1 * (disth - L1)
    uxh = dxh / disth
    uyh = dyh / disth
    rel_vxh = v3x - v1x
    rel_vyh = v3y - v1y
    v_rel_h = rel_vxh * uxh + rel_vyh * uyh
    F_damp_xh = -c1 * v_rel_h * uxh
    F_damp_yh = -c1 * v_rel_h * uyh
    F_spring_xh = F_spring_h * uxh
    F_spring_yh = F_spring_h * uyh

    # ---------- B. Seatbelt Force: Chest Belt (Pull-only) ----------
    dx1_anchor = x1 - x_anchor1
    dy1_anchor = y1 - y_anchor1
    d1 = np.sqrt(dx1_anchor**2 + dy1_anchor**2)
    if d1 < 1e-9:
        d1 = 1e-9
    # Generate force only if displacement exceeds slack length
    if d1 > L_slack1:
        e1 = d1 - L_slack1  # Excess displacement
        # Force direction: from chest towards the anchor
        ux1, uy1 = -dx1_anchor/d1, -dy1_anchor/d1
        F_seatbelt1 = F_max * sigmoid(e1, k_s, 0.0)
        v_rel1 = v1x*ux1 + v1y*uy1
        F_damp1 = -c_s * v_rel1
        F_s1_total = F_seatbelt1 + F_damp1
        if F_s1_total < 0:
            F_s1_total = 0.0
        F_s1x = F_s1_total * ux1
        F_s1y = F_s1_total * uy1
    else:
        F_s1x, F_s1y = 0.0, 0.0

    # ---------- C. Seatbelt Force: Abdomen Belt (Pull-only) ----------
    dx2_anchor = x2 - x_anchor2
    dy2_anchor = y2 - y_anchor2
    d2 = np.sqrt(dx2_anchor**2 + dy2_anchor**2)
    if d2 < 1e-9:
        d2 = 1e-9
    if d2 > L_slack2:
        e2 = d2 - L_slack2
        ux2, uy2 = -dx2_anchor/d2, -dy2_anchor/d2
        F_seatbelt2 = F_max * sigmoid(e2, k_s, 0.0)
        v_rel2 = v2x*ux2 + v2y*uy2
        F_damp2 = -c_s * v_rel2
        F_s2_total = F_seatbelt2 + F_damp2
        if F_s2_total < 0:
            F_s2_total = 0.0
        F_s2x = F_s2_total * ux2
        F_s2y = F_s2_total * uy2
    else:
        F_s2x, F_s2y = 0.0, 0.0

    # ---------- D. Impact Force (Gaussian Pulse) ----------
    # Main impulse along x-direction
    impulse_x = A * np.exp(-b * (t - t_impulse)**2)
    # Add a y-component (adjustable ratio)
    impulse_y = 0.44 * impulse_x

    # Let the abdomen receive part of the impulse
    ratio_abdomen = ratio_impulse

    # ---------- E. Combine Forces -> Acceleration ----------
    dv1x_dt = (F_spring_x - F_spring_xh + F_damp_x - F_damp_xh + F_s1x + impulse_x) / m1
    dv1y_dt = (F_spring_y - F_spring_yh + F_damp_y - F_damp_yh + F_s1y + impulse_y) / m1 -g
    dv2x_dt = (-F_spring_x - F_damp_x + F_s2x + ratio_abdomen * impulse_x) / m2
    dv2y_dt = (-F_spring_y - F_damp_y + F_s2y + ratio_abdomen * impulse_y) / m2 -g
    dv3x_dt = (F_spring_xh + F_damp_xh + impulse_x) / m3
    dv3y_dt = (F_spring_yh + F_damp_yh + impulse_y) / m3 -g
    return [dx1_dt, dv1x_dt, dy1_dt, dv1y_dt,
            dx2_dt, dv2x_dt, dy2_dt, dv2y_dt,dx3_dt, dv3x_dt, dy3_dt, dv3y_dt]


# =============== 3. Initial Integration (No Bounces) ===============
T = 0.3  # Simulation time (seconds)
t_span = (0, T)
t_eval = np.linspace(0, T, 1000)
init_conditions = [x1_0, v1x_0, y1_0, v1y_0,
                   x2_0, v2x_0, y2_0, v2y_0, x3_0, v3x_0, y3_0, v3y_0]

solution = solve_ivp(equations, t_span, init_conditions, t_eval=t_eval)
t = solution.t
x1, y1 = solution.y[0], solution.y[2]
x2, y2 = solution.y[4], solution.y[6]
x3, y3 = solution.y[8], solution.y[10]

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, 'b-', label='Chest (m1)')
plt.plot(x2, y2, 'r-', label='Abdomen (m2)')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Initial Trajectory (No Bounces)')
plt.grid()
plt.legend()
plt.close()


# =============== 4. Multi-phase Integration (With Floor Collisions) ===============
def floor_collision_m1(t, Y):
    return Y[2] - y_floor  # Detect when chest (y1) hits the floor

def floor_collision_m2(t, Y):
    return Y[6] - y_floor  # Detect when abdomen (y2) hits the floor

def floor_collision_m3(t, Y):
    return Y[8] - y_floor  # Detect when head (y3) hits the floor


floor_collision_m1.terminal = True
floor_collision_m1.direction = -1
floor_collision_m2.terminal = True
floor_collision_m2.direction = -1
floor_collision_m3.terminal = True
floor_collision_m3.direction = -1

def solve_with_bounces(t_max, init_conditions):
    t_all = []
    sol_all = []
    t0 = 0
    conditions = init_conditions
    while t0 < t_max:
        t_span = (t0, t_max)
        sol = solve_ivp(equations, t_span, conditions,
                        t_eval=np.linspace(t0, t_max, 1000),
                        events=[floor_collision_m1, floor_collision_m2, floor_collision_m3])
        t_all.append(sol.t)
        sol_all.append(sol.y)
        Y_collision = sol.y[:, -1]
        # Handle collisions: reverse vertical velocities upon impact
        if len(sol.t_events[0]) > 0:
            Y_collision[3] *= -restitution
        if len(sol.t_events[1]) > 0:
            Y_collision[7] *= -restitution
        if len(sol.t_events[1]) > 0:
            Y_collision[11] *= -restitution
        t0 = sol.t[-1] + 1e-3
        conditions = Y_collision
    t_full = np.concatenate(t_all)
    sol_full = np.concatenate(sol_all, axis=1)
    return t_full, sol_full

t_max_sim = 2.0
t, sol = solve_with_bounces(t_max_sim, init_conditions)
x1, y1 = sol[0], sol[2]
x2, y2 = sol[4], sol[6]
x3, y3 = sol[8], sol[10]

# =============== position plot======================

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, 'b-', label='Chest (m1)')
plt.plot(x2, y2, 'r-', label='Abdomen (m2)')
plt.plot(x3, y3, 'g', label = 'Head')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Initial Trajectory (No Bounces)')
plt.grid()
plt.legend()
plt.show()


# =============== 5. Animation Display ===============
x_min = min(x1.min(), x2.min(), x3.min())
x_max = max(x1.max(), x2.max(), x3.max())
y_min = min(y1.min(), y2.min(), y3.min())
y_max = max(y1.max(), y2.max(), y3.max())
padding = 0.1 * (x_max - x_min)
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Spring-Damper Motion with Seatbelts (Nonlinear Forces & Impulse)')
ax.grid()
ax.axhline(y_floor, color='black', linestyle='--', linewidth=2, label='Floor')

line1, = ax.plot([], [], 'bo-', markersize=8, label='Chest (m1)')
line2, = ax.plot([], [], 'go-', markersize=8, label='Abdomen (m2)')
line3, = ax.plot([], [], 'ro-', markersize=8, label='Head (m3)')
spring, = ax.plot([], [], 'r-', linewidth=2, label='Spring')
spring2, = ax.plot([], [], 'g-', linewidth=2, label='Head-Chest')

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    spring.set_data([], [])
    spring2.set_data([], [])
    return line1, line2, line3, spring, spring2

def update(frame):
    line1.set_data([x1[frame]], [y1[frame]])
    line2.set_data([x2[frame]], [y2[frame]])
    line3.set_data([x3[frame]], [y3[frame]])
    spring.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    spring2.set_data([x3[frame], x1[frame]], [y3[frame], y1[frame]])
    return line1, line2, spring, line3, spring2

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                              blit=True, interval=20)
plt.legend()
plt.show()

# =============== 6. Other Plots: Spring Extension, Velocity, Acceleration ===============
distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
extension = distance - L0
spring_force = -k * extension

distance1 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
extension1 = distance1 - L1
spring_force1 = -k1 * extension1

plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
plt.plot(t, extension, 'r', label="Extension In Torso")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Extension (m)')
plt.title('Spring Extension In Torso Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(t, spring_force, 'r', label="Force In Torso")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Spring Force In Torso Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(t, extension1, 'r', label="Extension In Neck")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Extension (m)')
plt.title('Spring Extension In Neck Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(t, spring_force1, 'r', label="Force In Neck")
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Spring Force In Neck Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()




x1, y1 = sol[0], sol[2]
v1x, v1y = sol[1], sol[3]
x2, y2 = sol[4], sol[6]
v2x, v2y = sol[5], sol[7]
x3, y3 = sol[8], sol[10]
v3x, v3y = sol[9], sol[11]

dv1x_dt = np.gradient(v1x, t)
dv1y_dt = np.gradient(v1y, t)
dv2x_dt = np.gradient(v2x, t)
dv2y_dt = np.gradient(v2y, t)
dv3x_dt = np.gradient(v3x, t)
dv3y_dt = np.gradient(v3y, t)

plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.plot(t, v1x, label='v1x')
plt.plot(t, v1y, label='v1y')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Chest Velocity Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 3, 4)
plt.plot(t, dv1x_dt, label='a1x')
plt.plot(t, dv1y_dt, label='a1y')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Chest Acceleration Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 3, 2)
plt.plot(t, v2x, label='v2x')
plt.plot(t, v2y, label='v2y')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Abdomen Velocity Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(t, dv2x_dt, label='a2x')
plt.plot(t, dv2y_dt, label='a2y')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Abdomen Acceleration Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 3, 3)
plt.plot(t, v3x, label='v2x')
plt.plot(t, v3y, label='v2y')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Head Velocity Over Time')
plt.legend()
plt.grid()

plt.subplot(2, 3, 6)
plt.plot(t, dv3x_dt, label='a2x')
plt.plot(t, dv3y_dt, label='a2y')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Head Acceleration Over Time')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
