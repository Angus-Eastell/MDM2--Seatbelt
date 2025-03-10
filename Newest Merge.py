import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from itertools import product


# =============== 1. Initial condition setting ===============

def sigmoid(x, k, x0):
    """Sigmoid function used for smooth transition of seatbelt force with displacement"""
    return 1 / (1 + np.exp(-k * (x - x0)))


# Initial conditions (position & velocity)
x1_0, y1_0 = 0.02, 0.51  # Initial position for chest
x2_0, y2_0 = 0.02, 0.01  # Initial position for abdomen
v1x_0, v1y_0 = 0.0, 0.0  # Initial velocity for chest
v2x_0, v2y_0 = 0.0, 0.0  # Initial velocity for abdomen
# Initial conditions for head
x3_0, y3_0 = 0.02, 0.66  # Position
v3x_0, v3y_0 = 0.0, 0.0  # Velocity


# =============== 2. Define the Differential Equations ===============
def equations(t, Y, params):
    """
    Y = [x1, v1x, y1, v1y, x2, v2x, y2, v2y, x3, v3x, y3, v3y]
    Returns dY/dt.
    """
    x1, v1x, y1, v1y, x2, v2x, y2, v2y, x3, v3x, y3, v3y = Y

    # Unpack parameters from the dictionary
    m1, m2, k, c, g, L0, y_floor, restitution, k_seat, c_seat, m3, L1, k1, c1, \
        F_max, k_s, L_slack1, L_slack2, c_s, x_anchor1, y_anchor1, x_anchor2, y_anchor2, \
        A, b, t_impulse, ratio_impulse, k_muscle, c_muscle = (
        params["m1"], params["m2"], params["k"], params["c"], params["g"], params["L0"],
        params["y_floor"], params["restitution"], params["k_seat"], params["c_seat"],
        params["m3"], params["L1"], params["k1"], params["c1"], params["F_max"],
        params["k_s"], params["L_slack1"], params["L_slack2"], params["c_s"],
        params["x_anchor1"], params["y_anchor1"], params["x_anchor2"], params["y_anchor2"],
        params["A"], params["b"], params["t_impulse"], params["ratio_impulse"],
        params["k_muscle"], params["c_muscle"]
    )

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
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist == 0:
        dist = 1e-9
    F_spring = -k * (dist - L0)
    ux, uy = dx / dist, dy / dist
    # Damping based on relative velocity (using original formulation)
    rel_vx = v1x - v2x
    rel_vy = v1y - v2y
    v_rel = rel_vx * ux + rel_vy * uy
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
    d1 = np.sqrt(dx1_anchor ** 2 + dy1_anchor ** 2)
    if d1 < 1e-9:
        d1 = 1e-9
    # Generate force only if displacement exceeds slack length
    if d1 > L_slack1:
        e1 = d1 - L_slack1  # Excess displacement
        # Force direction: from chest towards the anchor
        ux1, uy1 = -dx1_anchor / d1, -dy1_anchor / d1
        F_seatbelt1 = F_max * sigmoid(e1, k_s, L_slack1)
        v_rel1 = v1x * ux1 + v1y * uy1
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
    d2 = np.sqrt(dx2_anchor ** 2 + dy2_anchor ** 2)
    if d2 < 1e-9:
        d2 = 1e-9
    if d2 > L_slack2:
        e2 = d2 - L_slack2
        ux2, uy2 = -dx2_anchor / d2, -dy2_anchor / d2
        F_seatbelt2 = F_max * sigmoid(e2, k_s, L_slack2)
        v_rel2 = v2x * ux2 + v2y * uy2
        F_damp2 = -c_s * v_rel2
        F_s2_total = F_seatbelt2 + F_damp2
        if F_s2_total < 0:
            F_s2_total = 0.0
        F_s2x = F_s2_total * ux2
        F_s2y = F_s2_total * uy2
    else:
        F_s2x, F_s2y = 0.0, 0.0

    # ----------- Neck Force -------------------------------

    # finding ideal neck positon
    x_neck = ux * L1 + x1
    y_neck = uy * L1 + y1

    dx_neck = x3 - x_neck
    dy_neck = y3 - y_neck
    dist_neck = np.sqrt(dx_neck ** 2 + dy_neck ** 2)
    if dist_neck == 0:
        dist_neck = 1e-9
    F_spring_neck = -k_muscle * (dist_neck)
    ux_neck, uy_neck = dx_neck / dist_neck, dy_neck / dist_neck
    F_spring_x_neck = F_spring_neck * ux_neck
    F_spring_y_neck = F_spring_neck * uy_neck

    # Compute velocity of the ideal neck position
    d_norm_dt = ((dx * (v3x - v1x) + dy * (v3y - v1y)) / dist)
    dux_dt = ((v3x - v1x) * dist - dx * d_norm_dt) / (dist ** 2)
    duy_dt = ((v3y - v1y) * dist - dy * d_norm_dt) / (dist ** 2)

    v_neck_x = v1x + dux_dt * L1
    v_neck_y = v1y + duy_dt * L1

    # Compute relative velocity (head vs ideal neck point)
    v_rel_x_neck = v3x - v_neck_x
    v_rel_y_neck = v3y - v_neck_y
    # Compute damping force
    F_damp_x_neck = -c_muscle * v_rel_x_neck
    F_damp_y_neck = -c_muscle * v_rel_y_neck

    # ---------- D. Impact Force (Gaussian Pulse) ----------
    # Main impulse along x-direction
    impulse_x = A * np.exp(-b * (t - t_impulse) ** 2)
    # Add a y-component (adjustable ratio)
    impulse_y = 0.44 * impulse_x

    # Let the abdomen receive part of the impulse
    ratio_abdomen = ratio_impulse

    # -----------Seat collision ---------------------

    # back rest interaction (smooth force instead of bounce)
    if x1 < 0:
        F_seat_1_x = -k_seat * (x1) - c_seat * v1x
    else:
        F_seat_1_x = 0
    if x2 < 0:
        F_seat_2_x = -k_seat * (x2) - c_seat * v2x
    else:
        F_seat_2_x = 0
    if x3 < 0:
        F_seat_3_x = -k_seat * (x3) - c_seat * v3x
    else:
        F_seat_3_x = 0

    # seat Floor interaction (smooth force instead of bounce)
    if y1 < 0:
        F_seat_1_y = -k_seat * (y1) - c_seat * v1y
    else:
        F_seat_1_y = 0
    if y2 < 0:
        F_seat_2_y = -k_seat * (y2) - c_seat * v2y
    else:
        F_seat_2_y = 0
    if y3 < 0:
        F_seat_3_y = -k_seat * (y3) - c_seat * v3y
    else:
        F_seat_3_y = 0

    # ---------- E. Combine Forces -> Acceleration ----------
    dv1x_dt = (F_spring_x - F_spring_xh + F_damp_x - F_damp_xh + F_s1x + impulse_x + F_seat_1_x) / m1
    dv1y_dt = (F_spring_y - F_spring_yh + F_damp_y - F_damp_yh + F_s1y + impulse_y + F_seat_1_y) / m1 - g
    dv2x_dt = (-F_spring_x - F_damp_x + F_s2x + impulse_x + F_seat_2_x) / m2
    dv2y_dt = (-F_spring_y - F_damp_y + F_s2y + impulse_y + F_seat_2_y) / m2 - g
    dv3x_dt = (F_spring_xh + F_damp_xh + F_spring_x_neck + F_damp_x_neck + impulse_x + F_seat_3_x) / m3
    dv3y_dt = (F_spring_yh + F_damp_yh + F_spring_y_neck + F_damp_y_neck + impulse_y + F_seat_3_y) / m3 - g

    return [dx1_dt, dv1x_dt, dy1_dt, dv1y_dt,
            dx2_dt, dv2x_dt, dy2_dt, dv2y_dt, dx3_dt, dv3x_dt, dy3_dt, dv3y_dt]


# =============== 3. Initial Integration (No Bounces) ===============
def solution(t_max, ini_conditions, parameters):
    # Simulation time (seconds)
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 1000)

    solution = solve_ivp(equations, t_span, ini_conditions, args=(parameters,), t_eval=t_eval)

    return t_eval, solution.y


# =============== 4. Multi-phase Integration (With Floor Collisions) ===============
def floor_collision_m1(t, Y):
    return Y[2] - parameters["y_floor"]  # Detect when chest (y1) hits the floor


def floor_collision_m2(t, Y):
    return Y[6] - parameters["y_floor"]  # Detect when abdomen (y2) hits the floor


def floor_collision_m3(t, Y):
    return Y[8] - parameters["y_floor"]  # Detect when head (y3) hits the floor


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
            Y_collision[3] *= -parameters["restitution"]
        if len(sol.t_events[1]) > 0:
            Y_collision[7] *= -parameters["restitution"]
        if len(sol.t_events[2]) > 0:
            Y_collision[11] *= -parameters["restitution"]
        t0 = sol.t[-1] + 1e-3
        conditions = Y_collision
    t_full = np.concatenate(t_all)
    sol_full = np.concatenate(sol_all, axis=1)
    return t_full, sol_full


# =============== position plot======================
def position_plot(sol):
    x1, y1 = sol[0], sol[2]
    x2, y2 = sol[4], sol[6]
    x3, y3 = sol[8], sol[10]
    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, 'b-', label='Chest (m1)')
    plt.plot(x2, y2, 'r-', label='Abdomen (m2)')
    plt.plot(x3, y3, 'g', label='Head')
    # Add dashed lines at x = 0 and y = 0 to indicate seat
    plt.axhline(0, color='k', linestyle='--', label='Seat Floor')  # Dashed line for y = 0 (seat level)
    plt.axvline(0, color='k', linestyle='--', label='Seat Backing')  # Dashed line for x = 0 (seat level)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Initial Trajectory (No Bounces)')
    plt.grid()
    plt.legend()
    plt.show()


# =============== 5. Animation Display ===============
def animation_plot(time, sol, params):
    y_floor = params["y_floor"]
    x1, y1 = sol[0], sol[2]
    x2, y2 = sol[4], sol[6]
    x3, y3 = sol[8], sol[10]

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
    ax.axhline(y_floor, color='black', linestyle='--', linewidth=2, label='Seat Floor')
    ax.axvline(y_floor, color='black', linestyle='--', linewidth=2, label='Seat Backing')

    line1, = ax.plot([], [], 'bo-', markersize=8, label='Chest (m1)')
    line2, = ax.plot([], [], 'go-', markersize=8, label='Abdomen (m2)')
    line3, = ax.plot([], [], 'ro-', markersize=8, label='Head (m3)')
    spring, = ax.plot([], [], 'r-', linewidth=2, label='Spring')
    spring2, = ax.plot([], [], 'g-', linewidth=2, label='Head-Chest')
    spring3, = ax.plot([], [], 'k-', linewidth=2, label='Chest restraint')
    spring4, = ax.plot([], [], 'k-', linewidth=2, label='Abdomen restraint')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        spring.set_data([], [])
        spring2.set_data([], [])
        spring3.set_data([], [])
        spring4.set_data([], [])
        return line1, line2, line3, spring, spring2, spring3, spring4

    def update(frame):
        line1.set_data([x1[frame]], [y1[frame]])
        line2.set_data([x2[frame]], [y2[frame]])
        line3.set_data([x3[frame]], [y3[frame]])
        spring.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        spring2.set_data([x3[frame], x1[frame]], [y3[frame], y1[frame]])
        spring3.set_data([params['x_anchor1'], x1[frame]], [params['y_anchor1'], y1[frame]])
        spring4.set_data([params['x_anchor2'], x2[frame]], [params['y_anchor2'], y2[frame]])
        return line1, line2, spring, line3, spring2, spring3, spring4

    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init,
                                  blit=True, interval=20)
    plt.legend()
    plt.show()


# =============== 6. Other Plots: Spring Extension, Velocity, Acceleration ===============
def spring_plots(time, sol, params):
    L0, L1, k, k1 = params["L0"], params["L1"], params["k"], params["k1"]

    x1, y1 = sol[0], sol[2]
    x2, y2 = sol[4], sol[6]
    x3, y3 = sol[8], sol[10]

    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    extension = distance - L0
    spring_force = -k * extension

    distance1 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    extension1 = distance1 - L1
    spring_force1 = -k1 * extension1

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 2, 1)
    plt.plot(time, extension, 'r', label="Extension In Torso")
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Extension (m)')
    plt.title('Spring Extension In Torso Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(time, spring_force, 'r', label="Force In Torso")
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Spring Force In Torso Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(time, extension1, 'r', label="Extension In Neck")
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Extension (m)')
    plt.title('Spring Extension In Neck Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(time, spring_force1, 'r', label="Force In Neck")
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Spring Force In Neck Over Time')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def velocity_acceleration_plots(time, sol):
    # extracting solution
    v1x, v1y = sol[1], sol[3]
    v2x, v2y = sol[5], sol[7]
    v3x, v3y = sol[9], sol[11]

    dv1x_dt = np.gradient(v1x, time)
    dv1y_dt = np.gradient(v1y, time)
    dv2x_dt = np.gradient(v2x, time)
    dv2y_dt = np.gradient(v2y, time)
    dv3x_dt = np.gradient(v3x, time)
    dv3y_dt = np.gradient(v3y, time)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1)
    plt.plot(time, v1x, label='v1x')
    plt.plot(time, v1y, label='v1y')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Chest Velocity Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(time, dv1x_dt, label='a1x')
    plt.plot(time, dv1y_dt, label='a1y')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Chest Acceleration Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(time, v2x, label='v2x')
    plt.plot(time, v2y, label='v2y')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Abdomen Velocity Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(time, dv2x_dt, label='a2x')
    plt.plot(time, dv2y_dt, label='a2y')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Abdomen Acceleration Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(time, v3x, label='v2x')
    plt.plot(time, v3y, label='v2y')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Head Velocity Over Time')
    plt.legend()
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(time, dv3x_dt, label='a2x')
    plt.plot(time, dv3y_dt, label='a2y')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Head Acceleration Over Time')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


# ============= Functions for mass distribition ===================
def calculating_spring_force(sol, params):
    L0, L1, k, k1 = params["L0"], params["L1"], params["k"], params["k1"]

    x1, y1 = sol[0], sol[2]
    x2, y2 = sol[4], sol[6]
    x3, y3 = sol[8], sol[10]

    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    extension = distance - L0
    force = -k * extension

    distance1 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    extension1 = distance1 - L1
    force1 = -k1 * extension1

    return force, force1


def seatbelt_force(sol, params):
    x1, y1 = sol[0], sol[2]
    x2, y2 = sol[4], sol[6]
    x3, y3 = sol[8], sol[10]

    dx1_anchor = x1 - params["x_anchor1"]
    dy1_anchor = y1 - params["y_anchor1"]
    d1 = np.sqrt(dx1_anchor ** 2 + dy1_anchor ** 2)
    f_seatbelt1 = []
    for index, distance1 in enumerate(d1):
        # Generate force only if displacement exceeds slack length
        if distance1 > params["L_slack1"]:
            e1 = distance1 - params["L_slack1"]  # Excess displacement
            # Force direction: from chest towards the anchor
            F1 = params["F_max"] * sigmoid(e1, params["k_s"], params["L_slack1"])
            f_seatbelt1.append(F1)
        else:
            f_seatbelt1.append(0)

    # ---------- C. Seatbelt Force: Abdomen Belt (Pull-only) ----------
    dx2_anchor = x2 - params["x_anchor2"]
    dy2_anchor = y2 - params["y_anchor2"]
    d2 = np.sqrt(dx2_anchor ** 2 + dy2_anchor ** 2)
    f_seatbelt2 = []
    for index, distance2 in enumerate(d2):
        # Generate force only if displacement exceeds slack length
        if distance2 > params["L_slack2"]:
            e2 = distance2 - params["L_slack2"]  # Excess displacement
            # Force direction: from chest towards the anchor
            F2 = params["F_max"] * sigmoid(e2, params["k_s"], params["L_slack2"])
            f_seatbelt2.append(F2)
        else:
            f_seatbelt2.append(0)

    return f_seatbelt1, f_seatbelt2


def plotting_seatbelt_forces(time, sol, params):
    f_seatbelt_1, f_seatbelt_2 = seatbelt_force(sol, params)

    plt.figure(figsize=(8, 5))
    plt.plot(time, f_seatbelt_1, label='Force From Chest Seatbelt')
    plt.plot(time, f_seatbelt_2, label='Force From Abdomen Seatbelt')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Force Form Seatbelts During Collision')
    plt.show()


def center_of_mass(chest_masses, abdomen_masses, head_mass, total_mass, initial_conditions):
    # Assuming chest, abdomen, and head have mass and position at the start
    # Extract positions from init_conditions
    x1, y1 = initial_conditions[0], initial_conditions[2]  # Chest position
    x2, y2 = initial_conditions[4], initial_conditions[6]  # Abdomen position
    x3, y3 = initial_conditions[8], initial_conditions[10]  # Head position

    # Calculate weighted sum for x and y positions
    x_com = (chest_masses * x1 + abdomen_masses * x2 + head_mass * x3) / total_mass
    y_com = (chest_masses * y1 + abdomen_masses * y2 + head_mass * y3) / total_mass

    return x_com, y_com


def mass_distributions(t_max_sim, initial_conditions, params, total_mass=80):
    # assume mass of head is constant in the process
    # range of masses, total masses assumed to be but can be varied with gender
    # num steps to 20
    num_points = 20
    body_mass = total_mass - params["m3"]
    lower_bound, upper_bound = round(body_mass / 3), round((2 * body_mass) / 3) + 1
    chest_masses = np.linspace(lower_bound, upper_bound, num_points)
    abdomen_masses = np.flip(chest_masses)
    mass_ratio = chest_masses / abdomen_masses
    peak_list_torso = []
    peak_list_neck = []
    centre_of_mass_x, centre_of_mass_y = center_of_mass(chest_masses, abdomen_masses, params['m3'], total_mass,
                                                        initial_conditions)
    # iterating over mass combinations
    for i in range(len(chest_masses)):
        params["m1"] = chest_masses[i]
        params["m2"] = abdomen_masses[i]

        t, sol = solution(t_max_sim, initial_conditions, params)
        e_torso, e_neck = calculating_spring_force(sol, params)
        peak_list_torso.append(max(e_torso))
        peak_list_neck.append(max(e_neck))

    neck_injury = 20  # change for critical value of force
    torso_injury = 60

    # Create figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))  # 1 row, 2 columns

    # Plot in the first subplot (Torso Peak Force)
    ax1.plot(mass_ratio, peak_list_torso, 'ro-', label='Peak Force In Torso', alpha=0.7)
    ax1.axhline(torso_injury, color='grey', linestyle='--', linewidth=2,
                label='Critical Force For Serious Injury In Torso', alpha=0.7)
    ax1.set_xlabel('Mass Ratio Between Chest and Abdomen (Chest Mass / Abdomen Mass)')
    ax1.set_ylabel('Peak Force In Springs (N)')
    ax1.set_title(f'Peak Force For Torso (Total Mass: {total_mass} kg)')
    ax1.grid(True)
    ax1.legend()

    # Plot in the second subplot (Neck Peak Force)
    ax2.plot(mass_ratio, peak_list_neck, 'bo-', label='Peak Force in Neck', alpha=0.7)
    ax2.axhline(neck_injury, color='black', linestyle='--', linewidth=2,
                label='Critical Force For Serious Injury In Neck', alpha=0.7)
    ax2.set_xlabel('Mass Ratio Between Chest and Abdomen (Chest Mass / Abdomen Mass)')
    ax2.set_ylabel('Peak Force In Springs (N)')
    ax2.set_title(f'Peak Force For Neck (Total Mass: {total_mass} kg)')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Create figure and two subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Plot Peak Force in Torso in the first subplot
    ax1.plot(centre_of_mass_y, peak_list_torso, 'ro-', label='Peak Force In Torso', alpha=0.7)
    ax1.axhline(torso_injury, color='grey', linestyle='--', linewidth=2,
                label='Critical Force For Serious Injury In Torso', alpha=0.7)
    ax1.set_xlabel('Centre of Mass (y-direction) (m)')
    ax1.set_ylabel('Peak Force In Springs (N)')
    ax1.set_title(f'Peak Force For Torso (Total Mass: {total_mass} kg)')
    ax1.grid(True)
    ax1.legend()

    # Plot Peak Force in Neck in the second subplot
    ax2.plot(centre_of_mass_y, peak_list_neck, 'bo-', label='Peak Force in Neck', alpha=0.7)
    ax2.axhline(neck_injury, color='black', linestyle='--', linewidth=2,
                label='Critical Force For Serious Injury In Neck', alpha=0.7)
    ax2.set_xlabel('Centre of Mass (y-direction) (m)')
    ax2.set_ylabel('Peak Force In Springs (N)')
    ax2.set_title(f'Peak Force For Neck (Total Mass: {total_mass} kg)')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


# ============== Pregnacy simulation ============================

def pregnancy_simulation(t_max_sim, initial_conditions, params):
    # female simulation without pregancy
    # finds solution no bounce
    t, sol = solution(t_max_sim, initial_conditions, params)

    f_seatbelt_1_w, f_seatbelt_2_w = seatbelt_force(sol, params)

    f_torso_w, f_neck_w = calculating_spring_force(sol, params)

    # change in parameters for pregnancy
    params['m2'] = 10 + params['m2']  # extra 10 kg for pregnancy
    initial_conditions[4] += 0.15
    """ params['L_slack1'] = 0.1
    params['L_slack2'] += 0.15"""

    # finds solution no bounce
    t, sol = solution(t_max_sim, initial_conditions, params)
    # plots
    f_seatbelt_1_p, f_seatbelt_2_p = seatbelt_force(sol, params)

    f_torso_p, f_neck_p = calculating_spring_force(sol, params)

    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, f_seatbelt_1_w, label='Force From Chest Seatbelt')
    plt.plot(t, f_seatbelt_2_w, label='Force From Abdomen Seatbelt')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Woman: Force Form Seatbelts During Collision')

    plt.subplot(1, 2, 2)
    plt.plot(t, f_seatbelt_1_p, label='Force From Chest Seatbelt')
    plt.plot(t, f_seatbelt_2_p, label='Force From Abdomen Seatbelt')
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Pregnancy: Force Form Seatbelts During Collision')

    plt.tight_layout()
    plt.show()

    # plotting forces in body

    plt.subplot(1, 2, 1)
    plt.plot(t, f_torso_w, label="Not Pregnant")
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Spring Force In Torso Pregancy Comarison')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(t, f_neck_w, label="Not Pregnant")
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Spring Force In Neck Over Time Pregnancy Comparison')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 1)
    plt.plot(t, f_torso_p, label="Pregnant")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(t, f_neck_p, label="Pregnant")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    """ position_plot(sol)
    animation_plot(t, sol, params)
    spring_plots(t,sol,params)
    plotting_seatbelt_forces(t,sol,parameters)
    velocity_acceleration_plots(t,sol)"""


# ============== Effect of increasing crash forces =============

def crash_force(t_max_sim, initial_conditions, params):
    mass = params['m1'] + params['m2'] + params["m3"]
    force = np.linspace(500, 5000, 10)

    max_torso_force, max_neck_force, max_chest_seat_force, max_abdomen_seat_force = [], [], [], []
    av_torso_force, av_neck_force, av_chest_seat_force, av_abdomen_seat_force = [], [], [], []
    for f in force:
        params["A"] = f
        time, sol = solution(t_max_sim, initial_conditions, params)
        torso_force, neck_force = calculating_spring_force(sol, params)
        chest_seat_force, abdomen_seat_force = seatbelt_force(sol, params)
        max_torso_force.append(max(np.abs(torso_force)))
        max_neck_force.append(max(np.abs(neck_force)))
        max_chest_seat_force.append(max(np.abs(chest_seat_force)))
        max_abdomen_seat_force.append(max(np.abs(abdomen_seat_force)))
        av_torso_force.append(np.mean(np.array(torso_force)))
        av_neck_force.append(np.mean(np.array(neck_force)))
        av_chest_seat_force.append(np.mean(np.array(chest_seat_force)))
        av_abdomen_seat_force.append(np.mean(np.array(abdomen_seat_force)))

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(force, max_torso_force, label='Peak Force In Torso')
    plt.plot(force, max_neck_force, label='Peak Force In Neck')
    plt.plot(force, av_torso_force, label='Peak Force In Torso')
    plt.plot(force, av_neck_force, label='Average Force In Neck')
    plt.xlabel('Peak Crash Force (N)')
    plt.ylabel('Force In Limbs (N)')
    plt.title('Forces In Limbs Varying Peak Crash Force')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(force, max_chest_seat_force, label='Peak Force In Chest Seatbelt')
    plt.plot(force, max_abdomen_seat_force, label='Peak Force In Abdomen Seatbelt')
    plt.plot(force, av_chest_seat_force, label='Average Force In Chest Seatbelt')
    plt.plot(force, av_abdomen_seat_force, label='Average Force In Abdomen Seatbelt')
    plt.xlabel('Peak Crash Force (N)')
    plt.ylabel('Force In Seatbelts (N)')
    plt.title('Forces In Seatbelts Varying Peak Crash Force')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

    params["A"] = 1500
    width = np.linspace(100, 1000, 10)
    max_torso_force, max_neck_force, max_chest_seat_force, max_abdomen_seat_force = [], [], [], []
    av_torso_force, av_neck_force, av_chest_seat_force, av_abdomen_seat_force = [], [], [], []
    for w in width:
        params["b"] = w
        time, sol = solution(t_max_sim, initial_conditions, params)
        torso_force, neck_force = calculating_spring_force(sol, params)
        chest_seat_force, abdomen_seat_force = seatbelt_force(sol, params)
        max_torso_force.append(max(np.abs(torso_force)))
        max_neck_force.append(max(np.abs(neck_force)))
        max_chest_seat_force.append(max(np.abs(chest_seat_force)))
        max_abdomen_seat_force.append(max(abdomen_seat_force))
        av_torso_force.append(np.mean(np.array(torso_force)))
        av_neck_force.append(np.mean(np.array(neck_force)))
        av_chest_seat_force.append(np.mean(np.array(chest_seat_force)))
        av_abdomen_seat_force.append(np.mean(np.array(abdomen_seat_force)))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(force, max_torso_force, label='Peak Force In Torso')
    plt.plot(force, max_neck_force, label='Peak Force In Neck')
    plt.plot(force, av_torso_force, label='Peak Force In Torso')
    plt.plot(force, av_neck_force, label='Average Force In Neck')
    plt.xlabel('Peak Crash Force (N)')
    plt.ylabel('Force In Limbs (N)')
    plt.title('Forces In Limbs Varying Width of Crash Force Pulse')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(width, max_chest_seat_force, label='Peak Force In Chest Seatbelt')
    plt.plot(width, max_abdomen_seat_force, label='Peak Force In Abdomen Seatbelt')
    plt.plot(width, av_chest_seat_force, label='Average Force In Chest Seatbelt')
    plt.plot(width, av_abdomen_seat_force, label='Average Force In Abdomen Seatbelt')
    plt.xlabel('Width of Crash Force (s-1)')
    plt.ylabel('Force In Seatbelts (N)')
    plt.title('Forces In Seatbelts Varying Width of Crash Force Pulse')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


# ============== Question 4: compute_injury_metrics ===========================
def compute_HIC(t, a):
    # Convert acceleration units from m/s² to g
    a_g = a / 9.81
    dt_max = 0.036  # 36 ms

    # 累计积分
    cum_int = np.concatenate(([0], cumulative_trapezoid(a_g, t)))
    n = len(t)
    hic_value = 0.0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if t[j] - t[i] > dt_max:
                break
            dt_ij = t[j] - t[i]
            area_ij = cum_int[j] - cum_int[i]
            a_avg = area_ij / dt_ij  # Average acceleration in g
            hic_ij = (a_avg ** 2.5) * dt_ij
            if hic_ij > hic_value:
                hic_value = hic_ij

    return hic_value


def compute_injury_metrics(sol, t, params):
    """
    Extract key injury metrics:
    1. HIC (based on head resultant acceleration)
    2. Peak force generated by the chest-abdomen spring
    """
    # -- Head resultant acceleration --
    v3x = sol[9]  # Head x-velocity
    v3y = sol[11]  # Head y-velocity
    a3x = np.gradient(v3x, t)
    a3y = np.gradient(v3y, t)
    # 2D resultant acceleration
    a_head = np.sqrt(a3x ** 2 + a3y ** 2)
    # Calculate HIC
    hic_value = compute_HIC(t, a_head)

    # -- Peak force of the chest-abdomen spring --
    x1, y1 = sol[0], sol[2]  # Chest
    x2, y2 = sol[4], sol[6]  # Abdomen
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    extension = dist - params["L0"]
    force_torso = np.abs(params["k"] * extension)
    peak_torso_force = np.max(force_torso)

    return {
        "HIC": hic_value,
        "peak_torso_force": peak_torso_force
    }


# ============== Question 4：scan parameters ===========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


def scan_multiple_parameters(param_names, param_ranges, fixed_params, init_conditions, t_max):
    combos = []
    metrics_list = []
    for values in product(*param_ranges):
        params = fixed_params.copy()
        combo = {}
        for name, val in zip(param_names, values):
            params[name] = val
            combo[name] = val
        combos.append(combo)
        t, sol = solution(t_max, init_conditions, params)
        metrics = compute_injury_metrics(sol, t, params)
        metrics_list.append(metrics)
    return combos, metrics_list


# -------------------- Pareto--------------------
def pareto_front_indices(objectives):
    """
    给定目标值数组 (n_points, n_objectives)，返回 Pareto 前沿的索引（目标均需最小化）。
    """
    n_points = objectives.shape[0]
    is_dominated = np.zeros(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated[i] = True
                    break
    return np.where(~is_dominated)[0]


body_types = {
    "male": {"m1": 19.5, "m2": 21.7, "m3": 4.5},
    "female": {"m1": 17.0, "m2": 18.0, "m3": 4.0},
    "pregnant": {"m1": 19.5, "m2": 25.0, "m3": 4.5}  # 腹部质量增加
}

baseline_params = {
    "m1": 19.5,  # Chest mass（会根据体型更新）
    "m2": 21.7,  # Abdomen mass（会根据体型更新）
    "k": 650.0,  # Chest-Abdomen spring constant
    "c": 500.0,  # Damping for chest-abdomen
    "g": 9.81,  # Gravitational acceleration
    "L0": 0.5,  # Natural spring length for chest-abdomen
    "y_floor": 0.0,  # Floor height
    "restitution": 0.0,  # Collision restitution (设为0)
    "k_seat": 2e4,  # Seatbelt spring constant
    "c_seat": 2500,  # Seatbelt damping coefficient
    "m3": 4.5,  # Head mass（会根据体型更新）
    "L1": 0.15,  # Natural spring length for head-chest
    "k1": 800.0,  # Head-Chest spring constant
    "c1": 500.0,  # Damping for head-chest
    "k_muscle": 2000,  # Neck muscle spring constant
    "c_muscle": 100,  # Neck muscle damping
    "F_max": 2000.0,  # Maximum seatbelt force
    "k_s": 50.0,  # Sigmoid slope for seatbelt force
    "L_slack1": 0.1,  # Chest belt slack (待扫描)
    "L_slack2": 0.1,  # Abdomen belt slack (固定)
    "c_s": 1000.0,  # Seatbelt damping
    "x_anchor1": 0.01, "y_anchor1": 0.51,  # Chest belt anchor
    "x_anchor2": 0.01, "y_anchor2": 0.01,  # Abdomen belt anchor
    "A": 800.0,  # Impact impulse force
    "b": 500.0,  # Gaussian pulse width parameter
    "t_impulse": 0.04,  # Impulse center time (sec)
    "ratio_impulse": 0.3  # Ratio for y-component impulse
}

init_conditions = [x1_0, v1x_0, y1_0, v1y_0,
                   x2_0, v2x_0, y2_0, v2y_0,
                   x3_0, v3x_0, y3_0, v3y_0]
t_max_sim = 1

# -------------------- 设置扫描参数 --------------------
# 以同时扫描 L_slack1, k_seat, c_seat 为例
L_slack1_range = np.linspace(0.05, 0.1, 5)  # 例如，从 0.05 到 0.1 m（可根据需要调整）
k_seat_range = np.linspace(15000, 30000, 5)  # 15000 到 30000 N/m
c_seat_range = np.linspace(1500, 4000, 5)  # 1500 到 4000 Ns/m

param_names = ["L_slack1", "k_seat", "c_seat"]
param_ranges = [L_slack1_range, k_seat_range, c_seat_range]

# -------------------- 对每个体型进行扫描，并保存结果 --------------------
body_type_results = {}  # 用于存储每个体型的所有结果和 Pareto 前沿

for btype, mass_params in body_types.items():
    print("Processing body type:", btype)
    # 更新质量参数
    params_current = baseline_params.copy()
    params_current["m1"] = mass_params["m1"]
    params_current["m2"] = mass_params["m2"]
    params_current["m3"] = mass_params["m3"]

    # 多参数联合扫描
    combos, metrics_list = scan_multiple_parameters(param_names, param_ranges, params_current, init_conditions,
                                                    t_max_sim)

    # 只打印前 5 个组合供检查
    print("Top 5 parameter combinations and indicators for", btype, ":")
    for combo, metrics in zip(combos[:5], metrics_list[:5]):
        print("Parameter combinations：", combo)
        print("HIC:", metrics["HIC"], "Peak Torso Force:", metrics["peak_torso_force"])
        print("----------")

    # 将所有结果保存为 DataFrame
    data = []
    for combo, metrics in zip(combos, metrics_list):
        row = {}
        row.update(combo)
        row.update(metrics)
        data.append(row)
    df_all = pd.DataFrame(data)

    # 提取 Pareto 前沿：以 HIC 和 peak_torso_force 为目标（均需最小化）
    objectives = df_all[["HIC", "peak_torso_force"]].to_numpy()
    pareto_idx = pareto_front_indices(objectives)
    df_pareto = df_all.iloc[pareto_idx]

    body_type_results[btype] = {"all": df_all, "pareto": df_pareto}

    print("Pareto front for", btype, "：")
    print(df_pareto)
    print("===================================")

# -------------------- 绘制对比图 --------------------
# 比较不同体型下 Pareto 前沿中 L_slack1 对 HIC 的影响
plt.figure(figsize=(10, 6))
for btype, results in body_type_results.items():
    df_pareto = results["pareto"]
    plt.scatter(df_pareto["L_slack1"], df_pareto["HIC"], label=btype, s=100, alpha=0.7)
plt.xlabel("L_slack1 (m)")
plt.ylabel("HIC")
plt.title("Pareto Front Comparison: HIC vs. L_slack1")
plt.legend()
plt.grid(True)
plt.show()

# 比较不同体型下 Pareto 前沿中 L_slack1 对 Peak Torso Force 的影响
plt.figure(figsize=(10, 6))
for btype, results in body_type_results.items():
    df_pareto = results["pareto"]
    plt.scatter(df_pareto["L_slack1"], df_pareto["peak_torso_force"], label=btype, s=100, alpha=0.7)
plt.xlabel("L_slack1 (m)")
plt.ylabel("Peak Torso Force (N)")
plt.title("Pareto Front Comparison: Peak Torso Force vs. L_slack1")
plt.legend()
plt.grid(True)
plt.show()


# =============== Calculate seatbelt force ===============
def calculating_seatbelt_force(sol, params):
    """
    返回在每个时刻，胸部安全带 (seatbelt1) 和腹部安全带 (seatbelt2) 的力（绝对值）。
    """
    x1, y1, v1x, v1y = sol[0], sol[2], sol[1], sol[3]
    x2, y2, v2x, v2y = sol[4], sol[6], sol[5], sol[7]

    x_anchor1, y_anchor1 = params["x_anchor1"], params["y_anchor1"]
    x_anchor2, y_anchor2 = params["x_anchor2"], params["y_anchor2"]
    L_slack1, L_slack2 = params["L_slack1"], params["L_slack2"]
    F_max, k_s, c_s = params["F_max"], params["k_s"], params["c_s"]

    # Chest Belt
    d1 = np.sqrt((x1 - x_anchor1) ** 2 + (y1 - y_anchor1) ** 2)
    seatbelt_chest = np.zeros_like(d1)
    condition = d1 > L_slack1
    if np.any(condition):
        e1 = d1[condition] - L_slack1
        ux1 = -(x1[condition] - x_anchor1) / d1[condition]
        uy1 = -(y1[condition] - y_anchor1) / d1[condition]
        F_val1 = F_max * sigmoid(e1, k_s, 0.0)
        v_rel1 = v1x[condition] * ux1 + v1y[condition] * uy1
        F_damp1 = -c_s * v_rel1
        F_total1 = F_val1 + F_damp1
        F_total1[F_total1 < 0] = 0
        seatbelt_chest[condition] = np.abs(F_total1)

    # Abdomen Belt
    d2 = np.sqrt((x2 - x_anchor2) ** 2 + (y2 - y_anchor2) ** 2)
    seatbelt_abd = np.zeros_like(d2)
    condition2 = d2 > L_slack2
    if np.any(condition2):
        e2 = d2[condition2] - L_slack2
        ux2 = -(x2[condition2] - x_anchor2) / d2[condition2]
        uy2 = -(y2[condition2] - y_anchor2) / d2[condition2]
        F_val2 = F_max * sigmoid(e2, k_s, 0.0)
        v_rel2 = v2x[condition2] * ux2 + v2y[condition2] * uy2
        F_damp2 = -c_s * v_rel2
        F_total2 = F_val2 + F_damp2
        F_total2[F_total2 < 0] = 0
        seatbelt_abd[condition2] = np.abs(F_total2)

    return seatbelt_chest, seatbelt_abd


def sweep_spring_forces(t_max_sim, init_conditions, params,
                        L0_values, L1_values):
    original_L0 = params["L0"]
    original_L1 = params["L1"]

    peak_torso_vs_L0 = []
    peak_neck_vs_L1 = []

    for val_L0 in L0_values:
        params["L0"] = val_L0
        params["L1"] = original_L1
        t, sol = solution(t_max_sim, init_conditions, params)
        force_torso, force_neck = calculating_spring_force(sol, params)
        peak_torso_vs_L0.append(np.max(np.abs(force_torso)))

    params["L0"] = original_L0
    for val_L1 in L1_values:
        params["L1"] = val_L1
        t, sol = solution(t_max_sim, init_conditions, params)
        force_torso, force_neck = calculating_spring_force(sol, params)
        peak_neck_vs_L1.append(np.max(np.abs(force_neck)))

    params["L0"] = original_L0
    params["L1"] = original_L1

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Torso Force vs L0
    axes[0].plot(L0_values, peak_torso_vs_L0, 'ro-', label='Peak Torso Spring Force')
    axes[0].set_xlabel('L0 (Chest-Abdomen Distance) [m]')
    axes[0].set_ylabel('Peak Force [N]')
    axes[0].set_title('Torso Spring Force vs. L0')
    axes[0].grid(True)
    axes[0].legend()

    # Neck Force vs L1
    axes[1].plot(L1_values, peak_neck_vs_L1, 'bo-', label='Peak Neck Spring Force')
    axes[1].set_xlabel('L1 (Head-Chest Distance) [m]')
    axes[1].set_ylabel('Peak Force [N]')
    axes[1].set_title('Neck Spring Force vs. L1')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def sweep_seatbelt_forces(t_max_sim, init_conditions, params,
                          L0_values, L1_values):
    original_L0 = params["L0"]
    original_L1 = params["L1"]

    peak_abdomen_belt_vs_L0 = []
    peak_chest_belt_vs_L1 = []

    # 1) Scan L0 (Recording Abdominal Harness Force)
    for val_L0 in L0_values:
        params["L0"] = val_L0
        params["L1"] = original_L1
        t, sol = solution(t_max_sim, init_conditions, params)
        sb_chest, sb_abd = calculating_seatbelt_force(sol, params)
        peak_abdomen_belt_vs_L0.append(np.max(sb_abd))

    # 2) Scan L1 (record chest harness force)
    params["L0"] = original_L0
    for val_L1 in L1_values:
        params["L1"] = val_L1
        t, sol = solution(t_max_sim, init_conditions, params)
        sb_chest, sb_abd = calculating_seatbelt_force(sol, params)
        peak_chest_belt_vs_L1.append(np.max(sb_chest))

    params["L0"] = original_L0
    params["L1"] = original_L1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Abdomen Seatbelt Force vs L0
    axes[0].plot(L0_values, peak_abdomen_belt_vs_L0, 'go-', label='Peak Abdomen Seatbelt Force')
    axes[0].set_xlabel('L0 (Chest-Abdomen Distance) [m]')
    axes[0].set_ylabel('Peak Force [N]')
    axes[0].set_title('Abdomen Seatbelt Force vs. L0')
    axes[0].grid(True)
    axes[0].legend()

    # Chest Seatbelt Force vs L1
    axes[1].plot(L1_values, peak_chest_belt_vs_L1, 'mo-', label='Peak Chest Seatbelt Force')
    axes[1].set_xlabel('L1 (Head-Chest Distance) [m]')
    axes[1].set_ylabel('Peak Force [N]')
    axes[1].set_title('Chest Seatbelt Force vs. L1')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============== Solving and plotting ===========================

parameters = {
    # Physical parameters
    "m1": 19.5,  # Mass of the upper ball (chest)
    "m2": 21.7,  # Mass of the lower ball (abdomen)
    "k": 650.0,  # Spring constant (connecting chest and abdomen)
    "c": 500.0,  # Spring damping coefficient (using original formulation)
    "g": 9.81,  # Gravitational acceleration
    "L0": 0.5,  # Natural spring length
    "y_floor": 0.0,  # Floor height
    "restitution": 0.0,  # Energy loss factor upon collision (1 = perfect bounce, <1 = energy loss)
    "k_seat": 2e4,  # seat floor spring constant
    "c_seat": 250,  # Floor damping coefficient

    # Adding parameters for the head mass
    "m3": 4.5,  # Mass of the head
    "L1": 0.15,  # Natural length of spring between head and chest
    "k1": 800.0,  # Spring constant between head and chest
    "c1": 500.0,  # Damping coefficient between head and chest
    "k_muscle": 2000,  # Spring constant for muscle in neck
    "c_muscle": 100,  # Damping coefficient for muscle in neck

    # Seatbelt parameters (for nonlinear seatbelt force model)
    "F_max": 2000.0,  # Maximum seatbelt force
    "k_s": 50.0,  # Sigmoid slope
    "L_slack1": 0.1,  # Slack length for the chest seatbelt (no force when displacement is within slack)
    "L_slack2": 0.1,  # Slack length for the abdomen seatbelt
    "c_s": 1000.0,  # Seatbelt damping coefficient

    # Seatbelt anchor points (fixed points); adjust to match desired seatbelt direction
    "x_anchor1": 0.01, "y_anchor1": 0.51,  # Anchor for chest belt
    "x_anchor2": 0.01, "y_anchor2": 0.01,  # Anchor for abdomen belt

    # Impact (impulse) parameters (Gaussian pulse)
    "A": 1500.0,  # Peak impulse force
    "b": 500.0,  # Width control for Gaussian pulse
    "t_impulse": 0.04,  # Time at which impulse is centered (sec)
    "ratio_impulse": 0.3,  # Ratio of impulse force's y-component to x-component
}

# ===============  main function  ==========================
if __name__ == "__main__":
    # setting initial conditions and time
    init_conditions = [
        x1_0, v1x_0, y1_0, v1y_0,
        x2_0, v2x_0, y2_0, v2y_0,
        x3_0, v3x_0, y3_0, v3y_0
    ]
    t_max_sim = 1

    parameters = {
        # Physical parameters
        "m1": 19.5,  # Mass of the upper ball (chest)
        "m2": 21.7,  # Mass of the lower ball (abdomen)
        "k": 650.0,  # Spring constant (connecting chest and abdomen)
        "c": 500.0,  # Spring damping coefficient (using original formulation)
        "g": 9.81,  # Gravitational acceleration
        "L0": 0.5,  # Natural spring length
        "y_floor": 0.0,  # Floor height
        "restitution": 0.0,  # Energy loss factor upon collision
        "k_seat": 2e4,  # seat floor spring constant
        "c_seat": 2500,  # Floor damping coefficient

        # Head mass and spring
        "m3": 4.5,
        "L1": 0.15,
        "k1": 800.0,
        "c1": 500.0,
        "k_muscle": 2000,
        "c_muscle": 100,

        # Seatbelt parameters
        "F_max": 2000.0,
        "k_s": 50.0,
        "L_slack1": 0.1,
        "L_slack2": 0.1,
        "c_s": 1000.0,
        "x_anchor1": 0.01, "y_anchor1": 0.51,
        "x_anchor2": 0.01, "y_anchor2": 0.01,

        # Impact (Gaussian pulse)
        "A": 800.0,
        "b": 500.0,
        "t_impulse": 0.04,
        "ratio_impulse": 0.3,
    }

    t, sol = solution(t_max_sim, init_conditions, parameters)
    position_plot(sol)
    animation_plot(t, sol, parameters)
    # Spring force over time
    spring_plots(t, sol, parameters)
    # Velocity acceleration diagram
    velocity_acceleration_plots(t, sol)
    # Plots Seatbelt Forces
    plotting_seatbelt_forces(t, sol, parameters)
    # Mass distribution scans
    mass_distributions(t_max_sim, init_conditions, parameters, total_mass=80)
    # Pregnancy simulation
    pregnancy_simulation(t_max_sim, init_conditions, parameters)
    # Varying peak focre in crash and width of gaussian pulse
    crash_force(t_max_sim, init_conditions, parameters)

    # 3) Calculate the damage metric
    injury_metrics = compute_injury_metrics(sol, t, parameters)
    print("HIC:", injury_metrics["HIC"])
    print("Torso Peak Force:", injury_metrics["peak_torso_force"])

    L0_values = np.linspace(0.5, 0.9, 20)
    L1_values = np.linspace(0.15, 0.30, 20)

    # Torso and neck spring force peaks
    sweep_spring_forces(t_max_sim, init_conditions, parameters,
                        L0_values, L1_values)

    # Peak harness force in the abdomen and chest
    sweep_seatbelt_forces(t_max_sim, init_conditions, parameters,
                          L0_values, L1_values)

