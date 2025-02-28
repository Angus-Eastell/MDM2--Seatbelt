import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Constants
m1 = 35.0  # Mass of chest
m2 = 40.0 # Mass of abdomen
k = 15000  # Spring constant
c = 1500  # Damping coefficient
g = 9.81  # Gravity
L0 = 0.5  # Natural spring length
y_floor = 0.0  # Floor height
restitution = 0.9  # Energy loss factor (1 = perfect bounce, <1 = energy loss)

# Initial conditions
x1_0, y1_0 = 0.0, 0.6  # Initial position of mass 1
v1x_0, v1y_0 = 0.0, 0.0  # Initial velocity of mass 2

x2_0, y2_0 = 0.0, 0.01 # Inital position of mass 2
v2x_0, v2y_0 = 0.0, 0.0

# Force of seatbelt
Fs = 1000


# Define the equations of motion
def equations(t, y):
    x1, v1x, y1, v1y, x2, v2x, y2, v2y = y

    # Compute displacement from anchor (0, 0.5)
    dx = x1 - x2
    dy = y1 - y2
    distance = np.sqrt(dx**2 + dy**2)

    # Compute spring force (Hooke's Law)
    F_spring = -k * (distance - L0)

    # Avoid division by zero
    ux, uy = (dx / distance, dy / distance) if distance != 0 else (0, 0)

    # Compute force components
    F_spring_x = F_spring * ux
    F_spring_y = F_spring * uy

    # finding the component pf velocity in spring
    velocity_component = (v1x)*ux + (v1y)*uy

    # Compute damping forces
    F_damping_x = -c * velocity_component * ux
    F_damping_y = -c * velocity_component * uy

    # Equations of motion
    dv1x_dt = (F_spring_x + F_damping_x) / m1
    dv1y_dt = (F_spring_y + F_damping_y) / m1 - g  # Gravity acts in Y

    dv2x_dt = (-F_spring_x - F_damping_x) / m2
    dv2y_dt = (-F_spring_y - F_damping_y) / m2 - g

    return [v1x, dv1x_dt, v1y, dv1y_dt, v2x, dv2x_dt, v2y, dv2y_dt]

# Event function to detect floor collision
def floor_collision_m1(t, y):
    return y[2] - y_floor  # Detect when y1 = y_floor

def floor_collision_m2(t, y):
    return y[6] - y_floor # Detect when y2 = y_floor

floor_collision_m1.terminal = True  # Stop integration when event occurs
floor_collision_m1.direction = -1  # Detect only downward crossing

floor_collision_m2.terminal = True  # Stop integration when event occurs
floor_collision_m2.direction = -1  # Detect only downward crossing

# Function to solve ODE in multiple phases (handling bounces)
def solve_with_bounces(t_max, init_conditions):
    t_all = []
    sol_all = []
    t0 = 0
    conditions = init_conditions

    while t0 < t_max:
        t_span = (t0, t_max)
        sol = solve_ivp(equations, t_span, conditions, t_eval=np.linspace(t0, t_max, 1000), events=[floor_collision_m1, floor_collision_m2])

        # Store solution
        t_all.append(sol.t)
        sol_all.append(sol.y)

        # Get state at collision & reverse velocity
        y_collision = sol.y[:, -1]
       
        # Handle collision with the floor
        if len(sol.t_events[0]) > 0:  # If m1 hit the floor
            y_collision[3] *= -restitution  # Reverse v1y
        if len(sol.t_events[1]) > 0:  # If m2 hit the floor
            y_collision[7] *= -restitution  # Reverse v2y

        # Restart integration with new conditions
        t0 = sol.t[-1] + 1e-3 # Continue from last time step
        conditions = y_collision  # Update state

    # Combine results
    t_full = np.concatenate(t_all)
    sol_full = np.concatenate(sol_all, axis=1)
    return t_full, sol_full

# Solve system
t, sol = solve_with_bounces(2, [x1_0, v1x_0, y1_0, v1y_0, x2_0, v2x_0, y2_0, v2y_0])

# Extract positions
x1, y1 = sol[0], sol[2]
x2, y2 = sol[4], sol[6]


##############################################################################################
##############################################################################################
##############################################################################################
# Plotting 

# Compute dynamic axis limits
x_min = np.min([np.min(sol[0]), np.min(sol[4])])  # Min of x1 and x2
x_max = np.max([np.max(sol[0]), np.max(sol[4])])  # Max of x1 and x2
y_min = np.min([np.min(sol[2]), np.min(sol[6])])  # Min of y1 and y2
y_max = np.max([np.max(sol[2]), np.max(sol[6])])  # Max of y1 and y2

# Add some padding to the limits for better visualization
padding = 0.1 * (x_max - x_min)  # 10% padding
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding

# Set up animation with floor
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Spring-Damper Motion with Floor Collision")
ax.grid()

# Draw floor
ax.axhline(y_floor, color='black', linestyle='--', linewidth=2, label="Floor")

# Create plots
line1, = ax.plot([], [], 'bo-', markersize=8, label="Chest (m1)")
line2, = ax.plot([], [], 'go-', markersize=8, label="Abdomen Ball (m2)")
spring, = ax.plot([], [], 'r-', linewidth=2, label="Spring")

# Initialize animation
def init():
    line1.set_data([], [])
    line1.set_data([], [])
    spring.set_data([], [])
    return line1, line2, spring

# Update function
def update(frame):
    x_m1, y_m1 = x1[frame], y1[frame]
    x_m2, y_m2 = x2[frame], y2[frame]
    line1.set_data([x_m1], [y_m1])
    line2.set_data([x_m2], [y_m2])
    spring.set_data([x_m1, x_m2], [y_m1, y_m2])  # Spring from fixed point to mass
    return line1, line2, spring

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20)

plt.legend()
plt.show()

# Compute spring extension
distance = np.sqrt((x1- x2)**2 + (y1 - y2)**2)  # Distance from fixed point
extension = distance - L0  # Extension beyond natural length
spring_force = k * extension

plt.figure(figsize= (12,8))

# Plot extension over time
plt.subplot(1, 2, 1)
plt.plot(t, extension, label="Spring Extension", color='r')
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line
plt.xlabel("Time (s)")
plt.ylabel("Spring Extension (m)")
plt.title("Spring Extension Over Time")
plt.legend()
plt.grid()

# Plot Force in spring over time
plt.subplot(1, 2, 2)
plt.plot(t, spring_force, label="Force in Spring", color='r')
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Reference line
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.title("Spring Force Over Time")
plt.legend()
plt.grid()

plt.tight_layout
plt.show()


# Extract position, velocity, and acceleration
x1, y1 = sol[0], sol[2]
v1x, v1y = sol[1], sol[3]

# Extract position, velocity, and acceleration
x2, y2 = sol[4], sol[6]
v2x, v2y = sol[5], sol[7]

# Compute accelerations
# Using dv1x_dt and dv1y_dt from equations
dx1_dt = np.gradient(x1, t)  # Velocity in x-direction
dy1_dt = np.gradient(y1, t)  # Velocity in y-direction

dv1x_dt = np.gradient(v1x, t)  # Acceleration in x-direction
dv1y_dt = np.gradient(v1y, t)  # Acceleration in y-direction

# Using dv2x_dt and dv2y_dt from equations
dx2_dt = np.gradient(x2, t)  # Velocity in x-direction
dy2_dt = np.gradient(y2, t)  # Velocity in y-direction

dv2x_dt = np.gradient(v2x, t)  # Acceleration in x-direction
dv2y_dt = np.gradient(v2y, t)  # Acceleration in y-direction


# Plot velocity and acceleration
plt.figure(figsize=(10, 5))

# Plot velocity
plt.subplot(2, 2, 1)
plt.plot(t, v1x, label="v1x (Velocity in X)")
plt.plot(t, v1y, label="v1y (Velocity in Y)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity of Chest (Mass 1) Over Time")
plt.legend()
plt.grid()

# Plot acceleration
plt.subplot(2, 2, 2)
plt.plot(t, dv1x_dt, label="a1x (Acceleration in X)")
plt.plot(t, dv1y_dt, label="a1y (Acceleration in Y)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration of Chest (Mass 1) Over Time")
plt.legend()
plt.grid()

# Plot velocity
plt.subplot(2, 2, 3)
plt.plot(t, v2x, label="v1x (Velocity in X)")
plt.plot(t, v2y, label="v1y (Velocity in Y)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity of Abdomen (Mass 2) Over Time")
plt.legend()
plt.grid()

# Plot acceleration
plt.subplot(2, 2, 4)
plt.plot(t, dv2x_dt, label="a1x (Acceleration in X)")
plt.plot(t, dv2y_dt, label="a1y (Acceleration in Y)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration of Abdomen (Mass 2) Over Time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
