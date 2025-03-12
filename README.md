# **Seatbelt Optimization Model - README**

## **Overview**
This project evaluates seatbelt configurations using a **multi-mass spring-damping system** to model human impact dynamics during a crash. Our goal is to optimize seatbelt slack and stiffness to minimize **head injury criterion (HIC)** and **peak torso force**, using **multi-objective optimization and Pareto front analysis**.

## **Methodology**
The model consists of a **three-mass system (head, chest, abdomen)** connected by **internal spring-damper elements** to simulate tissue properties. **External seatbelt forces** are applied based on displacement beyond slack length. The equations of motion are solved using **ODE integration**.

### **Key Components of the Model**
1. **Internal Forces (Spring-Damping Model)**
   - Masses are linked with springs and dampers.
   - Governing equations:

     \[
     m \ddot{x} + c\dot{x} + kx = F_{external}
     \]

     \[
     m \ddot{y} + c\dot{y} + ky = F_{external}
     \]

2. **Seatbelt Force Modeling**
   - Seatbelt force is applied when displacement exceeds slack length.
   - Modeled using a **sigmoid function** for smooth force transition.

3. **Crash Impact Simulation**
   - Modeled as a **Gaussian impulse** acting on the body.
   - Force applied along both horizontal and vertical directions.

4. **Optimization and Pareto Analysis**
   - We analyze **seatbelt slack (L_slack1)** and **stiffness** to balance injury reduction.
   - Pareto front is extracted to identify the best trade-offs.

## **Code Structure**
| Function | Description |
|----------|-------------|
| `sigmoid(x, k, x0)` | Smooth transition function for seatbelt force application. |
| `equations(t, Y, params)` | Defines the system of **ODEs** governing body dynamics. |
| `solution(t_max, ini_conditions, parameters)` | Solves ODEs using `solve_ivp`. |
| `compute_injury_metrics(sol, t, params)` | Calculates **HIC** and **peak torso force**. |
| `scan_multiple_parameters()` | Performs **parameter sweeps** for optimization. |
| `pareto_front_indices()` | Extracts **Pareto optimal solutions**. |
| `animation_plot()` | Visualizes body motion over time. |

## **How to Run**
1. Install required dependencies:
   pip install numpy scipy matplotlib pandas
2. Run the script:
   python Newest_Merge.py
3. The script will:
（1）Simulate the spring-damper model under crash conditions.
（2）Compute injury metrics (HIC, torso force).
（3）Perform optimization to find the best seatbelt configurations.
（4）Generate plots and animations for visualization.

## **Results & Insights**
- **Increasing seatbelt slack reduces HIC** but raises **torso force**.
- **Pregnant individuals experience higher torso forces** when slack is excessive.
- **Balanced seatbelt stiffness and slack** yield the **best safety outcomes**.
- **Seatbelt force varies based on body proportions**, affecting **injury risk**.

## **Limitations and Future Work**
Although our study provides valuable insights, there are still areas for improvement in our model:

- **Rigid body assumption**:  
  Our model treats body masses as **rigid**. Incorporating **deformable and volumetric masses** would provide a more realistic representation of human impact dynamics.
  
- **Different age groups**:  
  Expanding to include **elderly or children** would enhance applicability and improve the model's ability to reflect real-world safety outcomes.

- **Individual variability**:  
  Accounting for **variations in height, weight, and muscle distribution** could refine the optimization process and allow for more **personalized safety recommendations**.

- **Diverse crash conditions**:  
  Testing under **different impact scenarios** (e.g., side impacts, rollovers) would make our model more robust and applicable to **varied real-world accidents**.

**Addressing these gaps will help refine seatbelt optimization for even greater safety across diverse populations.**
