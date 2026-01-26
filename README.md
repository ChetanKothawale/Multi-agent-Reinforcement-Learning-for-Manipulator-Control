# reinforcement-learning-for-redundant-robot-control-in-task‑space

This repository implements various control strategies for a **4-DOF redundant robotic manipulator** in task-space. It features a comparative analysis between traditional Jacobian-based methods and a fully cooperative **Multi-Agent Reinforcement Learning (MARL)** framework.

---

## 🔍 Overview

The project addresses the inverse kinematics (IK) problem for redundant robots, where the number of joint degrees-of-freedom () exceeds the task-space dimensions (). Traditional analytical methods often struggle with stability near singularities. This implementation demonstrates that a coordinated MARL approach can resolve these redundancies more effectively than classical Jacobian techniques.

## 🛠 Implemented Methods

### 1. Traditional Jacobian Control

* Moore-Penrose Pseudo-Inverse**: Utilizes a non-square Jacobian matrix () to map task-space velocity to joint velocities.


* Singularity-Avoidance (DLS)**: Implements **Damped Least Squares** by adding a damping factor () to ensure the matrix remains invertible at singular configurations.


* Numerical Approximation**: Both methods use numerical techniques to estimate the Jacobian columns based on small perturbations () in joint variables.



### 2. Multi-Agent Reinforcement Learning (MARL)

* Agent Definition**: Each robot joint is treated as an independent learning agent.


* Q-Learning Framework**: A critic-only, fully cooperative Q-learning mechanism where agents learn coordinated joint behaviors online.


* Reward Function**: Designed to minimize positional error while penalizing excessive or abrupt joint movements to ensure smooth convergence.



## 📂 Project Structure

* `jacobian_methods.m`: MATLAB implementation of PID control using Moore-Penrose and Singularity-Avoidance Jacobian methods.
* `marl_control_4dof.m`: Training and testing environment for the Q-Learning based MARL approach.
* `MR_Report.pdf`: Comprehensive term report detailing the methodology, D-H parameters, and simulation results.
* `MR_Presentation.pdf`: Summary presentation of the project findings and performance comparisons.

## 🚀 Key Results

* Faster Convergence**: MARL achieved faster settling times for trajectory tracking compared to traditional Jacobian-based methods. 
* Robustness**: The RL-based approach demonstrated superior stability near singular points where traditional pseudo-inverse methods typically fail.


* 
**Model-Free Potential**: MARL learned the inverse kinematics solution without requiring explicit symbolic differentiation.
