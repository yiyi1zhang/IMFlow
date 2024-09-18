# IMFlow
Code for IMFlow: Inverse Modeling with Conditional Normalizing Flows for Data-Driven Model Predictive Control

- openIMFlow: open-loop IMFlow for path planning
- closedIMFlow: closed-loop IMFlow for predictive control 
- PETS: modified probabilistic ensembles trajectory tracking, implemented in PyTorch, based on the official code of [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://github.com/kchua/handful-of-trials)
- APG: modified analytic plolicy gradient, based on the official code of [Training Efficient Controllers via Analytic Policy Gradient](https://github.com/lis-epfl/apg_trajectory_tracking)
- MPC: model predictive control using [do-mpc](https://github.com/do-mpc/do-mpc)
- vehiclemodels: cloned from [Commonroad](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/tree/master/PYTHON?ref_type=heads)

workflow: generate data --> train model --> validate model --> test in control