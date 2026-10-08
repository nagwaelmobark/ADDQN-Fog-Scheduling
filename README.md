# ADDQN Fog Scheduling

This repository contains the implementation and experimental results of an Attention-Enhanced Double Deep Q-Network (ADDQN) for deadline-aware task scheduling in heterogeneous fog computing environments.

## Overview

ADDQN combines Double DQN learning with a shared node encoder, learned node-wise attention, and dual-path feature processing. The scheduling objective considers response time, deadline satisfaction, overload avoidance, load balancing, and normalized energy depletion.

The evaluation includes fixed and unseen fog topologies, scalability experiments, dynamic workloads, reward sensitivity, component-level analysis, and CPU inference benchmarking.

## ADDQN Architecture

The proposed model includes:

- Shared node encoder for fog-node features
- Learned node-wise attention mechanism
- Attention-weighted global representation
- Flattened node-level representation
- Dual-path feature fusion
- Double DQN action-value learning
- Multi-objective reward shaping

Each fog node is represented by five features:

1. Queue-derived utilization proxy
2. Normalized queue occupancy
3. Communication latency
4. Remaining normalized energy
5. Processing-capacity factor

## Repository Structure

```text
ADDQN-Fog-Scheduling/
│
├── src/
│   ├── environment.py
│   ├── reward.py
│   ├── model.py
│   ├── agent.py
│   └── replay_buffer.py
│
├── experiments/
│   └── revision_experiments.py
│
├── data/
│   ├── fixed_topology/
│   ├── unseen_topology/
│   └── experimental datasets
│
├── results/
│   └── publication-level result tables
│
├── figures/
│   └── figures from the experimental analysis
│
├── train.py
├── requirements.txt
└── README.md
