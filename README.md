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
Experimental Setup
The main ADDQN configuration uses:
- 500 training episodes
- 200 scheduling steps per episode
- Replay buffer size: 100,000
- Batch size: 64
- Discount factor: 0.99
- Learning rate: 3 × 10⁻⁴
- Initial epsilon: 1.0
- Minimum epsilon: 0.05
- Epsilon decay: 0.995
- Target-network update frequency: 5 episodes
Experiments were conducted for fog systems with 15, 50, and 100 nodes. Separate models were trained for each network size.
Evaluated Scenarios
The repository includes results for:
- Fixed-topology scheduling
- Unseen-topology generalization
- Scalability at 15, 50, and 100 fog nodes
- Bursty workloads
- Nonstationary workloads
- Deadline-reward sensitivity
- CPU inference latency
- Parameter growth
- Component-level ablation analysis
Main Findings
ADDQN performs competitively in the 15-node fixed-topology setting and shows stronger performance under unseen topologies and at 50 nodes.
At 50 nodes, ADDQN achieves a mean response time of approximately 141.66 ms and a deadline-miss ratio of 8.82%.
The experiments also identify a scalability boundary at 100 nodes under the fixed 100,000-interaction training budget. This limitation is related to learning and representation scalability rather than inference speed, as mean CPU decision latency remains below 1 ms across the tested network sizes.
Installation
Install the required Python packages using:
pip install -r requirements.txt

Main dependencies include:
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib
Training
The main 15-node ADDQN training script can be started using:
python train.py

The script uses the default training configuration reported above and records training metrics to a CSV file.
Experimental Data
The data directory contains detailed evaluation data, including matched fixed-topology and unseen-topology runs.
The results directory contains publication-oriented summary tables, while figures contains the main experimental visualizations.
Implementation Note
The current ADDQN implementation uses a flattened node-level branch in addition to the attention-weighted global representation. Therefore, the architecture is not permutation-invariant and requires a separate model for each value of the number of fog nodes.
Citation
Citation information will be added after publication.
Authors
Nagwa Elmobark
Sara Elhishi
Alshaimaa M. Mohammed
