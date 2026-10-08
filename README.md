تمام ✅ ندخل على أهم خطوة دلوقتي: **تجهيز README.md** بحيث أي حد يدخل الـGitHub يفهم المشروع بسرعة.

افتحي **Notepad** واكتبي النص التالي بالكامل:

```markdown
# ADDQN Fog Scheduling

This repository contains the implementation and experimental evaluation of an Attention-Enhanced Double Deep Q-Network (ADDQN) for deadline-aware task scheduling in fog computing environments.

## Overview

The proposed ADDQN combines:

- Double Deep Q-Network learning
- Shared node encoding
- Node-wise attention
- Dual-path feature processing
- Multi-objective reward shaping

The scheduling objective considers response time, deadline satisfaction, load balancing, and normalized energy depletion.

## Experimental Evaluation

The repository includes experiments covering:

- Fixed-topology evaluation
- Generalization to unseen fog topologies
- Scalability at 15, 50, and 100 fog nodes
- Bursty and nonstationary workloads
- Reward-weight sensitivity
- Component-level ablation analysis
- CPU inference latency

## Repository Structure

```text
ADDQN-Fog-Scheduling/
├── src/            # Core ADDQN implementation
├── experiments/    # Experimental evaluation scripts
├── notebooks/      # Jupyter notebooks
├── data/           # Experimental data
├── results/        # Numerical results
├── figures/        # Figures used in the analysis
├── requirements.txt
└── README.md
```

## Requirements

The main Python dependencies are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt
```

## Reproducibility

The experiments use controlled random seeds and matched evaluation streams where applicable.

The main training configuration uses:

- 500 training episodes
- 200 scheduling steps per episode
- Replay buffer size: 100,000
- Batch size: 64
- Discount factor: 0.99
- Learning rate: 3e-4
- Target network update frequency: 5 episodes

Separate models are trained for different fog-network sizes.

## Notes

The 100-node experiments are evaluated under the same fixed 100,000-interaction training budget used at smaller network sizes.

The current ADDQN implementation is not permutation-invariant because the flattened node-level representation depends on the number and ordering of fog nodes.

## Citation

Citation information will be added after publication.

## Authors

Nagwa Elmobark  
Sara Elhishi  
Alshaimaa M. Mohammed
```

