# DP-FedSOFIM: Differentially Private Federated Stochastic Optimization using Regularized Fisher Information Matrix

This repository contains the official implementation accompanying the paper **"DP-FedSOFIM"**, accepted at **TMLR (2026)**. It provides a unified codebase for training differentially private federated learning models using second-order optimization (SOFIM) and a suite of baseline algorithms, with exact privacy accounting via the hockey-stick divergence.

---

## Overview

DP-FedSOFIM trains a linear classifier on top of a frozen, pre-trained ResNet backbone in a federated setting under record-level differential privacy. The key contribution is the **SOFIM server optimizer**, which applies a Sherman-Morrison Newton step on the aggregated noisy gradients — achieving second-order convergence at O(d) cost, the same as SGD with momentum.

**Architecture:**
- **Backbones** (frozen): CIFAR-100-pretrained ResNet-20 and VGG-16 (chenyaofo), producing low-dimensional features
- **Head** (trained federally): Linear classifier
- **Privacy**: Gaussian mechanism with per-example gradient clipping; noise calibrated via exact hockey-stick divergence accounting

---

## Repository Structure

| File | Description |
|---|---|
| [main.py](main.py) | CLI entry point — parses args, runs training, saves results |
| [train.py](train.py) | Core federated training loop; all client and server classes |
| [sanitizer.py](sanitizer.py) | DP noise mechanisms for all algorithms |
| [hockey_stick_accountant.py](hockey_stick_accountant.py) | Exact Gaussian DP accountant (hockey-stick divergence) |
| [dataset.py](dataset.py) | Feature extraction, federated data partitioning (IID / non-IID / Dirichlet) |
| [plot_comparison_accuracycurves.py](plot_comparison_accuracycurves.py) | Accuracy curve plotting across algorithms |
| [plot_comparison_losscurves.py](plot_comparison_losscurves.py) | Loss curve plotting across algorithms |

---

## Installation

```bash
pip install torch torchvision numpy scipy medmnist
```

Backbones are loaded automatically via `torch.hub` from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models) on first run.

---

## Quick Start

### DP-SOFIM (proposed method)

```bash
python main.py \
  --backbone cifar100_resnet20 \
  --dataset cifar10 \
  --num_clients 20 \
  --clients_per_round 20 \
  --federated_rounds 70 \
  --epsilon 5.0 \
  --use_sofim \
  --sofim_rho 0.5 \
  --sofim_beta 0.9 \
  --save_results
```

### DP-FedGD (baseline)

```bash
python main.py \
  --backbone cifar100_resnet20 \
  --num_clients 20 \
  --clients_per_round 20 \
  --federated_rounds 70 \
  --epsilon 5.0
```

### No-DP baseline (sanity check)

```bash
python main.py \
  --backbone cifar100_resnet20 \
  --num_clients 20 \
  --clients_per_round 20 \
  --federated_rounds 70 \
  --no_dp
```

---

## Supported Algorithms

| Flag | Algorithm | Description |
|---|---|---|
| *(default)* | DP-FedGD | Differentially private federated gradient descent |
| `--use_sofim` | **DP-SOFIM** | Second-order SOFIM server optimizer (proposed) |
| `--use_fedfc` | DP-FedFC | Feature covariance preconditioning (Algorithm 3) |
| `--use_fednew_fc` | DP-FedNew-FC | Client-side ADMM with covariance preconditioning |
| `--use_scaffold` | DP-SCAFFOLD | Control variates for variance reduction |
| `--use_fedavg_multi_iters` | DP-FedAvg | FedAvg with K > 1 local SGD steps |
| `--use_fedadam` | DP-FedAdam | Server-side Adam optimizer |
| `--use_fedyogi` | DP-FedYogi | Server-side Yogi optimizer (Reddi et al., 2021) |
| `--use_fedprox` | DP-FedProx | Proximal penalty regularization |
| `--use_ftrl` | DP-FTRL | Tree-aggregation based privacy accounting |
| `--use_adafedprox` | DP-AdaFedProx | Heterogeneity-robust adaptive proximal method |

---

## Datasets

| Dataset | Flag | Classes | Notes |
|---|---|---|---|
| CIFAR-10 | `--dataset cifar10` | 10 | Auto-downloaded |
| CIFAR-10 Binary | `--dataset cifar10_binary` | 2 | Classes 0 vs 1 |
| PathMNIST | `--dataset pathmnist` | 9 | RGB, requires `medmnist` |
| BloodMNIST | `--dataset bloodmnist` | 8 | RGB, requires `medmnist` |
| DermaMNIST | `--dataset dermamnist` | 7 | RGB, requires `medmnist` |
| ChestMNIST | `--dataset chestmnist` | 15 | Grayscale→3ch, multi-label→single-label |
| Tiny-ImageNet | `--dataset tinyimagenet` | 200 | Scalability/runtime evaluation |

---

## Data Partitioning

Three federated data distribution strategies are supported via `--partition_type`:

| Strategy | Flag | Description |
|---|---|---|
| IID | `iid` | Uniform random split across clients |
| Non-IID (class-based) | `non_iid_classes` | Each client receives `--classes_per_client` classes |
| Non-IID (Dirichlet) | `dirichlet` | Label distribution drawn from Dir(α); lower α = more heterogeneous |

Non-IID Dirichlet (α = 0.5) is the primary evaluation setting in the paper. Per-client, per-class partition statistics for CIFAR-10 and PathMNIST are reported in Appendix I of the paper.

---

## Key Arguments

### Federated Setup
| Argument | Default | Description |
|---|---|---|
| `--num_clients` | 20 | Total number of clients |
| `--clients_per_round` | 10 | Clients participating per round |
| `--federated_rounds` | 30 | Communication rounds |
| `--local_iterations` | 1 | Local gradient steps per client per round |

### Differential Privacy
| Argument | Default | Description |
|---|---|---|
| `--epsilon` | 5.0 | Target privacy parameter ε |
| `--delta` | 1e-5 | Target privacy parameter δ |
| `--gradient_clip_norm` | 1.0 | Per-example gradient clipping bound C_g |
| `--no_dp` | — | Disable DP (no clipping, no noise) |

### SOFIM Parameters
| Argument | Default | Description |
|---|---|---|
| `--sofim_rho` | 0.5 | FIM regularization ρ > 0 |
| `--sofim_beta` | 0.9 | EMA momentum β ∈ [0, 1) |
| `--sofim_warmup_rounds` | 20 | Rounds of EMA-only warmup before enabling Sherman-Morrison preconditioning (stabilizes early rounds under tight ε) |
| `--sofim_ablation_mode` | `full` | `full` / `ema_only` / `grad_only` |
| `--sofim_disable_bias_correction` | — | Disable bias correction (recommended for ε < 2) |

---

## Reproducing Paper Experiments

The full sweep and hyperparameter search runners (`run_dp_fedgd_experiments.py`, `run_hyperparam_search.py`) used internally for the paper have been removed from this repo. To reproduce a specific configuration, call `main.py` directly with the desired flags (see Key Arguments above), or write a thin wrapper script that loops over `--epsilon`, `--use_*`, and `--seed` values and calls `main.py` with `--save_results`.

Results are saved as JSON files to the directory set by `--results_dir`.

### Plotting

```bash
python plot_comparison_accuracycurves.py
python plot_comparison_losscurves.py
```

---

## Privacy Accounting

Privacy is tracked using an exact hockey-stick divergence accountant ([hockey_stick_accountant.py](hockey_stick_accountant.py)), which computes the closed-form Gaussian DP formula for T-round composition:

$$\delta(\varepsilon) = \Phi\!\left(-\frac{\varepsilon\sigma}{\sqrt{T}\Delta} + \frac{\sqrt{T}\Delta}{2\sigma}\right) - e^{\varepsilon}\,\Phi\!\left(-\frac{\varepsilon\sigma}{\sqrt{T}\Delta} - \frac{\sqrt{T}\Delta}{2\sigma}\right)$$

Noise multiplier σ_g is calibrated via binary search to meet the target (ε, δ) budget before training begins. DP-FedSOFIM incurs no additional privacy cost beyond the underlying privatized gradient release mechanism, since all curvature and preconditioning operations are server-side post-processing on already-privatized gradients.

---

## Algorithm Details (DP-SOFIM)

**Client side** (each round k, each client i):
1. Compute per-example gradients on local data D_i
2. Clip each gradient: clip_{C_g}(∇f(θ^k, x))
3. Average clipped gradients over |D_i|
4. Add Gaussian noise: u_i^k = avg + N(0, (C_g σ_g)²/n · I_d)

**Server side** (each round k):
1. Aggregate: U^k = (1/n) Σ u_i^k
2. EMA update: M_t = β · M_{t-1} + (1 − β) · U^k
3. Bias correction: M̂_t = M_t / (1 − β^t)
4. Sherman-Morrison preconditioner: F_t^{-1} · M̂_t
5. Update: θ^{k+1} = θ^k − η · F_t^{-1} · M̂_t

The Sherman-Morrison step avoids forming the d×d Fisher matrix explicitly, keeping complexity O(d) — identical to SGD with momentum.

---

## Evaluation Summary

The paper evaluates DP-FedSOFIM against eight baselines (DP-FedGD, DP-FedAvg, DP-SCAFFOLD, DP-FedFC, DP-FedAdam, DP-FedYogi, DP-FTRL, DP-AdaFedProx) across:
- Two datasets: CIFAR-10 and PathMNIST
- Two backbones: ResNet-20 and VGG-16
- Non-IID Dirichlet (α = 0.5) partitioning, 20 clients
- Four privacy budgets: ε ∈ {0.5, 1, 5, 10}
- Three random seeds, with McNemar's test for statistical significance

DP-FedSOFIM matches or leads the best competing method at the final round in the large majority of these settings, with the Sherman-Morrison preconditioning step providing its largest gains at moderate-to-high privacy noise (ε ≈ 1–5), where curvature estimation is reliable but isotropic gradient steps are still noise-degraded.

---

## Reproducibility

All experiments use fixed random seeds (3 seeds per configuration; `--seed 42` as the default). To reproduce a single run:

```bash
python main.py --seed 42 --backbone cifar100_resnet20 --dataset pathmnist \
  --num_clients 20 --clients_per_round 20 --federated_rounds 70 \
  --partition_type dirichlet --dirichlet_alpha 0.5 \
  --epsilon 1.0 --use_sofim --sofim_rho 10.0 --sofim_beta 0.8 \
  --save_results --results_dir ./results
```

---

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{nair2026dpfedsofim,
  title   = {{DP}-Fed{SOFIM}: Differentially Private Federated Stochastic Optimization
             using Regularized Fisher Information Matrix},
  author  = {Nair, Sidhant and Sen, Tanmay and Sen, Mrinmay and Banerjee, Sayantan},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=aDzj9DrwAR}
}
```

---

## License
Copyright (c) 2026 Sidhant Nair

Licensed under the Apache License, Version 2.0.
See the LICENSE file for details.
