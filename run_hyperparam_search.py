import os
import json
import copy
import itertools
import torch
import numpy as np
import sys

from train import run_dpfedgd_training, parse_args

# =============================================================================
# CONFIGURATION
# =============================================================================

ALGORITHM = "fedavg"   # options: fedgd, sofim, fedfc, scaffold, fedadam, fedyogi
STAGE     = "coarse"  # "coarse" first, then "fine" after reviewing coarse results

# --- Fixed search config ---
SEARCH_DATASET   = "pathmnist"
SEARCH_CLIENTS   = 20
SEARCH_PARTITION = "dirichlet"
SEARCH_ALPHA     = 0.1
SEARCH_SEED      = 42
SEARCH_ROUNDS    = 50
SEARCH_BACKBONE  = "cifar100_resnet20"
SEARCH_EPSILONS  = [0.5, 1, 5, 10]

# --- Output directory ---
results_dir = f"./results_hparam_alpha0.1/{ALGORITHM}/{STAGE}"
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# HYPERPARAM GRIDS
# Coarse: wide range, large steps — find the right region
# Fine:   narrow range around best coarse result, smaller steps
#
# After coarse search, read printed best params and update FINE_OVERRIDE below
# =============================================================================

GRIDS = {

    "fedgd": {
        "coarse": {
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        },
        "fine": {
            "learning_rate": [0.03, 0.05, 0.1, 0.3, 0.5],
        },
    },

    "fedavg": {
        "coarse": {
            "learning_rate":        [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0],
            "local_iterations":     [1, 3, 5, 10, 20],
        },
    "fine": {
            "learning_rate":        [0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            "local_iterations":     [1, 2, 5, 7, 10],
        },
    },

    "sofim": {
        "coarse": {
            "learning_rate": [ 0.1, 0.5, 1.0, 5.0],
            "sofim_rho":     [0.01, 0.1, 1.0, 5.0, 10.0],
            "sofim_beta":    [0.8, 0.9, 0.99],
        },
        "fine": {
            "learning_rate": [0.05, 0.1, 0.2, 0.5],
            "sofim_rho":     [0.1, 0.3, 0.5, 1.0, 2.0],
            "sofim_beta":    [0.85, 0.9, 0.93, 0.95],
        },
    },

    "fedfc": {
        "coarse": {
            "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0],
            "fc_gamma":      [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "fine": {
            "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.5],
            "fc_gamma":      [0.05, 0.1, 0.5, 1.0, 5.0],
        },
    },

    "scaffold": {
        "coarse": {
            "learning_rate":        [0.01, 0.1, 0.5, 1.0, 5.0],
            "scaffold_client_lr":   [0.001, 0.01, 0.1, 0.5],
            "scaffold_local_steps": [1, 3, 5],
        },
        "fine": {
            "learning_rate":        [0.05, 0.1, 0.3, 0.5, 1.0],
            "scaffold_client_lr":   [0.005, 0.01, 0.05, 0.1],
            "scaffold_local_steps": [1, 2, 3, 5],
        },
    },

    "fedadam": {
        "coarse": {
            "learning_rate":   [0.0001, 0.001, 0.01, 0.1, 1.0],
            "fedadam_beta1":   [0.0, 0.9, 0.99],
            "fedadam_beta2":   [0.99, 0.999],
            "fedadam_epsilon": [1e-8, 1e-4, 1e-2, 0.1],
        },
        "fine": {
            "learning_rate":   [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            "fedadam_beta1":   [0.9, 0.95, 0.99],
            "fedadam_beta2":   [0.99, 0.999, 0.9999],
            "fedadam_epsilon": [1e-6, 1e-4, 1e-3, 0.01],
        },
    },

    "fedyogi": {
        "coarse": {
            "learning_rate":   [0.0001, 0.001, 0.01, 0.1, 1.0],
            "fedyogi_beta1":   [0.0, 0.9, 0.99],
            "fedyogi_beta2":   [0.99, 0.999],
            "fedyogi_epsilon": [1e-4, 1e-3, 0.01, 0.1],
        },
        "fine": {
            "learning_rate":   [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            "fedyogi_beta1":   [0.9, 0.95, 0.99],
            "fedyogi_beta2":   [0.99, 0.999, 0.9999],
            "fedyogi_epsilon": [1e-3, 5e-3, 0.01, 0.05],
        },
    },
}

# =============================================================================
# FINE STAGE OVERRIDE
# After coarse search, fill this in per epsilon with narrowed ranges.
# Leave as {} to use the full fine grid above.
#
# Example:
# FINE_OVERRIDE = {
#     0.5: {"learning_rate": [0.05, 0.1, 0.2], "sofim_rho": [0.5, 1.0, 2.0]},
#     1:   {"learning_rate": [0.1, 0.2, 0.5],  "sofim_rho": [0.1, 0.5, 1.0]},
#     5:   {"learning_rate": [0.1, 0.5, 1.0],  "sofim_rho": [0.01, 0.1, 0.5]},
#     10:  {"learning_rate": [0.5, 1.0, 2.0],  "sofim_rho": [0.01, 0.05, 0.1]},
# }
# =============================================================================
# FINE_OVERRIDE = {
#     0.5: {"learning_rate": [0.1, 0.3, 0.5, 1.0],   "sofim_rho": [0.5, 1.0, 2.0],       "sofim_beta": [0.95, 0.97, 0.99]},
#     1:   {"learning_rate": [1.0, 3.0, 5.0, 10.0],  "sofim_rho": [5.0, 10.0, 20.0],     "sofim_beta": [0.7, 0.8, 0.9]},   # expanded — coarse hit boundary
#     5:   {"learning_rate": [0.5, 1.0, 2.0],         "sofim_rho": [0.5, 1.0, 2.0],       "sofim_beta": [0.7, 0.8, 0.9]},
#     10:  {"learning_rate": [0.5, 1.0, 2.0],         "sofim_rho": [0.5, 1.0, 2.0],       "sofim_beta": [0.7, 0.8, 0.9]},
# }

# =============================================================================
# ARGS FACTORY
# =============================================================================

sys.argv = ['train.py']
default_args = parse_args()


def make_args(eps, hparams):
    args = copy.deepcopy(default_args)

    # All flags off
    args.use_fednew_fc          = False
    args.use_fedfc              = False
    args.use_scaffold           = False
    args.use_fedavg_multi_iters = False
    args.use_fedadam            = False
    args.use_fedyogi            = False
    args.use_fedprox            = False
    args.use_sofim              = False
    args.no_dp                  = False

    # Enable algorithm
    if ALGORITHM == "sofim":
        args.use_sofim                     = True
        args.sofim_rho                     = hparams.get("sofim_rho", 0.5)
        args.sofim_beta                    = hparams.get("sofim_beta", 0.9)
        args.sofim_ablation_mode           = "full"
        args.sofim_disable_bias_correction = False
        args.sofim_adaptive_params         = False
        args.sofim_warmup_rounds           = 0

    elif ALGORITHM == "fedfc":
        args.use_fedfc  = True
        args.fc_cc      = 1.0
        args.fc_gamma   = hparams.get("fc_gamma", 1.0)
        args.fc_sigma_c = 1.0

    elif ALGORITHM == "scaffold":
        args.use_scaffold         = True
        args.scaffold_client_lr   = hparams.get("scaffold_client_lr", 0.1)
        args.scaffold_local_steps = hparams.get("scaffold_local_steps", 5)
        args.scaffold_server_lr   = None

    elif ALGORITHM == "fedadam":
        args.use_fedadam     = True
        args.fedadam_beta1   = hparams.get("fedadam_beta1", 0.9)
        args.fedadam_beta2   = hparams.get("fedadam_beta2", 0.99)
        args.fedadam_epsilon = hparams.get("fedadam_epsilon", 1e-8)

    elif ALGORITHM == "fedyogi":
        args.use_fedyogi     = True
        args.fedyogi_beta1   = hparams.get("fedyogi_beta1", 0.9)
        args.fedyogi_beta2   = hparams.get("fedyogi_beta2", 0.99)
        args.fedyogi_epsilon = hparams.get("fedyogi_epsilon", 1e-3)

    # SOFIM defaults to avoid missing attr errors
    for attr, default in [
        ('sofim_rho', 0.5), ('sofim_beta', 0.9),
        ('sofim_ablation_mode', 'full'), ('sofim_warmup_rounds', 0),
        ('sofim_disable_bias_correction', False),
        ('sofim_adaptive_params', False),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    # Shared config
    args.learning_rate      = hparams.get("learning_rate", 0.1)
    args.epsilon            = float(eps)
    args.delta              = 1e-5
    args.federated_rounds   = SEARCH_ROUNDS
    args.dataset            = SEARCH_DATASET
    args.binary             = False
    args.backbone           = SEARCH_BACKBONE
    args.num_clients        = SEARCH_CLIENTS
    args.clients_per_round  = SEARCH_CLIENTS
    args.partition_type     = SEARCH_PARTITION
    args.dirichlet_alpha    = SEARCH_ALPHA
    args.classes_per_client = 2
    args.local_iterations   = 1
    args.batch_size         = 64
    args.gradient_clip_norm = 10.0
    args.eval_every         = 10
    args.seed               = SEARCH_SEED
    args.verbose            = False
    args.save_results       = False

    return args


# =============================================================================
# BUILD GRID FOR CURRENT STAGE + EPSILON
# =============================================================================

def build_grid(eps):
    if STAGE == "coarse":
        grid = GRIDS[ALGORITHM]["coarse"]
    else:
        override = FINE_OVERRIDE.get(eps, {})
        base = GRIDS[ALGORITHM]["fine"]
        grid = {k: override.get(k, base[k]) for k in base}

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    return keys, combos


def hparam_tag(hparams):
    parts = []
    for k, v in sorted(hparams.items()):
        parts.append(f"{k}{v:.0e}" if isinstance(v, float) and v < 0.01 else f"{k}{v}")
    return "_".join(parts)


# =============================================================================
# MAIN SEARCH LOOP
# =============================================================================

total = sum(len(build_grid(eps)[1]) for eps in SEARCH_EPSILONS)

print(f"\n{'='*70}")
print(f"Hyperparam Search: {ALGORITHM.upper()} | Stage: {STAGE.upper()}")
print(f"Dataset: {SEARCH_DATASET} | Clients: {SEARCH_CLIENTS} | Partition: {SEARCH_PARTITION}")
print(f"Epsilons: {SEARCH_EPSILONS} | Rounds: {SEARCH_ROUNDS} | Seed: {SEARCH_SEED}")
print(f"Total runs: {total}")
print(f"{'='*70}\n")

best_per_eps = {eps: {"acc": -1, "hparams": None} for eps in SEARCH_EPSILONS}
run_count = 0

for eps in SEARCH_EPSILONS:
    keys, combos = build_grid(eps)
    print(f"\n{'='*60}")
    print(f"ε = {eps} | {len(combos)} configs")
    print(f"{'='*60}")

    eps_results = []

    for combo in combos:
        hparams = dict(zip(keys, combo))
        tag = hparam_tag(hparams)
        filepath = os.path.join(results_dir, f"{ALGORITHM}_eps{eps}_{tag}.json")
        run_count += 1

        print(f"\n[{run_count}/{total}] ε={eps} | {hparams}")

        if os.path.exists(filepath):
            print(f"  ⏭  Skipping (exists)")
            try:
                with open(filepath) as f:
                    data = json.load(f)
                acc = data.get("final_stats", {}).get("accuracy", 0)
                eps_results.append({"hparams": hparams, "acc": acc})
                if acc > best_per_eps[eps]["acc"]:
                    best_per_eps[eps] = {"acc": acc, "hparams": hparams}
            except Exception:
                pass
            continue

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            args = make_args(eps, hparams)
            results = run_dpfedgd_training(args)

            if results is not None:
                acc = results.get("final_stats", {}).get("accuracy", 0)
                results["hparam_config"] = {
                    "algorithm": ALGORITHM, "stage": STAGE,
                    "epsilon": eps, "hparams": hparams,
                    "final_accuracy": acc,
                }
                with open(filepath, "w") as f:
                    json.dump(results, f, indent=2)

                eps_results.append({"hparams": hparams, "acc": acc})
                if acc > best_per_eps[eps]["acc"]:
                    best_per_eps[eps] = {"acc": acc, "hparams": hparams}
                print(f"  ✅ Acc: {acc*100:.2f}%")
            else:
                print(f"  ❌ Failed")

        except Exception as e:
            print(f"  ❗ Error: {e}")
            import traceback
            traceback.print_exc()

    eps_results.sort(key=lambda x: x["acc"], reverse=True)
    print(f"\n--- Top 5 for ε={eps} ---")
    for i, r in enumerate(eps_results[:5]):
        print(f"  {i+1}. Acc={r['acc']*100:.2f}% | {r['hparams']}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

keys_all = list(GRIDS[ALGORITHM][STAGE].keys())

print(f"\n{'='*70}")
print(f"COMPLETE — {ALGORITHM.upper()} | {STAGE.upper()}")
print(f"{'='*70}")
print(f"\nBest per ε:")
for eps in SEARCH_EPSILONS:
    b = best_per_eps[eps]
    print(f"  ε={eps}: {b['acc']*100:.2f}% | {b['hparams']}")

print(f"\nConsistency across ε:")
for key in keys_all:
    vals = [best_per_eps[eps]["hparams"][key]
            for eps in SEARCH_EPSILONS
            if best_per_eps[eps]["hparams"] is not None]
    if len(set(str(v) for v in vals)) == 1:
        print(f"  {key} = {vals[0]}  ✅ consistent — use one value for all ε")
    else:
        print(f"  {key}: {dict(zip(SEARCH_EPSILONS, vals))}  ⚠ varies — use per-ε values")

if STAGE == "coarse":
    print(f"\nNEXT: Review best params → fill FINE_OVERRIDE → set STAGE='fine' → rerun")
else:
    print(f"\nNEXT: Update run_dp_fedgd_experiments.py with best params → run full sweep")

print(f"\nResults: {results_dir}/")