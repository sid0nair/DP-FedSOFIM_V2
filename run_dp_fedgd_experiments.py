import os
import json
import copy
import torch
import numpy as np
import sys

from train import run_dpfedgd_training, parse_args

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Backbone ---
SWEEP_BACKBONE = "cifar100_resnet20"

# --- Experiment Dimensions ---
DATASETS         = ["pathmnist"]
NUM_CLIENTS_LIST = [20]
epsilons         = [0.5,1,5,10]
SEEDS            = [42]#123, 7
PARTITION_TYPES  = ["dirichlet"]
DIRICHLET_ALPHA  = 0.5
rounds_list      = [70]

# --- Force overwrite existing results ---
FORCE_RERUN = True

# --- Algorithm Selection — set True for each you want to run ---
USE_FEDNEW_FC    = False
USE_FEDFC        = False
USE_SOFIM        = False
USE_SCAFFOLD     = True
USE_FEDAVG_MULTI = False
USE_FEDADAM      = False
USE_FEDYOGI      = False
USE_FEDPROX      = False
USE_FEDGD        = False
NO_DP_BASELINE   = False

# =============================================================================
# TUNED HYPERPARAMETERS
# cifar10, 20 clients, Dirichlet α=0.5, batch=64 (memory chunk only)
# All params from coarse→fine hyperparam search
# =============================================================================

# --- FedNew-FC defaults (not tuned) ---
FN_ALPHA    = 0.1
FN_RHO      = 0.1
FN_C1       = 1.0
FN_C2       = 1.0
FN_C_PRIMAL = 1.0

# --- FedGD ---
FEDGD_LR_PER_EPS = {0.5: 0.1, 1: 0.1, 5: 0.1, 10: 0.1}

# ---FedAvg
# Add these after FEDGD_LR_PER_EPS:
FEDAVG_LR_PER_EPS          = {0.5: 1.0, 0.75: 1.0, 1: 1.0, 2: 1.0, 5: 1.0, 10: 1.0}
FEDAVG_CLIENT_LR_PER_EPS   = {0.5: 0.1, 0.75: 0.1, 1: 0.1, 2: 0.1, 5: 0.1, 10: 0.1}
FEDAVG_LOCAL_STEPS_PER_EPS = {0.5: 1,   0.75: 1,   1: 1,   2: 1,   5: 1,   10: 1}

# --- FedFC ---
FEDFC_LR_PER_EPS    = {0.5: 0.4, 1: 0.3, 5: 2.0, 10: 0.5}
FEDFC_GAMMA_PER_EPS = {0.5: 2.0, 1: 1.0, 5: 5.0, 10: 5.0}
FEDFC_CC      = 1.0
FEDFC_SIGMA_C = 1.0

# --- SCAFFOLD ---
SCAFFOLD_LR_PER_EPS          = {0.5: 0.5, 1: 0.5, 5: 0.5, 10: 1}
SCAFFOLD_CLIENT_LR_PER_EPS   = {0.5: 0.2, 1: 0.1, 5: 0.1, 10: 0.05}
SCAFFOLD_LOCAL_STEPS_PER_EPS = {0.5: 1,   1: 5,   5: 7,   10: 7}

# --- FedAdam ---
FEDADAM_LR_PER_EPS      = {0.5: 0.02, 1: 0.05, 5: 0.1,    10: 0.05}
FEDADAM_BETA1_PER_EPS   = {0.5: 0.8, 1: 0.9,  5: 0.9,    10: 0.95}
FEDADAM_BETA2_PER_EPS   = {0.5: 0.8, 1: 0.999,5: 0.9999, 10: 0.9999}
FEDADAM_EPSILON_PER_EPS = {0.5: 0.1,  1: 0.05, 5: 0.01,   10: 1e-5}

# --- FedYogi ---
FEDYOGI_LR_PER_EPS      = {0.5: 0.02,  1: 0.2,  5: 0.05,   10: 0.05}
FEDYOGI_BETA1_PER_EPS   = {0.5: 0.90, 1: 0.9,  5: 0.9,    10: 0.95}
FEDYOGI_BETA2           = 0.9
FEDYOGI_EPSILON_PER_EPS = {0.5: 0.1,  1: 0.05, 5: 0.01,  10: 1e-5}

# --- SOFIM ---
SOFIM_LR_PER_EPS = {
    0.5: 0.2,
    0.75: 0.5,
    1: 3.0,
    2: 0.8,
    5: 0.8,
    10: 0.8,
}

SOFIM_RHO_PER_EPS = {
    0.5: 0.5,
    0.75: 0.5,
    1: 10,
    2: 0.5,
    5: 0.5,
    10: 0.5,
}

SOFIM_BETA_PER_EPS = {
    0.5: 0.95,
    0.75: 0.95,
    1: 0.8,
    2: 0.9,
    5: 0.9,
    10: 0.9,
}

BIAS_CORR = {
    0.5: True,
    0.75: True,
    1: True,
    2: True,
    5: False,
    10: False,
}

SOFIM_WARMUP_PER_EPS = {
    0.5: 10,
    0.75: 10,
    1: 0,
    2: 0,
    5: 0,
    10: 0,
}
SOFIM_WEIGHT_DECAY = 5e-4

# --- ρ Sweep (SOFIM only) ---
RHO_SWEEP     = False
RHO_VALUES    = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
RHO_DATASET   = "pathmnist"
RHO_CLIENTS   = 20
RHO_PARTITION = "dirichlet"
RHO_ROUNDS    = 70
RHO_EPSILONS  = [1, 5]

# --- Ablation Sweep (SOFIM only) ---
ABLATION_SWEEP     = False
ABLATION_MODES     = ["grad_only", "ema_only", "full"]
ABLATION_DATASET   = "pathmnist"
ABLATION_CLIENTS   = 20
ABLATION_PARTITION = "dirichlet"
ABLATION_ROUNDS    = 70

# --- Output directory ---
results_dir = "./results_sweep_postreview_pathmnist_final_runtime"
os.makedirs(results_dir, exist_ok=True)

# =============================================================================
# ACTIVE ALGORITHMS
# =============================================================================

ACTIVE_ALGOS = []
if USE_FEDGD:        ACTIVE_ALGOS.append("fedgd")
if USE_FEDFC:        ACTIVE_ALGOS.append("fedfc")
if USE_FEDAVG_MULTI: ACTIVE_ALGOS.append("fedavg")
if USE_SCAFFOLD:     ACTIVE_ALGOS.append("scaffold")
if USE_FEDADAM:      ACTIVE_ALGOS.append("fedadam")
if USE_FEDYOGI:      ACTIVE_ALGOS.append("fedyogi")
if USE_SOFIM:        ACTIVE_ALGOS.append("sofim")

print(f"Starting multi-algorithm sweep")
print(f"  Algorithms:  {ACTIVE_ALGOS}")
print(f"  Datasets:    {DATASETS}")
print(f"  Clients:     {NUM_CLIENTS_LIST}")
print(f"  Epsilons:    {epsilons}")
print(f"  Seeds:       {SEEDS}")
print(f"  Partitions:  {PARTITION_TYPES}")
print(f"  Rounds:      {rounds_list}")
print(f"  Force rerun: {FORCE_RERUN}")
if USE_SOFIM and RHO_SWEEP:
    print(f"  ρ sweep: {RHO_VALUES}")
if USE_SOFIM and ABLATION_SWEEP:
    print(f"  Ablation modes: {ABLATION_MODES}")

# =============================================================================
# ARGS FACTORY
# =============================================================================

sys.argv = ['train.py']
default_args = parse_args()


def make_args(algo, eps, rounds, seed, dataset, num_clients, partition_type,
              sofim_rho=None, sofim_ablation_mode="full"):
    """Build args for one algorithm + config combination."""
    args = copy.deepcopy(default_args)
    eps_f = float(eps)

    # --- All algorithm flags off ---
    args.use_fednew_fc          = False
    args.use_fedfc              = False
    args.use_scaffold           = False
    args.use_fedavg_multi_iters = False
    args.use_fedadam            = False
    args.use_fedyogi            = False
    args.use_fedprox            = False
    args.use_sofim              = False
    args.no_dp                  = NO_DP_BASELINE

    # --- Algorithm-specific params ---
    if algo == "fedgd":
        args.learning_rate = FEDGD_LR_PER_EPS.get(eps_f, 0.1)

    elif algo == "fedfc":
        args.use_fedfc     = True
        args.fc_cc         = FEDFC_CC
        args.fc_gamma      = FEDFC_GAMMA_PER_EPS.get(eps_f, 1.0)
        args.fc_sigma_c    = FEDFC_SIGMA_C
        args.learning_rate = FEDFC_LR_PER_EPS.get(eps_f, 0.1)

    elif algo == "fedavg":
        args.use_fedavg_multi_iters = True
        args.fedavg_local_steps = FEDAVG_LOCAL_STEPS_PER_EPS.get(eps_f, 5)
        args.fedavg_client_lr = FEDAVG_CLIENT_LR_PER_EPS.get(eps_f, 0.1)
        args.learning_rate = FEDAVG_LR_PER_EPS.get(eps_f, 0.1)

    elif algo == "scaffold":
        args.use_scaffold         = True
        args.scaffold_local_steps = SCAFFOLD_LOCAL_STEPS_PER_EPS.get(eps_f, 1)
        args.scaffold_client_lr   = SCAFFOLD_CLIENT_LR_PER_EPS.get(eps_f, 0.1)
        args.scaffold_server_lr   = None
        args.learning_rate        = SCAFFOLD_LR_PER_EPS.get(eps_f, 0.5)

    elif algo == "fedadam":
        args.use_fedadam     = True
        args.fedadam_beta1   = FEDADAM_BETA1_PER_EPS.get(eps_f, 0.9)
        args.fedadam_beta2   = FEDADAM_BETA2_PER_EPS.get(eps_f, 0.99)
        args.fedadam_epsilon = FEDADAM_EPSILON_PER_EPS.get(eps_f, 1e-8)
        args.learning_rate   = FEDADAM_LR_PER_EPS.get(eps_f, 0.1)

    elif algo == "fedyogi":
        args.use_fedyogi     = True
        args.fedyogi_beta1   = FEDYOGI_BETA1_PER_EPS.get(eps_f, 0.9)
        args.fedyogi_beta2   = FEDYOGI_BETA2
        args.fedyogi_epsilon = FEDYOGI_EPSILON_PER_EPS.get(eps_f, 1e-3)
        args.learning_rate   = FEDYOGI_LR_PER_EPS.get(eps_f, 0.1)


    elif algo == "sofim":

        args.use_sofim = True
        args.sofim_rho = sofim_rho if sofim_rho is not None else SOFIM_RHO_PER_EPS.get(eps_f, 0.5)
        args.sofim_beta = SOFIM_BETA_PER_EPS.get(eps_f, 0.9)
        args.sofim_ablation_mode = sofim_ablation_mode
        use_bias_corr = BIAS_CORR.get(eps_f, True)
        args.sofim_disable_bias_correction = not use_bias_corr
        args.sofim_adaptive_params = False
        args.sofim_weight_decay = SOFIM_WEIGHT_DECAY
        if sofim_ablation_mode in ("ema_only", "grad_only"):
            args.sofim_warmup_rounds = 0
            rho = SOFIM_RHO_PER_EPS.get(eps_f, 0.5)
            args.learning_rate = SOFIM_LR_PER_EPS.get(eps_f, 0.5) / rho
        else:
            args.sofim_warmup_rounds = SOFIM_WARMUP_PER_EPS.get(eps_f, 0)
            args.learning_rate = SOFIM_LR_PER_EPS.get(eps_f, 0.5)

    # --- SOFIM defaults to avoid missing attr errors ---
    if not hasattr(args, 'sofim_rho') or not args.use_sofim:
        args.sofim_rho                     = SOFIM_RHO_PER_EPS.get(eps_f, 0.5)
        args.sofim_beta                    = SOFIM_BETA_PER_EPS.get(eps_f, 0.9)
        args.sofim_ablation_mode           = sofim_ablation_mode
        args.sofim_disable_bias_correction = False
        args.sofim_adaptive_params         = False
        args.sofim_warmup_rounds           = 0
        args.sofim_weight_decay = 0.0

    # --- FedNew-FC defaults ---
    args.fn_alpha    = FN_ALPHA
    args.fn_rho      = FN_RHO
    args.fn_c1       = FN_C1
    args.fn_c2       = FN_C2
    args.fn_c_primal = FN_C_PRIMAL

    # --- Shared experiment config ---
    args.epsilon            = eps_f
    args.delta              = 1e-5
    args.federated_rounds   = int(rounds)
    args.dataset            = dataset
    args.binary             = (dataset == "cifar10_binary")
    args.backbone           = SWEEP_BACKBONE
    args.num_clients        = num_clients
    args.clients_per_round  = num_clients
    args.partition_type     = partition_type
    args.dirichlet_alpha    = DIRICHLET_ALPHA
    args.classes_per_client = 2
    args.local_iterations   = 1
    args.batch_size         = 64
    args.gradient_clip_norm = 10.0
    args.eval_every         = 10
    args.seed               = seed
    args.verbose            = False
    args.save_results       = False

    return args


# =============================================================================
# RUN + SAVE HELPER
# =============================================================================

def run_and_save(args, filepath, extra_config=None):
    if os.path.exists(filepath) and not FORCE_RERUN:
        print(f"  ⏭  Skipping (exists): {os.path.basename(filepath)}")
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = run_dpfedgd_training(args)

        if results is not None:
            results['sweep_config'] = {
                'dataset':             args.dataset,
                'num_clients':         args.num_clients,
                'partition_type':      args.partition_type,
                'dirichlet_alpha':     args.dirichlet_alpha,
                'epsilon':             args.epsilon,
                'rounds':              args.federated_rounds,
                'seed':                args.seed,
                'backbone':            args.backbone,
                'learning_rate':       args.learning_rate,
                'sofim_rho':           getattr(args, 'sofim_rho', None),
                'sofim_beta':          getattr(args, 'sofim_beta', None),
                'sofim_ablation_mode': getattr(args, 'sofim_ablation_mode', 'full'),
                'sofim_weight_decay': getattr(args, 'sofim_weight_decay', 0.0),
            }
            if extra_config:
                results['sweep_config'].update(extra_config)

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  ✅ Saved: {os.path.basename(filepath)}")
        else:
            print(f"  ❌ Failed: results is None")

    except Exception as e:
        print(f"  ❗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Always release GPU memory after each run, success or failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()


# =============================================================================
# MAIN RESULTS SWEEP
# =============================================================================

print("\n" + "=" * 70)
print("MAIN RESULTS SWEEP")
print("=" * 70)

for algo in ACTIVE_ALGOS:
    print(f"\n{'='*70}")
    print(f"Algorithm: {algo.upper()}")
    print(f"{'='*70}")

    for dataset in DATASETS:
        for num_clients in NUM_CLIENTS_LIST:
            for partition_type in PARTITION_TYPES:
                for eps in epsilons:
                    for rounds in rounds_list:
                        for seed in SEEDS:
                            tag = (f"{algo}_{dataset}_"
                                   f"c{num_clients}_{partition_type}_"
                                   f"eps{eps}_r{rounds}_seed{seed}")
                            filepath = os.path.join(results_dir, f"{tag}.json")
                            print(f"\n▶ {tag}")
                            args = make_args(algo, eps, rounds, seed,
                                             dataset, num_clients, partition_type)
                            run_and_save(args, filepath,
                                         extra_config={'algorithm': algo})

# =============================================================================
# RHO SENSITIVITY SWEEP (SOFIM only)
# =============================================================================

if USE_SOFIM and RHO_SWEEP:
    print("\n" + "=" * 70)
    print("RHO SENSITIVITY SWEEP")
    print("=" * 70)

    for eps in RHO_EPSILONS:
        for rho in RHO_VALUES:
            for seed in SEEDS:
                tag = (f"rho_sweep_sofim_{RHO_DATASET}_c{RHO_CLIENTS}_"
                       f"{RHO_PARTITION}_eps{eps}_rho{rho}_seed{seed}")
                filepath = os.path.join(results_dir, f"{tag}.json")
                print(f"\n▶ {tag}")
                args = make_args("sofim", eps, RHO_ROUNDS, seed,
                                 RHO_DATASET, RHO_CLIENTS, RHO_PARTITION,
                                 sofim_rho=rho)
                run_and_save(args, filepath,
                             extra_config={'algorithm': 'sofim',
                                           'sweep_type': 'rho', 'rho': rho})

# =============================================================================
# ABLATION SWEEP (SOFIM only)
# =============================================================================

if USE_SOFIM and ABLATION_SWEEP:
    print("\n" + "=" * 70)
    print("ABLATION SWEEP (Fisher preconditioning vs EMA momentum)")
    print("=" * 70)

    for mode in ABLATION_MODES:
        for eps in epsilons:
            for seed in SEEDS:
                tag = (f"ablation_{mode}_{ABLATION_DATASET}_c{ABLATION_CLIENTS}_"
                       f"{ABLATION_PARTITION}_eps{eps}_seed{seed}")
                filepath = os.path.join(results_dir, f"{tag}.json")
                print(f"\n▶ {tag}")
                args = make_args("sofim", eps, ABLATION_ROUNDS, seed,
                                 ABLATION_DATASET, ABLATION_CLIENTS,
                                 ABLATION_PARTITION,
                                 sofim_ablation_mode=mode)
                run_and_save(args, filepath,
                             extra_config={'algorithm': 'sofim',
                                           'sweep_type': 'ablation',
                                           'ablation_mode': mode})

print(f"\n🎉 All experiments completed.")
print(f"   Results saved to: {results_dir}/")