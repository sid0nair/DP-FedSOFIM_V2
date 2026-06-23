"""
Record-Level DP-FedGD Training Implementation (Algorithm 5)

CLIENT SIDE (for each client i, each round k):
1. Compute per-example gradients on local data
2. Clip each gradient: clip_C_g(∇f(θ^k, x))
3. Average clipped gradients: (1/|D_i|) Σ clip_C_g(∇f(θ^k, x))
4. Add Gaussian noise: u_i^k = average + E_i^k where E_i^k ~ N(0, I_d(C_g σ_g)²/n)
5. Send u_i^k to server

SERVER SIDE (each round k):
1. Aggregate noisy updates: U^k = (1/n) Σ_{i=1}^n u_i^k
2. Update model parameters: θ^{k+1} = θ^k - η · U^k
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import os

# Import dataset and sanitizer
from dataset import (
    get_federated_cifar10_feature_loaders,
    get_federated_cifar10_binary_features,
    get_federated_chestmnist_features,
    get_federated_pathmnist_features,
    get_federated_bloodmnist_features,
    get_federated_dermamnist_features,
    get_federated_tinyimagenet_features,
)

from sanitizer import DPFedGDSanitizer, DPFedNewSanitizer, DPFedFCSanitizer, DPFedScaffoldSanitizer, DPFedFTRLSanitizer

def parse_args():
    p = argparse.ArgumentParser(description="Record-Level DP-FedGD (Algorithm 5) on CIFAR-10 and MedMNIST")

    # Model
    p.add_argument("--backbone", type=str, default="cifar100_resnet20",
                   choices=["resnet20", "resnet32", "resnet44", "resnet56",
                            "cifar100_resnet20", "cifar100_resnet56",
                            "cifar100_vgg16_bn"],
                   help="Backbone architecture (chenyaofo CIFAR-pretrained)")

    # Federated setup
    p.add_argument("--num_clients", type=int, default=20,
                   help="Total number of clients (n in algorithm)")
    p.add_argument("--clients_per_round", type=int, default=20,
                   help="Clients per round (for Algorithm 5, typically n)")
    p.add_argument("--federated_rounds", type=int, default=100,
                   help="Training rounds (T in algorithm)")
    p.add_argument("--local_iterations", type=int, default=1,
                   help="Local gradient computations (typically 1 for Algorithm 5)")

    # Data
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar10_binary", "chestmnist", "pathmnist", "bloodmnist", "dermamnist",
                 "tinyimagenet"],
        help="Dataset to use for federated training."
    )
    p.add_argument("--partition_type", type=str, default="iid",
                   choices=["iid", "non_iid_classes", "dirichlet"])
    p.add_argument("--dirichlet_alpha", type=float, default=0.5,
                   help="Dirichlet concentration parameter for non-IID partitioning")
    p.add_argument("--classes_per_client", type=int, default=2,
                   help="Classes per client for non_iid_classes partitioning")
    p.add_argument("--binary", action="store_true",
                   help="Binary classification (classes 0 vs 1)")

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.1,
                   help="Server learning rate (η in algorithm)")
    # SOFIM parameters (server-side second-order optimization)
    # Privacy
    p.add_argument("--gradient_clip_norm", type=float, default=1.0,
                   help="Gradient clipping constant (C_g in algorithm)")
    p.add_argument("--epsilon", type=float, default=5.0,
                   help="Target privacy ε")
    p.add_argument("--delta", type=float, default=1e-5,
                   help="Target privacy δ")
    p.add_argument("--no_dp", action="store_true",
                   help="Disable DP sanitization (no clipping/noise) for sanity checks")

    # SOFIM parameters (server-side second-order optimization)
    p.add_argument("--use_sofim", action="store_true",
                   help="Use SOFIM (second-order) server optimizer instead of SGD")
    p.add_argument("--sofim_beta", type=float, default=0.9,
                   help="SOFIM momentum parameter β ∈ [0, 1)")
    p.add_argument("--sofim_rho", type=float, default=0.5,
                   help="SOFIM FIM regularization ρ > 0 (try 0.1, 0.5, or 1.0)")
    p.add_argument("--sofim_disable_bias_correction", action="store_true",
                   help="Disable bias correction in SOFIM (recommended for ε < 2)")
    p.add_argument("--sofim_adaptive_params", action="store_true",
                   help="Use epsilon-adaptive hyperparameters for SOFIM")
    p.add_argument("--sofim_weight_decay", type=float, default=0.0,
                   help="L2 weight decay for SOFIM server update (λ, adds λ·θ to preconditioned gradient)")
    # DP-SCAFFOLD parameters (record-level full-batch)
    p.add_argument("--use_scaffold", action="store_true",
                   help="Use record-level full-batch DP-SCAFFOLD (control variates)")
    p.add_argument("--scaffold_local_steps", type=int, default=5,
                   help="Number of local steps M for DP-SCAFFOLD")
    p.add_argument("--scaffold_client_lr", type=float, default=0.1,
                   help="Local learning rate η_l for DP-SCAFFOLD")
    p.add_argument("--scaffold_server_lr", type=float, default=None,
                   help="Global learning rate η_g for DP-SCAFFOLD (defaults to --learning_rate)")

    p.add_argument("--use_fedavg_multi_iters", action="store_true",
                   help="Use DP-FedAvg with K>1 local SGD steps")
    p.add_argument("--fedavg_local_steps", type=int, default=5,
                   help="Number of local steps K for DP-FedAvg")
    p.add_argument("--fedavg_client_lr", type=float, default=0.1,
                   help="Local learning rate eta_l for DP-FedAvg")
    # Evaluation
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sofim_warmup_rounds", type=int, default=20,
                   help="Number of rounds to use SGD before switching to SOFIM")
    p.add_argument("--sofim_warmup_use_ftrl", action="store_true",
                   help="Use DP-FTRL (binary tree) update rule during SOFIM warmup rounds instead of "
                        "DP-FedGD/EMA. Reduces effective compositions from T to "
                        "ceil(log2(warmup))+1 + (T-warmup), giving lower noise for the same budget.")
    p.add_argument("--sofim_ablation_mode", type=str, default="full",
                   choices=["full", "ema_only", "grad_only"],
                   help="SOFIM ablation: full=SOFIM, ema_only=EMA no SM, grad_only=raw gradient")

    # DP-FedFC parameters
    p.add_argument("--use_fedfc", action="store_true",
                   help="Use DP-FedFC (feature covariance preconditioning)")
    p.add_argument("--fc_cc", type=float, default=1.0,
                   help="FedFC clipping constant for features Cc")
    p.add_argument("--fc_gamma", type=float, default=1.0,
                   help="FedFC regularization parameter γ")
    p.add_argument("--fc_sigma_c", type=float, default=None,
                   help="FedFC noise multiplier for covariance")

    # FedNew-FC parameters (client-side ADMM / feature-covariance preconditioning)
    p.add_argument("--use_fednew_fc", action="store_true",
                   help="Use DP-FedNew-FC (client-side ADMM/preconditioning) instead of FedGD/SOFIM")
    p.add_argument("--fn_alpha", type=float, default=1.0,
                   help="FedNew-FC dual update step size α")
    p.add_argument("--fn_rho", type=float, default=1.0,
                   help="FedNew-FC ADMM penalty ρ")
    p.add_argument("--fn_c1", type=float, default=1.0,
                   help="FedNew-FC record-level gradient clipping C1")
    p.add_argument("--fn_c2", type=float, default=1.0,
                   help="FedNew-FC auxiliary gradient clipping C2")
    p.add_argument("--fn_c_primal", type=float, default=1.0,
                   help="FedNew-FC primal clipping bound C")

    p.add_argument("--use_fedadam", action="store_true",
                   help="Use DP-FedAdam with server-side Adam optimizer")
    p.add_argument("--fedadam_beta1", type=float, default=0.9,
                   help="Adam beta1 (first moment decay)")
    p.add_argument("--fedadam_beta2", type=float, default=0.99,
                   help="Adam beta2 (second moment decay)")
    p.add_argument("--fedadam_epsilon", type=float, default=1e-8,
                   help="Adam epsilon for numerical stability")
    p.add_argument("--use_fedyogi", action="store_true",
                   help="Use DP-FedYogi with server-side Yogi optimizer (Reddi et al. 2021)")
    p.add_argument("--fedyogi_beta1", type=float, default=0.9)
    p.add_argument("--fedyogi_beta2", type=float, default=0.99)
    p.add_argument("--fedyogi_epsilon", type=float, default=1e-3,
                   help="Yogi epsilon; also sets v_0=eps^2")

    p.add_argument("--use_fedprox", action="store_true",
                   help="Use DP-FedProx with proximal penalty")
    p.add_argument("--use_adafedprox", action="store_true",
                   help="Use DP-AdaFedProx: FedProx with diminishing proximal term "
                        "mu_t = max(mu_0 / sqrt(t), 0.1 * mu_0). Reuses --fedprox_* params.")
    p.add_argument("--fedprox_mu", type=float, default=0.01,
                   help="Initial proximal penalty mu_0 (FedProx / AdaFedProx)")
    p.add_argument("--fedprox_local_steps", type=int, default=5,
                   help="Number of local steps K for FedProx / AdaFedProx")
    p.add_argument("--fedprox_client_lr", type=float, default=0.1,
                   help="Local learning rate for FedProx / AdaFedProx")

    p.add_argument("--use_ftrl", action="store_true",
                   help="Use DP-FTRL with binary tree aggregation (Kairouz et al., ICML 2021). "
                        "Calibrates noise to O(log T) effective compositions instead of T, "
                        "giving sqrt(T/log T) noise reduction vs DP-FedGD.")

    # Tiny-ImageNet subset
    p.add_argument("--tinyimagenet_classes", type=int, default=50,
                   help="Number of classes to use from Tiny-ImageNet (default 50). "
                        "Random subset selected with --tinyimagenet_subset_seed.")
    p.add_argument("--tinyimagenet_subset_seed", type=int, default=0,
                   help="Fixed seed for Tiny-ImageNet class subset selection (default 0).")

    return p.parse_args()
# --- DP-SCAFFOLD Client Implementation ---
import torch
from typing import Optional, Tuple, Dict, List

class DPFedGDClient:
    """
    Client implementing Algorithm 5 (Record-Level DP-FedGD).

    Client step for round k:
    1. Set model to θ^k
    2. Compute per-example gradients on local data D_i
    3. Clip each: g̃ = clip_{C_g}(∇f(θ^k, x))
    4. Average: avg = (1/|D_i|) Σ g̃
    5. Add noise: u_i^k = avg + N(0, I_d(C_g σ_g)²/n)
    6. Return u_i^k
    """

    def __init__(self, client_id: int, train_loader: DataLoader,
                 feature_dim: int, num_classes: int, device: torch.device,
                 fc_cc: float = 1.0):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_samples = len(train_loader.dataset)

        # FedFC feature clipping constant (C_c) used for covariance computation
        self.fc_cc = float(fc_cc)

        # Local model
        self.model = LinearClassifier(feature_dim, num_classes).to(device)

        # --- DP-SCAFFOLD state (client control variate c_i) ---
        self.d = (feature_dim + 1) * (num_classes if num_classes > 1 else 1)
        self.c_i = torch.zeros(self.d, device=self.device)

        # Pre-load and store all client data
        self.features = None
        self.labels = None
        self._load_all_data()

    def _load_all_data(self):
        """Load all client data into CPU memory."""
        all_features = []
        all_labels = []

        for X, y in self.train_loader:
            all_features.append(X)
            all_labels.append(y)

        # Store on CPU — move to device per-batch during training
        self.features = torch.cat(all_features, dim=0).cpu()
        self.labels = torch.cat(all_labels, dim=0).cpu()

    def compute_noisy_update(self, global_weights: torch.Tensor,
                             sanitizer: Optional[DPFedGDSanitizer]) -> Tuple[torch.Tensor, Dict]:
        self.model.set_weights(global_weights)
        self.model.train()

        stats = {
            'loss': 0.0,
            'samples': self.num_samples,
            'gradient_norm_before_noise': 0.0,
            'gradient_norm_after_noise': 0.0
        }

        Cg = sanitizer.gradient_clip_norm if sanitizer is not None else None

        clipped_sum = None
        total_loss = 0.0

        # Accumulators for clip stats (scalars only — no [N,P] tensor kept)
        all_norms = []
        clip_count = 0
        factor_sum = 0.0
        total_count = 0

        # Auto-scale batch size so batch_grads never exceeds ~8 MB of GPU RAM.
        # For VGG16 + 200 classes: grad_dim=102,600 → batch≈20 (8 MB).
        # For ResNet20 + 10 classes: grad_dim=650 → batch=64 (normal).
        grad_dim = (self.feature_dim + 1) * (self.num_classes if self.num_classes > 1 else 1)
        batch_size = min(self.num_samples, max(4, (8 * 1024 * 1024) // (grad_dim * 4)))
        num_batches = (self.num_samples + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.num_samples)

            X_batch = self.features[start_idx:end_idx].to(self.device)
            y_batch = self.labels[start_idx:end_idx].to(self.device)

            batch_grads, batch_loss = self._compute_per_example_gradients(X_batch, y_batch)
            # batch_grads: [B, P]
            B = batch_grads.size(0)
            total_loss += batch_loss * B

            if Cg is not None:
                per_ex_norms = batch_grads.norm(p=2, dim=1)  # [B]
                factors = (Cg / (per_ex_norms + 1e-12)).clamp(max=1.0)
                batch_grads = batch_grads * factors.unsqueeze(1)

                all_norms.append(per_ex_norms.detach())
                clip_count += int((factors < 1.0).sum().item())
                factor_sum += factors.sum().item()
            else:
                all_norms.append(batch_grads.norm(p=2, dim=1).detach())

            total_count += B

            # Accumulate sum, free batch tensor immediately
            if clipped_sum is None:
                clipped_sum = batch_grads.sum(dim=0)
            else:
                clipped_sum += batch_grads.sum(dim=0)

            del batch_grads

        stats['loss'] = total_loss / self.num_samples

        # Clip stats from collected scalar norms
        all_norms_cat = torch.cat(all_norms)  # [N] — just float scalars, negligible memory
        stats["perex_grad_norm_mean"] = float(all_norms_cat.mean().item())
        stats["perex_grad_norm_p50"] = float(all_norms_cat.median().item())
        if total_count <= 1:
            stats["perex_grad_norm_p90"] = float(all_norms_cat[-1].item())
        else:
            k = max(1, min(int(math.ceil(0.9 * total_count)), total_count))
            stats["perex_grad_norm_p90"] = float(all_norms_cat.kthvalue(k).values.item())

        if Cg is not None:
            stats["clip_frac"] = clip_count / max(1, total_count)
            stats["clip_factor_mean"] = factor_sum / max(1, total_count)
        else:
            stats["clip_frac"] = 0.0
            stats["clip_factor_mean"] = 1.0

        stats["clipped_sum_norm"] = float(clipped_sum.norm().item())

        gradient_avg = clipped_sum / self.num_samples
        stats['gradient_norm_before_noise'] = gradient_avg.norm().item()

        if sanitizer is not None:
            noisy_update, noise_stats = sanitizer.add_client_noise(clipped_sum, self.num_samples)
            stats["sum_noise_std"] = noise_stats["sum_noise_std"]
        else:
            noisy_update = gradient_avg
            stats["sum_noise_std"] = 0.0

        stats["gradient_norm_after_noise"] = float(noisy_update.norm().item())
        return noisy_update, stats

    def _compute_per_example_gradients(self, X: torch.Tensor, y: torch.Tensor):
        """
        Vectorized computation of per-example gradients.
        Eliminates the slow Python loop over batch_size.
        """
        with torch.no_grad():  # ← add this
            batch_size = X.size(0)
            ones = torch.ones(batch_size, 1, device=self.device, dtype=X.dtype)
            X_with_bias = torch.cat([X, ones], dim=1)
            logits = X_with_bias @ self.model.w

        # 3. Compute Errors (dL/dz) and Gradients efficiently
        if self.num_classes == 1:
            # Binary classification
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)

            # BCE derivative w.r.t logits is (p - y)
            errors = (probs - y.float()).unsqueeze(1)  # Shape: [B, 1]

            # Calculate Sum Loss for stats
            loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='sum')

            # Vectorized Gradient: Broadcast multiply [B, D] * [B, 1] -> [B, D]
            per_sample_grads = X_with_bias * errors

        else:
            # Multi-class classification
            # CrossEntropy derivative w.r.t logits is (softmax(z) - y_onehot)
            probs = F.softmax(logits, dim=1)
            y_onehot = F.one_hot(y.long(), num_classes=self.num_classes).float()
            errors = probs - y_onehot  # Shape: [B, num_classes]

            # Calculate Sum Loss for stats
            loss = F.cross_entropy(logits, y.long(), reduction='sum')

            # Vectorized Gradient: Outer product for each sample
            # We want [B, D, C] -> flattened to [B, D*C]
            # Einstein summation: bi=batch,input; bj=batch,class -> bij
            per_sample_grads = torch.einsum('bi,bj->bij', X_with_bias, errors)

            # Flatten gradients to 2D [B, Total_Params] for clipping
            per_sample_grads = per_sample_grads.reshape(batch_size, -1)

        avg_loss = loss.item() / batch_size

        # Return tensors directly (faster than lists)
        return per_sample_grads, avg_loss

    def _compute_clipped_sum_and_loss(
            self,
            weights: torch.Tensor,
            sanitizer: Optional[DPFedGDSanitizer]
    ) -> Tuple[torch.Tensor, float, Dict]:
        """Compute per-example grads at `weights`, clip, and return clipped SUM along with avg loss and stats."""
        self.model.set_weights(weights)
        self.model.train()

        grad_dim = (self.feature_dim + 1) * (self.num_classes if self.num_classes > 1 else 1)
        batch_size = min(self.num_samples, max(4, (8 * 1024 * 1024) // (grad_dim * 4)))
        num_batches = (self.num_samples + batch_size - 1) // batch_size

        Cg = float(sanitizer.gradient_clip_norm) if sanitizer is not None else None

        clipped_sum = None
        total_loss = 0.0
        clip_count = 0
        factor_sum = 0.0
        total_count = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.num_samples)

            X_batch = self.features[start_idx:end_idx].to(self.device)
            y_batch = self.labels[start_idx:end_idx].to(self.device)

            batch_grads, batch_loss = self._compute_per_example_gradients(X_batch, y_batch)
            B = batch_grads.size(0)
            total_loss += batch_loss * B

            if Cg is not None:
                per_ex_norms = batch_grads.norm(p=2, dim=1)
                factors = (Cg / (per_ex_norms + 1e-12)).clamp(max=1.0)
                batch_grads = batch_grads * factors.unsqueeze(1)
                clip_count += int((factors < 1.0).sum().item())
                factor_sum += factors.sum().item()

            total_count += B

            if clipped_sum is None:
                clipped_sum = batch_grads.sum(dim=0)
            else:
                clipped_sum += batch_grads.sum(dim=0)

            del batch_grads

        avg_loss = total_loss / max(1, self.num_samples)

        stats = {}
        if sanitizer is not None:
            stats["perex_grad_norm_mean"] = 0.0
            stats["clip_frac"] = clip_count / max(1, total_count)
            stats["clip_factor_mean"] = factor_sum / max(1, total_count)
        else:
            stats["perex_grad_norm_mean"] = 0.0
            stats["clip_frac"] = 0.0
            stats["clip_factor_mean"] = 1.0

        stats["clipped_sum_norm"] = float(clipped_sum.norm().item())
        return clipped_sum, float(avg_loss), stats

    def compute_scaffold_update(
        self,
        global_weights: torch.Tensor,
        server_c: torch.Tensor,
        sanitizer: Optional[DPFedGDSanitizer],
        K: int,
        client_lr: float,
        round_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Record-level full-batch DP-SCAFFOLD (Algorithm 2).

        Returns:
            delta_x = y_i^M - theta^{k-1}
            delta_c = c_i^k - c_i^{k-1}
            client_stats
        """
        theta = global_weights.detach()
        y = theta.clone()

        dp_enabled = (sanitizer is not None)
        total_loss = 0.0
        last_sum_noise_std = 0.0

        # Local steps m = 1..K
        for m in range(1, K + 1):
            clipped_sum, avg_loss, _ = self._compute_clipped_sum_and_loss(y, sanitizer)
            total_loss += avg_loss

            # Add noise to SUM then divide by |D_i| via sanitizer.add_client_noise
            if dp_enabled:
                g_tilde, _noise_stats = sanitizer.add_client_noise(clipped_sum, self.num_samples)
                last_sum_noise_std = float(_noise_stats.get("sum_noise_std", 0.0))
            else:
                g_tilde = clipped_sum / float(self.num_samples)

            # y^m = y^{m-1} - η_l ( g_tilde - c_i + c )
            y = y - float(client_lr) * (g_tilde - self.c_i + server_c.detach())

        # Update client control variate
        c_i_old = self.c_i.clone()
        # Algorithm 2: c_i^k = c_i^{k-1} - c^{k-1} + (theta^{k-1} - y_i^K) / (K * eta_l)
        c_i_new = c_i_old - server_c + (theta - y) / (float(K) * float(client_lr) + 1e-12)
        self.c_i = c_i_new.detach()

        delta_x = (y - theta).detach()
        delta_c = (c_i_new - c_i_old).detach()

        client_stats = {
            "loss": total_loss / max(1, K),
            "sum_noise_std": float(last_sum_noise_std),
        }
        return delta_x, delta_c, client_stats

    def compute_local_covariance(self, sanitizer) -> torch.Tensor:
        n_i = self.num_samples
        X = self.features.float().to(self.device)
        ones = torch.ones(n_i, 1, device=self.device)
        X_wb = torch.cat([X, ones], dim=1)

        C_i = (X_wb.T @ X_wb) / n_i

        if sanitizer is not None:
            C_i = sanitizer.add_covariance_noise(C_i)

        del X, X_wb
        torch.cuda.empty_cache()

        return C_i  # keep on device, no .cpu()

    def compute_fedavg_update(
            self,
            global_weights: torch.Tensor,
            sanitizer: Optional[DPFedGDSanitizer],
            K: int,
            client_lr: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        DP-FedAvg with K local SGD steps (no control variates).

        Each local step:
          1. Compute full-batch per-example gradients at current local weights y
          2. Clip, add noise, divide by |D_i|
          3. y <- y - eta_l * g_tilde

        Returns weight delta: delta = y^K - theta^{k-1}
        DP accounting: T*K compositions (handled by sanitizer calibration).
        """
        theta = global_weights.detach()
        y = theta.clone()

        dp_enabled = (sanitizer is not None)
        total_loss = 0.0
        last_sum_noise_std = 0.0

        for m in range(1, K + 1):
            clipped_sum, avg_loss, step_stats = self._compute_clipped_sum_and_loss(y, sanitizer)
            total_loss += avg_loss

            if dp_enabled:
                g_tilde, noise_stats = sanitizer.add_client_noise(clipped_sum, self.num_samples)
                last_sum_noise_std = float(noise_stats.get("sum_noise_std", 0.0))
            else:
                g_tilde = clipped_sum / float(self.num_samples)

            # Local weight update (no control variate correction)
            y = y - float(client_lr) * g_tilde

        delta_x = (y - theta).detach()

        client_stats = {
            "loss": total_loss / max(1, K),
            "sum_noise_std": float(last_sum_noise_std),
            "clip_frac": step_stats.get("clip_frac", 0.0),
            "clip_factor_mean": step_stats.get("clip_factor_mean", 1.0),
            "perex_grad_norm_mean": step_stats.get("perex_grad_norm_mean", 0.0),
            "gradient_norm_before_noise": float(
                (clipped_sum / self.num_samples).norm().item()
            ),
            "gradient_norm_after_noise": float(g_tilde.norm().item()),
            "clipped_sum_norm": float(clipped_sum.norm().item()),
        }
        return delta_x, client_stats

    def compute_fedprox_update(
            self,
            global_weights: torch.Tensor,
            sanitizer: Optional[DPFedGDSanitizer],
            K: int,
            client_lr: float,
            mu: float,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        DP-FedProx with K local steps.
        Identical to FedAvg but adds proximal term mu/2 * ||y - theta||^2
        to the local objective, which adds mu*(y - theta) to the gradient.

        Each local step:
          1. Compute clipped noisy gradient g_tilde at current y
          2. Add proximal correction: g_prox = g_tilde + mu * (y - theta)
          3. y <- y - eta_l * g_prox

        DP accounting: T*K compositions (same as FedAvg).
        Proximal term is non-private (depends only on global weights, not data).
        """
        theta = global_weights.detach()
        y = theta.clone()

        dp_enabled = (sanitizer is not None)
        total_loss = 0.0
        last_sum_noise_std = 0.0

        for m in range(1, K + 1):
            clipped_sum, avg_loss, step_stats = self._compute_clipped_sum_and_loss(y, sanitizer)
            total_loss += avg_loss

            if dp_enabled:
                g_tilde, noise_stats = sanitizer.add_client_noise(clipped_sum, self.num_samples)
                last_sum_noise_std = float(noise_stats.get("sum_noise_std", 0.0))
            else:
                g_tilde = clipped_sum / float(self.num_samples)

            # Proximal correction — no noise needed, depends only on global weights
            proximal_grad = mu * (y - theta)
            g_prox = g_tilde + proximal_grad

            y = y - float(client_lr) * g_prox

        delta_x = (y - theta).detach()

        client_stats = {
            "loss": total_loss / max(1, K),
            "sum_noise_std": float(last_sum_noise_std),
            "clip_frac": step_stats.get("clip_frac", 0.0),
            "clip_factor_mean": step_stats.get("clip_factor_mean", 1.0),
            "perex_grad_norm_mean": step_stats.get("perex_grad_norm_mean", 0.0),
            "gradient_norm_before_noise": float(
                (clipped_sum / self.num_samples).norm().item()
            ),
            "gradient_norm_after_noise": float(g_tilde.norm().item()),
            "clipped_sum_norm": float(clipped_sum.norm().item()),
        }
        return delta_x, client_stats


class DPFedGDServer:
    """
    Server implementing Algorithm 5 (Record-Level DP-FedGD).

    Server step for round k:
    1. Receive noisy updates {u_i^k}_{i=1}^n from clients
    2. Aggregate: U^k = (1/n) Σ u_i^k
    3. Update: θ^{k+1} = θ^k - η · U^k
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 learning_rate: float, device: torch.device):
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Global model
        self.global_model = LinearClassifier(feature_dim, num_classes).to(device)

    def get_global_weights(self) -> torch.Tensor:
        """Get current global model weights."""
        return self.global_model.get_weights()

    def aggregate_and_update(self, client_updates: List[torch.Tensor],
                             sanitizer: Optional[DPFedGDSanitizer]) -> Dict:
        """
        Algorithm 5 server aggregation and update.

        Args:
            client_updates: List of noisy client updates {u_i^k}
            sanitizer: Sanitizer (for statistics tracking)

        Returns:
            Statistics dictionary
        """
        # Step 1: Aggregate client updates
        # U^k = (1/n) Σ_{i=1}^n u_i^k
        if sanitizer is not None:
            aggregated_update = sanitizer.aggregate_client_updates(client_updates)
        else:
            # No DP: simple mean of client updates
            aggregated_update = torch.stack(client_updates).mean(dim=0)

        # Track statistics
        update_norm = aggregated_update.norm().item()

        # Step 2: Update model parameters
        # θ^{k+1} = θ^k - η · U^k
        current_weights = self.get_global_weights()
        new_weights = current_weights - self.learning_rate * aggregated_update

        # Compute weight change
        weight_change = (new_weights - current_weights).norm().item()

        # Update global model
        self.global_model.set_weights(new_weights)

        return {
            'update_norm': update_norm,
            'weight_change_norm': weight_change,
            'learning_rate': self.learning_rate
        }

    def evaluate(self, test_loader: DataLoader, is_binary: bool,
                 return_mask: bool = False) -> Dict[str, float]:
        """Evaluate global model."""
        self.global_model.eval()
        correct = total = total_loss = 0
        tp = fp = tn = fn = 0
        correct_mask = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.global_model(X)

                if is_binary:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    batch_correct = (preds == y.float())
                    correct += batch_correct.sum().item()
                    loss = F.binary_cross_entropy_with_logits(logits, y.float())
                    tp += ((preds == 1) & (y == 1)).sum().item()
                    tn += ((preds == 0) & (y == 0)).sum().item()
                    fp += ((preds == 1) & (y == 0)).sum().item()
                    fn += ((preds == 0) & (y == 1)).sum().item()
                else:
                    preds = torch.argmax(logits, dim=1)
                    batch_correct = (preds == y)
                    correct += batch_correct.sum().item()
                    loss = F.cross_entropy(logits, y.long())

                correct_mask.extend(batch_correct.cpu().tolist())
                total += X.size(0)
                total_loss += loss.item() * X.size(0)

        self.global_model.train()
        if is_binary:
            accuracy = correct / total
            avg_loss = total_loss / total
            recall_pos = tp / (tp + fn + 1e-12)
            precision_pos = tp / (tp + fp + 1e-12)
            f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-12)
            result = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'recall_pos': recall_pos,
                'precision_pos': precision_pos,
                'f1_pos': f1_pos,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
            }
        else:
            result = {
                'accuracy': correct / total,
                'loss': total_loss / total,
            }

        if return_mask:
            result['test_correct_mask'] = correct_mask
        return result

class DPFedScaffoldServer(DPFedGDServer):
    """Server for record-level full-batch DP-SCAFFOLD (Algorithm 2)."""

    def __init__(self, feature_dim: int, num_classes: int,
                 server_lr: float, device: torch.device):
        super().__init__(feature_dim, num_classes, learning_rate=server_lr, device=device)
        self.server_c = torch.zeros_like(self.get_global_weights())

    def get_server_control(self) -> torch.Tensor:
        return self.server_c

    def aggregate_and_update(self, deltas_x: List[torch.Tensor], deltas_c: List[torch.Tensor]) -> Dict:
        # Δθ^k = (1/n) Σ (y_i^M - θ^{k-1})
        delta_theta = torch.stack(deltas_x).mean(dim=0)
        # Δc^k = (1/n) Σ (c_i^k - c_i^{k-1})
        delta_c = torch.stack(deltas_c).mean(dim=0)

        old_w = self.get_global_weights()
        # θ^k = θ^{k-1} + η_g Δθ^k
        new_w = old_w + self.learning_rate * delta_theta
        self.global_model.set_weights(new_w)

        # c^k = c^{k-1} + Δc^k
        self.server_c = self.server_c + delta_c

        return {
            "update_norm": float(delta_theta.norm().item()),
            "weight_change_norm": float((new_w - old_w).norm().item()),
            "server_c_norm": float(self.server_c.norm().item()),
            "learning_rate": float(self.learning_rate),
        }

class DPFedAvgServer(DPFedGDServer):
    """
    Server for DP-FedAvg with multiple local steps.
    Applies weight deltas directly (not gradients).
    θ^{k+1} = θ^k + η_g * (1/n) Σ delta_i^k
    """

    def aggregate_and_update(self, deltas_x: List[torch.Tensor],
                              sanitizer=None) -> Dict:
        delta_avg = torch.stack(deltas_x).mean(dim=0)

        old_w = self.get_global_weights()
        new_w = old_w + self.learning_rate * delta_avg
        self.global_model.set_weights(new_w)

        return {
            'update_norm': float(delta_avg.norm().item()),
            'weight_change_norm': float((new_w - old_w).norm().item()),
            'learning_rate': float(self.learning_rate),
        }

class DPFedGDServerSOFIM:
    """
    Server with SOFIM (Second-Order Federated Information Matrix) optimization.

    Implements Algorithm 1 from SOFIM paper:
    - Uses regularized Fisher Information Matrix (FIM)
    - Sherman-Morrison formula for efficient matrix inversion
    - Momentum on gradients (like Adam)
    - Same O(d) time and space complexity as SGD with momentum

    Client side remains unchanged - only server-side optimization changes.
    """

    def __init__(self, feature_dim: int, num_classes: int, learning_rate: float,
                 device: torch.device,
                 beta: float = 0.9,
                 rho: float = 0.5,
                 use_bias_correction: bool = True,
                 warmup_rounds: int = 0,
                 ablation_mode: str = "full",
                 weight_decay: float = 0.0,
                 warmup_use_ftrl: bool = False):
        """
        Initialize SOFIM server.

        Args:
            feature_dim: Feature dimension
            num_classes: Number of classes
            learning_rate: Learning rate η
            device: Computing device
            beta: Momentum decay rate β ∈ [0, 1) (default: 0.9)
            rho: FIM regularization parameter ρ > 0 (default: 0.5)
            use_bias_correction: Whether to use bias correction (default: True)
        """
        self.global_model = LinearClassifier(feature_dim, num_classes).to(device)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        self.learning_rate = learning_rate

        # SOFIM parameters
        self.beta = beta
        self.rho = rho
        self.use_bias_correction = use_bias_correction
        self.warmup_rounds = warmup_rounds
        # ablation_mode: "full" | "ema_only" | "grad_only"
        # full:     standard SOFIM — EMA + Sherman-Morrison
        # ema_only: EMA momentum only, skip Sherman-Morrison (identity preconditioner)
        # grad_only: raw aggregated gradient, no EMA, no SM
        self.ablation_mode = ablation_mode
        self.weight_decay = weight_decay

        # SOFIM state
        self.M = None  # First moment buffer Mt
        self.step_count = 0  # For bias correction

        # Model dimension
        self.d = (feature_dim + 1) * (num_classes if num_classes > 1 else 1)

        # DP-FTRL warmup state (only used when warmup_use_ftrl=True)
        self.warmup_use_ftrl = warmup_use_ftrl
        if warmup_use_ftrl:
            self.theta_0 = self.global_model.get_weights().clone().detach()
            self._tree_stack: List[tuple] = []
            self._prefix_sum: Optional[torch.Tensor] = None
            self._ftrl_round = 0

    def get_global_weights(self) -> torch.Tensor:
        """Get current global model weights."""
        return self.global_model.get_weights()

    def aggregate_and_update(self, client_updates: List[torch.Tensor],
                             sanitizer, current_round: int) -> Dict:
        """
        SOFIM update following Algorithm 1 from paper.

        Steps:
        1. Aggregate client gradients: U^k = mean(u_i^k)
        2. Update first moment with momentum: Mt = β·Mt-1 + (1-β)·U^k
        3. Bias correction: Mct = Mt / (1 - β^t)
        4. Apply Sherman-Morrison preconditioner: Ft^(-1)·Mct
        5. Update weights: θ^{k+1} = θ^k - η·Ft^(-1)·Mct

        Args:
            client_updates: List of noisy client updates
            sanitizer: DP sanitizer (not used, for compatibility)

        Returns:
            Dictionary with update statistics
        """
        # Step 1: Aggregate client gradients (same as first-order)
        if sanitizer is not None:
            aggregated_gradient = sanitizer.aggregate_client_updates(client_updates)
        else:
            aggregated_gradient = torch.stack(client_updates).mean(dim=0)

        # Flatten to vector form
        gt = aggregated_gradient.view(-1).detach()  # [d]

        # Step 2: Update first moment (momentum)
        if self.M is None:
            self.M = torch.zeros_like(gt)

        self.M = self.beta * self.M + (1.0 - self.beta) * gt
        self.step_count += 1

        # Step 3: Bias correction (like Adam)
        if self.use_bias_correction:
            Mct = self.M / (1.0 - self.beta ** self.step_count)
        else:
            Mct = self.M.clone()

        # Step 4: Apply SOFIM preconditioner using Sherman-Morrison formula
        is_warmup = False
        is_ftrl_warmup = False
        preconditioned_update = None

        if self.ablation_mode == "grad_only":
            # Raw gradient — no EMA, no SM (equivalent to FedGD)
            preconditioned_update = gt
        elif self.ablation_mode == "ema_only":
            # EMA momentum only — no Sherman-Morrison step
            preconditioned_update = Mct
        elif current_round <= self.warmup_rounds:
            is_warmup = True
            if self.warmup_use_ftrl:
                # FTRL warmup: binary tree prefix-sum update
                is_ftrl_warmup = True
                self._ftrl_round += 1
                self._tree_push(gt)
            else:
                # EMA-only warmup (original behaviour)
                preconditioned_update = Mct
        else:
            # Full SOFIM: EMA + Sherman-Morrison
            preconditioned_update = self._apply_sofim_preconditioner(Mct)

        # Step 5: Update model weights
        global_weights = self.get_global_weights()
        if is_ftrl_warmup:
            # θ = θ_0 - η * running_mean(all warmup gradients)
            running_mean = self._prefix_sum / self._ftrl_round
            new_weights = self.theta_0 - self.learning_rate * running_mean
            update_norm = running_mean.norm().item()
            tree_depth = max((h for h, _ in self._tree_stack), default=0)
        else:
            if self.weight_decay > 0.0:
                # L2 reg: effective update = preconditioned_update + λ·θ
                preconditioned_update = preconditioned_update + self.weight_decay * global_weights
            new_weights = global_weights - self.learning_rate * preconditioned_update
            update_norm = preconditioned_update.norm().item()
            tree_depth = 0

        self.global_model.set_weights(new_weights)
        weight_change = (new_weights - global_weights).norm().item()

        return {
            'update_norm': update_norm,
            'weight_change_norm': weight_change,
            'gradient_norm': gt.norm().item(),
            'is_warmup': is_warmup,
            'is_ftrl_warmup': is_ftrl_warmup,
            'ftrl_tree_depth': tree_depth,
            'moment_norm': self.M.norm().item(),
            'moment_corrected_norm': Mct.norm().item(),
            'bias_correction_factor': (1.0 - self.beta ** self.step_count),
            'learning_rate': self.learning_rate
        }


    def _tree_push(self, gradient: torch.Tensor) -> None:
        """Insert a new leaf into the streaming binary tree (Honaker 2015)."""
        node = gradient.clone()
        height = 0
        while self._tree_stack and self._tree_stack[-1][0] == height:
            _, prev = self._tree_stack.pop()
            node = prev + node
            height += 1
        self._tree_stack.append((height, node))
        self._prefix_sum = sum(v for _, v in self._tree_stack)

    def _apply_sofim_preconditioner(self, Mct: torch.Tensor) -> torch.Tensor:
        """
        Apply SOFIM preconditioner using the Sherman-Morrison formula.

        SOFIM uses the rank-1 Fisher approximation

            F_t = Mct Mct^T + rho I

        and computes

            F_t^{-1} Mct

        without explicitly forming or inverting the d×d matrix F_t.

        Using Sherman-Morrison:

            (rho I + uu^T)^{-1}
                = (1/rho)I
                  - uu^T / (rho^2 * (1 + u^T u / rho))

        where u = Mct.

        Therefore,

            F_t^{-1} Mct
                = Mct/rho
                  - (Mct (Mct^T Mct))
                    / (rho^2 (1 + Mct^T Mct / rho))

        Args:
            Mct: Bias-corrected first-moment vector.

        Returns:
            Preconditioned update direction.
        """

        # First term: Mct / rho
        term1 = Mct / self.rho

        # Squared norm ||Mct||² = Mct^T Mct
        Mct_norm_sq = torch.dot(Mct, Mct)

        # Second Sherman-Morrison correction term
        numerator = Mct * Mct_norm_sq
        denominator = self.rho ** 2 * (1.0 + Mct_norm_sq / self.rho)

        term2 = numerator / (denominator + 1e-8)

        # Final preconditioned direction
        preconditioned = term1 - term2

        # Diagnostics (every 10 rounds)
        if hasattr(self, "step_count") and self.step_count % 10 == 0:
            precond_norm = torch.norm(preconditioned).item()
            moment_norm = torch.norm(Mct).item()

            print(
                f"[SOFIM] "
                f"moment={moment_norm:.3e}, "
                f"precond={precond_norm:.3e}, "
                f"ratio={precond_norm / (moment_norm + 1e-12):.3e}, "
                f"rho={self.rho:.3e}, "
                f"norm_sq={Mct_norm_sq.item():.3e}"
            )

        return preconditioned

    def evaluate(self, test_loader: DataLoader, is_binary: bool,
                 return_mask: bool = False) -> Dict[str, float]:
        """Evaluate global model on test set."""
        self.global_model.eval()
        correct = total = total_loss = 0
        tp = fp = tn = fn = 0
        correct_mask = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.global_model(X)

                if is_binary:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    batch_correct = (preds == y.float())
                    correct += batch_correct.sum().item()
                    loss = F.binary_cross_entropy_with_logits(logits, y.float())
                    tp += ((preds == 1) & (y == 1)).sum().item()
                    tn += ((preds == 0) & (y == 0)).sum().item()
                    fp += ((preds == 1) & (y == 0)).sum().item()
                    fn += ((preds == 0) & (y == 1)).sum().item()
                else:
                    preds = torch.argmax(logits, dim=1)
                    batch_correct = (preds == y)
                    correct += batch_correct.sum().item()
                    loss = F.cross_entropy(logits, y.long())

                correct_mask.extend(batch_correct.cpu().tolist())
                total += X.size(0)
                total_loss += loss.item() * X.size(0)

        self.global_model.train()
        if is_binary:
            accuracy = correct / total
            avg_loss = total_loss / total
            recall_pos = tp / (tp + fn + 1e-12)
            precision_pos = tp / (tp + fp + 1e-12)
            f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-12)
            result = {
                'accuracy': accuracy,
                'loss': avg_loss,
                'recall_pos': recall_pos,
                'precision_pos': precision_pos,
                'f1_pos': f1_pos,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            }
        else:
            result = {
                'accuracy': correct / total,
                'loss': total_loss / total,
            }

        if return_mask:
            result['test_correct_mask'] = correct_mask
        return result

class DPFedAdamServer(DPFedGDServer):
    """
    Server for DP-FedAdam.
    Applies Adam-style adaptive updates server-side on aggregated gradients.
    Same client as FedGD — purely server-side change.
    θ^{k+1} = θ^k - η * m̂_k / (√v̂_k + ε_adam)
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 learning_rate: float, device: torch.device,
                 beta1: float = 0.9, beta2: float = 0.99,
                 epsilon_adam: float = 1e-8):
        super().__init__(feature_dim, num_classes, learning_rate, device)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon_adam = epsilon_adam
        self.t = 0  # step counter for bias correction

        self.m = None
        self.v = None

    def aggregate_and_update(self, client_updates: List[torch.Tensor],
                              sanitizer=None) -> Dict:
        # Aggregate
        if sanitizer is not None:
            g = sanitizer.aggregate_client_updates(client_updates).detach()
        else:
            g = torch.stack(client_updates).mean(dim=0)

        self.t += 1

        # Adam moment updates
        if self.m is None:
            self.m = torch.zeros_like(g)
            self.v = torch.zeros_like(g)

        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update
        old_w = self.get_global_weights()
        new_w = old_w - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon_adam)
        self.global_model.set_weights(new_w)

        return {
            'update_norm': float(g.norm().item()),
            'weight_change_norm': float((new_w - old_w).norm().item()),
            'learning_rate': self.learning_rate,
        }

class DPFedYogiServer(DPFedGDServer):
    """
    Server for DP-FedYogi.
    Reddi et al., "Adaptive Federated Optimization", ICLR 2021 (Algorithm 2).

    Differs from Adam only in the second moment update:
      Adam:  v_t = β₂·v_{t-1} + (1-β₂)·g²
      Yogi:  v_t = v_{t-1} - (1-β₂)·sign(v_{t-1} - g²)·g²

    Yogi prevents v from growing too fast when gradients are consistent,
    giving better control over the effective learning rate under DP noise.
    v_0 = ε² (not zero) per Reddi et al. to avoid near-zero denominator.
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 learning_rate: float, device: torch.device,
                 beta1: float = 0.9, beta2: float = 0.99,
                 epsilon_yogi: float = 1e-3):
        super().__init__(feature_dim, num_classes, learning_rate, device)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon_yogi = epsilon_yogi
        self.t = 0
        self.m = None
        self.v = None  # initialised to ε² on first call

    def aggregate_and_update(self, client_updates: List[torch.Tensor],
                              sanitizer=None) -> Dict:
        if sanitizer is not None:
            g = sanitizer.aggregate_client_updates(client_updates).detach()
        else:
            g = torch.stack(client_updates).mean(dim=0)

        self.t += 1

        if self.m is None:
            self.m = torch.zeros_like(g)
            self.v = torch.full_like(g, self.epsilon_yogi ** 2)

        # First moment: same as Adam
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g

        # Second moment: Yogi sign-based update
        g_sq = g ** 2
        self.v = self.v - (1.0 - self.beta2) * torch.sign(self.v - g_sq) * g_sq

        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        old_w = self.get_global_weights()
        new_w = old_w - self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon_yogi)
        self.global_model.set_weights(new_w)

        return {
            'update_norm': float(g.norm().item()),
            'weight_change_norm': float((new_w - old_w).norm().item()),
            'learning_rate': self.learning_rate,
        }

class DPFedFTRLServer(DPFedGDServer):
    """
    Server implementing DP-FTRL with binary tree aggregation.

    Kairouz et al., "Practical and Private (Deep) Learning without Auditing",
    ICML 2021.

    Two changes vs DP-FedGD:
    1. FTRL update rule: θ_{t+1} = θ_0 - η * (1/t) Σ_{i=1}^t g_i
       Uses the running mean of ALL past gradients instead of only the current one.
       This averages out DP noise over time and is more stable under high noise.
    2. Binary tree prefix-sum accumulation (Honaker 2015): gradients are summed
       via a streaming binary tree so that each training example participates in
       only O(log T) tree nodes.  Paired with DPFedFTRLSanitizer (which calibrates
       noise to T_eff = ceil(log2 T)+1 compositions), this gives sqrt(T/log T)
       noise reduction over DP-FedGD for the same privacy budget.
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 learning_rate: float, device: torch.device):
        super().__init__(feature_dim, num_classes, learning_rate, device)
        # FTRL anchor: regularisation centre (initial weights)
        self.theta_0 = self.global_model.get_weights().clone().detach()
        self._round = 0
        # Streaming binary tree state: list of (height, partial_sum) pairs
        self._tree_stack: List[tuple] = []
        self._prefix_sum: Optional[torch.Tensor] = None

    def _tree_push(self, gradient: torch.Tensor) -> None:
        """
        Insert a new leaf into the streaming binary tree and update prefix sum.

        Implements the binary-counter approach (Honaker 2015): two nodes of the
        same height are merged into one node of height+1, exactly like carrying
        a bit in binary addition.  No extra noise is added here — all noise is
        already present in `gradient` from the client-side mechanism.
        """
        node = gradient.clone()
        height = 0
        while self._tree_stack and self._tree_stack[-1][0] == height:
            _, prev = self._tree_stack.pop()
            node = prev + node
            height += 1
        self._tree_stack.append((height, node))
        self._prefix_sum = sum(v for _, v in self._tree_stack)

    def aggregate_and_update(self, client_updates: List[torch.Tensor],
                              sanitizer=None) -> Dict:
        self._round += 1

        # Standard server-side aggregation (noise already in client updates)
        if sanitizer is not None:
            g_t = sanitizer.aggregate_client_updates(client_updates)
        else:
            g_t = torch.stack(client_updates).mean(dim=0)

        # Accumulate via binary tree (O(log T) structure for noise accounting)
        self._tree_push(g_t)

        # FTRL update: θ = θ_0 - η * running_mean_of_all_gradients
        running_mean = self._prefix_sum / self._round
        old_w = self.get_global_weights()
        new_w = self.theta_0 - self.learning_rate * running_mean
        self.global_model.set_weights(new_w)

        tree_depth = max((h for h, _ in self._tree_stack), default=0)
        return {
            'update_norm': float(running_mean.norm().item()),
            'weight_change_norm': float((new_w - old_w).norm().item()),
            'learning_rate': self.learning_rate,
            'ftrl_tree_depth': tree_depth,
        }


def compute_lemma4_xi(a: torch.Tensor, b: torch.Tensor, C: float) -> float:
    """
    Computes exact scalar xi such that ||a + xi*b||_2 = C[cite: 1351, 1352, 1804].
    Ensures the auxiliary gradient is within sensitivity bounds.
    """
    b_norm = b.norm(p=2).item()
    if b_norm < 1e-10: return 1.0

    b_unit = b / b_norm
    a_dot_b_unit = torch.dot(a, b_unit).item()
    a_norm_sq = torch.norm(a).item() ** 2

    # Quadratic formula for xi
    discriminant = 4 * (a_dot_b_unit ** 2) + 4 * (C ** 2 - a_norm_sq)
    if discriminant < 0: return 1.0

    xi = (-2 * a_dot_b_unit + np.sqrt(discriminant)) / (2 * b_norm)
    return xi


# --- 2. FedNew-FC Client Implementation ---
class DPFedNewFCClient:
    """
    [cite_start]Client for Record-Level DP-FedNew-FC[cite: 1363].
    1. Precomputes Feature Covariance matrix A_i once.
    2. Performs iterative ADMM updates with local dual variables.
    """

    def __init__(self, client_id: int, train_loader: DataLoader,
                 feature_dim: int, num_classes: int, device: torch.device,
                 alpha: float, rho: float):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.rho = rho
        self.gamma = alpha + rho  # Combined regularization [cite: 1342]

        # Load and store data, with L2 normalization (clip features to unit norm)
        all_x, all_y = [], []
        for X, y in train_loader:
            # Clip features to L2 norm <= 1 to stabilize covariance / Hessian
            norms = X.norm(p=2, dim=1, keepdim=True)
            # Avoid division by zero; clamp minimum norm to 1.0
            norms_clamped = torch.clamp(norms, min=1.0)
            X = X / norms_clamped
            all_x.append(X)
            all_y.append(y)
        self.features = torch.cat(all_x, dim=0).to(device)
        self.labels = torch.cat(all_y, dim=0).to(device)
        self.n_i = self.features.size(0)

        # 1. Precompute A_i = (S_i + gamma*I)^-1
        self.A_i = self._precompute_preconditioner()

        # [cite_start]2. ADMM Variables: same size as flattened weights [cite: 1288]
        self.d = (feature_dim + 1) * (num_classes if num_classes > 1 else 1)
        self.y_i = torch.zeros(self.d, device=device)  # Current primal
        self.lambda_i = torch.zeros(self.d, device=device)  # Current dual

    def _precompute_preconditioner(self):
        """Form A_i matrix based on feature-covariance Hessian approximation[cite: 1358]."""
        ones = torch.ones(self.n_i, 1, device=self.device)
        X_with_bias = torch.cat([self.features, ones], dim=1)  # [n_i, dx+1]

        # S_i = (1/n_i) * X^T @ X
        S_i = (X_with_bias.T @ X_with_bias) / self.n_i

        # A_i = (S_i + gamma * I)^-1
        reg_I = torch.eye(S_i.size(0), device=self.device) * self.gamma
        return torch.inverse(S_i + reg_I)

    def compute_primal_update(self, theta_k: torch.Tensor, sanitizer, args) -> Tuple[torch.Tensor, float]:
        """Main round update."""
        dp_enabled = (sanitizer is not None) and (not getattr(args, "no_dp", False))

        # Step 1: Record-level gradient clipping (C1) [DP only]
        per_sample_grads, avg_loss = self._compute_vectorized_grads(theta_k)
        if dp_enabled:
            norms = per_sample_grads.norm(p=2, dim=1, keepdim=True)
            factors = (args.fn_c1 / (norms + 1e-12)).clamp(max=1.0)
            g_i_k = (per_sample_grads * factors).mean(dim=0)
        else:
            # No DP: raw mean gradient
            g_i_k = per_sample_grads.mean(dim=0)

        # Step 2: Form auxiliary gradient g_sum = g - lambda + rho*y
        # ADMM terms: b = -lambda + rho*y
        admm_bias = -self.lambda_i + self.rho * self.y_i
        g_sum = g_i_k + admm_bias

        # [cite_start]Exact scaling via Lemma 4 if norm > C2 [cite: 1352] (DP only)
        if dp_enabled and (g_sum.norm().item() > args.fn_c2):
            xi = compute_lemma4_xi(g_i_k, admm_bias, args.fn_c2)
            g_sum = g_i_k + xi * admm_bias

        # Step 3: Non-DP primal update y_hat = A_i @ g_sum
        # Must handle matrix shape if multiclass
        if self.num_classes > 1:
            g_sum_mat = g_sum.view(self.feature_dim + 1, self.num_classes)
            y_hat_mat = self.A_i @ g_sum_mat
            y_hat = y_hat_mat.flatten()
        else:
            y_hat = self.A_i @ g_sum

        # Step 4: Primal clipping (C) [DP only]
        if dp_enabled:
            y_hat_norm = y_hat.norm().item()
            y_bar = y_hat * min(1.0, args.fn_c_primal / (y_hat_norm + 1e-12))

            # Step 5: Distributed Gaussian Noise E_i
            sigma_release = sanitizer.get_client_noise_std()  # Calibrated per rounds T
            noise = torch.randn_like(y_bar) * sigma_release
            y_tilde_i = y_bar + noise
        else:
            # No DP: no clipping, no noise
            y_tilde_i = y_hat

        # Step 6: Dual update lambda = lambda + alpha * (y_tilde - theta)
        self.lambda_i = self.lambda_i + self.alpha * (y_tilde_i - theta_k)
        self.y_i = y_tilde_i.clone()  # Store for next round iteration

        return y_tilde_i, avg_loss

    def _compute_vectorized_grads(self, weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute per-example gradients and average loss for the current client
        using the flattened weight vector `weights`.
        """
        # Rebuild linear model from weights
        model = LinearClassifier(self.feature_dim, self.num_classes).to(self.device)
        model.set_weights(weights)

        ones = torch.ones(self.n_i, 1, device=self.device)
        X_wb = torch.cat([self.features, ones], dim=1)
        logits = X_wb @ model.w

        if self.num_classes == 1:
            # Binary classification
            logits_flat = logits.squeeze(-1)
            probs = torch.sigmoid(logits_flat)
            errors = (probs - self.labels.float()).unsqueeze(1)
            per_sample_grads = X_wb * errors
            loss = F.binary_cross_entropy_with_logits(logits_flat, self.labels.float(), reduction="mean")
        else:
            # Multi-class classification
            probs = F.softmax(logits, dim=1)
            y_oh = F.one_hot(self.labels.long(), num_classes=self.num_classes).float()
            errors = probs - y_oh  # [n_i, C]
            per_sample_grads = torch.einsum("bi,bj->bij", X_wb, errors).reshape(self.n_i, -1)
            loss = F.cross_entropy(logits, self.labels.long(), reduction="mean")

        return per_sample_grads, float(loss.item())

# --- 3. FedNew Server Implementation ---
class DPFedNewFCServer:
    """Server for DP-FedNew-FC."""

    def __init__(self, feature_dim: int, num_classes: int, device: torch.device):
        self.global_model = LinearClassifier(feature_dim, num_classes).to(device)
        self.device = device

    def aggregate_and_update(self, client_primals: List[torch.Tensor]):
        """theta_{k+1} = (1/n) * sum(y_tilde_i)."""
        new_theta = torch.stack(client_primals).mean(dim=0)
        self.global_model.set_weights(new_theta)

        # Include weight_change_norm to satisfy the logging logic in run_dpfedgd_training
        return {
            'update_norm': new_theta.norm().item(),
            'weight_change_norm': 0.0  # Placeholder for FedNew-FC
        }

    def get_global_weights(self) -> torch.Tensor:
        """
        Get current global model weights.
        Required by the main training loop to sync clients.
        """
        return self.global_model.get_weights()

    def evaluate(self, test_loader: DataLoader, is_binary: bool) -> Dict[str, float]:
        """Evaluate global model on test set."""
        self.global_model.eval()
        correct = total = total_loss = 0
        tp = fp = tn = fn = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.global_model(X)

                if is_binary:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    correct += (preds == y.float()).sum().item()
                    loss = F.binary_cross_entropy_with_logits(logits, y.float())
                    tp += ((preds == 1) & (y == 1)).sum().item()
                    tn += ((preds == 0) & (y == 0)).sum().item()
                    fp += ((preds == 1) & (y == 0)).sum().item()
                    fn += ((preds == 0) & (y == 1)).sum().item()
                else:
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    loss = F.cross_entropy(logits, y.long())

                total += X.size(0)
                total_loss += loss.item() * X.size(0)

        self.global_model.train()
        if is_binary:
            accuracy = correct / total
            avg_loss = total_loss / total
            recall_pos = tp / (tp + fn + 1e-12)
            precision_pos = tp / (tp + fp + 1e-12)
            f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos + 1e-12)
            return {
                'accuracy': accuracy,
                'loss': avg_loss,
                'recall_pos': recall_pos,
                'precision_pos': precision_pos,
                'f1_pos': f1_pos,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
            }
        else:
            return {
                'accuracy': correct / total,
                'loss': total_loss / total
            }


class DPFedFCServer(DPFedGDServer):
    def __init__(self, *args, fc_gamma: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = fc_gamma
        self.C_inv = None

    def set_global_covariance(self, local_covariances: List[torch.Tensor]):
        """Algorithm 3 Line 8: Aggregate Ci and compute global preconditioning matrix."""
        n = len(local_covariances)
        # Average the local covariances
        C_global = torch.stack(local_covariances).sum(dim=0) / n

        # Add regularization gamma * Id and invert
        reg_I = torch.eye(C_global.size(0), device=self.device) * self.gamma
        self.C_inv = torch.inverse(C_global + reg_I)

    def aggregate_and_update(self, client_updates: List[torch.Tensor], sanitizer):
        """
        Steps 13–14: Compute global noisy update Uk = C^-1 @ (sum(ui)/n)
        """

        # --- Raw averaged gradient ---
        u_avg = torch.stack(client_updates).mean(dim=0)
        u_avg_norm = u_avg.norm().item()

        # --- Preconditioned update ---
        u_avg_mat = u_avg.view(self.feature_dim + 1, -1)
        U_k = (self.C_inv @ u_avg_mat).flatten()
        U_k_norm = U_k.norm().item()

        # --- Ratio ---
        ratio = U_k_norm / (u_avg_norm + 1e-12)

        # --- Parameter update ---
        current_w = self.get_global_weights()
        new_w = current_w - self.learning_rate * U_k
        self.global_model.set_weights(new_w)

        return {
            "u_avg_norm": u_avg_norm,
            "U_k_norm": U_k_norm,
            "precond_ratio": ratio,
            "update_norm": U_k_norm,
            "weight_change_norm": (new_w - current_w).norm().item(),
        }


def get_adaptive_sofim_params(epsilon: float, user_beta: float = None,
                              user_rho: float = None,
                              user_bias_correction: bool = None):
    """
    Get SOFIM hyperparameters adapted to privacy level.

    Args:
        epsilon: Privacy parameter
        user_beta: User-specified beta (overrides adaptive if not None)
        user_rho: User-specified rho (overrides adaptive if not None)
        user_bias_correction: User-specified bias correction (overrides adaptive if not None)

    Returns:
        Dictionary with beta, rho, and use_bias_correction
    """
    # Default adaptive strategy based on privacy level
    if epsilon >= 10:
        default_beta = 0.9
        default_rho = 0.5
        default_bias_correction = True
        regime = "high-ε (low noise)"
    elif epsilon >= 2:
        default_beta = 0.93
        default_rho = 2.0
        default_bias_correction = True
        regime = "medium-high-ε (moderate noise)"
    elif epsilon >= 1:
        default_beta = 0.95
        default_rho = 5.0
        default_bias_correction = False
        regime = "medium-ε (high noise)"
    else:  # ε < 1
        default_beta = 0.97
        default_rho = 10.0
        default_bias_correction = False
        regime = "low-ε (extreme noise)"

    # User overrides take precedence
    final_beta = user_beta if user_beta is not None else default_beta
    final_rho = user_rho if user_rho is not None else default_rho
    final_bias_correction = user_bias_correction if user_bias_correction is not None else default_bias_correction

    return {
        'beta': final_beta,
        'rho': final_rho,
        'use_bias_correction': final_bias_correction,
        'regime': regime
    }

class LinearClassifier(nn.Module):
    """Linear classifier for federated learning."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Weight matrix [feature_dim + 1, num_classes] (includes bias)
        self.w = nn.Parameter(torch.zeros(feature_dim + 1, num_classes))

        # Xavier initialization
        with torch.no_grad():
            if num_classes == 1:
                std = math.sqrt(2.0 / (feature_dim + 1)) * 0.5
                self.w.normal_(0, std)
            else:
                std = math.sqrt(2.0 / (feature_dim + 1 + num_classes)) * 0.5
                self.w.normal_(0, std)

    def forward(self, x):
        # Add bias term
        batch_size = x.size(0)
        ones = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        x_with_bias = torch.cat([x, ones], dim=1)

        # Linear transformation
        logits = x_with_bias @ self.w
        return logits.squeeze(-1) if self.num_classes == 1 else logits

    def get_weights(self) -> torch.Tensor:
        """Get flattened weights."""
        return self.w.data.flatten()

    def set_weights(self, weights: torch.Tensor):
        """Set weights from flattened tensor."""
        target_shape = (self.feature_dim + 1, self.num_classes if self.num_classes > 1 else 1)
        self.w.data = weights.view(target_shape)

import torch
import numpy as np


# def compute_fisher_alignment(
#         clients, global_weights, momentum_buffer, device,
#         max_clients=10, power_iter=40, trace_probe=30):
#
#     subset = clients[:min(max_clients, len(clients))]
#     num_classes = subset[0].num_classes
#     d = momentum_buffer.numel()
#     m = momentum_buffer.flatten().float().to(device)
#
#     if m.norm() < 1e-12:
#         return {
#             'fisher_cosine_sim': 0, 'fisher_spectral_gap': 1,
#             'fisher_variance_explained': 0, 'fisher_lambda1': 0, 'fisher_lambda2': 0
#         }
#
#     client_data = []
#     total_n = 0
#
#     for client in subset:
#         n_i = client.num_samples
#         X = client.features.float()  # keep on CPU
#         ones = torch.ones(n_i, 1)
#         X_wb = torch.cat([X, ones], dim=1)  # CPU
#         del X
#
#         d_feat1 = X_wb.shape[1]
#         C = num_classes if num_classes > 1 else 1
#         W = global_weights.view(d_feat1, C).float().cpu()
#
#         with torch.no_grad():
#             logits = X_wb @ W
#             p = torch.softmax(logits, dim=1) if num_classes > 1 else torch.sigmoid(logits)
#
#         del logits, W
#         client_data.append((X_wb, p, n_i))  # all CPU
#         total_n += n_i
#
#     def fisher_vec_product(v):
#         C = num_classes if num_classes > 1 else 1
#         V = v.view(-1, C)
#         Fv = torch.zeros_like(V)
#         for X, p, n_i in client_data:
#             X_g = X.to(device)
#             p_g = p.to(device)
#             XV = X_g @ V
#             term1 = p_g * XV
#             term2 = p_g * (p_g * XV).sum(dim=1, keepdim=True)
#             Fv += X_g.T @ (term1 - term2)
#             del X_g, p_g, XV, term1, term2
#         return Fv.flatten() / total_n  # ← moved inside fisher_vec_product
#
#     # Top eigenvector
#     v1 = m / m.norm()
#     prev = 0
#     lam = torch.tensor(0.0, device=device)
#     for i in range(power_iter):
#         with torch.no_grad():
#             Fv = fisher_vec_product(v1)
#         norm = Fv.norm()
#         if norm < 1e-12:
#             break
#         v1 = Fv / norm
#         with torch.no_grad():
#             lam = torch.dot(v1, fisher_vec_product(v1))
#         if abs(lam.item() - prev) < 1e-8:
#             break
#         prev = lam.item()
#
#     lambda1 = lam.item()
#
#     # lambda2
#     v2 = torch.randn_like(v1)
#     v2 -= torch.dot(v2, v1) * v1
#     v2 = v2 / v2.norm()
#     prev = 0
#     lam2 = torch.tensor(0.0, device=device)
#     for i in range(power_iter):
#         with torch.no_grad():
#             Fv2 = fisher_vec_product(v2)
#         Fv2 -= torch.dot(v1, Fv2) * v1
#         norm = Fv2.norm()
#         if norm < 1e-12:
#             break
#         v2 = Fv2 / norm
#         with torch.no_grad():
#             lam2 = torch.dot(v2, fisher_vec_product(v2))
#         if abs(lam2.item() - prev) < 1e-8:
#             break
#         prev = lam2.item()
#
#     lambda2 = lam2.item()
#
#     # Trace estimate
#     trace = 0
#     for _ in range(trace_probe):
#         z = torch.randint(0, 2, (d,), device=device).float()
#         z = 2 * z - 1
#         with torch.no_grad():
#             trace += torch.dot(z, fisher_vec_product(z))
#     trace /= trace_probe
#     trace = max(trace.item(), 1e-12)
#
#     cosine = float(abs(torch.dot(m / m.norm(), v1)).item())
#     spectral_gap = max(lambda1 / max(lambda2, 1e-12), 1.0)
#     variance = min(lambda1 / trace, 1.0)
#
#     print("\nFisher diagnostics")
#     print("---------------------")
#     print("dim =", d)
#     print("lambda1 =", lambda1)
#     print("lambda2 =", lambda2)
#     print("gap =", spectral_gap)
#     print("variance explained =", variance)
#     print("cos(M,v1) =", cosine)
#
#     # --- Free all GPU tensors before returning ---
#     del client_data, v1, v2, m
#     torch.cuda.empty_cache()
#
#     return {
#         'fisher_cosine_sim': cosine,
#         'fisher_spectral_gap': spectral_gap,
#         'fisher_variance_explained': variance,
#         'fisher_lambda1': lambda1,
#         'fisher_lambda2': lambda2
#     }

# def compute_fisher_alignment(clients, global_weights, momentum_buffer, device, max_clients=10):
#     """
#     Compute alignment between SOFIM momentum buffer M_t and the top
#     eigenvector of the empirical Fisher Information Matrix.
#
#     Uses the power method with Fisher-vector products — O(n*d) per iteration,
#     never forms the d×d matrix.
#
#     Returns dict with:
#         cosine_sim       : |cos(M_t, v1)| where v1 = top Fisher eigenvector
#         spectral_gap     : lambda1 / lambda2
#         variance_explained: lambda1 / Tr(F) via Hutchinson estimator
#         lambda1          : dominant eigenvalue
#     """
#     subset = clients[:min(max_clients, len(clients))]
#     num_classes = subset[0].num_classes
#     d = momentum_buffer.shape[0]
#
#     # Build X_wb (features with bias) for each client and set model weights
#     client_data = []
#     total_n = 0
#     for client in subset:
#         n_i = client.num_samples
#         ones = torch.ones(n_i, 1, device=device, dtype=client.features.dtype)
#         X_wb = torch.cat([client.features, ones], dim=1).float()  # [n_i, d_feat+1]
#
#         # Softmax probabilities at current weights
#         w = global_weights.view(client.feature_dim + 1,
#                                 num_classes if num_classes > 1 else 1).float()
#         logits = X_wb @ w  # [n_i, C]
#         if num_classes == 1:
#             p = torch.sigmoid(logits)          # [n_i, 1]
#             W_diag = (p * (1 - p)).squeeze(1)  # [n_i]  scalar weight per sample
#         else:
#             p = torch.softmax(logits, dim=1)   # [n_i, C]
#
#         client_data.append((X_wb, p, n_i))
#         total_n += n_i
#
#     def fisher_vec_product(v):
#         """Compute F·v without forming F. v is [d] flat vector."""
#         Fv = torch.zeros(d, device=device, dtype=torch.float32)
#         if num_classes == 1:
#             # Binary: F = (1/n) Σ_i X_i^T diag(p_i(1-p_i)) X_i
#             # F·v = (1/n) Σ_i X_i^T (w_i ⊙ (X_i·v))
#             for X_wb, p, n_i in client_data:
#                 Xv = X_wb @ v          # [n_i]
#                 W_diag = (p * (1 - p)).squeeze(1)
#                 Fv += X_wb.T @ (W_diag * Xv) / total_n
#         else:
#             # Multiclass: F·v = (1/n) Σ_i X_i^T (P_i ⊙ XV - P_i ⊙ (P_i^T XV·1)) reshaped
#             # v is [d_feat+1 * C] — reshape to [d_feat+1, C]
#             d_feat_1 = client_data[0][0].shape[1]  # d_feat + 1
#             C = num_classes
#             V = v.view(d_feat_1, C)  # [d_feat+1, C]
#             Fv_mat = torch.zeros(d_feat_1, C, device=device, dtype=torch.float32)
#             for X_wb, p, n_i in client_data:
#                 XV = X_wb @ V              # [n_i, C]
#                 pXV = p * XV              # [n_i, C]  element-wise
#                 # subtract p * (sum_c p_c * XV_c) — the pp^T XV term
#                 ppXV = p * (p * XV).sum(dim=1, keepdim=True)  # [n_i, C]
#                 Fv_mat += X_wb.T @ (pXV - ppXV) / total_n
#             Fv = Fv_mat.flatten()
#         return Fv
#
#     # --- Power method for lambda1, v1 ---
#     # Warm-start from M_t direction
#     m = momentum_buffer.float().view(-1)
#     m_norm = m.norm()
#     if m_norm < 1e-12:
#         return {'cosine_sim': 0.0, 'spectral_gap': 1.0,
#                 'variance_explained': 0.0, 'lambda1': 0.0}
#
#     v1 = m / m_norm
#     lambda1 = 0.0
#     for _ in range(30):
#         Fv = fisher_vec_product(v1)
#         lambda1_new = float(torch.dot(v1, Fv).item())
#         v1 = Fv / (Fv.norm().clamp(min=1e-12))
#         if abs(lambda1_new - lambda1) < 1e-6:
#             break
#         lambda1 = lambda1_new
#     lambda1 = float(torch.dot(v1, fisher_vec_product(v1)).item())
#
#     # --- Deflated power method for lambda2 ---
#     v2 = torch.randn(d, device=device, dtype=torch.float32)
#     v2 = v2 - torch.dot(v2, v1) * v1
#     v2 = v2 / v2.norm().clamp(min=1e-12)
#     lambda2 = 0.0
#     for _ in range(20):
#         Fv2 = fisher_vec_product(v2)
#         Fv2 = Fv2 - lambda1 * torch.dot(Fv2, v1) * v1  # deflate
#         lambda2_new = float(torch.dot(v2, Fv2).item())
#         v2 = Fv2 / (Fv2.norm().clamp(min=1e-12))
#         if abs(lambda2_new - lambda2) < 1e-6:
#             break
#         lambda2 = lambda2_new
#
#     # --- Hutchinson trace estimate for variance_explained ---
#     n_probe = 20
#     trace_est = 0.0
#     for _ in range(n_probe):
#         z = torch.randint(0, 2, (d,), device=device,
#                           dtype=torch.float32) * 2 - 1
#         trace_est += float(torch.dot(z, fisher_vec_product(z)).item())
#     trace_est = max(trace_est / n_probe, 1e-12)
#
#     cosine_sim = float(abs(torch.dot(m / m_norm, v1)).item())
#     spectral_gap = lambda1 / max(abs(lambda2), 1e-12)
#     variance_explained = min(lambda1 / trace_est, 1.0)
#
#     return {
#         'fisher_cosine_sim': cosine_sim,
#         'fisher_spectral_gap': float(spectral_gap),
#         'fisher_variance_explained': float(variance_explained),
#         'fisher_lambda1': float(lambda1),
#         'fisher_lambda2': float(lambda2),
#     }

def run_dpfedgd_training(args):
    """
    Unified training loop supporting:
    1. Standard DP-FedGD (Algorithm 5)
    2. Server-side SOFIM (Second-order momentum)
    3. DP-FedNew-FC (Client-side ADMM/preconditioning)
    """

    # --- 1. System Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("Federated DP Training Framework")
    print("=" * 70)
    print(f"Backbone: {args.backbone}")
    print(f"Clients: {args.num_clients}, Participating per round: {args.clients_per_round}")

    if args.clients_per_round != args.num_clients:
        raise ValueError("This run assumes full client participation: set clients_per_round = num_clients.")
    print(f"Rounds: {args.federated_rounds}")
    if getattr(args, "no_dp", False):
        print("Privacy: DISABLED (--no_dp)")
    else:
        print(f"Privacy Budget: ε={args.epsilon}, δ={args.delta}")

    # Resolve resize_to_224
    resize_to_224 = False  # all supported backbones are CIFAR-style 32x32

    # --- 2. Data Loading ---
    dataset_name = getattr(args, "dataset", "cifar10")
    print(f"Loading federated dataset features: dataset={dataset_name} | backbone={args.backbone}")

    if dataset_name == "cifar10":
        client_loaders, global_test_loader, feature_dim, _ = get_federated_cifar10_feature_loaders(
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            device=str(device),
            partition_type=args.partition_type,
            resize_to_224=resize_to_224,
            alpha=getattr(args, "dirichlet_alpha", 0.5),
            classes_per_client=getattr(args, "classes_per_client", 2),
            backbone=args.backbone,
            seed=args.seed,
        )
        num_classes = 10
        is_binary = False

    elif dataset_name == "cifar10_binary" or getattr(args, "binary", False):
        client_loaders, global_test_loader, feature_dim, _ = get_federated_cifar10_binary_features(
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            device=str(device),
            resize_to_224=resize_to_224,
            partition_type=args.partition_type,
            alpha=getattr(args, "dirichlet_alpha", 0.5),
            classes_per_client=getattr(args, "classes_per_client", 2),
            backbone=args.backbone,
            seed=args.seed,
        )
        num_classes = 1
        is_binary = True

    elif dataset_name == "chestmnist":
        client_loaders, global_test_loader, feature_dim, num_classes, _ = get_federated_chestmnist_features(
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            device=str(device),
            partition_type=args.partition_type,
            alpha=getattr(args, "dirichlet_alpha", 0.5),
            classes_per_client=getattr(args, "classes_per_client", 2),
            seed=args.seed,
            backbone=args.backbone,
        )
        is_binary = False

    elif dataset_name == "pathmnist":
        client_loaders, global_test_loader, feature_dim, num_classes, _ = get_federated_pathmnist_features(
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            device=str(device),
            partition_type=args.partition_type,
            alpha=getattr(args, "dirichlet_alpha", 0.5),
            classes_per_client=getattr(args, "classes_per_client", 2),
            seed=args.seed,
            backbone=args.backbone,
        )
        is_binary = False

    elif dataset_name == "bloodmnist":
        client_loaders, global_test_loader, feature_dim, num_classes, _ = get_federated_bloodmnist_features(
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            device=str(device),
            partition_type=args.partition_type,
            alpha=getattr(args, "dirichlet_alpha", 0.5),
            classes_per_client=getattr(args, "classes_per_client", 2),
            seed=args.seed,
            backbone=args.backbone,
        )
        is_binary = False

    elif dataset_name == "dermamnist":
        client_loaders, global_test_loader, feature_dim, num_classes, _ = get_federated_dermamnist_features(
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            device=str(device),
            partition_type=args.partition_type,
            alpha=getattr(args, "dirichlet_alpha", 0.5),
            classes_per_client=getattr(args, "classes_per_client", 2),
            seed=args.seed,
            backbone=args.backbone,
        )
        is_binary = False

    elif dataset_name == "tinyimagenet":
        client_loaders, global_test_loader, feature_dim, num_classes, _ = get_federated_tinyimagenet_features(
            data_dir=args.data_dir,
            num_clients=args.num_clients,
            batch_size=args.batch_size,
            device=str(device),
            partition_type=args.partition_type,
            alpha=getattr(args, "dirichlet_alpha", 0.5),
            classes_per_client=getattr(args, "classes_per_client", 10),
            seed=args.seed,
            backbone=args.backbone,
            num_classes_subset=getattr(args, "tinyimagenet_classes", 50),
            subset_seed=getattr(args, "tinyimagenet_subset_seed", 0),
        )
        is_binary = False

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Feature dimension: {feature_dim} | Classes: {num_classes} | Dataset: {dataset_name}")

    # --- 3. Optimizer Initialization ---
    print("Initializing optimizer and client state...")

    # CASE: DP-SCAFFOLD (Control Variates)
    # --- Inside run_dpfedgd_training (train.py) ---

    # Step 1: Initialize Case for DP-FedFC Server/Clients
    if args.use_fedfc:
        print("Mode: DP-FedFC (Feature covariance preconditioning enabled)")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device, fc_cc=args.fc_cc)
                   for i, loader in enumerate(client_loaders)]
        server = DPFedFCServer(feature_dim, num_classes,
                               learning_rate=args.learning_rate,
                               fc_gamma=args.fc_gamma,
                               device=device)

    # Step 2: Initialize Sanitizer for FedFC (later in the function)
    # --- Move this after line 1132 in train.py ---


    # CASE: DP-FedNew-FC
    elif args.use_fednew_fc:
        print("Mode: DP-FedNew-FC (Client-side preconditioning enabled)")
        clients = [
            DPFedNewFCClient(i, loader, feature_dim, num_classes, device,
                             args.fn_alpha, args.fn_rho)
            for i, loader in enumerate(client_loaders)
        ]
        server = DPFedNewFCServer(feature_dim, num_classes, device)

    # CASE: DP-SCAFFOLD (Control Variates)
    elif getattr(args, "use_scaffold", False):
        print("Mode: DP-SCAFFOLD (record-level full-batch control variates)")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device) for i, loader in enumerate(client_loaders)]
        server_lr = args.learning_rate if args.scaffold_server_lr is None else args.scaffold_server_lr
        server = DPFedScaffoldServer(feature_dim, num_classes, server_lr=server_lr, device=device)

    # CASE: SOFIM (Server-side Second-Order)
    elif args.use_sofim:
        print("Mode: DP-SOFIM (Server-side acceleration enabled)")
        if args.sofim_adaptive_params:
            sofim_params = get_adaptive_sofim_params(args.epsilon)
            print(f"  Adaptive Regime: {sofim_params['regime']}")
        else:
            sofim_params = {
                'beta': args.sofim_beta,
                'rho': args.sofim_rho,
                'use_bias_correction': not args.sofim_disable_bias_correction
            }

        warmup_use_ftrl = getattr(args, 'sofim_warmup_use_ftrl', False)
        if warmup_use_ftrl and args.sofim_warmup_rounds > 0:
            w = min(args.sofim_warmup_rounds, args.federated_rounds)
            warmup_t_eff = math.ceil(math.log2(max(w, 2))) + 1
            noise_reduction = (w / warmup_t_eff) ** 0.5
            print(f"  Warmup FTRL: {w} warmup rounds → T_eff={warmup_t_eff} "
                  f"(vs {w} sequential); noise reduction ≈ {noise_reduction:.2f}x over warmup phase")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device) for i, loader in
                   enumerate(client_loaders)]
        server = DPFedGDServerSOFIM(
            feature_dim=feature_dim, num_classes=num_classes,
            learning_rate=args.learning_rate, device=device,
            beta=sofim_params['beta'], rho=sofim_params['rho'],
            use_bias_correction=sofim_params['use_bias_correction'],
            warmup_rounds=args.sofim_warmup_rounds,
            ablation_mode=getattr(args, 'sofim_ablation_mode', 'full'),
            weight_decay=args.sofim_weight_decay,
            warmup_use_ftrl=warmup_use_ftrl,
        )

    elif getattr(args, "use_fedavg_multi_iters", False):
        print(f"Mode: DP-FedAvg (K={args.fedavg_local_steps} local steps)")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device)
                   for i, loader in enumerate(client_loaders)]
        server = DPFedAvgServer(feature_dim, num_classes,
                                learning_rate=args.learning_rate, device=device)

    elif getattr(args, "use_fedadam", False):
        print("Mode: DP-FedAdam (server-side Adam)")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device)
                   for i, loader in enumerate(client_loaders)]
        server = DPFedAdamServer(
            feature_dim=feature_dim, num_classes=num_classes,
            learning_rate=args.learning_rate, device=device,
            beta1=args.fedadam_beta1,
            beta2=args.fedadam_beta2,
            epsilon_adam=args.fedadam_epsilon,
        )
    elif getattr(args, "use_fedyogi", False):
        print("Mode: DP-FedYogi (server-side Yogi, Reddi et al. 2021)")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device)
                   for i, loader in enumerate(client_loaders)]
        server = DPFedYogiServer(
            feature_dim=feature_dim, num_classes=num_classes,
            learning_rate=args.learning_rate, device=device,
            beta1=args.fedyogi_beta1,
            beta2=args.fedyogi_beta2,
            epsilon_yogi=args.fedyogi_epsilon,
        )
    elif getattr(args, "use_fedprox", False):
        print(f"Mode: DP-FedProx (K={args.fedprox_local_steps}, mu={args.fedprox_mu})")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device)
                   for i, loader in enumerate(client_loaders)]
        server = DPFedAvgServer(feature_dim, num_classes,
                                learning_rate=args.learning_rate, device=device)

    elif getattr(args, "use_adafedprox", False):
        print(f"Mode: DP-AdaFedProx (K={args.fedprox_local_steps}, "
              f"mu_0={args.fedprox_mu}, mu_t=mu_0/sqrt(t))")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device)
                   for i, loader in enumerate(client_loaders)]
        server = DPFedAvgServer(feature_dim, num_classes,
                                learning_rate=args.learning_rate, device=device)

    elif getattr(args, "use_ftrl", False):
        print("Mode: DP-FTRL (binary tree aggregation, Kairouz et al. ICML 2021)")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device)
                   for i, loader in enumerate(client_loaders)]
        server = DPFedFTRLServer(feature_dim, num_classes,
                                 learning_rate=args.learning_rate, device=device)

    # CASE: Standard DP-FedGD (First-Order)
    else:
        print("Mode: First-order SGD")
        clients = [DPFedGDClient(i, loader, feature_dim, num_classes, device) for i, loader in
                   enumerate(client_loaders)]
        server = DPFedGDServer(feature_dim, num_classes, learning_rate=args.learning_rate, device=device)

    # --- 4. Privacy Calibration ---
    min_dataset_size = min(len(loader.dataset) for loader in client_loaders)
    # The sensitivity bound for FedGD/SOFIM is args.gradient_clip_norm
    # For FedNew-FC, the noise is added to the primal with sensitivity args.fn_c_primal
    clip_for_accountant = args.fn_c_primal if args.use_fednew_fc else args.gradient_clip_norm

    dp_enabled = not getattr(args, "no_dp", False)
    if not dp_enabled:
        print("WARNING: DP is DISABLED (no clipping, no noise). Running clean baseline for sanity check.")

    if args.use_fednew_fc:
        if dp_enabled:
            # Use the specialized FedNew sanitizer
            sanitizer = DPFedNewSanitizer(
                num_clients=args.num_clients,
                clients_per_round=args.clients_per_round,
                federated_rounds=args.federated_rounds,
                fn_c2=args.fn_c2,
                fn_gamma=args.fn_alpha + args.fn_rho,  # gamma = alpha + rho
                fn_c_primal=args.fn_c_primal,
                device=device
            )
            sanitizer.calibrate_noise(args.epsilon, args.delta)
        else:
            sanitizer = None
    else:
        # DP-SCAFFOLD adds DP noise K times per training round, so privacy composes over T*K releases.
        total_dp_releases = args.federated_rounds
        if getattr(args, "use_scaffold", False):
            total_dp_releases = args.federated_rounds * args.scaffold_local_steps
        elif getattr(args, "use_fedavg_multi_iters", False):
            total_dp_releases = args.federated_rounds * args.fedavg_local_steps
        elif getattr(args, "use_fedprox", False) or getattr(args, "use_adafedprox", False):
            total_dp_releases = args.federated_rounds * args.fedprox_local_steps
        elif args.use_sofim and getattr(args, "sofim_warmup_use_ftrl", False) and args.sofim_warmup_rounds > 0:
            # FTRL tree mechanism during warmup reduces warmup_rounds to T_eff compositions.
            # Total effective compositions = T_eff(warmup) + (T - warmup) < T.
            w = min(args.sofim_warmup_rounds, args.federated_rounds)
            warmup_t_eff = math.ceil(math.log2(max(w, 2))) + 1
            total_dp_releases = warmup_t_eff + (args.federated_rounds - w)

        if args.use_fedfc:
            # If DP is enabled, Algorithm 3 requires covariance noise sigma_c.
            # If DP is disabled (clean baseline), allow sigma_c to be omitted and treat it as 0.
            if dp_enabled:
                if args.fc_sigma_c is None:
                    raise ValueError("--use_fedfc requires --fc_sigma_c (sigma_c) to implement Algorithm 3 exactly")
                sanitizer = DPFedFCSanitizer(
                    num_clients=args.num_clients,
                    clients_per_round=args.clients_per_round,
                    federated_rounds=total_dp_releases,
                    dataset_size=min_dataset_size,
                    gradient_clip_norm=args.gradient_clip_norm,
                    fn_cc=args.fc_cc,
                    fn_cg=args.gradient_clip_norm,
                    sigma_c=args.fc_sigma_c,
                    neighbor_type="replace_one",
                    device=device,
                )
            else:
                # Clean baseline: no DP noise or clipping
                if args.fc_sigma_c is None:
                    args.fc_sigma_c = 0.0
                sanitizer = None
        else:
            if getattr(args, "use_ftrl", False):
                sanitizer_cls = DPFedFTRLSanitizer
            elif getattr(args, "use_scaffold", False):
                sanitizer_cls = DPFedScaffoldSanitizer
            else:
                sanitizer_cls = DPFedGDSanitizer
            sanitizer = sanitizer_cls(
                num_clients=args.num_clients,
                clients_per_round=args.clients_per_round,
                federated_rounds=total_dp_releases,
                gradient_clip_norm=args.gradient_clip_norm,
                dataset_size=min_dataset_size,
                neighbor_type="replace_one",
                display_rounds=args.federated_rounds,
                device=device
            )

        if dp_enabled:
            sanitizer.calibrate_noise(args.epsilon, args.delta)
        else:
            sanitizer = None

    # --- FedFC Phase 1: One-time global covariance precomputation (Algorithm 3, Lines 4-8) ---
    if args.use_fedfc:
        print(">>> Phase 1 (FedFC): Pre-computing Global Noisy Covariance")
        # Algorithm 3 says: for user i=1..n do ... share C_i
        # Use all clients (not just a sampled subset) for the one-time covariance phase.
        local_covariances = [client.compute_local_covariance(sanitizer) for client in clients]
        server.set_global_covariance(local_covariances)
        print(">>> Phase 1 (FedFC): Done. Global preconditioner C_inv is set.")

    def _agg_client_stats(client_stats_list):
        """Return mean/min/max aggregates for numeric client stats."""
        if not client_stats_list:
            return {}

        def _vals(key):
            vs = []
            for cs in client_stats_list:
                v = cs.get(key, None)
                if v is None:
                    continue
                # allow torch scalars
                if hasattr(v, "item"):
                    v = v.item()
                vs.append(float(v))
            return vs

        def _mean(key, default=0.0):
            vs = _vals(key)
            return float(np.mean(vs)) if vs else default

        def _min(key, default=0.0):
            vs = _vals(key)
            return float(np.min(vs)) if vs else default

        def _max(key, default=0.0):
            vs = _vals(key)
            return float(np.max(vs)) if vs else default

        return {
            # clipping behavior
            "clip_frac_mean": _mean("clip_frac"),
            "clip_frac_max": _max("clip_frac"),
            "clip_factor_mean": _mean("clip_factor_mean"),

            # per-example grad norms
            "perex_grad_norm_mean": _mean("perex_grad_norm_mean"),
            "perex_grad_norm_p50_mean": _mean("perex_grad_norm_p50"),
            "perex_grad_norm_p90_mean": _mean("perex_grad_norm_p90"),

            # signal + noise magnitudes
            "clipped_sum_norm_mean": _mean("clipped_sum_norm"),
            "clipped_sum_norm_max": _max("clipped_sum_norm"),
            "grad_norm_before_noise_mean": _mean("gradient_norm_before_noise"),
            "grad_norm_after_noise_mean": _mean("gradient_norm_after_noise"),
            "sum_noise_std_mean": _mean("sum_noise_std"),
        }

    # --- 5. Main Training Loop ---
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    results = {'config': vars(args), 'round_results': []}
    start_time = time.time()

    for round_num in range(1, args.federated_rounds + 1):
        round_start = time.time()
        if sanitizer is not None and not getattr(args, "use_scaffold", False):
            sanitizer.start_round()
        global_weights = server.get_global_weights()

        client_updates = []
        round_stats = {'total_loss': 0.0, 'total_samples': 0}
        client_debug_stats = []

        # Select clients
        # Enforce full client participation (no subsampling)
        selected_indices = np.arange(args.num_clients)
        selected_clients = [clients[i] for i in selected_indices]

        # --- CLIENT UPDATE PHASE ---
        deltas_x = []
        deltas_c = []
        client_updates = []  # For non-SCAFFOLD
        client_phase_start = time.time()

        for client in selected_clients:
            if args.use_fednew_fc:
                # Specialized primal-dual update with meaningful loss
                update, loss = client.compute_primal_update(global_weights, sanitizer, args)
                client_updates.append(update)
                round_stats['total_loss'] += loss * len(client.train_loader.dataset)
                round_stats['total_samples'] += len(client.train_loader.dataset)

            elif getattr(args, "use_scaffold", False):
                delta_x, delta_c_i, client_stats = client.compute_scaffold_update(
                    global_weights=global_weights,
                    server_c=server.get_server_control(),
                    sanitizer=sanitizer,
                    K=args.scaffold_local_steps,
                    client_lr=args.scaffold_client_lr,
                    round_num=round_num
                )
                deltas_x.append(delta_x)
                deltas_c.append(delta_c_i)
                round_stats['total_loss'] += client_stats['loss'] * len(client.train_loader.dataset)
                round_stats['total_samples'] += len(client.train_loader.dataset)
                client_debug_stats.append(client_stats)

            elif getattr(args, "use_fedavg_multi_iters", False):
                delta_x, client_stats = client.compute_fedavg_update(
                    global_weights=global_weights,
                    sanitizer=sanitizer,
                    K=args.fedavg_local_steps,
                    client_lr=args.fedavg_client_lr,
                )
                deltas_x.append(delta_x)
                round_stats['total_loss'] += client_stats['loss'] * len(client.train_loader.dataset)
                round_stats['total_samples'] += len(client.train_loader.dataset)
                client_debug_stats.append(client_stats)

            elif getattr(args, "use_fedprox", False):
                delta_x, client_stats = client.compute_fedprox_update(
                    global_weights=global_weights,
                    sanitizer=sanitizer,
                    K=args.fedprox_local_steps,
                    client_lr=args.fedprox_client_lr,
                    mu=args.fedprox_mu,
                )

            elif getattr(args, "use_adafedprox", False):
                # Diminishing proximal: mu_t = mu_0 / sqrt(t), floored at 10% of mu_0
                mu_t = max(args.fedprox_mu / math.sqrt(round_num),
                           args.fedprox_mu * 0.1)
                delta_x, client_stats = client.compute_fedprox_update(
                    global_weights=global_weights,
                    sanitizer=sanitizer,
                    K=args.fedprox_local_steps,
                    client_lr=args.fedprox_client_lr,
                    mu=mu_t,
                )
                client_stats['adaptive_mu'] = mu_t
                deltas_x.append(delta_x)
                round_stats['total_loss'] += client_stats['loss'] * len(client.train_loader.dataset)
                round_stats['total_samples'] += len(client.train_loader.dataset)
                client_debug_stats.append(client_stats)

            else:
                # Standard gradient computation (Algorithm 5)
                update, client_stats = client.compute_noisy_update(global_weights, sanitizer)
                client_updates.append(update)
                round_stats['total_loss'] += client_stats['loss'] * len(client.train_loader.dataset)
                round_stats['total_samples'] += len(client.train_loader.dataset)
                client_debug_stats.append(client_stats)

        client_phase_time = time.time() - client_phase_start

        # --- SERVER AGGREGATION PHASE ---
        server_phase_start = time.time()

        if args.use_fednew_fc:
            # Simple average of noisy primal variables
            server_stats = server.aggregate_and_update(client_updates)
        elif getattr(args, "use_scaffold", False):
            server_stats = server.aggregate_and_update(deltas_x, deltas_c)
        elif args.use_sofim:
            # Momentum + Sherman-Morrison preconditioning
            server_stats = server.aggregate_and_update(client_updates, sanitizer, current_round=round_num)
        elif getattr(args, "use_fedavg_multi_iters", False):
            server_stats = server.aggregate_and_update(deltas_x)
        elif getattr(args, "use_fedadam", False):
            server_stats = server.aggregate_and_update(client_updates, sanitizer)
        elif getattr(args, "use_fedyogi", False):
            server_stats = server.aggregate_and_update(client_updates, sanitizer)
        elif getattr(args, "use_fedprox", False) or getattr(args, "use_adafedprox", False):
            server_stats = server.aggregate_and_update(deltas_x)
        elif getattr(args, "use_ftrl", False):
            server_stats = server.aggregate_and_update(client_updates, sanitizer)
        else:
            # Standard First-Order Update
            server_stats = server.aggregate_and_update(client_updates, sanitizer)
        server_phase_time = time.time() - server_phase_start
        round_time = time.time() - round_start

        # Prepare round logging
        round_result = {
            'round': round_num,
            'train_loss': round_stats['total_loss'] / max(1, round_stats['total_samples']),
            'update_norm': server_stats['update_norm'],
            'weight_change_norm': server_stats['weight_change_norm'],
            'round_time': round_time,
            'client_time': client_phase_time,
            'server_time': server_phase_time,
        }
        # Merge server-side stats (e.g., FedFC u_avg_norm / U_k_norm / precond_ratio)
        if isinstance(server_stats, dict):
            round_result.update(server_stats)
        round_result.update(_agg_client_stats(client_debug_stats))

        # --- Fisher Alignment Diagnostic (SOFIM full mode only, every 10 rounds) ---
        # if (args.use_sofim
        #         and getattr(args, 'sofim_ablation_mode', 'full') == 'full'
        #         and server.M is not None
        #         and round_num % 10 == 0):
        #     align_stats = compute_fisher_alignment(
        #         clients=selected_clients,
        #         global_weights=server.get_global_weights(),
        #         momentum_buffer=server.M,
        #         device=device,
        #         max_clients=min(5, len(selected_clients)),
        #     )
        #     torch.cuda.empty_cache()
        #     round_result.update(align_stats)

        # --- Periodic Evaluation ---
        if round_num % args.eval_every == 0 or round_num == args.federated_rounds:
            eval_metrics = server.evaluate(global_test_loader, is_binary)
            round_result.update(eval_metrics)

            if args.use_sofim:
                optimizer_log = (
                    f"||g_agg||={round_result.get('gradient_norm', 0.0):.3e}, "
                    f"||M||={round_result.get('moment_norm', 0.0):.3e}, "
                    f"||Mhat||={round_result.get('moment_corrected_norm', 0.0):.3e}, "
                    f"update={round_result.get('update_norm', 0.0):.3e}, "
                )
            else:
                optimizer_log = (
                    f"||u_avg||={round_result.get('u_avg_norm', 0.0):.3e}, "
                    f"||U_k||={round_result.get('U_k_norm', 0.0):.3e}, "
                    f"ratio={round_result.get('precond_ratio', 0.0):.3e}, "
                )

            print(
                f"Round {round_num:3d}/{args.federated_rounds}: "
                f"Loss={round_result['train_loss']:.4f}, "
                f"Acc={eval_metrics['accuracy']:.4f}, "
                f"clip%={round_result.get('clip_frac_mean', 0.0):.2f}, "
                f"||g||={round_result.get('grad_norm_before_noise_mean', 0.0):.3f}, "
                f"{optimizer_log}"
                f"σ_server={round_result.get('sum_noise_std_mean', 0.0):.3e}, "
                f"Time={round_time:.1f}s"
            )

        results['round_results'].append(round_result)

    # --- Final Stats ---
    final_metrics = server.evaluate(global_test_loader, is_binary, return_mask=True)
    total_time = time.time() - start_time

    dp_enabled_final = sanitizer is not None
    if dp_enabled_final:
        achieved_epsilon, achieved_delta = sanitizer.get_privacy_cost()
    else:
        achieved_epsilon, achieved_delta = None, None

    results['privacy_stats'] = {
        'dp_enabled': dp_enabled_final,
        'target_epsilon': None if getattr(args, 'no_dp', False) else float(getattr(args, 'epsilon', 0.0)),
        'target_delta': None if getattr(args, 'no_dp', False) else float(getattr(args, 'delta', 0.0)),
        'achieved_epsilon': achieved_epsilon,
        'achieved_delta': achieved_delta,
    }

    round_times = [r['round_time'] for r in results['round_results']]
    client_times = [r.get('client_time', 0.0) for r in results['round_results']]
    server_times = [r.get('server_time', 0.0) for r in results['round_results']]

    results['final_stats'] = {
        **final_metrics,
        'total_time': total_time,
        'avg_round_time': float(np.mean(round_times)),
        'avg_client_time': float(np.mean(client_times)),
        'avg_server_time': float(np.mean(server_times)),
        'dp_enabled': dp_enabled_final,
        'achieved_epsilon': achieved_epsilon,
        'achieved_delta': achieved_delta,
    }

    print("\n" + "=" * 70)
    print("Training Complete!")
    if dp_enabled_final and achieved_epsilon is not None:
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f} | Privacy Cost: ε={achieved_epsilon:.4f}")
    else:
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f} | Privacy: DISABLED")
    print("=" * 70)

    import gc
    del clients, server
    if sanitizer is not None:
        del sanitizer
    torch.cuda.empty_cache()
    gc.collect()
    return results

def main():
    args = parse_args()

    try:
        results = run_dpfedgd_training(args)
        return results
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        return None
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys

    result = main()
    sys.exit(0 if result is not None else 1)