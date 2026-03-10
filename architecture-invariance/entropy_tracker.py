"""
Unified entropy tracker for architecture invariance experiment.
Tracks per-unit output entropy across training steps for any architecture.
"""

import torch
import torch.nn.functional as F
import json
import os


class EntropyTracker:
    """Track per-unit output entropy for any architecture."""

    def __init__(self, n_layers, n_units_per_layer, device='cpu'):
        self.n_layers = n_layers
        self.n_units = n_units_per_layer
        self.device = device

        # Per-unit entropy EMA
        self.ema = torch.ones(n_layers, n_units_per_layer, device=device)
        # Per-unit entropy variance EMA (for crystallization detection)
        self.ema_var = torch.ones(n_layers, n_units_per_layer, device=device) * 0.1
        # Observed max entropy per unit
        self.observed_max = torch.ones(n_layers, n_units_per_layer, device=device) * 0.1

        # History for analysis
        self.history = []  # list of (step, per_layer_crystal_frac, overall_crystal_frac)
        self.entropy_history = []  # list of (step, layer_mean_entropies)

    def update(self, layer_idx, unit_entropies):
        """
        Update EMA for a given layer.
        unit_entropies: [n_units] tensor of entropy values
        """
        unit_entropies = unit_entropies.detach()

        # Update observed max
        self.observed_max[layer_idx] = torch.max(
            self.observed_max[layer_idx], unit_entropies
        )

        # Normalize by observed max
        normed = unit_entropies / (self.observed_max[layer_idx] + 1e-8)

        # Adaptive decay based on variance
        variance = (normed - self.ema[layer_idx]).pow(2)
        self.ema_var[layer_idx] = 0.99 * self.ema_var[layer_idx] + 0.01 * variance

        # High variance = slow decay (still learning), low variance = fast decay (crystallized)
        alpha = 0.95 + 0.045 * torch.sigmoid(50 * self.ema_var[layer_idx])

        # Update EMA
        self.ema[layer_idx] = alpha * self.ema[layer_idx] + (1 - alpha) * normed

    def crystal_fraction(self, threshold=0.1):
        """What % of units have low entropy variance (crystallized)?"""
        crystallized = (self.ema_var < threshold).float()
        per_layer = crystallized.mean(dim=1)  # [n_layers]
        overall = crystallized.mean().item()
        return per_layer.tolist(), overall

    def layer_ordering(self, threshold=0.1):
        """Which layers have highest crystal fraction? Returns sorted indices."""
        per_layer, _ = self.crystal_fraction(threshold)
        # Sort by crystal fraction descending (most crystallized first)
        ordering = sorted(range(self.n_layers), key=lambda i: per_layer[i], reverse=True)
        return ordering, per_layer

    def log_step(self, step):
        """Record current state for analysis."""
        per_layer, overall = self.crystal_fraction()
        self.history.append({
            'step': step,
            'per_layer': per_layer,
            'overall': overall,
        })

        layer_means = self.ema.mean(dim=1).tolist()
        self.entropy_history.append({
            'step': step,
            'layer_means': layer_means,
            'layer_vars': self.ema_var.mean(dim=1).tolist(),
        })

    def save(self, path):
        """Save tracking history to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'n_layers': self.n_layers,
            'n_units': self.n_units,
            'history': self.history,
            'entropy_history': self.entropy_history,
            'final_ordering': self.layer_ordering()[0],
            'final_crystal_fraction': self.crystal_fraction(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def compute_attention_entropy(attn_weights):
    """
    Compute per-head entropy from attention weights.
    attn_weights: [batch, n_heads, seq_len, seq_len]
    Returns: [n_heads] mean entropy per head
    """
    # Clamp for numerical stability
    attn_weights = attn_weights.clamp(min=1e-8)
    # Entropy per position: -sum(p * log(p))
    entropy = -(attn_weights * attn_weights.log()).sum(dim=-1)  # [batch, heads, seq]
    # Normalize by max possible entropy at each position
    seq_len = attn_weights.shape[-1]
    positions = torch.arange(1, seq_len + 1, device=attn_weights.device, dtype=attn_weights.dtype)
    max_entropy = positions.log()
    max_entropy[0] = 1.0  # avoid div by zero
    normalized = entropy / max_entropy.unsqueeze(0).unsqueeze(0)  # [batch, heads, seq]
    # Mean over batch and sequence
    return normalized.mean(dim=(0, 2))  # [n_heads]


def compute_hidden_entropy(hidden_states, n_bins=32):
    """
    Compute per-unit entropy from hidden state activations.
    hidden_states: [batch, seq_len, n_units]
    Returns: [n_units] entropy per unit

    Uses histogram-based entropy estimation over the batch*seq dimension.
    """
    B, T, D = hidden_states.shape
    flat = hidden_states.reshape(-1, D)  # [B*T, D]

    entropies = torch.zeros(D, device=hidden_states.device)
    for i in range(D):
        vals = flat[:, i]
        # Adaptive binning based on value range
        lo, hi = vals.min().item(), vals.max().item()
        if hi - lo < 1e-6:
            entropies[i] = 0.0
            continue
        counts = torch.histc(vals, bins=n_bins, min=lo, max=hi)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies[i] = -(probs * probs.log()).sum()

    # Normalize by max entropy (log(n_bins))
    max_ent = torch.tensor(n_bins, dtype=torch.float32).log()
    return entropies / max_ent


def compute_gate_entropy(gate_activations):
    """
    Compute per-unit entropy from LSTM gate activations (sigmoid outputs).
    gate_activations: [batch, seq_len, n_units] values in (0, 1)
    Returns: [n_units] binary entropy per unit

    Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
    """
    # Mean activation per unit across batch and time
    p = gate_activations.mean(dim=(0, 1)).clamp(1e-6, 1 - 1e-6)
    entropy = -p * p.log() - (1 - p) * (1 - p).log()
    # Normalize by max binary entropy (log(2))
    return entropy / torch.tensor(2.0).log()
