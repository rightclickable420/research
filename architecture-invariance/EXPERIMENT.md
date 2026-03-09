# Architecture Invariance Experiment

**Question:** Does crystallization ordering depend on the model architecture or the data?

If ordering is architecture-invariant → crystallization is a data measurement (spectroscopy).
If ordering is architecture-specific → crystallization is a transformer property.

## Design

### Architectures (all ~1M parameters, 6 layers)

| Model | Units to Track | Crystallization Metric |
|-------|---------------|----------------------|
| Transformer (baseline) | 36 attention heads (6L × 6H) | Per-head attention entropy EMA |
| LSTM | 384 hidden units × 6 layers | Per-unit output entropy EMA |
| Mamba/S4 | 384 channels × 6 layers | Per-channel output entropy EMA |
| MLP | 384 neurons × 6 layers | Per-neuron activation entropy EMA |

### Dataset
- Shakespeare char-level (same as existing experiments)
- Same tokenizer, same block_size=256, same batch_size=64

### Training
- Same optimizer (AdamW), same LR schedule
- 10,000 steps (same as existing crystallize config)
- Log per-unit entropy every 50 steps

### Measurements
1. **Crystal fraction** — % of units below entropy threshold at convergence
2. **Ordering** — which layers crystallize first (is it middle-out for all?)
3. **Ceiling** — does each architecture converge to similar crystal %?
4. **Velocity** — how fast does crystallization proceed?

### Success Criteria
- **Strong positive:** All 4 architectures show middle-out ordering on Shakespeare → data property
- **Weak positive:** Crystal ceiling is similar across architectures, ordering varies → partially data
- **Negative:** Each architecture has completely different crystallization dynamics → probe property

## Implementation Plan

### Files to Create

```
projects/research/architecture-invariance/
├── EXPERIMENT.md          (this file)
├── models/
│   ├── transformer.py     (extract from nanoGPT model.py, add entropy logging)
│   ├── lstm.py            (6-layer LSTM with per-unit entropy tracking)
│   ├── mamba.py           (6-layer Mamba/S4 with per-channel entropy tracking)
│   └── mlp.py             (6-layer MLP with per-neuron entropy tracking)
├── train_invariance.py    (unified training script, selects model by arg)
├── entropy_tracker.py     (shared entropy measurement + logging)
├── analyze.py             (generate crystallization curves + comparison plots)
└── data/                  (symlink to nanoGPT shakespeare_char data)
```

### Entropy Measurement (Generalized)

The key insight: "entropy" means different things per architecture, but the *measurement* is the same — how stable is each unit's output distribution over recent batches?

```python
class EntropyTracker:
    """Track per-unit output entropy for any architecture."""
    
    def __init__(self, n_layers, n_units_per_layer):
        self.ema = torch.ones(n_layers, n_units_per_layer)
        self.ema_var = torch.ones(n_layers, n_units_per_layer) * 0.1
        self.observed_max = torch.ones(n_layers, n_units_per_layer) * 0.1
    
    def update(self, layer_idx, unit_outputs):
        """
        unit_outputs: [batch, seq_len, n_units]
        Compute entropy of each unit's output distribution over the sequence.
        """
        # Softmax over sequence dimension to get distribution
        probs = F.softmax(unit_outputs, dim=1)  
        entropy = -(probs * probs.log()).sum(dim=1).mean(dim=0)  # [n_units]
        # Update EMA with adaptive decay (same as crystallize attention)
        ...
    
    def crystal_fraction(self, threshold=0.1):
        """What % of units have low entropy variance (crystallized)?"""
        ...
    
    def layer_ordering(self):
        """Which layers crystallized first?"""
        ...
```

### Per-Architecture Specifics

**Transformer:** Already implemented. Extract per-head attention entropy from CausalSelfAttention. The attention weight matrix directly gives a distribution to compute entropy over.

**LSTM:** Per hidden unit, track the gate activation entropy. Specifically, the forget gate and output gate activations form distributions that can be entropy-measured. Alternative: track hidden state stability (how much each unit's output changes between timesteps).

**Mamba/S4:** Per channel in the SSM, track the selectivity parameter (dt) entropy. Mamba's selective mechanism is analogous to attention — it decides what to remember and forget. Track per-channel dt stability.

**MLP:** Per neuron per layer, track activation distribution entropy across the sequence. A crystallized neuron fires consistently for certain inputs. A fluid neuron varies.

### What We Need

- [ ] Python 3 + PyTorch (already have)
- [ ] Shakespeare char-level data (already have in nanoGPT/data/)
- [ ] mamba-ssm package (pip install mamba-ssm, or minimal S4 implementation)
- [ ] ~4-8 hours of CPU training (4 models × ~1-2 hrs each on our VPS)

### Risk: CPU Training Time

Our VPS has no GPU. nanoGPT on Shakespeare takes ~30 min on MPS (Apple Silicon). On CPU it'll be slower — estimate 1-2 hours per model. Total ~4-8 hours. We can run them in parallel (4 cores available) or sequential overnight.

**Mitigation:** Reduce max_iters to 3000 for initial signal check. If ordering emerges by step 3000, we don't need 10000.

## Timeline

- Day 1: Implement LSTM + MLP models with entropy tracking (~2-3 hours code)
- Day 1: Start training runs overnight
- Day 2: Implement Mamba (or minimal S4) + start training
- Day 2: Analysis script + comparison plots
- Day 3: Write up results

## References

- Existing crystallize configs: `projects/nanoGPT/config/train_crystallize*.py`
- Existing model: `projects/nanoGPT/model.py` (CausalSelfAttention with entropy tracking)
- Paper draft: `projects/research/crystallizing-attention.md`
