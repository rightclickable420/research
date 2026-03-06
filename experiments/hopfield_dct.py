"""
Hopfield-DCT Collective Compression Experiment
Phase 1: Mechanism proof on Kevin's 80 memory chunks

Tests whether Hopfield associative reconstruction improves
retrieval quality of DCT-degraded embeddings.
"""

import numpy as np
from scipy.fft import dct, idct
import json

# Load embeddings
all_embeddings = np.load('/tmp/memory_embeddings.npy')

# Filter out zero-padded vectors
norms = np.linalg.norm(all_embeddings, axis=1)
mask = norms > 0.01
embeddings = all_embeddings[mask]
N, D = embeddings.shape
print(f"Real embeddings: {N} x {D}")
print(f"Norm range: {norms[mask].min():.3f} - {norms[mask].max():.3f}")

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# ─── DCT Compression ─────────────────────────────────────

def dct_compress(emb, keep_k):
    """DCT compress: keep top-k frequency coefficients."""
    coeffs = dct(emb, type=2, norm='ortho')
    compressed = np.zeros_like(coeffs)
    compressed[:keep_k] = coeffs[:keep_k]
    return idct(compressed, type=2, norm='ortho')

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# ─── Continuous Hopfield Network ──────────────────────────

class ContinuousHopfield:
    """
    Modern continuous Hopfield network (Ramsauer et al., 2021).
    Stores patterns as memories, retrieves by attention-like energy minimization.
    """
    def __init__(self, memories, beta=1.0):
        """
        memories: (N, D) array of stored patterns
        beta: inverse temperature (higher = sharper retrieval)
        """
        self.memories = memories  # (N, D)
        self.beta = beta
    
    def retrieve(self, query, steps=5):
        """
        Retrieve from partial/degraded query.
        Uses the attention-like update rule from modern Hopfield networks.
        """
        xi = query.copy()
        for _ in range(steps):
            # Compute similarities (attention scores)
            scores = self.beta * self.memories @ xi  # (N,)
            # Softmax
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            # Weighted combination of memories (attention output)
            xi = weights @ self.memories  # (D,)
            # Normalize
            xi = xi / (np.linalg.norm(xi) + 1e-10)
        return xi

# ─── Experiment: DCT alone vs DCT + Hopfield ─────────────

# Compression levels to test (fraction of 768 coefficients kept)
keep_fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

results = []

for frac in keep_fractions:
    keep_k = max(1, int(D * frac))
    
    dct_sims = []
    hopfield_sims = []
    
    for i in range(N):
        original = embeddings[i]
        
        # DCT compress
        degraded = dct_compress(original, keep_k)
        degraded_norm = degraded / (np.linalg.norm(degraded) + 1e-10)
        
        # Hopfield reconstruct (exclude self from memory bank to be fair)
        memory_bank = np.delete(embeddings, i, axis=0)
        hopfield = ContinuousHopfield(memory_bank, beta=8.0)
        reconstructed = hopfield.retrieve(degraded_norm)
        
        # Measure quality
        dct_sim = cosine_sim(original, degraded_norm)
        hop_sim = cosine_sim(original, reconstructed)
        
        dct_sims.append(dct_sim)
        hopfield_sims.append(hop_sim)
    
    dct_mean = np.mean(dct_sims)
    hop_mean = np.mean(hopfield_sims)
    delta = hop_mean - dct_mean
    
    result = {
        'keep_frac': frac,
        'keep_k': keep_k,
        'dct_quality': round(float(dct_mean), 4),
        'hopfield_quality': round(float(hop_mean), 4),
        'delta': round(float(delta), 4),
        'improvement_pct': round(float(delta / (1 - dct_mean + 1e-10) * 100), 1),
    }
    results.append(result)
    
    marker = "✓" if delta > 0 else "✗"
    print(f"{marker} keep={frac:.0%} ({keep_k:3d} coeffs) | DCT: {dct_mean:.4f} | Hopfield: {hop_mean:.4f} | Δ: {delta:+.4f}")

# ─── Summary ──────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY: Hopfield-DCT Collective Compression (N={}, D={})".format(N, D))
print("="*70)

positive = sum(1 for r in results if r['delta'] > 0)
print(f"\nHopfield improved retrieval in {positive}/{len(results)} compression levels")

# Find crossover point
for r in results:
    if r['delta'] <= 0 and r['keep_frac'] > 0.01:
        print(f"Crossover at keep={r['keep_frac']:.0%}: DCT alone is sufficient")
        break

# The key metric: at what compression can Hopfield maintain 0.9 quality?
for r in results:
    if r['hopfield_quality'] >= 0.9:
        print(f"Hopfield reaches 0.9 quality at keep={r['keep_frac']:.0%} ({r['keep_k']} coeffs)")
        break

for r in results:
    if r['dct_quality'] >= 0.9:
        print(f"DCT alone reaches 0.9 quality at keep={r['keep_frac']:.0%} ({r['keep_k']} coeffs)")
        break

# Save results
with open('/tmp/hopfield_dct_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /tmp/hopfield_dct_results.json")
