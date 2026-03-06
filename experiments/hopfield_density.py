"""
Hopfield-DCT Density Scaling Experiment
Phase 2: Does reconstruction quality improve with corpus density?

Uses Cohere/Wikipedia Simple English embeddings (1024-dim).
Tests at: 50, 100, 500, 1000, 5000 corpus sizes.
"""

import numpy as np
from scipy.fft import dct, idct
from datasets import load_dataset
import json, time

# ─── Load Wikipedia embeddings ────────────────────────────

print("Loading Wikipedia embeddings (streaming)...")
ds = load_dataset("Cohere/wikipedia-2023-11-embed-multilingual-v3", "simple", split="train", streaming=True)

MAX_LOAD = 6000  # Load enough for our largest test + held-out
embeddings_list = []
for i, sample in enumerate(ds):
    if i >= MAX_LOAD:
        break
    embeddings_list.append(sample['emb'])
    if (i+1) % 1000 == 0:
        print(f"  Loaded {i+1}...")

all_embeddings = np.array(embeddings_list, dtype=np.float32)
# Normalize
norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
all_embeddings = all_embeddings / (norms + 1e-10)

N_total, D = all_embeddings.shape
print(f"Total embeddings: {N_total} x {D}")

# ─── Helpers ──────────────────────────────────────────────

def dct_compress(emb, keep_k):
    coeffs = dct(emb, type=2, norm='ortho')
    c = np.zeros_like(coeffs)
    c[:keep_k] = coeffs[:keep_k]
    return idct(c, type=2, norm='ortho')

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

class Hopfield:
    def __init__(self, memories, beta):
        self.memories = memories
        self.beta = beta
    
    def retrieve(self, query, steps=5):
        xi = query.copy()
        for _ in range(steps):
            scores = self.beta * self.memories @ xi
            scores -= scores.max()
            w = np.exp(scores)
            w /= w.sum()
            xi = w @ self.memories
            xi /= (np.linalg.norm(xi) + 1e-10)
        return xi

# ─── Density Scaling Experiment ───────────────────────────

# Hold out 50 test memories (never in the corpus)
test_set = all_embeddings[:50]
corpus_pool = all_embeddings[50:]

corpus_sizes = [50, 100, 500, 1000, 5000]
keep_fracs = [0.02, 0.05, 0.1, 0.2, 0.5]
beta = 128.0

results = []

for n_corpus in corpus_sizes:
    if n_corpus > len(corpus_pool):
        print(f"Skipping n={n_corpus}, not enough data")
        continue
    
    corpus = corpus_pool[:n_corpus]
    t0 = time.time()
    
    for frac in keep_fracs:
        keep_k = max(1, int(D * frac))
        dct_sims = []
        hop_sims = []
        
        # Test on held-out set
        for i in range(len(test_set)):
            original = test_set[i]
            degraded = dct_compress(original, keep_k)
            degraded_norm = degraded / (np.linalg.norm(degraded) + 1e-10)
            
            h = Hopfield(corpus, beta=beta)
            reconstructed = h.retrieve(degraded_norm)
            
            dct_sims.append(cosine_sim(original, degraded_norm))
            hop_sims.append(cosine_sim(original, reconstructed))
        
        dct_mean = np.mean(dct_sims)
        hop_mean = np.mean(hop_sims)
        delta = hop_mean - dct_mean
        
        result = {
            'corpus_size': n_corpus,
            'keep_frac': frac,
            'keep_k': keep_k,
            'dct_quality': round(float(dct_mean), 4),
            'hopfield_quality': round(float(hop_mean), 4),
            'delta': round(float(delta), 4),
        }
        results.append(result)
    
    elapsed = time.time() - t0
    # Print summary for this corpus size
    print(f"\n{'='*60}")
    print(f"Corpus size: {n_corpus} | Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"{'keep':>6} | {'DCT':>7} {'Hop':>7} {'Δ':>7}")
    print(f"{'-'*35}")
    for r in results:
        if r['corpus_size'] == n_corpus:
            marker = "↑" if r['delta'] > 0 else "↓"
            print(f"{r['keep_frac']:>5.0%} | {r['dct_quality']:>7.4f} {r['hopfield_quality']:>7.4f} {r['delta']:>+7.4f} {marker}")

# ─── Density Scaling Summary ─────────────────────────────

print(f"\n{'='*60}")
print("DENSITY SCALING SUMMARY")
print(f"{'='*60}")
print(f"\nFixed compression at 5% ({int(D*0.05)} coefficients):")
print(f"{'Corpus':>8} | {'DCT':>7} {'Hop':>7} {'Δ':>7}")
print(f"{'-'*35}")
for r in results:
    if r['keep_frac'] == 0.05:
        print(f"{r['corpus_size']:>8} | {r['dct_quality']:>7.4f} {r['hopfield_quality']:>7.4f} {r['delta']:>+7.4f}")

print(f"\nFixed compression at 10%:")
print(f"{'Corpus':>8} | {'DCT':>7} {'Hop':>7} {'Δ':>7}")
print(f"{'-'*35}")
for r in results:
    if r['keep_frac'] == 0.1:
        print(f"{r['corpus_size']:>8} | {r['dct_quality']:>7.4f} {r['hopfield_quality']:>7.4f} {r['delta']:>+7.4f}")

# Save
with open('/tmp/hopfield_density_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to /tmp/hopfield_density_results.json")
