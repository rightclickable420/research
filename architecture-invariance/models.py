"""
Architecture invariance models: Transformer, LSTM, MLP
All ~1M parameters, 6 layers, character-level language modeling.
Each exposes per-unit activations for entropy tracking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Transformer (simplified from nanoGPT)
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
        self.last_attn_weights = None

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.attn(h)
        q, k, v = qkv.split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self.last_attn_weights = att.detach()
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=6, n_embd=384,
                 block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout, block_size)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self.n_layer = n_layer
        self.n_head = n_head
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_attn_weights(self):
        """Return list of [batch, heads, seq, seq] attention weights per layer."""
        return [block.last_attn_weights for block in self.blocks]


# ============================================================================
# LSTM (6-layer with per-unit tracking)
# ============================================================================

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_embd=384, dropout=0.2, **kwargs):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Stack of LSTM layers (not using nn.LSTM to access per-layer gates)
        self.lstm_layers = nn.ModuleList([
            nn.LSTMCell(n_embd, n_embd) for _ in range(n_layer)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(n_embd) for _ in range(n_layer)
        ])
        self.layer_drops = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

        # Store hidden states for entropy tracking
        self.last_hidden_states = [None] * n_layer
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))

        # Process sequence step by step through stacked LSTMs
        all_hidden = [[] for _ in range(self.n_layer)]
        states = [(torch.zeros(B, self.n_embd, device=idx.device),
                    torch.zeros(B, self.n_embd, device=idx.device))
                   for _ in range(self.n_layer)]

        for t in range(T):
            inp = x[:, t, :]
            for l in range(self.n_layer):
                h, c = self.lstm_layers[l](inp, states[l])
                h = self.layer_norms[l](h)
                h = self.layer_drops[l](h)
                if l > 0:
                    h = h + inp  # residual connection
                states[l] = (h, c)
                all_hidden[l].append(h)
                inp = h

        # Stack hidden states for entropy tracking: [batch, seq, n_embd]
        for l in range(self.n_layer):
            self.last_hidden_states[l] = torch.stack(all_hidden[l], dim=1).detach()

        # Final output
        output = torch.stack(all_hidden[-1], dim=1)
        output = self.ln_f(output)
        logits = self.head(output)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_hidden_states(self):
        """Return list of [batch, seq, n_embd] hidden states per layer."""
        return self.last_hidden_states


# ============================================================================
# MLP (6-layer with per-neuron tracking)
# ============================================================================

class MLPLM(nn.Module):
    """
    6-layer MLP language model. Processes each position independently
    but uses a context window via flattened input.
    """
    def __init__(self, vocab_size, n_layer=6, n_embd=384, block_size=256,
                 dropout=0.2, context_window=16, **kwargs):
        super().__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.context_window = context_window
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # First layer takes context_window * n_embd and projects to n_embd
        self.input_proj = nn.Linear(context_window * n_embd, n_embd)

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            ))
            self.layer_norms.append(nn.LayerNorm(n_embd))

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Track activations per layer
        self.last_activations = [None] * n_layer
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)  # [B, T, n_embd]
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)

        # Create context windows: for each position, gather previous context_window tokens
        # Pad the beginning
        padded = F.pad(x, (0, 0, self.context_window - 1, 0))  # [B, T+ctx-1, n_embd]
        windows = padded.unfold(1, self.context_window, 1)  # [B, T, n_embd, ctx]
        windows = windows.permute(0, 1, 3, 2).reshape(B, T, -1)  # [B, T, ctx*n_embd]

        x = self.input_proj(windows)  # [B, T, n_embd]

        for i, (layer, ln) in enumerate(zip(self.layers, self.layer_norms)):
            residual = x
            x = ln(x)
            x = layer(x) + residual
            self.last_activations[i] = x.detach()

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_activations(self):
        """Return list of [batch, seq, n_embd] activations per layer."""
        return self.last_activations


def get_model(arch, vocab_size, **kwargs):
    """Factory function to create model by architecture name."""
    if arch == 'transformer':
        return TransformerLM(vocab_size, **kwargs)
    elif arch == 'lstm':
        return LSTMLM(vocab_size, **kwargs)
    elif arch == 'mlp':
        return MLPLM(vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
