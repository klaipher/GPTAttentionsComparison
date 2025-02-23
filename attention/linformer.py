import math
import torch
import torch.nn as nn


def gen_causal_mask(input_size, dim_k, full_attention=False):
    """
    Generates a causal mask of size (input_size, dim_k) for Linformer.
    When full_attention is True, returns an (input_size, input_size) mask.
    """
    if full_attention:
        return (torch.triu(torch.ones(input_size, input_size)) == 1).transpose(0, 1)
    return (torch.triu(torch.ones(dim_k, input_size)) == 1).transpose(0, 1)


class LinformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Ensure the embedding is divisible by number of heads
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        # linformer_k: reduced sequence length (e.g., 64) for keys/values projections
        self.linformer_k = getattr(config, "linformer_k", config.block_size // 4)
        self.causal = getattr(config, "causal", True)

        # Single linear layer to produce queries, keys, and values (as in Nanogpt)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropouts as in the vanilla version
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Learnable projection matrices for keys and values:
        # They map from the original sequence length (block_size) to the lower dimension (linformer_k)
        self.E = nn.Parameter(torch.randn(self.block_size, self.linformer_k))
        self.F = nn.Parameter(torch.randn(self.block_size, self.linformer_k))

        # Precompute causal mask if needed (mask shape: (block_size, linformer_k))
        if self.causal:
            self.register_buffer("causal_mask",
                                 gen_causal_mask(self.block_size, self.linformer_k, full_attention=False))

    def forward(self, x):
        """
        x: (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()

        # Compute queries, keys, values in one go and split them up
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape and transpose to get shape (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply Linformer projection: project along the sequence dimension.
        # k_proj: (B, n_head, linformer_k, head_dim)
        # v_proj: (B, n_head, linformer_k, head_dim)
        k_proj = torch.einsum('bhtd,tk->bhkd', k, self.E)
        v_proj = torch.einsum('bhtd,tk->bhkd', v, self.F)

        # Compute attention scores using the projected keys.
        # q: (B, n_head, T, head_dim)
        # k_proj.transpose(-2, -1): (B, n_head, head_dim, linformer_k)
        att = (q @ k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            # Use only the first T rows of the precomputed causal mask.
            # causal_mask shape: (T, linformer_k)
            mask = self.causal_mask[:T, :]
            att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        att = att.softmax(dim=-1)
        att = self.attn_dropout(att)

        # Multiply attention weights with projected values.
        # Resulting shape: (B, n_head, T, head_dim)
        y = att @ v_proj

        # Reassemble multi-head outputs.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
