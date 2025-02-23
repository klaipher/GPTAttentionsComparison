import math

import torch
import torch.nn as nn


class LinformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.linformer_k = 128

        # Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Linformer projection matrices E, F to go from T -> k
        # shape (n_head, linformer_k, block_size)
        self.E = nn.Parameter(torch.randn(config.n_head, self.linformer_k, config.block_size))
        self.F = nn.Parameter(torch.randn(config.n_head, self.linformer_k, config.block_size))

        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        # (B, T, 3*n_embd) -> split -> each (B, T, n_embd)
        q, k, v = self.c_attn(x).split(C, dim=2)

        # shape them into heads
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, n_head, T, head_size)

        # Project K and V from T->k
        # E,F each is (n_head, linformer_k, block_size), slice to current T
        E = self.E[:, :, :T]  # => (n_head, linformer_k, T)
        F = self.F[:, :, :T]  # => (n_head, linformer_k, T)

        # Now do the batch multiplication:
        k_projected = torch.einsum('hkt, bhtd -> bhkd', E, k)  # (B, n_head, linformer_k, head_size)
        v_projected = torch.einsum('hkt, bhtd -> bhkd', F, v)  # (B, n_head, linformer_k, head_size)

        # Compute attention
        # q is (B, n_head, T, head_size)
        # k_projected is (B, n_head, linformer_k, head_size)
        # so q @ k_projected^T => (B, n_head, T, linformer_k)
        att = torch.matmul(
            q, k_projected.transpose(-2, -1)
        ) * (1.0 / math.sqrt(self.head_size))

        att = nn.functional.softmax(att, -1)
        att = self.attn_dropout(att)

        # Then multiply by v_projected => (B, n_head, T, head_size)
        y = att @ v_projected

        # re-combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final linear out
        y = self.resid_dropout(self.c_proj(y))
        return y 