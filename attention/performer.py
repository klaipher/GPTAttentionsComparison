import math
import torch
import torch.nn as nn


class PerformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = self.n_embd // self.n_head
        self.dropout = config.dropout

        # Number of random features for Performer
        self.n_features = config.attention_config.get('performer_features', 64) \
            if config.attention_config else 64

        # Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Random projection matrix for the feature map
        # Typically not trained, so we register it as a buffer
        # Shape: (head_size, n_features)
        # We'll scale it by 1 / (some_std) for stable exponent magnitudes
        proj = torch.randn(self.head_size, self.n_features) * 0.1
        self.register_buffer("proj", proj)

    def forward(self, x):
        B, T, C = x.size()

        # 1) Compute Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)  # each (B, T, C)

        # 2) Split heads: (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3) Map Q, K with the Performer random feature map
        #    We'll do a simple Gaussian random features approach:
        #    phi(x) = exp( x @ proj - ||x||^2/2 ) (with normalization)
        #    For numerical stability, we usually add an epsilon as well.
        q_prime = self._prime(q)  # (B, n_head, T, n_features)
        k_prime = self._prime(k)  # (B, n_head, T, n_features)

        # 4) Approximate "softmax(QK^T) V" with:
        #    out = (q_prime * (k_prime^T V)) / (q_prime * k_prime^T 1)
        #    We'll implement that in steps:

        # 4a) Compute K'^T V by merging the T dimension in K'^T
        #    k_prime: (B, n_head, T, n_features)
        #    v:       (B, n_head, T, head_size)
        # We want (B, n_head, n_features, head_size)
        # i.e. sum_{t} [k_prime(t) * v(t)]
        kv_ = torch.einsum('b h t f, b h t d -> b h f d', k_prime, v)  # (B, n_head, n_features, head_size)

        # 4b) Compute normalizing denominator = q_prime * sum_{t}(k_prime(t))
        #    which is  (B, n_head, T, n_features) x (B, n_head, n_features) -> (B, n_head, T)
        # But we first sum over T in k_prime => shape (B, n_head, n_features)
        k_sum = k_prime.sum(dim=2)  # (B, n_head, n_features)
        # Then multiply by q_prime => (B, n_head, T)
        denom = torch.einsum('b h t f, b h f -> b h t', q_prime, k_sum)  # (B, n_head, T)
        denom = 1.0 / (denom + 1e-6)  # avoid division by zero

        # 4c) Now compute the numerator: q_prime @ (k_prime^T V)
        # q_prime: (B, n_head, T, n_features)
        # kv_:     (B, n_head, n_features, head_size)
        out = torch.einsum('b h t f, b h f d -> b h t d', q_prime, kv_)  # (B, n_head, T, head_size)

        # 4d) Multiply by denom as broadcast: (B, n_head, T, head_size)
        out = out * denom.unsqueeze(-1)

        out = self.attn_dropout(out)

        # 5) Re-combine heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 6) Final linear + residual dropout
        out = self.resid_dropout(self.c_proj(out))
        return out

    def _prime(self, x):
        """
        Performer random feature map:
          phi(x) = exp( x @ proj - ||x||^2 / 2 )
        with shape transformations.

        x: (B, n_head, T, head_size)
        returns: (B, n_head, T, n_features)
        """
        # 1) squared norm
        # ||x||^2 along the last dim: shape (B, n_head, T, 1)
        norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)  # (B, n_head, T, 1)

        # 2) project x => (B, n_head, T, n_features)
        #    self.proj has shape (head_size, n_features)
        x_proj = torch.einsum('b h t d, d f -> b h t f', x, self.proj)

        # 3) exponent
        # We do: exp(x_proj - norm_sq/2)
        # This is the approximate kernel for softmax
        x_exp = torch.exp(x_proj - 0.5 * norm_sq)

        # 4) normalization factor: typically 1/sqrt(n_features), or we can do it later
        x_exp = x_exp * (1.0 / math.sqrt(self.n_features))
        return x_exp
