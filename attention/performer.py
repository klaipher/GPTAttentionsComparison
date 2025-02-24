import math
import torch
import torch.nn as nn


class CausalPerformerAttention(nn.Module):
    """
    Causal Performer attention (FAVOR+) for GPT-style models.
    Uses random feature maps + prefix sums to enforce autoregressive masking.

    NOTE:
    - This is a *vectorized* implementation. For long T,
      memory consumption can be high (we store prefix sums of shape ~ (B, n_head, T, ...)).
    - If you need incremental generation, you would maintain prefix sums
      in a stateful manner instead of computing them for the entire sequence at once.
    """

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

        # Q, K, V projections (linear)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Create a random projection matrix for the feature map
        proj = torch.randn(self.head_size, self.n_features) * 0.1
        self.register_buffer("proj", proj)

    def forward(self, x):
        """
        x: (B, T, C), where C = n_embd
        returns: (B, T, C)
        """
        B, T, C = x.size()

        # Compute Q, K, V in one fused linear op
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2) 
        
        # Reshape into heads: (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Map K, Q via the Performer random feature map: phi(k), phi(q)
        k_prime = self._prime(k)  # (B, n_head, T, n_features)
        q_prime = self._prime(q)  # (B, n_head, T, n_features)

        # Compute prefix sums for enforcing causality.
        k_prime_expanded = k_prime.unsqueeze(-1)
        v_expanded = v.unsqueeze(-2)
        # Compute k_prime * v => shape (B, n_head, T, n_features, head_size)
        kprime_v = k_prime_expanded * v_expanded 

        # prefix sums along T
        prefix_k = torch.cumsum(k_prime, dim=2)
        prefix_kprime_v = torch.cumsum(kprime_v, dim=2)

        numerator = torch.einsum(
            'b n t f, b n t f d -> b n t d', 
            q_prime,
            prefix_kprime_v
        )
        # denominator shape => (B, n_head, T)
        denominator = torch.einsum(
            'b n t f, b n t f -> b n t',  
            q_prime,
            prefix_k
        ) + 1e-6  # avoid division by zero

        out = numerator / denominator.unsqueeze(-1)  # broadcast over 'd'

        # Dropout on the attention output
        out = self.attn_dropout(out)

        # Re-combine the heads: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection + residual dropout
        out = self.resid_dropout(self.c_proj(out))
        return out

    def _prime(self, x):
        """
        Performer random feature map:
           phi(x) = exp(x * W - ||x||^2 / 2) / sqrt(n_features)
        where W is self.proj (shape [head_size, n_features]).

        x: shape (B, n_head, T, head_size)
        returns: (B, n_head, T, n_features)
        """
        # squared norm of x => (B, n_head, T, 1)
        norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)  # ||x||^2

        # x_proj => (B, n_head, T, n_features)
        x_proj = torch.einsum('b n t d, d f -> b n t f', x, self.proj)

        # exponent => exp(x_proj - norm_sq/2)
        x_exp = torch.exp(x_proj - 0.5 * norm_sq)

        # scale by 1 / sqrt(n_features)
        x_exp = x_exp * (1.0 / math.sqrt(self.n_features))
        return x_exp
