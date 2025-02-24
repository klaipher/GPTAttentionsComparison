import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinformerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.dropout = config.dropout
        
        # Linformer projection dimension (k)
        self.projection_length = config.attention_config["linformer_k"]
        
        # Linformer projections
        self.E = nn.Parameter(torch.randn(self.n_head, self.projection_length, self.block_size))
        self.F = nn.Parameter(torch.randn(self.n_head, self.projection_length, self.block_size))
        
        # Initialize projections using scaled normal distribution
        nn.init.normal_(self.E, mean=0.0, std=0.02)
        nn.init.normal_(self.F, mean=0.0, std=0.02)
        
        # Create a causal mask for the projected space
        projected_mask = torch.tril(torch.ones(self.block_size, self.projection_length))
        self.register_buffer("proj_mask", projected_mask.view(1, 1, self.block_size, self.projection_length))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # Project keys and values using E and F
        k_projected = torch.matmul(self.E[:, :, :T], k) # (nh, k, T) x (B, nh, T, hs) -> (B, nh, k, hs)
        v_projected = torch.matmul(self.F[:, :, :T], v) # (nh, k, T) x (B, nh, T, hs) -> (B, nh, k, hs)
        
        # Compute attention scores
        att = (q @ k_projected.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal masking in the projected space
        causal_mask = self.proj_mask[:, :, :T, :self.projection_length]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to projected values
        y = att @ v_projected  # (B, nh, T, k) x (B, nh, k, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
