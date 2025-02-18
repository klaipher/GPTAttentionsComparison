import torch
import torch.nn as nn
from torch.nn import functional as F

class NystromAttention(nn.Module):
    """
    Linformer self-attention mechanism with linear complexity.
    Projects keys and values to a lower dimensional space for efficiency.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Default Linformer config
        self.n_landmarks = config.attention_config.get('nystrom_landmarks', 32) if config.attention_config else 32

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        params = {'B': B, 'nh': self.n_head, 'T': T, 'hs': C // self.n_head}
        
        # Project keys and values to lower dimensional space
        q_landmarks = self.__get_landmark_representation(q, self.n_landmarks, **params)
        k_landmarks = self.__get_landmark_representation(k, self.n_landmarks, **params)
        
        # Compute the attention matrix
        L = F.softmax(torch.matmul(q, k_landmarks.transpose(-1, -2)), dim=-1)
        P = self.__iterative_inv(F.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1))
        N = F.softmax(torch.matmul(q_landmarks, k.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim=-1)
        
        # Compute attention scores
        att = L @ P @ N
        att = self.attn_dropout(att)

        # Apply attention to values and reshape
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    def __get_landmark_representation(self, tensor, num_landmarks, B, nh, T, hs):
        tensor_reshaped = tensor.reshape(B, nh, num_landmarks, T // num_landmarks, hs).transpose(1, 2) # (B, nh, T, hs)
        tensor_landmarks = tensor_reshaped.mean(dim=-2)
        return tensor_landmarks

    def __iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        V = 1 / torch.max(torch.sum(K, dim=-2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V