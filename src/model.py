"""
Mixture of Experts (MoE) Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert network"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    """Routes tokens to experts"""
    def __init__(self, dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
    
    def forward(self, x):
        return self.gate(x)


class MoELayer(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, dim, hidden_dim, num_experts=4, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        self.router = Router(dim, num_experts)
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Get routing weights
        router_logits = self.router(x)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Process through experts
        output = torch.zeros_like(x)
        
        for i in range(self.num_experts):
            # Find tokens for this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            expert_input = x[expert_mask]
            
            if expert_input.shape[0] > 0:
                # Apply capacity limit
                capacity = int(self.capacity_factor * seq_len * batch_size / self.num_experts)
                if expert_input.shape[0] > capacity:
                    expert_input = expert_input[:capacity]
                    expert_mask_limited = expert_mask.clone()
                    expert_mask_limited[expert_mask.nonzero(as_tuple=True)[0][capacity:]] = False
                    expert_mask = expert_mask_limited
                
                # Process through expert
                expert_output = self.experts[i](expert_input)
                
                # Apply gating and accumulate
                output[expert_mask] += expert_output
        
        # Calculate load balancing loss
        tokens_per_expert = torch.histc(
            top_k_indices.float(), 
            bins=self.num_experts, 
            min=0, 
            max=self.num_experts-1
        )
        load_balancing_loss = tokens_per_expert.std() / tokens_per_expert.mean()
        
        return output, load_balancing_loss


class MixtureOfExperts(nn.Module):
    """Complete MoE model"""
    def __init__(self, vocab_size, dim=256, hidden_dim=1024, num_experts=4, 
                 num_layers=2, top_k=2, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.top_k = top_k
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        
        # MoE layers
        self.layers = nn.ModuleList([
            MoELayer(dim, hidden_dim, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        
        # Output
        self.output = nn.Linear(dim, vocab_size)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Process through MoE layers
        total_aux_loss = 0
        for layer, ln in zip(self.layers, self.layer_norms):
            residual = x
            x, aux_loss = layer(x)
            x = ln(x + residual)
            total_aux_loss += aux_loss
        
        # Output
        logits = self.output(x)
        
        return logits, total_aux_loss / self.num_layers