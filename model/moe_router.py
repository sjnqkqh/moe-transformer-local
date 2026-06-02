import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        """
        Top-K Router for Mixture of Experts (MoE).
        
        Args:
            d_model (int): Model embedding dimension (768).
            num_experts (int): Total number of experts (4).
            k (int): Number of experts to route each token to (2).
        """
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        # Gating projection layer (no bias is standard to avoid shifting expert preferences)
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Flattened token representations of shape (S, d_model),
                              where S = batch_size * seq_len.
                              
        Returns:
            top_k_probs (torch.Tensor): Gating probabilities for top-k experts of shape (S, k).
            top_k_indices (torch.Tensor): Expert indices for top-k experts of shape (S, k).
            router_logits (torch.Tensor): Raw routing logits of shape (S, num_experts).
        """
        # Compute raw logits: (S, num_experts)
        router_logits = self.gate(x)
        
        # Softmax to get probability distribution over experts: (S, num_experts)
        probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts and their probabilities
        top_k_probs, top_k_indices = torch.topk(probs, k=self.k, dim=-1)
        
        # Re-normalize top-k probabilities to sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices, router_logits

def load_balancing_loss(router_logits: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
    """
    Computes the load balancing auxiliary loss (Shazeer et al., 2017).
    
    Args:
        router_logits (torch.Tensor): Raw routing logits of shape (S, num_experts).
        top_k_indices (torch.Tensor): Expert indices of top-k experts, shape (S, k).
        
    Returns:
        torch.Tensor: Auxiliary loss (scalar).
    """
    S, E = router_logits.shape
    probs = F.softmax(router_logits, dim=-1)
    
    # Binary mask indicating if expert e is selected in top-k
    # Shape: (S, E)
    mask = torch.zeros_like(probs)
    mask.scatter_(1, top_k_indices, 1.0)
    
    # f: fraction of tokens sent to expert e (mean across tokens)
    # P: average probability allocated to expert e (mean across tokens)
    f = mask.mean(dim=0)
    P = probs.mean(dim=0)
    
    # Perfectly balanced: f_e = k/E, P_e = 1/E.
    # Total loss is E * sum(f_e * P_e). For k=2, E=4, balanced loss is 2.0.
    aux_loss = E * torch.sum(f * P)
    return aux_loss

def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the router z-loss to prevent logit explosion (ST-MoE / DeepSeek style).
    
    Args:
        router_logits (torch.Tensor): Raw routing logits of shape (S, num_experts).
        
    Returns:
        torch.Tensor: Z-loss (scalar).
    """
    # z_loss = mean( log(sum(exp(x)))^2 )
    log_z = torch.logsumexp(router_logits, dim=-1)
    return torch.mean(log_z ** 2)
