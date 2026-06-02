import torch
import torch.nn as nn
import torch.nn.functional as F
from model.normalization import RMSNorm
from model.rope import precompute_freqs_cis
from model.transformer_block import TransformerBlock
from model.moe_layer import MoETransformerBlock

class MoETransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int = 32000, 
                 d_model: int = 768, 
                 n_layers: int = 8, 
                 n_heads: int = 8, 
                 d_ff: int = 2048, 
                 num_experts: int = 4, 
                 k: int = 2, 
                 max_seq_len: int = 1024, 
                 eps: float = 1e-6):
        """
        Decoder-only MoE Transformer model.
        
        Args:
            vocab_size (int): Size of BPE vocabulary (32000).
            d_model (int): Hidden dimension (768).
            n_layers (int): Number of layers (8).
            n_heads (int): Number of attention heads (8).
            d_ff (int): FFN/Expert intermediate dimension (2048).
            num_experts (int): Total experts in MoE layers (4).
            k (int): Top-K routing experts (2).
            max_seq_len (int): Context window length (1024).
            eps (float): Norm epsilon.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token Embeddings (untied from output projections)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        
        # Interleaved Layers: Even layers = MoE, Odd layers = Dense
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.layers.append(MoETransformerBlock(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                    num_experts=num_experts, k=k, max_seq_len=max_seq_len, eps=eps
                ))
            else:
                self.layers.append(TransformerBlock(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                    max_seq_len=max_seq_len, eps=eps
                ))
                
        # Final RMSNorm
        self.norm = RMSNorm(d_model, eps=eps)
        
        # LM Head (untied)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Precompute RoPE frequency cosines/sines table
        freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        """
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, seq_len).
            labels (torch.Tensor, optional): Token IDs of shape (batch_size, seq_len).
            
        Returns:
            logits (torch.Tensor): LM output logits of shape (batch_size, seq_len, vocab_size).
            loss (torch.Tensor, optional): Total loss (scalar), if labels are provided.
            main_loss (torch.Tensor, optional): Cross entropy loss (scalar).
            total_aux_loss (torch.Tensor, optional): Load balancing loss sum (scalar).
            total_z_loss (torch.Tensor, optional): Z-loss sum (scalar).
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        
        # Embed tokens
        x = self.token_embeddings(input_ids)
        
        # Slice precomputed RoPE frequencies
        freqs_cis = self.freqs_cis[:T]
        
        # Router loss accumulators
        total_aux_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)
        
        # Forward through layers
        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                x, aux_l, z_l = layer(x, freqs_cis)
                total_aux_loss = total_aux_loss + aux_l
                total_z_loss = total_z_loss + z_l
            else:
                x = layer(x, freqs_cis)
                
        # Apply final norm and project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        main_loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Cross entropy calculation (default ignore_index=-100)
            main_loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size), 
                shift_labels.view(-1), 
                ignore_index=-100
            )
            
            # total_loss = main_loss + 0.01 * aux_loss + 0.001 * z_loss
            loss = main_loss + 0.01 * total_aux_loss + 0.001 * total_z_loss
            
        return logits, loss, main_loss, total_aux_loss, total_z_loss
