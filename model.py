import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class QuarkLlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 1152
    intermediate_size: int = 3456
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute sinusoidal frequencies for Rotary Position Embeddings (RoPE)."""
    assert dim % 2 == 0, f"Dimension {dim} must be even for RoPE"
    # shape: (dim / 2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs) # shape: (max_seq_len, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex form: e^(i * theta)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to Query and Key representations."""
    # xq: (bsz, seqlen, n_heads, head_dim)
    # freqs_cis: (seqlen, head_dim // 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis to (1, seqlen, 1, head_dim // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network as used in LLaMA / TinyLlama"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(x * W_gate) * (x * W_up)) * W_down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with Causal Masking"""
    def __init__(self, config: QuarkLlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Transpose for attention computation: (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Expand KV heads to match query heads (GQA)
        if self.num_kv_groups > 1:
            xk = xk.repeat_interleave(self.num_kv_groups, dim=1)
            xv = xv.repeat_interleave(self.num_kv_groups, dim=1)

        # PyTorch 2.0+ Scaled Dot Product Attention with Causal Masking (FlashAttention-backed)
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

        # Reshape back to (bsz, seqlen, hidden_size)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)

class QuarkLlamaBlock(nn.Module):
    """Single Transformer Decoder Layer"""
    def __init__(self, config: QuarkLlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUFFN(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        # Pre-LN Residual Attention
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis)
        # Pre-LN Residual MLP
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class QuarkLlamaForCausalLM(nn.Module):
    """Full QuarkLlama Autoregressive Language Model"""
    def __init__(self, config: QuarkLlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([QuarkLlamaBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies
        head_dim = config.hidden_size // config.num_attention_heads
        freqs_cis = precompute_rope_frequencies(head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights (Xavier / normal)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:seqlen].to(x.device)

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return logits, loss

if __name__ == "__main__":
    config = QuarkLlamaConfig()
    model = QuarkLlamaForCausalLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.3f} B)")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with dummy data
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    labels = input_ids.clone()
    logits, loss = model(input_ids, labels=labels)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss value: {loss.item():.4f}")
    print("Verification passed successfully!")
