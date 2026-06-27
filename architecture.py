import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =====================================================================
# 1. CORE ARCHITECTURE LAYERS
# =====================================================================

class RMSNorm(nn.Module):
    """Modern LLaMA-style Root Mean Square Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Compute-efficient SwiGLU FFN layer replacing traditional MLP."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =====================================================================
# 2. POSITIONAL EMBEDDINGS (RoPE)
# =====================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) as implemented in LLaMA."""
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculate theta frequencies: base^(-2(i-1)/dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cosine and sine tables across max context window
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)

        # Parallel embedding concatenation for [cos(x), cos(x)] and [sin(x), sin(x)] splits
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return sliced matrices fitting the current batch sequence length
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def _rotate_half_tensor(x: torch.Tensor) -> torch.Tensor:
    """Helper function to split and rotate dimensions for RoPE calculation."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE rotation to Q and K tensors."""
    # Shapes: [batch_size, num_heads, seq_len, head_dim]
    # Broadcast cos and sin across batch and head dimensions: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half_tensor(q) * sin)
    k_embed = (k * cos) + (_rotate_half_tensor(k) * sin)
    return q_embed, k_embed

# =====================================================================
# GQA
# =====================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) layer incorporating Rotary Embeddings.
    Optimizes KV Cache memory consumption by grouping Query heads around shared Key/Value heads.
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "Number of Q heads must be divisible by KV heads."

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = dim // num_heads

        # Projections matrices
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        # RoPE initialization for this head dimension size
        self.rope = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=max_seq_len)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats Key/Value states across grouped Query heads to allow matrix multiplication."""
        if n_rep == 1:
            return x
        bs, num_kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, num_kv_heads, n_rep, seq_len, head_dim)
            .reshape(bs, num_kv_heads * n_rep, seq_len, head_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.shape

        # 1. Project inputs into Q, K, V spaces
        q = self.q_proj(x) # [bs, seq_len, num_heads * head_dim]
        k = self.k_proj(x) # [bs, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(x) # [bs, seq_len, num_kv_heads * head_dim]

        # 2. Reshape for multi-head alignment
        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Positional Embeddings
        cos, sin = self.rope(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Expand K and V to match Q head grouping topology
        k = self._repeat_kv(k, self.num_queries_per_kv) # [bs, num_heads, seq_len, head_dim]
        v = self._repeat_kv(v, self.num_queries_per_kv) # [bs, num_heads, seq_len, head_dim]

        # 5. Scaled Dot-Product Attention with Modern SDPA Backing Manager
        # Replaces deprecated sdp_kernel manager to comply with modern PyTorch APIs
        from torch.nn.attention import sdpa_kernel, SDPBackend

        # Enforce highly accelerated hardware paths explicitly
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        with sdpa_kernel(backends):
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0 if self.training else 0.0,
                is_causal=True
            )

        # 6. Collapse heads and project out
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output)

# =====================================================================
# 3. TRAINING INFRASTRUCTURE TEMPLATE (For execution abstraction)
# =====================================================================
def run_training_step_template(model, data_loader, optimizer, scaler, grad_accum_steps=4):
    """
    Template demonstrating how the training execution loop uses these components.
    This will be completely integrated once our data loaders are built.
    """
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(grad_accum_steps):
        x, y = data_loader.get_batch()

        # Enforce highly efficient AMP training execution
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps

        # Backward pass tracking
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

class TransformerBlock(nn.Module):
    """Full modern LLaMA block isolating GQA and SwiGLU with pre-normalization."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, hidden_dim: int, max_seq_len: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, max_seq_len)

        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual paths to preserve gradient flow at depth
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LLaMABase(nn.Module):
    """
    Master LLaMA-style Language Model Core Container.
    Features padded vocab layout for Tensor Core alignment and GQA blocks.
    """
    def __init__(self, vocab_size: int, dim: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Vocab sizing padding rule to ensure multi-element Tensor Core alignment (multiple of 128)
        if vocab_size % 128 != 0:
            vocab_size = ((vocab_size // 128) + 1) * 128
        self.vocab_size = vocab_size

        # Token embedding matrix
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # Stacked transformer layers
        hidden_dim = int(2 * (4 * dim) / 3) # SwiGLU heuristic scaling
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, num_kv_heads, hidden_dim, max_seq_len)
            for _ in range(num_layers)
        ])

        # Final output optimization tracking
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Weight tie execution paths to save additional parameters if needed
        # self.output.weight = self.tok_embeddings.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _bs, seq_len = tokens.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max window {self.max_seq_len}"

        # 1. Look up token representations
        h = self.tok_embeddings(tokens)

        # 2. Sequential transmission through GQA blocks
        for layer in self.layers:
            h = layer(h)

        # 3. Output structural norm
        h = self.norm(h)

        # 4. Final projection layer to vocabulary logit distribution space
        logits = self.output(h)

        # 5. Native Loss Calculation branch built right into forward execution
        loss = None
        if targets is not None:
            # Flatten cross entropy evaluation windows for high throughput computation
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss
