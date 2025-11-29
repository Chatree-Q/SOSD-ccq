from __future__ import annotations
import numpy as np
import math
import torch.nn.functional as F
import os
import sys
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# --- 关键修改：引入你的 tokenizer ---
# 将当前文件的父目录（即项目根目录）加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import BPE_Tokenizer, train_bpe

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    output = F.linear(in_features, weights, bias=None)
    return output

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    output = F.embedding(token_ids, weights)
    return output

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    x = F.linear(in_features, w1_weight)  
    y = F.linear(in_features, w3_weight)  
    gated = F.silu(x) * y 
    output = F.linear(gated, w2_weight)
    return output

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, -1e9)
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.size(-2)
    d_in = in_features.size(-1)
    d_k = q_proj_weight.size(0) // num_heads 
    d_v = v_proj_weight.size(0) // num_heads
    
    Q = F.linear(in_features, q_proj_weight) 
    K = F.linear(in_features, k_proj_weight) 
    V = F.linear(in_features, v_proj_weight) 
    
    Q = Q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_v).transpose(-3, -2)
    
    attn_output = run_scaled_dot_product_attention(Q, K, V)
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, num_heads*d_v)
    output = F.linear(attn_output, o_proj_weight) 
    return output

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.size(-2)
    d_k = q_proj_weight.size(0) // num_heads
    d_v = v_proj_weight.size(0) // num_heads
    
    Q = F.linear(in_features, q_proj_weight)
    K = F.linear(in_features, k_proj_weight)
    V = F.linear(in_features, v_proj_weight)
    
    Q = Q.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2) 
    K = K.view(*batch_dims, seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_v).transpose(-3, -2)
    
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device).expand(*batch_dims, seq_len)
    
    Q_rope = Q.flatten(0, -3) 
    K_rope = K.flatten(0, -3)
    token_positions_rope = token_positions.unsqueeze(1).expand(*batch_dims, num_heads, seq_len).flatten(0, -2)
    Q = run_rope(d_k, theta, max_seq_len, Q_rope, token_positions_rope).view_as(Q)
    K = run_rope(d_k, theta, max_seq_len, K_rope, token_positions_rope).view_as(K)
    
    attn_output = run_scaled_dot_product_attention(Q, K, V)
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, num_heads*d_v)
    output = F.linear(attn_output, o_proj_weight)
    return output

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2)[: (d_k // 2)] / d_k))
    seq_len = in_query_or_key.size(-2)
    positions = token_positions.unsqueeze(-1) 
    freqs = freqs.unsqueeze(0) 
    pos_freqs = positions * freqs 
    cos = pos_freqs.cos()
    sin = pos_freqs.sin()
    qk_even = in_query_or_key[..., ::2] 
    qk_odd = in_query_or_key[..., 1::2] 
    rotated_even = qk_even * cos - qk_odd * sin
    rotated_odd = qk_even * sin + qk_odd * cos
    output = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
    return output

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    ln1_out = run_rmsnorm(d_model, eps=1e-5, weights=weights["ln1.weight"], in_features=in_features)
    attn_out = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=ln1_out,
    )
    residual1 = in_features + attn_out
    ln2_out = run_rmsnorm(d_model, eps=1e-5, weights=weights["ln2.weight"], in_features=residual1)
    ffn_out = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=ln2_out,
    )
    output = residual1 + ffn_out
    return output

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    token_embeds = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices,
    )
    x = token_embeds
    for layer_idx in range(num_layers):
        layer_weights = {
            "attn.q_proj.weight": weights[f"layers.{layer_idx}.attn.q_proj.weight"],
            "attn.k_proj.weight": weights[f"layers.{layer_idx}.attn.k_proj.weight"],
            "attn.v_proj.weight": weights[f"layers.{layer_idx}.attn.v_proj.weight"],
            "attn.output_proj.weight": weights[f"layers.{layer_idx}.attn.output_proj.weight"],
            "ln1.weight": weights[f"layers.{layer_idx}.ln1.weight"],
            "ffn.w1.weight": weights[f"layers.{layer_idx}.ffn.w1.weight"],
            "ffn.w2.weight": weights[f"layers.{layer_idx}.ffn.w2.weight"],
            "ffn.w3.weight": weights[f"layers.{layer_idx}.ffn.w3.weight"],
            "ln2.weight": weights[f"layers.{layer_idx}.ln2.weight"],
        }
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )
    ln_final_out = run_rmsnorm(d_model, eps=1e-5, weights=weights["ln_final.weight"], in_features=x)
    logits = F.linear(ln_final_out, weights["lm_head.weight"])
    return logits

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    output = in_features / rms * weights
    return output

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return F.silu(in_features)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.stack([torch.from_numpy(dataset[i:i+context_length]) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i+1:i+context_length+1]) for i in ix])
    return x.to(device), y.to(device)

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return F.softmax(in_features, dim=dim)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    loss = F.cross_entropy(inputs, targets)
    return loss

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)

def get_adamw_cls() -> Any:
    return torch.optim.AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * (it + 1) / warmup_iters
    if it > warmup_iters + cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / cosine_cycle_iters
    lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    return lr

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return BPE_Tokenizer(vocab, merges, special_tokens)

# --- 关键修改：调用 tokenizer.py 的 train_bpe ---
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer using the
    implementation in tokenizer.py"""
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 这里的 train_bpe 来自文件开头的 from tokenizer import ...
    vocab, merges = train_bpe(text, vocab_size, special_tokens)
    
    return vocab, merges
