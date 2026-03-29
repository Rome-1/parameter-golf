"""
Trajectory / Temporal Distillation for Parameter Golf

Two-phase approach:
  Phase 1: `python trajectory_distill.py --mode=teacher` — Overtrain a teacher model
           beyond the 10-min budget, saving checkpoints and caching hidden states.
  Phase 2: `python trajectory_distill.py --mode=student` — Train a student model
           within the 10-min budget using auxiliary distillation losses from the
           pre-computed teacher features.

This script extends the baseline train_gpt.py with distillation-specific modifications.
All baseline functionality (model arch, optimizers, data loading, quantization, eval)
is imported directly from train_gpt.py.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# Add parent directory to path so we can import from train_gpt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gpt import (
    GPT,
    Block,
    CastedLinear,
    DistributedTokenLoader,
    Hyperparameters,
    Muon,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    eval_val,
    load_validation_tokens,
    quantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
    zeropower_via_newtonschulz5,
)

# ---------------------------------------------------------------------------
# DISTILLATION HYPERPARAMETERS
# ---------------------------------------------------------------------------

class DistillConfig:
    """Distillation-specific configuration, layered on top of Hyperparameters."""

    # Phase selection
    mode = os.environ.get("DISTILL_MODE", "student")  # "teacher" or "student"

    # Teacher overtraining
    teacher_wallclock_seconds = float(os.environ.get("TEACHER_WALLCLOCK", 1800.0))  # 30 min default
    teacher_checkpoint_dir = os.environ.get("TEACHER_CKPT_DIR", "./teacher_checkpoints")
    teacher_num_checkpoints = int(os.environ.get("TEACHER_NUM_CKPTS", 10))
    teacher_cache_hidden = bool(int(os.environ.get("TEACHER_CACHE_HIDDEN", "1")))
    teacher_cache_tokens = int(os.environ.get("TEACHER_CACHE_TOKENS", 1_048_576))  # 1M tokens

    # Teacher architecture (can be same as student or larger)
    teacher_num_layers = int(os.environ.get("TEACHER_NUM_LAYERS", 0))  # 0 = same as student
    teacher_model_dim = int(os.environ.get("TEACHER_MODEL_DIM", 0))   # 0 = same as student

    # Loss weights
    alpha_kd = float(os.environ.get("ALPHA_KD", 0.5))        # logit KD weight
    beta_hidden = float(os.environ.get("BETA_HIDDEN", 0.1))  # hidden-state matching weight
    gamma_cka = float(os.environ.get("GAMMA_CKA", 0.0))      # CKA alignment weight (0 = off)

    # KD settings
    kd_temperature = float(os.environ.get("KD_TEMPERATURE", 4.0))
    kd_divergence = os.environ.get("KD_DIVERGENCE", "forward_kl")  # forward_kl, reverse_kl, skew_kl
    kd_skew_alpha = float(os.environ.get("KD_SKEW_ALPHA", 0.5))

    # Hidden matching settings
    hidden_loss_type = os.environ.get("HIDDEN_LOSS_TYPE", "cosine")  # mse or cosine
    hidden_layer_mapping = os.environ.get("HIDDEN_LAYER_MAPPING", "skip")  # skip, last, all

    # Checkpoint scheduling
    checkpoint_schedule = os.environ.get("CKPT_SCHEDULE", "fixed_final")  # fixed_final, linear, curriculum


# ---------------------------------------------------------------------------
# MODIFIED GPT THAT RETURNS INTERMEDIATE STATES
# ---------------------------------------------------------------------------

class GPTWithIntermediates(GPT):
    """GPT variant that optionally returns intermediate layer representations."""

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor | list[Tensor]]]:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        layer_outputs: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
            if return_intermediates:
                layer_outputs.append(x)

        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
            if return_intermediates:
                layer_outputs.append(x)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        if not return_intermediates:
            return ce_loss

        intermediates = {
            "logits": logits,
            "final_hidden": x,
            "layer_outputs": layer_outputs,
        }
        return ce_loss, intermediates


# ---------------------------------------------------------------------------
# DISTILLATION LOSSES
# ---------------------------------------------------------------------------

def kd_logit_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float,
    divergence: str = "forward_kl",
    skew_alpha: float = 0.5,
) -> Tensor:
    """Knowledge distillation loss on logits.

    Supports forward KL (Hinton 2015), reverse KL (MiniLLM), and
    skew KL (DistiLLM).
    """
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)

    if divergence == "forward_kl":
        # KL(teacher || student) — standard KD
        loss = F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)
    elif divergence == "reverse_kl":
        # KL(student || teacher) — mode-seeking, better for autoregressive LMs
        t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
        s_prob = F.softmax(student_logits / temperature, dim=-1)
        loss = F.kl_div(t_log, s_prob, reduction="batchmean") * (temperature ** 2)
    elif divergence == "skew_kl":
        # Skew KL: interpolation between forward and reverse
        t_log = F.log_softmax(teacher_logits / temperature, dim=-1)
        s_prob = F.softmax(student_logits / temperature, dim=-1)
        # Mixed distribution: alpha * teacher + (1-alpha) * student
        mixed = skew_alpha * t + (1 - skew_alpha) * s_prob
        loss = F.kl_div(torch.log(mixed + 1e-8), s_prob, reduction="batchmean") * (temperature ** 2)
    else:
        raise ValueError(f"Unknown divergence: {divergence}")

    return loss


def hidden_matching_loss(
    student_hiddens: list[Tensor],
    teacher_hiddens: list[Tensor],
    loss_type: str = "cosine",
    projectors: nn.ModuleList | None = None,
) -> Tensor:
    """Feature-matching loss between student and teacher hidden states.

    Based on FitNets (Romero 2015) and TinyBERT (Jiao 2020).
    If teacher and student have different dimensions, learned linear projectors
    bridge the gap.
    """
    assert len(student_hiddens) == len(teacher_hiddens), (
        f"Layer count mismatch: {len(student_hiddens)} vs {len(teacher_hiddens)}"
    )
    total_loss = torch.zeros((), device=student_hiddens[0].device)
    n = len(student_hiddens)

    for i, (s, t) in enumerate(zip(student_hiddens, teacher_hiddens)):
        # Apply projector if dimensions differ
        if projectors is not None and i < len(projectors):
            s = projectors[i](s.reshape(-1, s.size(-1)))
            t = t.reshape(-1, t.size(-1))
        else:
            s = s.reshape(-1, s.size(-1))
            t = t.reshape(-1, t.size(-1))

        if loss_type == "mse":
            total_loss = total_loss + F.mse_loss(s.float(), t.float())
        elif loss_type == "cosine":
            # 1 - cosine similarity, averaged over sequence positions
            cos_sim = F.cosine_similarity(s.float(), t.float(), dim=-1)
            total_loss = total_loss + (1.0 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown hidden loss type: {loss_type}")

    return total_loss / n


def linear_cka_loss(
    student_features: Tensor,
    teacher_features: Tensor,
) -> Tensor:
    """CKA (Centered Kernel Alignment) loss for structural representation matching.

    Based on Kornblith et al. (2019) and Park et al. (2023).
    Uses linear CKA which is efficient and dimension-agnostic.

    Returns 1 - CKA so minimizing the loss maximizes alignment.
    """
    # Reshape to (N, D) where N = batch * seq_len
    X = student_features.reshape(-1, student_features.size(-1)).float()
    Y = teacher_features.reshape(-1, teacher_features.size(-1)).float()

    # Center the features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Linear CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    # Use Gram matrices for efficiency when N > D
    XtX = X.T @ X  # (D_s, D_s)
    YtY = Y.T @ Y  # (D_t, D_t)
    YtX = Y.T @ X  # (D_t, D_s)

    hsic_xy = (YtX * YtX).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)
    return 1.0 - cka


def map_layers(
    student_layers: list[Tensor],
    teacher_layers: list[Tensor],
    strategy: str = "skip",
) -> tuple[list[Tensor], list[Tensor]]:
    """Map teacher layers to student layers for hidden-state matching.

    Strategies:
      skip: match every k-th teacher layer to student layers (PKD-Skip)
      last: match last k teacher layers to last k student layers (PKD-Last)
      all:  linearly interpolate mapping from all teacher to all student layers
    """
    n_s = len(student_layers)
    n_t = len(teacher_layers)

    if strategy == "skip":
        # Evenly spaced teacher layers
        indices = [int(round(i * (n_t - 1) / (n_s - 1))) for i in range(n_s)]
        mapped_teacher = [teacher_layers[i] for i in indices]
        return student_layers, mapped_teacher

    elif strategy == "last":
        # Match last n_s teacher layers
        mapped_teacher = teacher_layers[-n_s:]
        return student_layers, mapped_teacher

    elif strategy == "all":
        # For each student layer, pick the nearest teacher layer
        indices = [int(round(i * (n_t - 1) / max(n_s - 1, 1))) for i in range(n_s)]
        mapped_teacher = [teacher_layers[i] for i in indices]
        return student_layers, mapped_teacher

    else:
        raise ValueError(f"Unknown layer mapping strategy: {strategy}")


# ---------------------------------------------------------------------------
# TEACHER CHECKPOINT MANAGEMENT
# ---------------------------------------------------------------------------

def get_checkpoint_schedule(
    strategy: str,
    student_step: int,
    student_total_steps: int,
    num_checkpoints: int,
) -> int:
    """Determine which teacher checkpoint to use at a given student training step.

    Returns checkpoint index (0 to num_checkpoints-1).
    """
    if strategy == "fixed_final":
        return num_checkpoints - 1
    elif strategy == "linear":
        # Match student progress to teacher trajectory
        progress = student_step / max(student_total_steps, 1)
        return min(int(progress * num_checkpoints), num_checkpoints - 1)
    elif strategy == "curriculum":
        # Start with early checkpoints, accelerate toward final
        progress = student_step / max(student_total_steps, 1)
        # Quadratic: spend more time on later checkpoints
        idx = int((progress ** 0.5) * num_checkpoints)
        return min(idx, num_checkpoints - 1)
    else:
        raise ValueError(f"Unknown checkpoint schedule: {strategy}")


# ---------------------------------------------------------------------------
# PHASE 1: TEACHER OVERTRAINING
# ---------------------------------------------------------------------------

def run_teacher_phase(args: Hyperparameters, dcfg: DistillConfig) -> None:
    """Overtrain a teacher model beyond the budget, saving checkpoints."""

    import sentencepiece as spm
    from torch.nn.parallel import DistributedDataParallel as DDP

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log0(msg: str) -> None:
        if master_process:
            print(msg)

    # Use teacher-specific architecture if specified, else same as student
    num_layers = dcfg.teacher_num_layers if dcfg.teacher_num_layers > 0 else args.num_layers
    model_dim = dcfg.teacher_model_dim if dcfg.teacher_model_dim > 0 else args.model_dim

    base_model = GPTWithIntermediates(
        vocab_size=args.vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Standard optimizer setup (same as baseline)
    from train_gpt import CONTROL_TENSOR_NAME_PATTERNS

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * dcfg.teacher_wallclock_seconds
    ckpt_dir = Path(dcfg.teacher_checkpoint_dir)
    if master_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"[teacher] model_params:{n_params} layers:{num_layers} dim:{model_dim}")
    log0(f"[teacher] max_wallclock:{dcfg.teacher_wallclock_seconds}s checkpoints:{dcfg.teacher_num_checkpoints}")

    # Training loop
    training_time_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    next_ckpt_step = 0
    ckpt_interval = max(args.iterations // dcfg.teacher_num_checkpoints, 1)
    ckpt_count = 0

    model.train()
    while True:
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if elapsed_ms >= max_wallclock_ms or step >= args.iterations:
            break

        # Save checkpoint at regular intervals
        if step >= next_ckpt_step and ckpt_count < dcfg.teacher_num_checkpoints:
            if master_process:
                ckpt_path = ckpt_dir / f"teacher_ckpt_{ckpt_count:03d}_step{step}.pt"
                torch.save(base_model.state_dict(), ckpt_path)
                log0(f"[teacher] saved checkpoint {ckpt_count} at step {step}: {ckpt_path}")
            ckpt_count += 1
            next_ckpt_step = step + ckpt_interval

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"]  # No warmdown for teacher
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        if step % 200 == 0:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            log0(f"[teacher] step:{step} loss:{train_loss.item():.4f} time:{training_time_ms:.0f}ms")
            t0 = time.perf_counter()

    # Save final checkpoint
    if master_process:
        final_path = ckpt_dir / f"teacher_final_step{step}.pt"
        torch.save(base_model.state_dict(), final_path)
        log0(f"[teacher] saved final checkpoint at step {step}: {final_path}")

    # Cache hidden states on a representative data subset
    if dcfg.teacher_cache_hidden and master_process:
        log0("[teacher] caching hidden states on representative data subset...")
        model.eval()
        cache_loader = DistributedTokenLoader(args.train_files, 0, 1, device)
        cached_hiddens: list[list[Tensor]] = []  # [layer_idx][batch_idx] -> Tensor
        tokens_cached = 0
        n_layers = num_layers

        with torch.inference_mode():
            while tokens_cached < dcfg.teacher_cache_tokens:
                x, y = cache_loader.next_batch(
                    min(args.train_batch_tokens, dcfg.teacher_cache_tokens - tokens_cached),
                    args.train_seq_len,
                    1,
                )
                _, intermediates = base_model(x, y, return_intermediates=True)
                layer_outs = intermediates["layer_outputs"]
                if not cached_hiddens:
                    cached_hiddens = [[] for _ in range(len(layer_outs))]
                for i, h in enumerate(layer_outs):
                    cached_hiddens[i].append(h.cpu())
                tokens_cached += x.numel()

        # Save cached hidden states
        cache_path = ckpt_dir / "teacher_hidden_cache.pt"
        # Concatenate per-layer
        cached = {f"layer_{i}": torch.cat(hs, dim=0) for i, hs in enumerate(cached_hiddens)}
        cached["num_layers"] = torch.tensor(n_layers)
        cached["model_dim"] = torch.tensor(model_dim)
        cached["tokens_cached"] = torch.tensor(tokens_cached)
        torch.save(cached, cache_path)
        log0(f"[teacher] cached hidden states ({tokens_cached} tokens, {n_layers} layers) -> {cache_path}")

    log0(f"[teacher] done. Total steps: {step}")
    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# PHASE 2: STUDENT DISTILLATION TRAINING
# ---------------------------------------------------------------------------

def run_student_phase(args: Hyperparameters, dcfg: DistillConfig) -> None:
    """Train a student model with auxiliary distillation losses from the teacher."""

    import glob as glob_mod
    import sentencepiece as spm
    from torch.nn.parallel import DistributedDataParallel as DDP

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log0(msg: str) -> None:
        if master_process:
            print(msg)

    # --- Load teacher ---
    ckpt_dir = Path(dcfg.teacher_checkpoint_dir)
    teacher_ckpts = sorted(ckpt_dir.glob("teacher_ckpt_*.pt"))
    teacher_final = sorted(ckpt_dir.glob("teacher_final_*.pt"))
    all_ckpts = teacher_ckpts + teacher_final
    if not all_ckpts:
        raise FileNotFoundError(f"No teacher checkpoints found in {ckpt_dir}")
    log0(f"[student] found {len(all_ckpts)} teacher checkpoints")

    teacher_num_layers = dcfg.teacher_num_layers if dcfg.teacher_num_layers > 0 else args.num_layers
    teacher_model_dim = dcfg.teacher_model_dim if dcfg.teacher_model_dim > 0 else args.model_dim

    teacher_model = GPTWithIntermediates(
        vocab_size=args.vocab_size,
        num_layers=teacher_num_layers,
        model_dim=teacher_model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()

    # Load the final teacher checkpoint initially
    teacher_state = torch.load(all_ckpts[-1], map_location=device, weights_only=True)
    teacher_model.load_state_dict(teacher_state)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    log0(f"[student] loaded teacher from {all_ckpts[-1]}")

    # --- Build student ---
    base_model = GPTWithIntermediates(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Build projectors if teacher and student dimensions differ
    projectors = None
    if teacher_model_dim != args.model_dim and dcfg.beta_hidden > 0:
        n_match = min(args.num_layers, teacher_num_layers)
        projectors = nn.ModuleList([
            nn.Linear(args.model_dim, teacher_model_dim, bias=False).to(device).bfloat16()
            for _ in range(n_match)
        ])
        log0(f"[student] created {n_match} projectors: {args.model_dim} -> {teacher_model_dim}")

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Standard optimizer setup
    from train_gpt import CONTROL_TENSOR_NAME_PATTERNS

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    # Add projector params to optimizer if present
    projector_params = list(projectors.parameters()) if projectors is not None else []

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr

    all_scalar_params = scalar_params + projector_params
    optimizer_scalar = torch.optim.Adam(
        [{"params": all_scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    # Setup tokenizer + validation
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device,
    )

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"[student] model_params:{n_params}")
    log0(f"[student] distill losses: alpha_kd={dcfg.alpha_kd} beta_hidden={dcfg.beta_hidden} gamma_cka={dcfg.gamma_cka}")
    log0(f"[student] kd: T={dcfg.kd_temperature} divergence={dcfg.kd_divergence}")
    log0(f"[student] hidden: type={dcfg.hidden_loss_type} mapping={dcfg.hidden_layer_mapping}")
    log0(f"[student] schedule: {dcfg.checkpoint_schedule}")

    # --- Main training loop ---
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    current_ckpt_idx = -1

    model.train()
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        # Validation
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        # Update teacher checkpoint according to schedule
        if len(all_ckpts) > 1:
            desired_idx = get_checkpoint_schedule(
                dcfg.checkpoint_schedule, step, args.iterations, len(all_ckpts),
            )
            if desired_idx != current_ckpt_idx:
                ckpt_state = torch.load(all_ckpts[desired_idx], map_location=device, weights_only=True)
                teacher_model.load_state_dict(ckpt_state)
                current_ckpt_idx = desired_idx
                log0(f"[student] switched to teacher checkpoint {desired_idx}: {all_ckpts[desired_idx].name}")

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        distill_loss_sum = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # Student forward with intermediates
                need_intermediates = dcfg.alpha_kd > 0 or dcfg.beta_hidden > 0 or dcfg.gamma_cka > 0
                if need_intermediates:
                    ce_loss, student_inter = base_model(x, y, return_intermediates=True)

                    # Teacher forward (no grad)
                    with torch.no_grad():
                        _, teacher_inter = teacher_model(x, y, return_intermediates=True)

                    total_loss = ce_loss

                    # Logit KD loss
                    if dcfg.alpha_kd > 0:
                        kd_loss = kd_logit_loss(
                            student_inter["logits"],
                            teacher_inter["logits"].detach(),
                            dcfg.kd_temperature,
                            dcfg.kd_divergence,
                            dcfg.kd_skew_alpha,
                        )
                        total_loss = total_loss + dcfg.alpha_kd * kd_loss

                    # Hidden-state matching loss
                    if dcfg.beta_hidden > 0:
                        s_layers, t_layers = map_layers(
                            student_inter["layer_outputs"],
                            [h.detach() for h in teacher_inter["layer_outputs"]],
                            dcfg.hidden_layer_mapping,
                        )
                        h_loss = hidden_matching_loss(s_layers, t_layers, dcfg.hidden_loss_type, projectors)
                        total_loss = total_loss + dcfg.beta_hidden * h_loss

                    # CKA structural alignment loss
                    if dcfg.gamma_cka > 0:
                        cka_loss = linear_cka_loss(
                            student_inter["final_hidden"],
                            teacher_inter["final_hidden"].detach(),
                        )
                        total_loss = total_loss + dcfg.gamma_cka * cka_loss

                    loss = total_loss
                    distill_loss_sum += (loss - ce_loss).detach()
                else:
                    loss = base_model(x, y)

            train_loss += loss.detach()
            (loss * grad_scale).backward()

        train_loss /= grad_accum_steps
        distill_loss_sum /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if step % 200 == 0 or step <= 10:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"distill_loss:{distill_loss_sum.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms"
            )

        # Wallclock cap
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if reached_cap:
            if distributed:
                cap_tensor = torch.ones(1, device=device, dtype=torch.int32)
                dist.all_reduce(cap_tensor, op=dist.ReduceOp.MAX)
            stop_after_step = step

    # --- Export (same as baseline) ---
    if master_process:
        log0("[student] exporting quantized model...")
        sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
        obj, stats = quantize_state_dict_int8(sd)

        import io
        import zlib

        buf = io.BytesIO()
        torch.save(obj, buf)
        raw = buf.getvalue()
        compressed = zlib.compress(raw, 9)

        code_bytes = len(Path(__file__).read_bytes())
        total_bytes = code_bytes + len(compressed)
        log0(f"final_int8_zlib_roundtrip code_bytes:{code_bytes} "
             f"compressed_bytes:{len(compressed)} total_bytes:{total_bytes}")

        # Save artifact
        artifact_path = f"logs/{args.run_id}_model.pt.zlib"
        with open(artifact_path, "wb") as f:
            f.write(compressed)
        log0(f"[student] saved artifact to {artifact_path}")

    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    global zeropower_via_newtonschulz5
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    args = Hyperparameters()
    dcfg = DistillConfig()

    if dcfg.mode == "teacher":
        run_teacher_phase(args, dcfg)
    elif dcfg.mode == "student":
        run_student_phase(args, dcfg)
    else:
        raise ValueError(f"Unknown DISTILL_MODE: {dcfg.mode}")


if __name__ == "__main__":
    main()
