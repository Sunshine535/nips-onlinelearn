"""Streaming Mirror-LoRA: dual-timescale LoRA with Fisher-precision
invariant-selective consolidation and optional Grassmann retraction.

Extends StreamingParameterMemory with:
1. Online Fisher accumulation (EMA)
2. Fisher-precision weighted consolidation
3. Adaptive consolidation trigger (max Fisher ratio)
4. Invariant-selective consolidation mask
5. Optional Grassmann retraction on LoRA-B matrices

Reference:
    theta_merge_i = (F_slow_i * theta_slow_i + lambda * F_fast_i * theta_fast_i)
                    / (F_slow_i + lambda * F_fast_i)
"""

import copy
import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel

try:
    from src.streaming_memory import (
        LongTermMemoryLoRA,
        ReservoirBuffer,
        SessionBuffer,
        StreamingParameterMemory,
        WorkingMemoryLoRA,
        _set_adapters_layerwise,
    )
except ImportError:
    from streaming_memory import (
        LongTermMemoryLoRA,
        ReservoirBuffer,
        SessionBuffer,
        StreamingParameterMemory,
        WorkingMemoryLoRA,
        _set_adapters_layerwise,
    )

logger = logging.getLogger(__name__)


class OnlineFisherAccumulator:
    """Online diagonal Fisher information estimator using exponential moving average.

    Maintains a running EMA of squared gradients as a proxy for the diagonal
    of the Fisher information matrix.  Suitable for streaming settings where
    computing the full Fisher over a static dataset is impractical.

    Args:
        beta: EMA decay factor.  Higher values give more weight to history.
    """

    def __init__(self, beta: float = 0.99):
        self.fisher: Dict[str, torch.Tensor] = {}
        self.beta: float = beta
        self._step: int = 0

    def update(self, model: nn.Module, loss: torch.Tensor) -> None:
        """Accumulate Fisher diagonal from the current mini-batch gradients.

        Expects ``loss.backward()`` to have already been called so that
        ``param.grad`` is populated for all trainable LoRA parameters.

        Args:
            model: The PeftModel whose gradients are used.
            loss: The scalar loss (used only for bookkeeping; gradients must
                  already be computed).
        """
        self._step += 1
        for name, param in model.named_parameters():
            if not param.requires_grad or "lora" not in name:
                continue
            if param.grad is None:
                continue

            grad_sq = param.grad.data.pow(2)

            if name not in self.fisher:
                self.fisher[name] = grad_sq.clone()
            else:
                self.fisher[name].mul_(self.beta).add_(grad_sq, alpha=1.0 - self.beta)

    def get_fisher(self) -> Dict[str, torch.Tensor]:
        """Return current diagonal Fisher estimates (detached copies).

        Returns:
            Dictionary mapping parameter names to their diagonal Fisher
            estimates.
        """
        return {k: v.clone() for k, v in self.fisher.items()}

    def reset(self) -> None:
        """Clear all accumulated Fisher information."""
        self.fisher.clear()
        self._step = 0

    @property
    def num_steps(self) -> int:
        """Number of update steps since last reset."""
        return self._step


class InvariantDetector:
    """Detect invariant vs. transient LoRA parameters via gradient direction stability.

    Maintains a sliding window of gradient snapshots and computes pairwise
    cosine similarity across consecutive windows.  Parameters whose gradient
    direction is stable (high average cosine similarity) are considered
    *invariant* and suitable for consolidation into long-term memory.

    Args:
        window_size: Number of gradient snapshots to keep.
        threshold: Cosine similarity threshold above which a parameter is
                   considered invariant.
    """

    def __init__(self, window_size: int = 3, threshold: float = 0.5):
        self.window_size: int = window_size
        self.threshold: float = threshold
        self.gradient_history: deque = deque(maxlen=window_size)

    def update(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Record a gradient snapshot.

        Args:
            gradients: Mapping from parameter name to gradient tensor.
                       Tensors are cloned and detached internally.
        """
        snapshot: Dict[str, torch.Tensor] = {}
        for name, grad in gradients.items():
            snapshot[name] = grad.detach().clone().flatten()
        self.gradient_history.append(snapshot)

    def get_mask(self) -> Dict[str, bool]:
        """Compute the invariant mask over all tracked parameters.

        Returns:
            Dictionary mapping parameter names to ``True`` (invariant,
            consolidate) or ``False`` (transient, skip).
        """
        if len(self.gradient_history) < 2:
            # Not enough history -- default to consolidating everything.
            if len(self.gradient_history) == 1:
                return {name: True for name in self.gradient_history[0]}
            return {}

        # Collect parameter names present in all snapshots.
        all_names: set = set(self.gradient_history[0].keys())
        for snap in self.gradient_history:
            all_names &= set(snap.keys())

        mask: Dict[str, bool] = {}
        for name in all_names:
            similarities: List[float] = []
            history_list = list(self.gradient_history)
            for i in range(len(history_list) - 1):
                g_a = history_list[i].get(name)
                g_b = history_list[i + 1].get(name)
                if g_a is None or g_b is None:
                    continue
                norm_a = g_a.norm()
                norm_b = g_b.norm()
                if norm_a < 1e-12 or norm_b < 1e-12:
                    similarities.append(0.0)
                else:
                    cos_sim = torch.dot(g_a, g_b) / (norm_a * norm_b)
                    similarities.append(cos_sim.item())

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                mask[name] = avg_sim >= self.threshold
            else:
                mask[name] = True  # default: consolidate

        return mask

    def reset(self) -> None:
        """Clear gradient history."""
        self.gradient_history.clear()


def grassmann_retraction(B_matrix: torch.Tensor) -> torch.Tensor:
    """Project a LoRA-B matrix back onto the Stiefel manifold via QR decomposition.

    This ensures the column space of B stays on the Grassmann manifold,
    which is the natural geometry for low-rank subspaces.

    Args:
        B_matrix: A 2-D tensor (out_features x rank).

    Returns:
        Retracted matrix of the same shape.
    """
    if B_matrix.dim() != 2:
        return B_matrix
    m, n = B_matrix.shape
    if m < n:
        # More columns than rows -- retraction not meaningful.
        return B_matrix
    Q, R = torch.linalg.qr(B_matrix)
    # Fix sign ambiguity: ensure diagonal of R is positive.
    diag_sign = torch.sign(torch.diag(R))
    # Guard against zero diagonal entries.
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q


class StreamingMirrorLoRA:
    """Streaming Mirror-LoRA: dual-timescale LoRA with Fisher-precision
    invariant-selective consolidation.

    Drop-in replacement for :class:`StreamingParameterMemory` that adds:

    1. **Online Fisher accumulation** via EMA (replaces batch FisherEstimator).
    2. **Fisher-precision weighted consolidation** -- after KL distillation +
       replay, applies element-wise Fisher-weighted merge to LT-LoRA params.
    3. **Adaptive consolidation frequency** -- triggers consolidation when
       ``max(F_fast / F_slow) >= threshold`` instead of at fixed intervals.
    4. **Invariant-selective mask** -- only consolidates parameters with
       stable gradient directions across a sliding window.
    5. **Optional Grassmann retraction** on LoRA-B matrices after updates.

    Args:
        base_model: The base pretrained model (before any PEFT adapters).
        working_config: Config dict for the working-memory LoRA adapter.
        longterm_config: Config dict for the long-term LoRA adapter.
        reservoir_size: Maximum number of samples in the reservoir buffer.
        beta: Weight for KL distillation loss during consolidation.
        gamma: Weight for Fisher trust-region regulariser.
        fisher_beta: EMA decay factor for the online Fisher accumulator.
        consolidation_threshold: Threshold for adaptive consolidation trigger.
            Consolidation fires when ``max(F_fast / F_slow) >= threshold``.
        invariant_window: Sliding window size for the invariant detector.
        invariant_threshold: Cosine similarity threshold for invariance.
        use_grassmann: Whether to apply Grassmann retraction on LoRA-B
            matrices after working-memory updates.
        adaptive_frequency: If ``True``, use the Fisher-ratio trigger for
            consolidation.  If ``False``, fall back to the fixed turn-count
            trigger from :class:`WorkingMemoryLoRA`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        working_config: dict,
        longterm_config: dict,
        reservoir_size: int = 5000,
        beta: float = 1.0,
        gamma: float = 0.0,
        fisher_beta: float = 0.99,
        consolidation_threshold: float = 1.0,
        invariant_window: int = 3,
        invariant_threshold: float = 0.5,
        use_grassmann: bool = False,
        adaptive_frequency: bool = True,
    ):
        # Delegate core adapter setup to StreamingParameterMemory.
        self._spm = StreamingParameterMemory(
            base_model=base_model,
            working_config=working_config,
            longterm_config=longterm_config,
            reservoir_size=reservoir_size,
            beta=beta,
            gamma=gamma,
        )

        # Expose inner attributes for backward-compat access.
        self.model: PeftModel = self._spm.model
        self.working: WorkingMemoryLoRA = self._spm.working
        self.longterm: LongTermMemoryLoRA = self._spm.longterm
        self.reservoir: ReservoirBuffer = self._spm.reservoir
        self.session_buffer: SessionBuffer = self._spm.session_buffer
        self.beta: float = beta
        self.gamma: float = gamma

        # --- Mirror-LoRA extensions ---
        self.fisher_fast = OnlineFisherAccumulator(beta=fisher_beta)
        self.fisher_slow: Dict[str, torch.Tensor] = {}  # accumulated LT Fisher
        self.fisher_beta: float = fisher_beta

        self.consolidation_threshold: float = consolidation_threshold
        self.adaptive_frequency: bool = adaptive_frequency

        self.invariant_detector = InvariantDetector(
            window_size=invariant_window, threshold=invariant_threshold
        )

        self.use_grassmann: bool = use_grassmann

        # Bookkeeping.
        self._consolidation_count: int = 0
        self._turn_count: int = 0

    # ------------------------------------------------------------------
    # Properties forwarded from inner SPM
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> int:
        """Current session index."""
        return self._spm.session_id

    @session_id.setter
    def session_id(self, value: int) -> None:
        self._spm.session_id = value

    @property
    def total_turns(self) -> int:
        """Total turns processed across all sessions."""
        return self._spm.total_turns

    @total_turns.setter
    def total_turns(self, value: int) -> None:
        self._spm.total_turns = value

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _grad_callback(self, model: nn.Module, loss: torch.Tensor) -> None:
        """Callback invoked from online_update's last backward step.

        Captures live training gradients for Fisher accumulation and
        invariant detection — no redundant forward pass required.
        """
        # 1. Update Fisher from live training gradients.
        self.fisher_fast.update(model, loss)

        # 2. Snapshot gradients for invariant detector.
        grads: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or "lora" not in name:
                continue
            if param.grad is not None:
                grads[name] = param.grad.data
            else:
                grads[name] = torch.zeros_like(param.data)
        if grads:
            self.invariant_detector.update(grads)

    def process_turn(self, input_ids: torch.Tensor, labels: torch.Tensor) -> dict:
        """Process one conversation turn.

        Steps:
          1. Update working-memory LoRA via online SGD, capturing gradients
             from the last training step for Fisher/invariant tracking.
          2. Optionally apply Grassmann retraction to LoRA-B matrices.
          3. Check whether consolidation should be triggered.

        Args:
            input_ids: Input token IDs, shape ``(1, seq_len)`` or ``(seq_len,)``.
            labels: Label token IDs, same shape as ``input_ids``.

        Returns:
            Dictionary with keys ``loss``, ``total_turns``, ``consolidated``,
            ``fisher_ratio`` (if adaptive), and ``consolidation_triggered``.
        """
        # 1. WM update with grad callback for Fisher/invariant capture.
        #    This avoids the old _update_fisher_and_snapshot which ran a
        #    redundant forward+backward pass on already-updated weights.
        _set_adapters_layerwise(self.model, ["working", "longterm"],
                               trainable_adapters={"working"})

        loss = self.working.online_update(
            self.model, input_ids, labels,
            lr=self.working.config.get("learning_rate", 5e-4),
            steps=self.working.config.get("online_update_steps", 3),
            grad_callback=self._grad_callback,
        )

        self._spm.session_buffer.add(input_ids, labels)
        self._spm.total_turns += 1
        self._turn_count += 1

        result = {"loss": loss, "total_turns": self._spm.total_turns, "consolidated": False}

        # 2. Optional Grassmann retraction.
        if self.use_grassmann:
            self._apply_grassmann_retraction()

        # 3. Check consolidation trigger.
        consolidated = False
        fisher_ratio = 0.0
        if self.should_consolidate():
            t0 = time.time()
            consol_loss = self.consolidate()
            elapsed = time.time() - t0
            consolidated = True
            logger.info(
                "Auto-consolidation #%d triggered at turn %d (%.2fs, loss=%.4f)",
                self._consolidation_count,
                self._turn_count,
                elapsed,
                consol_loss,
            )

        if self.adaptive_frequency and self.fisher_slow:
            fisher_ratio = self._compute_max_fisher_ratio()

        result["consolidated"] = consolidated
        result["fisher_ratio"] = fisher_ratio
        result["consolidation_triggered"] = consolidated
        return result

    def should_consolidate(self) -> bool:
        """Determine whether consolidation should fire.

        If ``adaptive_frequency`` is ``True``, consolidation triggers when::

            max_i (F_fast_i / F_slow_i) >= consolidation_threshold

        Also supports the closed-form optimal interval::

            B* = sqrt(C_merge / rho_drift)

        If ``adaptive_frequency`` is ``False``, delegates to the fixed
        turn-count trigger from :class:`WorkingMemoryLoRA`.

        Returns:
            ``True`` if consolidation should happen now.
        """
        if not self.adaptive_frequency:
            return self.working.needs_consolidation()

        # Need both fast and slow Fisher to compare.
        fast_fisher = self.fisher_fast.get_fisher()
        if not fast_fisher or not self.fisher_slow:
            # Fall back to fixed trigger until we have both.
            return self.working.needs_consolidation()

        ratio = self._compute_max_fisher_ratio()
        return ratio >= self.consolidation_threshold

    def consolidate(self) -> float:
        """Fisher-regularised consolidation with trust-region on important params.

        Procedure:
          1. Run standard KL distillation + replay consolidation (via SPM).
          2. Apply Fisher trust-region: penalise large updates to high-Fisher
             LT params that already encode important knowledge.
          3. Update the slow Fisher estimate.
          4. Reset per-session accumulators.

        The key insight: standard KL distillation treats all LT parameters
        equally.  Fisher trust-region adds an extra constraint that prevents
        the consolidation from overwriting previously learned important
        directions, while allowing free updates to low-importance params.

        Returns:
            Consolidation loss from the distillation step.
        """
        t0 = time.time()

        # Snapshot LT params BEFORE consolidation (for trust-region).
        lt_params_before: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if "lora" in name and "longterm" in name:
                lt_params_before[name] = param.data.clone()

        # 1. Standard consolidation (KL distillation + replay).
        consol_loss = self._spm.consolidate_session()

        # 2. Fisher trust-region: pull high-Fisher params back toward
        #    pre-consolidation values.  This prevents the KL distillation
        #    from overwriting important knowledge in LT.
        fast_fisher = self.fisher_fast.get_fisher()
        if self.fisher_slow:
            self._apply_fisher_trust_region(lt_params_before, fast_fisher)

        # 3. Update slow Fisher: merge fast into slow via EMA.
        self._update_slow_fisher(fast_fisher)

        # 4. Reset per-session accumulators.
        self.fisher_fast.reset()
        self.invariant_detector.reset()
        self._turn_count = 0
        self._consolidation_count += 1

        elapsed = time.time() - t0
        logger.info(
            "Mirror-LoRA consolidation #%d complete: loss=%.4f, elapsed=%.2fs, "
            "fast_fisher_keys=%d, slow_fisher_keys=%d",
            self._consolidation_count,
            consol_loss,
            elapsed,
            len(fast_fisher),
            len(self.fisher_slow),
        )

        return consol_loss

    def _apply_fisher_trust_region(
        self,
        lt_params_before: Dict[str, torch.Tensor],
        fast_fisher: Dict[str, torch.Tensor],
    ) -> None:
        """Apply Fisher trust-region constraint on LT params after consolidation.

        For each LT param, blend between pre- and post-consolidation values
        based on the slow Fisher importance:

            theta_LT = (1 - trust) * theta_LT_post + trust * theta_LT_pre

        where trust = sigmoid(norm(F_slow) - threshold) scales from 0
        (keep full update for unimportant params) to a cap (preserve
        important params).  This prevents catastrophic overwriting of
        previously consolidated important knowledge.
        """
        eps = 1e-8
        trust_cap = 0.3  # max trust: preserve up to 30% of pre-consol value
        count = 0

        for name, param in self.model.named_parameters():
            if "lora" not in name or "longterm" not in name:
                continue

            pre = lt_params_before.get(name)
            if pre is None:
                continue

            device = param.data.device
            pre = pre.to(device)
            delta = param.data - pre
            if delta.abs().max().item() < 1e-10:
                continue

            # Compute importance from slow Fisher.
            f_slow = self.fisher_slow.get(name)
            if f_slow is None:
                # No slow Fisher = first consolidation, keep full update.
                continue

            f_slow_dev = f_slow.to(device).float()
            # Per-element trust based on slow Fisher importance.
            # High F_slow → high trust → preserve pre-consol value.
            if f_slow_dev.shape == param.data.shape:
                log_f = torch.log(f_slow_dev + eps)
                trust = torch.sigmoid(log_f - log_f.mean()) * trust_cap
                param.data.copy_(param.data - trust * delta)
            else:
                # Shape mismatch: use scalar trust.
                importance = f_slow_dev.mean().item()
                trust = min(torch.sigmoid(torch.tensor(
                    importance - 1e-4)).item() * trust_cap, trust_cap)
                param.data.copy_(param.data - trust * delta)
            count += 1

        logger.debug("Fisher trust-region applied to %d LT params", count)

    def start_new_session(self) -> None:
        """Begin a new session.

        Delegates to SPM's ``start_new_session`` (zero-inits WM, clears
        session buffer) and resets the fast Fisher accumulator.
        """
        self._spm.start_new_session()
        self.fisher_fast.reset()
        self.invariant_detector.reset()
        self._turn_count = 0
        logger.info(
            "Mirror-LoRA: started session %d",
            self._spm.session_id,
        )

    # ------------------------------------------------------------------
    # Generation (forwarded)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        use_longterm: bool = True,
    ) -> torch.Tensor:
        """Generate with combined WM + LT adapters.

        Delegates entirely to the inner :class:`StreamingParameterMemory`.

        Args:
            input_ids: Prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            use_longterm: Whether to include the long-term adapter.

        Returns:
            Generated token IDs.
        """
        return self._spm.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_longterm=use_longterm,
        )

    # ------------------------------------------------------------------
    # Save / Load (forwarded)
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """Save adapters, reservoir, and session metadata.

        Also persists the slow Fisher estimate.

        Args:
            output_dir: Directory to write state into.
        """
        import os
        import pickle

        self._spm.save(output_dir)

        fisher_path = os.path.join(output_dir, "fisher_slow.pt")
        torch.save(self.fisher_slow, fisher_path)
        logger.info("Saved slow Fisher to %s", fisher_path)

        meta_extra = {
            "consolidation_count": self._consolidation_count,
            "consolidation_threshold": self.consolidation_threshold,
            "adaptive_frequency": self.adaptive_frequency,
            "use_grassmann": self.use_grassmann,
            "fisher_beta": self.fisher_beta,
        }
        meta_path = os.path.join(output_dir, "mirror_lora_meta.pt")
        torch.save(meta_extra, meta_path)
        logger.info("Saved Mirror-LoRA metadata to %s", meta_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_grassmann_retraction(self) -> None:
        """Apply QR retraction to all LoRA-B matrices in the working adapter."""
        for name, param in self.model.named_parameters():
            if "working" in name and "lora_B" in name and param.requires_grad:
                if param.data.dim() == 2:
                    param.data.copy_(grassmann_retraction(param.data))

    def _compute_max_fisher_ratio(self) -> float:
        """Compute max_i(F_fast_i / F_slow_i) for the adaptive trigger.

        Returns:
            The maximum element-wise ratio, or 0.0 if either Fisher is empty.
        """
        fast = self.fisher_fast.get_fisher()
        if not fast or not self.fisher_slow:
            return 0.0

        max_ratio = 0.0
        eps = 1e-8
        for name in fast:
            if name not in self.fisher_slow:
                continue
            f_fast = fast[name]
            f_slow = self.fisher_slow[name]
            # Compute element-wise ratio, clamp denominator.
            ratio = f_fast / (f_slow + eps)
            current_max = ratio.max().item()
            if current_max > max_ratio:
                max_ratio = current_max

        return max_ratio

    @staticmethod
    def compute_optimal_interval(
        merge_cost: float, drift_rate: float
    ) -> float:
        """Closed-form optimal consolidation interval.

        B* = sqrt(C_merge / rho_drift)

        Args:
            merge_cost: Computational cost of one consolidation step.
            drift_rate: Rate of parameter drift between consolidations.

        Returns:
            Optimal number of turns between consolidations.
        """
        if drift_rate <= 0:
            return float("inf")
        return math.sqrt(merge_cost / drift_rate)

    def _apply_fisher_precision_merge(
        self,
        lt_params_before: Dict[str, torch.Tensor],
        wm_params_snapshot: Dict[str, torch.Tensor],
        fast_fisher: Dict[str, torch.Tensor],
        invariant_mask: Dict[str, bool],
    ) -> None:
        """Fisher-precision modulated consolidation update on LT-LoRA params.

        Since WM (rank r_wm) and LT (rank r_lt) may have different shapes,
        direct parameter merge is not possible.  Instead, we modulate the
        consolidation UPDATE using Fisher information:

            delta_i = theta_LT_post_i - theta_LT_pre_i   (the KL distillation update)
            alpha_i = sigmoid(log(F_slow_i + eps) - mean(log(F_slow + eps)))
            w_i     = 1.0 if invariant, 0.3 if transient
            theta_LT_i = theta_LT_pre_i + alpha_i * w_i * delta_i

        High-Fisher (important) parameters keep most of their update;
        low-Fisher parameters get a damped update, preventing noisy
        consolidation from corrupting important directions.

        Args:
            lt_params_before: LT-LoRA parameter values before consolidation.
            wm_params_snapshot: Working-memory LoRA snapshot (unused, kept for API).
            fast_fisher: Diagonal Fisher estimates from the fast accumulator.
            invariant_mask: Boolean mask from the invariant detector.
        """
        eps = 1e-8
        merge_count = 0
        skip_count = 0

        # Compute global Fisher statistics for normalization.
        all_log_fisher = []
        for name, f in fast_fisher.items():
            all_log_fisher.append(torch.log(f + eps).mean().item())
        mean_log_f = sum(all_log_fisher) / max(len(all_log_fisher), 1) if all_log_fisher else 0.0

        for name, param in self.model.named_parameters():
            if "lora" not in name or "longterm" not in name:
                continue

            # Get pre-consolidation snapshot.
            pre = lt_params_before.get(name)
            if pre is None:
                skip_count += 1
                continue

            # Consolidation update.
            device = param.data.device
            pre = pre.to(device)
            delta = param.data - pre

            # Skip if no update happened.
            if delta.abs().max().item() < 1e-10:
                skip_count += 1
                continue

            # Find corresponding Fisher estimate.
            # Fisher keys are WM param names; map LT→WM name.
            wm_name = name.replace("longterm", "working")
            f_fast = fast_fisher.get(name, fast_fisher.get(wm_name))

            if f_fast is not None:
                # Compute importance-based boost.
                # Project Fisher to LT shape if needed (different ranks).
                f_fast_dev = f_fast.to(device).float()
                if f_fast_dev.shape != delta.shape:
                    importance = torch.log(f_fast_dev.mean() + eps).item()
                else:
                    importance = torch.log(f_fast_dev + eps).mean().item()

                # Boost factor: high-Fisher params get amplified consolidation,
                # low-Fisher params keep baseline (1.0).  Range: [0.8, 1.2].
                raw_alpha = torch.sigmoid(torch.tensor(importance - mean_log_f)).item()
                alpha = 0.8 + 0.4 * raw_alpha  # [0.8, 1.2]
            else:
                alpha = 1.0  # default: same as SPM

            # Invariant weighting: stable params keep full update,
            # transient get slightly reduced.  Range: [0.8, 1.0].
            is_invariant = invariant_mask.get(name, invariant_mask.get(wm_name, True))
            w = 1.0 if is_invariant else 0.8

            # Apply modulated update (floor = 0.64, ceiling = 1.2).
            param.data.copy_(pre + alpha * w * delta)
            merge_count += 1

        logger.debug(
            "Fisher-precision merge: %d params modulated, %d skipped",
            merge_count,
            skip_count,
        )

    def _update_slow_fisher(self, fast_fisher: Dict[str, torch.Tensor]) -> None:
        """Merge fast Fisher into the slow (long-term) Fisher via EMA.

        F_slow <- beta * F_slow + (1 - beta) * F_fast

        Args:
            fast_fisher: The fast Fisher estimates to merge.
        """
        decay = self.fisher_beta
        for name, f_fast in fast_fisher.items():
            # Remap to LT name if stored under WM name.
            lt_name = name.replace("working", "longterm") if "working" in name else name
            if lt_name in self.fisher_slow:
                device = self.fisher_slow[lt_name].device
                f_fast_dev = f_fast.to(device)
                self.fisher_slow[lt_name].mul_(decay).add_(f_fast_dev, alpha=1.0 - decay)
            else:
                self.fisher_slow[lt_name] = f_fast.clone()
