"""Synthetic benchmark for the separation theorem of dual-timescale streaming low-rank adaptation.

Controlled linear streaming prediction environment:
  x_t ~ N(0, I_d),  y_t = (U_t @ a_t)^T x_t + noise
  U_t: slowly drifting orthogonal subspace on Gr(d,r) (Grassmann random walk)
  a_t: fast-varying coefficient vector in R^r (periodic jumps of magnitude Delta)

All learners use rank-r parameterization theta = U_hat @ a_hat.

The CORE difference between single-timescale and dual-timescale:
  Single: coefficient update uses eta_a = c / sqrt(t)  (decaying)
  Dual:   coefficient update uses sliding-window OLS    (constant effective rate)

Both use the SAME slow subspace update (decaying eta_U / sqrt(t)).
This isolates the fast-state mechanism.

Why dual wins: after a coefficient jump at time tau, the single-timescale
learner must wait O(1/eta_tau) = O(sqrt(tau)) steps to recover because its
step size has decayed.  The dual learner's window OLS recovers in exactly
window_size steps regardless of tau.  With K_T jumps spread across [0,T],
single pays sum_k sqrt(k*J) while dual pays K_T * window_size.

Theoretical reference: see proofs/theorem1.tex (Necessity) and
proofs/theorem2.tex (Sufficiency) for formal statements and proofs."""

import math
from typing import Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Data stream
# ---------------------------------------------------------------------------

class TwoRateDriftStream:
    """Streaming environment with slow subspace drift and fast coefficient jumps.

    Parameters
    ----------
    d : int
        Input dimension.
    r : int
        True rank.
    rho_s : float
        Slow subspace rotation rate (QR retraction scale).
    rho_f : float
        Fast coefficient drift rate (unused when jumps active).
    jump_interval : int
        Steps between fast coefficient jumps.
    jump_size : float
        L2 magnitude of each coefficient jump (Delta).
    noise_std : float
        Observation noise sigma.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        d: int = 100,
        r: int = 8,
        rho_s: float = 0.01,
        rho_f: float = 0.1,
        jump_interval: int = 50,
        jump_size: float = 1.0,
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        self.d = d
        self.r = r
        self.rho_s = rho_s
        self.rho_f = rho_f
        self.jump_interval = jump_interval
        self.jump_size = jump_size
        self.noise_std = noise_std
        self.seed = seed

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self.t = 0

        Z = torch.randn(d, r, generator=self._rng)
        self.U, _ = torch.linalg.qr(Z)
        self.a = torch.randn(r, generator=self._rng)
        self.a = self.a / self.a.norm() * jump_size

        self._U0 = self.U.clone()
        self._a0 = self.a.clone()
        self._rng_state0 = self._rng.get_state()

    def step(self) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """Returns (x_t, y_t, theta_star_t)."""
        theta_star = self.U @ self.a
        x = torch.randn(self.d, generator=self._rng)
        noise = torch.randn(1, generator=self._rng).item() * self.noise_std
        y = (theta_star @ x).item() + noise
        self._evolve_subspace()
        self._evolve_coefficients()
        self.t += 1
        return x, y, theta_star

    def get_optimal_subspace(self) -> torch.Tensor:
        return self.U.clone()

    def get_optimal_coefficients(self) -> torch.Tensor:
        return self.a.clone()

    def reset(self):
        self.t = 0
        self.U = self._U0.clone()
        self.a = self._a0.clone()
        self._rng.set_state(self._rng_state0)

    def _evolve_subspace(self):
        Z = torch.randn(self.d, self.r, generator=self._rng)
        Z = Z - self.U @ (self.U.T @ Z)
        self.U = self.U + self.rho_s * Z
        self.U, _ = torch.linalg.qr(self.U)

    def _evolve_coefficients(self):
        if (self.t + 1) % self.jump_interval == 0:
            direction = torch.randn(self.r, generator=self._rng)
            direction = direction / direction.norm()
            # Replacement jump: new coefficient, bounded by A_max (fixes Issue #29)
            self.a = self.jump_size * direction


class DecreasingJumpStream(TwoRateDriftStream):
    """Canonical Theorem 1 test stream with K_T = Theta(sqrt(T)) jumps.

    Jumps are placed at quadratically-spaced times tau_k = k^2 to match
    the adversarial construction in Theorem 1 (proofs/theorem1.tex).
    This ensures non-overlapping recovery periods and maximizes the
    cumulative regret gap between single-timescale and dual-timescale.

    Fixes Issue #10: original stream used fixed jump_interval giving
    K_T = Theta(T), not the K_T = Theta(sqrt(T)) required by Theorem 1.
    """

    def __init__(
        self,
        d: int = 100,
        r: int = 8,
        rho_s: float = 0.0,
        jump_size: float = 1.0,
        noise_std: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(
            d=d, r=r, rho_s=rho_s, rho_f=0.0,
            jump_interval=999999999,
            jump_size=jump_size, noise_std=noise_std, seed=seed,
        )
        self._jump_times: List[int] = []  # populated on first use

    def _compute_jump_times(self, T: int) -> List[int]:
        """Quadratic spacing: tau_k = k^2 for k = 1, ..., floor(sqrt(T))."""
        K_T = int(math.sqrt(T))
        return [k * k for k in range(1, K_T + 1) if k * k <= T]

    def set_horizon(self, T: int):
        """Pre-compute jump schedule for horizon T."""
        self._jump_times = self._compute_jump_times(T)

    def _evolve_coefficients(self):
        if (self.t + 1) in self._jump_times:
            direction = torch.randn(self.r, generator=self._rng)
            direction = direction / direction.norm()
            self.a = self.jump_size * direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regret_step(y_hat: float, y: float, y_star: float) -> float:
    return 0.5 * (y_hat - y) ** 2 - 0.5 * (y_star - y) ** 2


def _init_subspace(d: int, r: int, ref_U: torch.Tensor, noise: float = 0.1) -> torch.Tensor:
    Z = ref_U + noise * torch.randn(d, r)
    Q, _ = torch.linalg.qr(Z)
    return Q


def _slow_U_update(U: torch.Tensor, x: torch.Tensor, a: torch.Tensor,
                    residual: float, lr_slow: float, t: int) -> torch.Tensor:
    """Shared slow subspace gradient step used by all rank-r learners."""
    x_sq = (x @ x).item() + 1e-8
    grad_U = (residual / x_sq) * torch.outer(x, a)
    eta = lr_slow / math.sqrt(max(t, 1))
    U = U - eta * grad_U
    U, _ = torch.linalg.qr(U)
    return U


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------

class OnlineLowRankLearner:
    """Single-timescale: BOTH a and U share one decaying schedule eta/sqrt(t).

    Coefficient recovery from jump at tau takes O(sqrt(tau)) steps because
    eta_tau = lr / sqrt(tau) -> recovery ~ Delta^2 / eta_tau = Delta^2 * sqrt(tau).
    """

    def __init__(self, d: int, r: int, lr: float = 1.0,
                 U_init: torch.Tensor = None):
        self.d = d
        self.r = r
        self.lr = lr
        if U_init is not None:
            self.U_hat = U_init.clone()
        else:
            Z = torch.randn(d, r) * 0.01
            self.U_hat, _ = torch.linalg.qr(Z)
        self.a_hat = torch.zeros(r)
        self._regret: List[float] = []
        self._cum = 0.0
        self._t = 0

    def predict(self, x: torch.Tensor) -> float:
        return ((self.U_hat @ self.a_hat) @ x).item()

    def update(self, x: torch.Tensor, y: float, theta_star: torch.Tensor):
        self._t += 1
        y_hat = self.predict(x)
        y_star = (theta_star @ x).item()
        self._cum += _regret_step(y_hat, y, y_star)
        self._regret.append(self._cum)

        residual = y_hat - y
        proj_x = self.U_hat.T @ x
        px_sq = (proj_x @ proj_x).item() + 1e-8
        x_sq = (x @ x).item() + 1e-8

        # SINGLE shared decaying rate for BOTH a and U
        eta = self.lr / math.sqrt(self._t)
        self.a_hat = self.a_hat - eta * residual * proj_x / px_sq
        grad_U = (residual / x_sq) * torch.outer(x, self.a_hat)
        self.U_hat = self.U_hat - eta * grad_U
        self.U_hat, _ = torch.linalg.qr(self.U_hat)

    def get_regret(self) -> List[float]:
        return list(self._regret)


class DualTimescaleLearner:
    """Dual-timescale: fast a via window OLS, slow U via decaying gradient.

    The sliding-window OLS computes:
      a_hat = (sum_{s in window} z_s z_s^T + lam I)^{-1} (sum_{s in window} z_s y_s)
    where z_s = U_hat^T x_s.

    Recovery from any jump = window_size steps (old data ages out).
    Steady-state noise = O(sigma^2 * r / window_size).
    """

    def __init__(
        self,
        d: int,
        r: int,
        lr_fast: float = 0.5,
        lr_slow: float = 0.01,
        consolidation_period: int = 20,
        fisher_weighted: bool = True,
        selective: bool = False,
        U_init: torch.Tensor = None,
        window_size: int = 30,
    ):
        self.d = d
        self.r = r
        self.lr_fast = lr_fast
        self.lr_slow = lr_slow
        self.consolidation_period = consolidation_period
        self.fisher_weighted = fisher_weighted
        self.selective = selective
        self.window_size = window_size

        if U_init is not None:
            self.U_hat = U_init.clone()
        else:
            Z = torch.randn(d, r) * 0.01
            self.U_hat, _ = torch.linalg.qr(Z)
        self.a_hat = torch.zeros(r)

        # Sliding window
        self._zs: List[torch.Tensor] = []
        self._ys: List[float] = []

        # Slow accumulation
        self._cov_proj = torch.zeros(r, r)
        self._U_grad_accum = torch.zeros(d, r)
        self._consolidation_count = 0

        # Gradient-sign history for selective consolidation (Cor3)
        self._grad_sign_history: List[torch.Tensor] = []
        self._grad_sign_window = 5

        self._regret: List[float] = []
        self._cum = 0.0
        self._t = 0

    def predict(self, x: torch.Tensor) -> float:
        return ((self.U_hat @ self.a_hat) @ x).item()

    def update(self, x: torch.Tensor, y: float, theta_star: torch.Tensor):
        self._t += 1
        y_hat = self.predict(x)
        y_star = (theta_star @ x).item()
        self._cum += _regret_step(y_hat, y, y_star)
        self._regret.append(self._cum)

        z = self.U_hat.T @ x  # (r,)
        residual = y_hat - y

        # --- Fast: window OLS ---
        self._zs.append(z.clone())
        self._ys.append(y)
        if len(self._zs) > self.window_size:
            self._zs.pop(0)
            self._ys.pop(0)
        n = len(self._zs)
        if n >= self.r + 1:
            Z_mat = torch.stack(self._zs)
            Y_vec = torch.tensor(self._ys)
            lam = 1e-3
            ZtZ = Z_mat.T @ Z_mat + lam * torch.eye(self.r)
            ZtY = Z_mat.T @ Y_vec
            self.a_hat = torch.linalg.solve(ZtZ, ZtY)

        # --- Slow: accumulate subspace gradient using POST-OLS residual ---
        x_sq = (x @ x).item() + 1e-8
        self._cov_proj = self._cov_proj + torch.outer(z, z)
        # Recompute residual with updated a_hat for unbiased U gradient
        new_residual = ((self.U_hat @ self.a_hat) @ x).item() - y
        grad_U = (new_residual / x_sq) * torch.outer(x, self.a_hat)
        self._U_grad_accum = self._U_grad_accum + grad_U
        self._consolidation_count += 1

        if self._consolidation_count >= self.consolidation_period:
            self.consolidate()

    def consolidate(self):
        if self._consolidation_count == 0:
            return
        avg_grad = self._U_grad_accum / self._consolidation_count

        # Record gradient sign for selective consolidation (Cor3)
        col_sign = torch.sign(avg_grad.mean(dim=0))  # (r,)
        self._grad_sign_history.append(col_sign)
        if len(self._grad_sign_history) > self._grad_sign_window:
            self._grad_sign_history.pop(0)

        if self.fisher_weighted and self._t > 0:
            # Fisher-precision weighting: scale each column of U gradient
            # by inverse of its importance (fixes Issue #25).
            # Importance ~ ||grad_U[:,j]||^2 as proxy for diagonal Fisher.
            col_norms = avg_grad.norm(dim=0) + 1e-8  # (r,)
            # Normalize to trust-region style: high-norm cols get less update
            scale = col_norms / col_norms.mean()
            avg_grad = avg_grad / scale.unsqueeze(0)

        if self.selective:
            # Gradient-sign stability: consolidate only coordinates with
            # consistent gradient direction (fixes Issue #24, matches Cor3).
            if hasattr(self, '_grad_sign_history') and len(self._grad_sign_history) >= 2:
                signs = torch.stack(self._grad_sign_history)  # (W, r)
                stability = signs.mean(dim=0).abs()  # s_j in [0,1]
                mask = (stability >= 0.5).float()  # tau_s = 0.5
            else:
                mask = torch.ones(self.r)
            avg_grad = avg_grad * mask.unsqueeze(0)

        eta = self.lr_slow / math.sqrt(max(self._t, 1))
        self.U_hat = self.U_hat - eta * avg_grad
        self.U_hat, _ = torch.linalg.qr(self.U_hat)
        self._U_grad_accum.zero_()
        self._consolidation_count = 0

    def get_regret(self) -> List[float]:
        return list(self._regret)


class EMALearner:
    """Rank-r single-timescale with EMA on theta for prediction."""

    def __init__(self, d: int, r: int, lr: float = 1.0,
                 beta: float = 0.99, U_init: torch.Tensor = None):
        self.d = d
        self.r = r
        self.lr = lr
        self.beta = beta
        if U_init is not None:
            self.U_hat = U_init.clone()
        else:
            Z = torch.randn(d, r) * 0.01
            self.U_hat, _ = torch.linalg.qr(Z)
        self.a_hat = torch.zeros(r)
        self.theta_ema = torch.zeros(d)
        self._regret: List[float] = []
        self._cum = 0.0
        self._t = 0

    def predict(self, x: torch.Tensor) -> float:
        return (self.theta_ema @ x).item()

    def update(self, x: torch.Tensor, y: float, theta_star: torch.Tensor):
        self._t += 1
        y_hat = self.predict(x)
        y_star = (theta_star @ x).item()
        self._cum += _regret_step(y_hat, y, y_star)
        self._regret.append(self._cum)

        theta = self.U_hat @ self.a_hat
        residual = (theta @ x).item() - y
        proj_x = self.U_hat.T @ x
        px_sq = (proj_x @ proj_x).item() + 1e-8
        x_sq = (x @ x).item() + 1e-8
        eta = self.lr / math.sqrt(self._t)
        self.a_hat = self.a_hat - eta * residual * proj_x / px_sq
        grad_U = (residual / x_sq) * torch.outer(x, self.a_hat)
        self.U_hat = self.U_hat - eta * grad_U
        self.U_hat, _ = torch.linalg.qr(self.U_hat)
        self.theta_ema = self.beta * self.theta_ema + (1.0 - self.beta) * (self.U_hat @ self.a_hat)

    def get_regret(self) -> List[float]:
        return list(self._regret)


class PeriodicAveragingLearner:
    """Rank-r single-timescale with periodic averaging of theta."""

    def __init__(self, d: int, r: int, lr: float = 1.0,
                 avg_period: int = 50, U_init: torch.Tensor = None):
        self.d = d
        self.r = r
        self.lr = lr
        self.avg_period = avg_period
        if U_init is not None:
            self.U_hat = U_init.clone()
        else:
            Z = torch.randn(d, r) * 0.01
            self.U_hat, _ = torch.linalg.qr(Z)
        self.a_hat = torch.zeros(r)
        self.theta_slow = torch.zeros(d)
        self._accum = torch.zeros(d)
        self._accum_n = 0
        self._regret: List[float] = []
        self._cum = 0.0
        self._t = 0

    def predict(self, x: torch.Tensor) -> float:
        return (0.5 * (self.U_hat @ self.a_hat + self.theta_slow) @ x).item()

    def update(self, x: torch.Tensor, y: float, theta_star: torch.Tensor):
        self._t += 1
        y_hat = self.predict(x)
        y_star = (theta_star @ x).item()
        self._cum += _regret_step(y_hat, y, y_star)
        self._regret.append(self._cum)

        residual = y_hat - y
        proj_x = self.U_hat.T @ x
        px_sq = (proj_x @ proj_x).item() + 1e-8
        x_sq = (x @ x).item() + 1e-8
        eta = self.lr / math.sqrt(self._t)
        self.a_hat = self.a_hat - eta * residual * proj_x / px_sq
        grad_U = (residual / x_sq) * torch.outer(x, self.a_hat)
        self.U_hat = self.U_hat - eta * grad_U
        self.U_hat, _ = torch.linalg.qr(self.U_hat)

        self._accum = self._accum + self.U_hat @ self.a_hat
        self._accum_n += 1
        if self._accum_n >= self.avg_period:
            self.theta_slow = self._accum / self._accum_n
            self._accum = torch.zeros(self.d)
            self._accum_n = 0

    def get_regret(self) -> List[float]:
        return list(self._regret)


class SubspaceTrackingLearner:
    """Oja subspace + decaying coefficient OGD."""

    def __init__(self, d: int, r: int, lr_oja: float = 0.01, lr_coeff: float = 1.0,
                 U_init: torch.Tensor = None):
        self.d = d
        self.r = r
        self.lr_oja = lr_oja
        self.lr_coeff = lr_coeff
        if U_init is not None:
            self.U_hat = U_init.clone()
        else:
            Z = torch.randn(d, r) * 0.01
            self.U_hat, _ = torch.linalg.qr(Z)
        self.a_hat = torch.zeros(r)
        self._regret: List[float] = []
        self._cum = 0.0
        self._t = 0

    def predict(self, x: torch.Tensor) -> float:
        return ((self.U_hat @ self.a_hat) @ x).item()

    def update(self, x: torch.Tensor, y: float, theta_star: torch.Tensor):
        self._t += 1
        y_hat = self.predict(x)
        y_star = (theta_star @ x).item()
        self._cum += _regret_step(y_hat, y, y_star)
        self._regret.append(self._cum)

        residual = y_hat - y
        proj_x = self.U_hat.T @ x
        px_sq = (proj_x @ proj_x).item() + 1e-8
        eta_a = self.lr_coeff / math.sqrt(self._t)
        self.a_hat = self.a_hat - eta_a * residual * proj_x / px_sq

        signal_x = y * x
        proj = self.U_hat.T @ signal_x
        oja = torch.outer(signal_x, proj) - self.U_hat @ torch.outer(proj, proj)
        eta_oja = self.lr_oja / math.sqrt(self._t)
        self.U_hat = self.U_hat + eta_oja * oja
        self.U_hat, _ = torch.linalg.qr(self.U_hat)

    def get_regret(self) -> List[float]:
        return list(self._regret)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def _pre_generate(stream: TwoRateDriftStream, T: int):
    stream.reset()
    return [stream.step() for _ in range(T)]


def run_experiment(
    stream: TwoRateDriftStream,
    learners: Dict[str, object],
    T: int = 10000,
) -> Dict[str, List[float]]:
    """Run all learners on same stream, return {name: cumulative_dynamic_regret}."""
    data = _pre_generate(stream, T)
    results = {}
    for name, learner in learners.items():
        for x, y, theta_star in data:
            learner.update(x, y, theta_star)
        results[name] = learner.get_regret()
    return results


def run_phase_diagram(
    rho_s_range: List[float],
    rho_f_range: List[float],
    T: int = 5000,
    d: int = 100,
    r: int = 8,
) -> Dict[str, object]:
    """Grid over (rho_s, rho_f). ratio = single/dual (>1 = dual wins)."""
    single_reg, dual_reg, ratio = [], [], []
    for rho_s in rho_s_range:
        row_s, row_d, row_r = [], [], []
        for rho_f in rho_f_range:
            ji = max(2, int(1.0 / rho_f))
            stream = TwoRateDriftStream(d=d, r=r, rho_s=rho_s, rho_f=rho_f,
                                        jump_interval=ji, jump_size=1.0,
                                        noise_std=0.1, seed=42)
            U0 = _init_subspace(d, r, stream.get_optimal_subspace(), noise=0.1)
            s = OnlineLowRankLearner(d, r, lr=1.0, U_init=U0.clone())
            du = DualTimescaleLearner(d, r, lr_slow=0.1, consolidation_period=1,
                                      fisher_weighted=False, U_init=U0.clone(),
                                      window_size=min(ji - 1, 30))
            res = run_experiment(stream, {"single": s, "dual": du}, T=T)
            sf, df = res["single"][-1], res["dual"][-1]
            row_s.append(sf); row_d.append(df)
            row_r.append(sf / df if abs(df) > 1e-3 else 1.0)
        single_reg.append(row_s); dual_reg.append(row_d); ratio.append(row_r)
    return {"rho_s_values": list(rho_s_range), "rho_f_values": list(rho_f_range),
            "single_regret": single_reg, "dual_regret": dual_reg, "ratio": ratio}


def run_frequency_sweep(
    B_range: List[int],
    stream: TwoRateDriftStream,
    T: int = 5000,
) -> Dict[str, object]:
    """Sweep consolidation period B."""
    U0 = _init_subspace(stream.d, stream.r, stream.get_optimal_subspace(), noise=0.1)
    frs, curves = [], {}
    for B in B_range:
        stream.reset()
        learner = DualTimescaleLearner(
            stream.d, stream.r, lr_slow=0.1,
            consolidation_period=B, fisher_weighted=False, U_init=U0.clone(),
            window_size=30)
        res = run_experiment(stream, {"dual": learner}, T=T)
        c = res["dual"]
        frs.append(c[-1] if c else 0.0); curves[B] = c
    return {"B_values": list(B_range), "final_regret": frs, "regret_curves": curves}


def run_selectivity_test(
    stream: TwoRateDriftStream,
    T: int = 5000,
) -> Dict[str, List[float]]:
    """Compare consolidate-all vs consolidate-invariant-only."""
    U0 = _init_subspace(stream.d, stream.r, stream.get_optimal_subspace(), noise=0.1)
    stream.reset()
    return run_experiment(stream, {
        "dual_all": DualTimescaleLearner(
            stream.d, stream.r, lr_slow=0.1,
            consolidation_period=1, fisher_weighted=False,
            selective=False, U_init=U0.clone(), window_size=30),
        "dual_selective": DualTimescaleLearner(
            stream.d, stream.r, lr_slow=0.1,
            consolidation_period=1, fisher_weighted=False,
            selective=True, U_init=U0.clone(), window_size=30),
    }, T=T)


def run_merge_rule_test(
    stream: TwoRateDriftStream,
    T: int = 5000,
) -> Dict[str, List[float]]:
    """Compare Fisher-weighted vs uniform consolidation vs EMA baseline."""
    U0 = _init_subspace(stream.d, stream.r, stream.get_optimal_subspace(), noise=0.1)
    stream.reset()
    return run_experiment(stream, {
        "fisher_weighted": DualTimescaleLearner(
            stream.d, stream.r, lr_slow=0.1,
            consolidation_period=20, fisher_weighted=True, U_init=U0.clone(),
            window_size=30),
        "uniform": DualTimescaleLearner(
            stream.d, stream.r, lr_slow=0.1,
            consolidation_period=20, fisher_weighted=False, U_init=U0.clone(),
            window_size=30),
        "ema_baseline": EMALearner(stream.d, stream.r, lr=1.0,
                                    beta=0.99, U_init=U0.clone()),
    }, T=T)


# ---------------------------------------------------------------------------
# Main: smoke test + demonstration experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # -----------------------------------------------------------------------
    # Smoke test
    # -----------------------------------------------------------------------
    print("=== Smoke test ===")
    stream = TwoRateDriftStream(d=50, r=4, rho_s=0.01, rho_f=0.1,
                                jump_interval=50, jump_size=1.0,
                                noise_std=0.1, seed=42)
    for i in range(3):
        x, y, theta_star = stream.step()
        print(f"  t={i}: ||x||={x.norm():.2f}, y={y:.4f}, ||theta*||={theta_star.norm():.4f}")
    U = stream.get_optimal_subspace()
    print(f"  U: {U.shape}, orthonormality err = {(U.T @ U - torch.eye(stream.r)).norm():.1e}")

    # -----------------------------------------------------------------------
    # Coefficient-only isolation test (fixed known subspace)
    # This is the cleanest demonstration: only the coefficient learner differs.
    # -----------------------------------------------------------------------
    print("\n=== Coefficient-only isolation (fixed known subspace, T scaling) ===")
    print(f"  {'T':>6s} {'single':>10s} {'dual':>10s} {'ratio':>7s}")
    for T in [500, 1000, 2000, 5000, 10000, 20000]:
        stream = TwoRateDriftStream(d=50, r=4, rho_s=0.0, rho_f=0.1,
                                    jump_interval=40, jump_size=2.0,
                                    noise_std=0.1, seed=42)
        # Give both the TRUE subspace -- only coefficient learning differs
        U_true = stream.get_optimal_subspace()
        single = OnlineLowRankLearner(50, 4, lr=1.0, U_init=U_true.clone())
        dual = DualTimescaleLearner(50, 4, lr_slow=0.0,
                                    consolidation_period=100000,
                                    U_init=U_true.clone(), window_size=30)
        res = run_experiment(stream, {"single": single, "dual": dual}, T=T)
        s, dv = res["single"][-1], res["dual"][-1]
        print(f"  {T:>6d} {s:>10.1f} {dv:>10.1f} {s/max(dv,1e-10):>7.2f}")

    # -----------------------------------------------------------------------
    # Theorem 1 canonical test: DecreasingJumpStream (K_T = sqrt(T))
    # -----------------------------------------------------------------------
    print("\n=== Theorem 1 canonical: DecreasingJumpStream (K_T=sqrt(T)) ===")
    print(f"  {'T':>6s} {'K_T':>5s} {'single':>10s} {'dual':>10s} {'ratio':>7s}")
    for T in [500, 1000, 2000, 5000, 10000]:
        stream = DecreasingJumpStream(d=50, r=4, rho_s=0.0,
                                       jump_size=2.0, noise_std=0.1, seed=42)
        stream.set_horizon(T)
        U_true = stream.get_optimal_subspace()
        single = OnlineLowRankLearner(50, 4, lr=1.0, U_init=U_true.clone())
        dual = DualTimescaleLearner(50, 4, lr_slow=0.0,
                                    consolidation_period=100000,
                                    U_init=U_true.clone(), window_size=30)
        res = run_experiment(stream, {"single": single, "dual": dual}, T=T)
        K_T = int(math.sqrt(T))
        s, dv = res["single"][-1], res["dual"][-1]
        print(f"  {T:>6d} {K_T:>5d} {s:>10.1f} {dv:>10.1f} {s/max(dv,1e-10):>7.2f}")

    # -----------------------------------------------------------------------
    # Full two-rate experiment with both subspace drift and coefficient jumps
    # -----------------------------------------------------------------------
    print("\n=== Full two-rate (subspace drift + jumps, T scaling) ===")
    print(f"  {'T':>6s} {'single':>10s} {'dual':>10s} {'ratio':>7s}")
    for T in [1000, 2000, 5000, 10000]:
        stream = TwoRateDriftStream(d=50, r=4, rho_s=0.005, rho_f=0.1,
                                    jump_interval=40, jump_size=2.0,
                                    noise_std=0.1, seed=42)
        U0 = _init_subspace(50, 4, stream.get_optimal_subspace(), noise=0.1)
        single = OnlineLowRankLearner(50, 4, lr=1.0, U_init=U0.clone())
        dual = DualTimescaleLearner(50, 4, lr_slow=0.1,
                                    consolidation_period=1, fisher_weighted=False,
                                    U_init=U0.clone(), window_size=30)
        res = run_experiment(stream, {"single": single, "dual": dual}, T=T)
        s, dv = res["single"][-1], res["dual"][-1]
        print(f"  {T:>6d} {s:>10.1f} {dv:>10.1f} {s/max(dv,1e-10):>7.2f}")

    # -----------------------------------------------------------------------
    # Full learner comparison
    # -----------------------------------------------------------------------
    print("\n=== Full comparison (T=5000) ===")
    T = 5000
    stream = TwoRateDriftStream(d=50, r=4, rho_s=0.005, rho_f=0.1,
                                jump_interval=40, jump_size=2.0,
                                noise_std=0.1, seed=42)
    U0 = _init_subspace(50, 4, stream.get_optimal_subspace(), noise=0.1)
    learners = {
        "single_timescale": OnlineLowRankLearner(50, 4, lr=1.0, U_init=U0.clone()),
        "dual_timescale": DualTimescaleLearner(
            50, 4, lr_slow=0.1, consolidation_period=1,
            fisher_weighted=False, U_init=U0.clone(), window_size=30),
        "ema": EMALearner(50, 4, lr=1.0, beta=0.99, U_init=U0.clone()),
        "periodic_avg": PeriodicAveragingLearner(50, 4, lr=1.0,
                                                  avg_period=50, U_init=U0.clone()),
        "subspace_tracking": SubspaceTrackingLearner(50, 4, lr_oja=0.01, lr_coeff=1.0,
                                                      U_init=U0.clone()),
    }
    results = run_experiment(stream, learners, T=T)
    print(f"  {'Learner':<25s} {'Regret':>10s} {'Reg/T':>8s}")
    print(f"  {'-'*43}")
    for name, curve in sorted(results.items(), key=lambda kv: kv[1][-1]):
        f = curve[-1]
        print(f"  {name:<25s} {f:>10.1f} {f/T:>8.4f}")

    # -----------------------------------------------------------------------
    # Frequency sweep, selectivity, merge rules, phase diagram
    # -----------------------------------------------------------------------
    print("\n=== Frequency sweep (B) ===")
    stream = TwoRateDriftStream(d=50, r=4, rho_s=0.005, rho_f=0.1,
                                jump_interval=40, jump_size=2.0,
                                noise_std=0.1, seed=42)
    sweep = run_frequency_sweep([5, 10, 20, 50, 100, 200], stream, T=5000)
    for B, reg in zip(sweep["B_values"], sweep["final_regret"]):
        print(f"  B={B:>4d}: {reg:.1f}")

    print("\n=== Selectivity test ===")
    stream = TwoRateDriftStream(d=50, r=4, rho_s=0.005, rho_f=0.1,
                                jump_interval=40, jump_size=2.0,
                                noise_std=0.1, seed=42)
    sel = run_selectivity_test(stream, T=5000)
    for name, curve in sorted(sel.items(), key=lambda kv: kv[1][-1]):
        print(f"  {name:<20s}: {curve[-1]:.1f}")

    print("\n=== Merge rule test ===")
    stream = TwoRateDriftStream(d=50, r=4, rho_s=0.005, rho_f=0.1,
                                jump_interval=40, jump_size=2.0,
                                noise_std=0.1, seed=42)
    merge = run_merge_rule_test(stream, T=5000)
    for name, curve in sorted(merge.items(), key=lambda kv: kv[1][-1]):
        print(f"  {name:<20s}: {curve[-1]:.1f}")

    print("\n=== Phase diagram (T=2000) ===")
    phase = run_phase_diagram(
        rho_s_range=[0.001, 0.01, 0.05],
        rho_f_range=[0.02, 0.1, 0.5],
        T=2000, d=50, r=4,
    )
    print("  rho_s \\ rho_f", [f"{v:.3f}" for v in phase["rho_f_values"]])
    for i, rho_s in enumerate(phase["rho_s_values"]):
        ratios = [f"{r:.2f}" for r in phase["ratio"][i]]
        print(f"  {rho_s:.3f}       {ratios}")

    print("\nDone.")
