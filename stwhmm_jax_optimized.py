"""JAX implementation of the Spatiotemporal Warped Hidden Markov Model (STWHMM).

Transition convention (preserved from the original notebook):
    T[next_z, current_z]
    H[next_tau, current_tau]
so T and H are column-stochastic.

Joint transition:
    P(i,j | l,w) = H[j,w] * ((1-s[w]) I[i,l] + s[w] T[i,l])

The numerically intensive operations are implemented in JAX and JIT compiled:
Poisson emissions, master-transition construction, forward/backward recursions,
posterior sufficient statistics, M-step calculations, scoring, and Viterbi.
Python is retained for the estimator API, validation, convergence control, and
verbose printing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# Enable 64-bit before importing jax.numpy.  The original NumPy/SciPy model
# uses float64; this substantially improves long-sequence log-space checks.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import gammaln, logsumexp

Array = jax.Array


@dataclass
class EMResult:
    """Summary and parameter snapshot returned by :meth:`STWHMM.fit`."""
    log_likelihoods: np.ndarray
    losses: np.ndarray
    converged: bool
    n_iter: int
    log_T: np.ndarray
    log_H: np.ndarray
    log_Lambda: np.ndarray
    log_pi: np.ndarray


@jax.jit
def _build_log_warped(log_T: Array, s: Array) -> Array:
    """log A[w,i,l], where A_w=(1-s[w])I+s[w]T."""
    M = log_T.shape[0]
    log_I = jnp.where(jnp.eye(M, dtype=bool), 0.0, -jnp.inf)
    log_s = jnp.where(s > 0.0, jnp.log(s), -jnp.inf)
    log_1ms = jnp.where(s < 1.0, jnp.log1p(-s), -jnp.inf)
    return jnp.logaddexp(
        log_1ms[:, None, None] + log_I[None, :, :],
        log_s[:, None, None] + log_T[None, :, :],
    )


def _build_log_master_tensor(log_T: Array, log_H: Array, s: Array) -> Array:
    """Compatibility helper; inference does not use this 4-D tensor."""
    log_A = _build_log_warped(log_T, s)  # w,i,l
    return log_A.transpose(1, 2, 0)[:, None, :, :] + log_H[None, :, None, :]


@jax.jit
def _compute_log_emissions(Y: Array, log_Lambda: Array) -> Array:
    """log p(Y[t] | z=m), shape (T,M), for independent Poisson counts."""
    y = Y[:, None, :]
    lr = log_Lambda[None, :, :]
    return jnp.sum(y * lr - jnp.exp(lr) - gammaln(y + 1.0), axis=-1)


@jax.jit
def _forward_factorized(log_B: Array, log_A: Array, log_H: Array, log_pi: Array) -> Array:
    """Exact factorized forward recursion; output (M,S,T)."""
    alpha0 = log_pi + log_B[0, :, None]
    def step(prev, b_t):
        # q[w,i] = logsumexp_l A[w,i,l] + prev[l,w]
        q = logsumexp(log_A + prev.T[:, None, :], axis=2)
        # cur[i,j] = b[i] + logsumexp_w H[j,w] + q[w,i]
        cur = b_t[:, None] + logsumexp(log_H[None, :, :] + q.T[:, None, :], axis=2)
        return cur, cur
    _, rest = lax.scan(step, alpha0, log_B[1:])
    return jnp.concatenate((alpha0[None], rest), axis=0).transpose(1, 2, 0)


@jax.jit
def _backward_factorized(log_B: Array, log_A: Array, log_H: Array) -> Array:
    """Exact factorized backward recursion; output (M,S,T)."""
    M, S = log_B.shape[1], log_H.shape[0]
    beta_last = jnp.zeros((M, S), dtype=log_B.dtype)
    def step(beta_next, b_next):
        # r[i,w] = logsumexp_j H[j,w] + b[i] + beta_next[i,j]
        r = b_next[:, None] + logsumexp(log_H[None, :, :] + beta_next[:, :, None], axis=1)
        # beta[l,w] = logsumexp_i A[w,i,l] + r[i,w]
        beta = logsumexp(log_A + r.T[:, :, None], axis=1).T
        return beta, beta
    _, rev = lax.scan(step, beta_last, log_B[1:][::-1])
    return jnp.concatenate((rev[::-1], beta_last[None]), axis=0).transpose(1, 2, 0)


@jax.jit
def _log_gamma(log_alpha: Array, log_beta: Array) -> Array:
    g = log_alpha + log_beta
    return g - logsumexp(g, axis=(0, 1), keepdims=True)


@jax.jit
def _transition_statistics_factorized(log_B, log_alpha, log_beta, log_A, log_H, ll, T_mask, H_mask):
    """Exact marginalized transition counts without a (M,S,M,S) zeta tensor."""
    M, S = log_alpha.shape[:2]
    init_T = jnp.full((M, M), -jnp.inf, dtype=log_B.dtype)
    init_H = jnp.full((S, S), -jnp.inf, dtype=log_B.dtype)
    init_err = jnp.array(0.0, dtype=log_B.dtype)
    xs = (log_alpha[:, :, :-1].transpose(2,0,1), log_B[1:], log_beta[:, :, 1:].transpose(2,0,1))
    def step(carry, x):
        tc, hc, max_err = carry
        a, b, bn = x
        # future[i,w] marginalizes next tau j.
        future = b[:, None] + logsumexp(log_H[None,:,:] + bn[:,:,None], axis=1)  # i,w
        # z-count[i,l] = sum_w,j zeta[i,j,l,w]
        z_by_w = a.T[:,None,:] + log_A + future.T[:,:,None] - ll  # w,i,l
        zc = logsumexp(z_by_w, axis=0)
        # incoming[w,i] marginalizes previous z l.
        incoming = logsumexp(log_A + a.T[:,None,:], axis=2)  # w,i
        # h-count[j,w] = H[j,w] * sum_i incoming[w,i] B[i] beta[i,j]
        hc_step = log_H + logsumexp(incoming.T[:,None,:] + b[:,None,None] + bn[:,:,None], axis=0)
        # Normalization from either marginalized count matrix.
        norm = logsumexp(zc)
        zc = jnp.where(T_mask, zc, -jnp.inf)
        hc_step = jnp.where(H_mask, hc_step, -jnp.inf)
        return (jnp.logaddexp(tc, zc), jnp.logaddexp(hc, hc_step), jnp.maximum(max_err, jnp.abs(norm))), None
    (tc,hc,err), _ = lax.scan(step, (init_T,init_H,init_err), xs)
    return tc,hc,err


@jax.jit
def _m_step_arrays(Y, log_alpha, log_beta, log_T_counts, log_H_counts, T_mask, H_mask, clone_membership, min_rate):
    log_T = log_T_counts - logsumexp(log_T_counts, axis=0, keepdims=True)
    log_T = jnp.where(T_mask, log_T, -jnp.inf)
    log_H = log_H_counts - logsumexp(log_H_counts, axis=0, keepdims=True)
    log_H = jnp.where(H_mask, log_H, -jnp.inf)
    gamma = _log_gamma(log_alpha, log_beta)
    gamma_z = jnp.exp(logsumexp(gamma, axis=1))
    group_gamma = clone_membership @ gamma_z
    denom = jnp.sum(group_gamma, axis=1)
    group_rates = (group_gamma @ Y) / denom[:,None]
    Lambda = clone_membership.T @ group_rates
    return log_T, log_H, jnp.log(jnp.maximum(Lambda, min_rate)), gamma, denom


@jax.jit
def _e_step_and_stats(Y, log_T, log_H, log_Lambda, log_pi, s, T_mask, H_mask):
    log_A = _build_log_warped(log_T, s)
    log_B = _compute_log_emissions(Y, log_Lambda)
    alpha = _forward_factorized(log_B, log_A, log_H, log_pi)
    beta = _backward_factorized(log_B, log_A, log_H)
    ll = logsumexp(alpha[:,:,-1])
    tc,hc,zeta_err = _transition_statistics_factorized(log_B, alpha, beta, log_A, log_H, ll, T_mask, H_mask)
    # Check a sparse set of time points: absolute FB discrepancies accumulate with T,
    # so fit() applies a length-aware tolerance.
    fb_vals = logsumexp(alpha + beta, axis=(0,1)) - ll
    fb_err = jnp.max(jnp.abs(fb_vals))
    gamma = _log_gamma(alpha,beta)
    gamma_err = jnp.max(jnp.abs(logsumexp(gamma, axis=(0,1))))
    return log_A, log_B, alpha, beta, ll, tc, hc, zeta_err, fb_err, gamma_err


@jax.jit
def _score_kernel(Y, log_T, log_H, log_Lambda, log_pi, s):
    A = _build_log_warped(log_T,s)
    B = _compute_log_emissions(Y,log_Lambda)
    a = _forward_factorized(B,A,log_H,log_pi)
    return logsumexp(a[:,:,-1])


@jax.jit
def _viterbi_kernel(Y, log_Lambda, log_T, log_H, log_pi, s):
    """Factorized max-product Viterbi with exact joint-state backpointers."""
    B = _compute_log_emissions(Y,log_Lambda)
    A = _build_log_warped(log_T,s)
    M,S = log_pi.shape
    delta0 = log_pi + B[0,:,None]
    def step(delta,b_t):
        # For each w,i choose best previous z l.
        scores_l = A + delta.T[:,None,:]  # w,i,l
        best_l = jnp.argmax(scores_l,axis=2)
        best_l_score = jnp.max(scores_l,axis=2)  # w,i
        # For each i,j choose best previous tau w.
        scores_w = log_H[None,:,:] + best_l_score.T[:,None,:]  # i,j,w
        best_w = jnp.argmax(scores_w,axis=2)  # i,j
        new_delta = b_t[:,None] + jnp.max(scores_w,axis=2)
        # Gather l corresponding to selected w for every i,j.
        i_idx = jnp.arange(M)[:,None]
        chosen_l = best_l[best_w, i_idx]
        pointer = chosen_l * S + best_w
        return new_delta, pointer.astype(jnp.int32)
    delta_final,pointers = lax.scan(step,delta0,B[1:])
    terminal = jnp.argmax(delta_final.reshape(-1)).astype(jnp.int32)
    Tobs=Y.shape[0]
    path=jnp.zeros((Tobs,),dtype=jnp.int32).at[-1].set(terminal)
    def body(k,path):
        t=(Tobs-2)-k
        nxt=path[t+1]; z=nxt//S; tau=nxt%S
        return path.at[t].set(pointers[t,z,tau])
    path=lax.fori_loop(0,Tobs-1,body,path)
    return jnp.stack((path//S,path%S),axis=0), jnp.max(delta_final)


class STWHMM:
    """JAX-accelerated STWHMM with the same public interface as the NumPy class."""

    def __init__(
        self,
        M: int,
        S: int,
        s: Optional[Sequence[float]] = None,
        sequence_lengths: Optional[Sequence[int]] = None,
        T_mask: Optional[np.ndarray] = None,
        H_mask: Optional[np.ndarray] = None,
        random_state=0,
        min_rate: float = 1e-12,
    ):
        self.M, self.S = int(M), int(S)
        if self.M < 3 or (self.M - 1) % 2:
            raise ValueError("M must be one hub plus an even number of clone states.")
        if self.S < 1:
            raise ValueError("S must be >= 1.")
        self.n_forward = (self.M - 1) // 2
        s_np = np.linspace(0.0, 1.0, self.S) if s is None else np.asarray(s, dtype=np.float64)
        if s_np.shape != (self.S,) or np.any((s_np < 0) | (s_np > 1)):
            raise ValueError(f"s must have shape ({self.S},) with values in [0,1].")
        self.s = jnp.asarray(s_np)
        self.min_rate = float(min_rate)
        if self.min_rate <= 0:
            raise ValueError("min_rate must be positive.")

        # JAX PRNG. A NumPy Generator is accepted only by drawing one seed from it.
        if isinstance(random_state, np.random.Generator):
            seed = int(random_state.integers(0, 2**31 - 1))
        elif random_state is None:
            seed = 0
        else:
            seed = int(random_state)
        self.key = jax.random.PRNGKey(seed)

        self.clone_groups = self.get_clone_groups(self.M)
        if T_mask is None:
            if sequence_lengths is None:
                sequence_lengths = [self.n_forward]
            self.sequence_lengths = tuple(int(x) for x in sequence_lengths)
            T_mask = self.build_T_mask(self.M, self.sequence_lengths)
        else:
            self.sequence_lengths = None if sequence_lengths is None else tuple(sequence_lengths)
            T_mask = np.asarray(T_mask, dtype=bool)
            if T_mask.shape != (self.M, self.M):
                raise ValueError(f"T_mask must have shape ({self.M},{self.M}).")
            self._validate_mask_columns(T_mask, "T_mask")

        if H_mask is None:
            H_mask = np.ones((self.S, self.S), dtype=bool)
        else:
            H_mask = np.asarray(H_mask, dtype=bool)
            if H_mask.shape != (self.S, self.S):
                raise ValueError(f"H_mask must have shape ({self.S},{self.S}).")
            self._validate_mask_columns(H_mask, "H_mask")

        self.T_mask = jnp.asarray(T_mask)
        self.H_mask = jnp.asarray(H_mask)

        membership = np.zeros((len(self.clone_groups), self.M), dtype=np.float64)
        for g, group in enumerate(self.clone_groups):
            membership[g, group] = 1.0
        self.clone_membership = jnp.asarray(membership)

        self.log_T_ = self.log_H_ = self.log_Lambda_ = self.log_pi_ = self.log_P_ = None
        self.n_features_in_ = None
        self.log_likelihood_history_ = []
        self.loss_history_ = []
        self.converged_ = False
        self.n_iter_ = 0

    @staticmethod
    def get_clone_groups(M: int):
        if (M - 1) % 2:
            raise ValueError("M must contain one hub plus an even number of sequence states.")
        n = (M - 1) // 2
        return [[0]] + [[m, m + n] for m in range(1, n + 1)]

    @staticmethod
    def _validate_mask_columns(mask, name):
        bad = np.flatnonzero(np.asarray(mask).sum(axis=0) == 0)
        if bad.size:
            raise ValueError(f"{name} has no allowed outgoing transition for columns {bad.tolist()}.")

    @classmethod
    def build_T_mask(cls, M: int, sequence_lengths: Sequence[int]):
        n = (M - 1) // 2
        lengths = [int(x) for x in sequence_lengths]
        if not lengths or any(x <= 0 for x in lengths) or sum(lengths) != n:
            raise ValueError(f"sequence_lengths must be positive and sum to {n}.")
        mask = np.zeros((M, M), dtype=bool)
        mask[1:, 0] = True
        mask[0, 1:] = True
        offset = 0
        for length in lengths:
            start, end = 1 + offset, offset + length
            for current in range(start, end):
                mask[current + 1, current] = True
            r_start, r_end = start + n, end + n
            for current in range(r_end, r_start, -1):
                mask[current - 1, current] = True
            offset += length
        cls._validate_mask_columns(mask, "T_mask")
        return mask

    @staticmethod
    def random_walk_H_mask(S: int, radius: int = 1, connect_state_zero: bool = True):
        idx = np.arange(S)
        mask = np.abs(idx[:, None] - idx[None, :]) <= int(radius)
        if connect_state_zero:
            mask[0, :] = True
            mask[:, 0] = True
        return mask

    def _next_keys(self, n):
        keys = jax.random.split(self.key, n + 1)
        self.key = keys[0]
        return keys[1:]

    def initialize(self, Y, emission_sigma: float = 0.25):
        Y = self._validate_Y(Y)
        Yj = jnp.asarray(Y, dtype=jnp.float64)
        self.n_features_in_ = Y.shape[1]
        kT, kH, kL, kpi = self._next_keys(4)

        def random_stochastic(mask, key):
            vals = jax.random.uniform(key, mask.shape, dtype=jnp.float64, minval=1e-12, maxval=1.0)
            A = jnp.where(mask, vals, 0.0)
            return A / jnp.sum(A, axis=0, keepdims=True)

        T = random_stochastic(self.T_mask, kT)
        H = random_stochastic(self.H_mask, kH)
        self.log_T_ = jnp.where(self.T_mask, jnp.log(T), -jnp.inf)
        self.log_H_ = jnp.where(self.H_mask, jnp.log(H), -jnp.inf)

        mean_rates = jnp.maximum(jnp.mean(Yj, axis=0), self.min_rate)
        group_noise = jax.random.normal(kL, (len(self.clone_groups), Y.shape[1]), dtype=jnp.float64)
        group_rates = mean_rates[None, :] * jnp.exp(emission_sigma * group_noise)
        Lambda = self.clone_membership.T @ group_rates
        self.log_Lambda_ = jnp.log(jnp.maximum(Lambda, self.min_rate))

        pi = jax.random.uniform(kpi, (self.M, self.S), dtype=jnp.float64, minval=1e-12, maxval=1.0)
        pi = pi / jnp.sum(pi)
        self.log_pi_ = jnp.log(pi)
        self.log_P_ = None  # full 4-D transition tensor intentionally not materialized
        self._check_parameters()
        return self

    def fit(
        self,
        Y,
        n_iter: int = 20,
        tol: Optional[float] = 1e-4,
        emission_sigma: float = 0.25,
        reinitialize: bool = True,
        update_pi: bool = False,
        check: bool = True,
        atol: float = 1e-6,
        verbose: bool = False,
    ) -> EMResult:
        Y_np = self._validate_Y(Y)
        Yj = jnp.asarray(Y_np, dtype=jnp.float64)
        if reinitialize or self.log_T_ is None:
            self.initialize(Y_np, emission_sigma)
        else:
            self._validate_Y(Y_np, check_features=True)

        self.log_likelihood_history_, self.loss_history_ = [], []
        self.converged_ = False
        previous = None

        for step in range(int(n_iter)):
            (self.log_P_, log_B, alpha, beta, ll, tc, hc,
             zeta_err, fb_err, gamma_err) = _e_step_and_stats(
                Yj, self.log_T_, self.log_H_, self.log_Lambda_, self.log_pi_,
                self.s, self.T_mask, self.H_mask
            )
            ll_f = float(ll)
            self.log_likelihood_history_.append(ll_f)
            self.loss_history_.append(-ll_f)

            if check:
                ze, fe, ge = float(zeta_err), float(fb_err), float(gamma_err)
                # Recursive log-space roundoff grows slowly with sequence length.
                # Keep atol as the floor, but scale the FB diagnostic for long recordings.
                fb_atol = max(float(atol), 1e-8 * np.sqrt(Y_np.shape[0]))
                if fe > fb_atol:
                    raise RuntimeError(f"Forward-backward likelihood mismatch at EM iteration {step}; max error={fe:.12e}, tolerance={fb_atol:.12e}.")
                if ze > atol:
                    raise RuntimeError(f"Zeta normalization failed at EM iteration {step}; max |logsumexp|={ze:.12e}.")
                if ge > atol:
                    raise RuntimeError(f"Gamma normalization failed at EM iteration {step}; max error={ge:.12e}.")

            log_T, log_H, log_Lambda, gamma, group_denoms = _m_step_arrays(
                Yj, alpha, beta, tc, hc, self.T_mask, self.H_mask,
                self.clone_membership, jnp.asarray(self.min_rate, dtype=jnp.float64)
            )
            if check:
                if np.any(np.asarray(group_denoms) <= 0):
                    raise RuntimeError("At least one clone group has zero posterior occupancy.")
                if np.any(~np.isfinite(np.asarray(logsumexp(tc, axis=0)))):
                    raise RuntimeError("At least one T* column has zero expected transition count.")
                if np.any(~np.isfinite(np.asarray(logsumexp(hc, axis=0)))):
                    raise RuntimeError("At least one H column has zero expected transition count.")

            self.log_T_, self.log_H_, self.log_Lambda_ = log_T, log_H, log_Lambda
            if update_pi:
                lp = gamma[:, :, 0]
                self.log_pi_ = lp - logsumexp(lp)
            self.log_P_ = None  # full 4-D transition tensor intentionally not materialized

            if check:
                self._check_parameters(atol)

            if verbose:
                delta = np.nan if previous is None else ll_f - previous
                print(f"EM {step + 1:3d}: loglik={ll_f:.6f}  delta={delta:.6g}")
            if previous is not None and tol is not None and abs(ll_f - previous) <= tol * (1.0 + abs(previous)):
                self.converged_ = True
                break
            previous = ll_f

        self.n_iter_ = len(self.log_likelihood_history_)
        return EMResult(
            log_likelihoods=np.asarray(self.log_likelihood_history_),
            losses=np.asarray(self.loss_history_),
            converged=self.converged_, n_iter=self.n_iter_,
            log_T=np.asarray(self.log_T_).copy(), log_H=np.asarray(self.log_H_).copy(),
            log_Lambda=np.asarray(self.log_Lambda_).copy(), log_pi=np.asarray(self.log_pi_).copy(),
        )

    def restore(self, result: EMResult):
        """Restore the parameter snapshot stored in an EMResult (useful for random restarts)."""
        self.log_T_ = jnp.asarray(result.log_T)
        self.log_H_ = jnp.asarray(result.log_H)
        self.log_Lambda_ = jnp.asarray(result.log_Lambda)
        self.log_pi_ = jnp.asarray(result.log_pi)
        self.log_P_ = None  # full 4-D transition tensor intentionally not materialized
        self.n_features_in_ = result.log_Lambda.shape[1]
        return self

    def compute_log_emissions(self, Y):
        self._require_initialized()
        Y = self._validate_Y(Y, check_features=True)
        return _compute_log_emissions(jnp.asarray(Y, dtype=jnp.float64), self.log_Lambda_)

    def compute_log_alpha(self, Y):
        B = self.compute_log_emissions(Y)
        return _forward_factorized(B, _build_log_warped(self.log_T_, self.s), self.log_H_, self.log_pi_)

    def compute_log_beta(self, Y):
        B = self.compute_log_emissions(Y)
        return _backward_factorized(B, _build_log_warped(self.log_T_, self.s), self.log_H_)

    def posterior(self, Y, marginalize_tau: bool = False, return_log: bool = False):
        B = self.compute_log_emissions(Y)
        A = _build_log_warped(self.log_T_, self.s)
        a, b = _forward_factorized(B, A, self.log_H_, self.log_pi_), _backward_factorized(B, A, self.log_H_)
        g = _log_gamma(a, b)
        if marginalize_tau:
            g = logsumexp(g, axis=1)  # (M,T)
        return g if return_log else jnp.exp(g)

    def score(self, Y) -> float:
        self._require_initialized()
        Y = self._validate_Y(Y, check_features=True)
        return float(_score_kernel(jnp.asarray(Y, dtype=jnp.float64), self.log_T_, self.log_H_, self.log_Lambda_, self.log_pi_, self.s))

    def viterbi(self, Y, return_log_probability: bool = False):
        self._require_initialized()
        Y = self._validate_Y(Y, check_features=True)
        states, lp = _viterbi_kernel(jnp.asarray(Y, dtype=jnp.float64), self.log_Lambda_, self.log_T_, self.log_H_, self.log_pi_, self.s)
        states_np = np.asarray(states)
        return (states_np, float(lp)) if return_log_probability else states_np

    predict = viterbi

    @property
    def T_(self):
        self._require_initialized(); return np.asarray(jnp.exp(self.log_T_))
    @property
    def H_(self):
        self._require_initialized(); return np.asarray(jnp.exp(self.log_H_))
    @property
    def Lambda_(self):
        self._require_initialized(); return np.asarray(jnp.exp(self.log_Lambda_))
    @property
    def pi_(self):
        self._require_initialized(); return np.asarray(jnp.exp(self.log_pi_))

    def get_params(self):
        self._require_initialized()
        return {"log_T": np.asarray(self.log_T_).copy(), "log_H": np.asarray(self.log_H_).copy(),
                "log_Lambda": np.asarray(self.log_Lambda_).copy(), "log_pi": np.asarray(self.log_pi_).copy()}

    def _validate_Y(self, Y, check_features=False):
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2 or min(Y.shape) < 1:
            raise ValueError("Y must be a non-empty (time, neurons/features) array.")
        if not np.all(np.isfinite(Y)) or np.any(Y < 0) or not np.allclose(Y, np.round(Y)):
            raise ValueError("Poisson observations must be finite, nonnegative integer-valued counts.")
        if check_features and self.n_features_in_ is not None and Y.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {Y.shape[1]}.")
        return Y

    def _require_initialized(self):
        if any(x is None for x in (self.log_T_, self.log_H_, self.log_Lambda_, self.log_pi_)):
            raise RuntimeError("Model parameters are not initialized. Call initialize() or fit() first.")

    def _check_parameters(self, atol=1e-8):
        self._require_initialized()
        T, H, pi = map(np.asarray, (jnp.exp(self.log_T_), jnp.exp(self.log_H_), jnp.exp(self.log_pi_)))
        if not np.allclose(T.sum(axis=0), 1.0, atol=atol): raise RuntimeError("T* normalization failed.")
        if not np.allclose(H.sum(axis=0), 1.0, atol=atol): raise RuntimeError("H normalization failed.")
        if not np.allclose(pi.sum(), 1.0, atol=atol): raise RuntimeError("pi normalization failed.")
        A = np.asarray(jnp.exp(_build_log_warped(self.log_T_, self.s)))
        if not np.allclose(A.sum(axis=1), 1.0, atol=atol): raise RuntimeError("Warped z-transition normalization failed.")
        L = np.asarray(self.log_Lambda_)
        for group in self.clone_groups:
            for m in group[1:]:
                if not np.allclose(L[m], L[group[0]], atol=1e-12, rtol=0):
                    raise RuntimeError(f"Shared-emission constraint failed for clone group {group}.")


# Explicit alias if both NumPy and JAX versions are imported in one notebook.
JAXSTWHMM = STWHMM
