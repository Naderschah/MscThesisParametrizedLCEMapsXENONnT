"""
Author: Felix Semler

This file contains all layer redefinitions for usage of the trained LCE function 

There is a numpy version and a jax version available that mirror one another, for jax find this file with the _jax.py file ending

Almost all layers are implemeneted as function generators to allow static compillation of constants

Jax is implemented ot work on fully static array sizes so everything needs to be batched prior to input 

The structure is as follows:
- Input Layers (xy to input transformations) and normalization
- Dense Layer Implementations
- Special Trainable Layers (Radial LCE , I0, etc.)
- Likelihood Function
- Miscellaneous helpers - batching, model generators, weigth loading, test cases, etc.


Note:
- Typing is not implemented for compilation only for legibility

"""
# --------------------------------------------------------------------------------
# --------------------------------- Imports --------------------------------------
# --------------------------------------------------------------------------------

import numpy as np
import scipy 
from typing import Callable, Tuple, Sequence, Optional
from collections import defaultdict
import json

# --------------------------------------------------------------------------------
# --------------------------------- Input Layers ---------------------------------
# --------------------------------------------------------------------------------

def get_input_functions(
            pmt_pos_top, 
            tpc_r = 66.4, 
            wall_dist_tolerance = 0.1, 
            wall_dist_step = 0.1, 
            pmt_r = 7.62 / 2, 
            ) -> Tuple[
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray]
            ]:
    """
    Generates the functions used for model inputs - just generates all at once
    Note all are assumed to be fully statically compiled (X is always the same size)

    pmt_pos_top: [n_pmts, 2] array of (used) PMT positions
    tpc_r : radius of the TPC
    """
    n_pmts = pmt_pos_top.shape[0]
    pmt_pos_reshape = np.reshape(pmt_pos_top, (1, n_pmts, 2))

    compute_radial_distance = direct_detection_input(pmt_pos_top)
    compute_event_wall_dist = wall_reflection_input(tpc_r, wall_dist_tolerance = wall_dist_tolerance, wall_dist_step = wall_dist_step)
    compute_n_perp_wires_in_way = make_perpendicular_wire_input(pmt_pos_top, pmt_r)
    compute_anode_mesh_input = make_anode_mesh_input(pmt_pos_top, pmt_r)

    return compute_radial_distance, compute_event_wall_dist, compute_n_perp_wires_in_way, compute_anode_mesh_input

def direct_detection_input(pmt_pos_top):
    """
    pmt_pos_top: [n_pmts, 2] array of (used) PMT positions
    """
    n_pmts = pmt_pos_top.shape[0]
    pmt_pos_reshape = np.reshape(pmt_pos_top, (1, n_pmts, 2))

    def compute_radial_distance(X):
        """
        X: [N, 2] array of coordinates

        Outputs : [N, n_pmts, 1] array of distances

        Computes the event to PMT distance for each PMT 
        """
        X_reshaped = np.reshape(X, (X.shape[0], 1, 2))
        distances = np.linalg.norm(pmt_pos_reshape - X_reshaped, axis=2, keepdims=True)
        return distances
    return compute_radial_distance
    
def wall_reflection_input(pmt_pos_top, 
                          tpc_r,
                          wall_dist_tolerance=0.1, 
                          wall_dist_step=0.1, 
                          max_iter=10
                          ) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function compute_event_wall_dist(X) that takes [N, 2] input coordinates
    and returns [N, n_pmts, 2] distances for event→wall and wall→PMT.
    All constants are statically compiled.
    """
    n_pmts = pmt_pos_top.shape[0]  # static
    def compute_event_wall_dist(X):
        # X: [N, 2]
        batch_size = X.shape[0]  # assumed static (autobatched externally)

        event_exp = np.repeat(X, repeats=n_pmts, axis=0)               # [N * n_pmts, 2]
        pmts_exp = np.tile(pmt_pos_top, (batch_size, 1))               # [N * n_pmts, 2]

        theta_event = np.arctan2(event_exp[:, 1], event_exp[:, 0])     # [N * n_pmts]
        theta_pmt = np.arctan2(pmts_exp[:, 1], pmts_exp[:, 0])         # [N * n_pmts]
        theta = np.arctan2(np.sin(theta_event + theta_pmt),
                            np.cos(theta_event + theta_pmt)) / 2.0     # [N * n_pmts]

        def compute_loss(th):
            W = tpc_r * np.stack([np.cos(th), np.sin(th)], axis=1)   # [N * n_pmts, 2]
            d_event = np.linalg.norm(W - event_exp, axis=1)            # [N * n_pmts]
            d_pmt = np.linalg.norm(W - pmts_exp, axis=1)               # [N * n_pmts]
            return d_event + d_pmt                                      # [N * n_pmts]

        def body_fn(carry):
            i, theta, step = carry
            loss = compute_loss(theta)
            loss_plus = compute_loss(theta + step)
            loss_minus = compute_loss(theta - step)

            better_plus = loss_plus < loss
            better_minus = loss_minus < loss

            move = np.where(better_plus, step,
                             np.where(better_minus, -step, 0.0))       # [N * n_pmts]
            theta_new = theta + move
            step_new = np.where(np.any(move != 0.0), step, step * 0.5)
            return (i + 1, theta_new, step_new), None

        def cond_fn(carry):
            i, _, step = carry
            return (i < max_iter) & (step > wall_dist_tolerance)

        init_carry = (0, theta, wall_dist_step)
        #(final_i, theta_final, _), _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        carry = init_carry
        while cond_fn(*carry):
            carry, _ = body_fn(carry)
        final_i, theta_final, _ = carry

        W_final = tpc_r * np.stack([np.cos(theta_final), np.sin(theta_final)], axis=1)  # [N * n_pmts, 2]
        d_event = np.linalg.norm(W_final - event_exp, axis=1)  # [N * n_pmts]
        d_pmt = np.linalg.norm(W_final - pmts_exp, axis=1)     # [N * n_pmts]

        result = np.stack([d_event, d_pmt], axis=1)            # [N * n_pmts, 2]
        return result.reshape((batch_size, n_pmts, 2))          # [N, n_pmts, 2]

    return compute_event_wall_dist

def make_perpendicular_wire_input(pmt_pos, pmt_r) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function f(events: [N, 2]) -> [N, n_pmts, 1]
    representing the total fraction of wire obstructing each event→PMT path.
    """
    angle = np.float32( -(np.pi / 3) + (np.pi / 2) ) # scalar
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]], dtype=np.float32)  # [2, 2]
    pmt_rot = np.asarray(pmt_pos, dtype=np.float32) @ rot_matrix                              # [n_pmts, 2]

    # Constants
    h = np.float32(0.027)
    H = np.float32(68.58 / 10)
    anode_r = np.float32(0.0304 / 2)
    dz = h + H
    _pmt_r = np.float32(pmt_r)

    wire_start = np.reshape(np.array([31.8, -31.8, 28.3, -28.3]) - anode_r, (1, 4), dtype=np.float32)  # [1, 4]
    wire_end   = np.reshape(np.array([31.8, -31.8, 28.3, -28.3]) + anode_r, (1, 4), dtype=np.float32)  # [1, 4]

    def compute_n_perp_wires_in_way(events):
        # events: [N, 2]
        events_rot = events @ rot_matrix                     # [N, 2]

        grad_minus = np.abs((pmt_rot[:, 0] - _pmt_r) - events_rot[:, 0, None]) / dz  # [N, n_pmts]
        grad_plus  = np.abs((pmt_rot[:, 0] + _pmt_r) - events_rot[:, 0, None]) / dz  # [N, n_pmts]

        bounds_minus = events_rot[:, 0, None] - grad_minus * h  # [N, n_pmts]
        bounds_plus  = events_rot[:, 0, None] + grad_plus  * h  # [N, n_pmts]

        start_overlap = np.maximum(bounds_minus[..., None], wire_start)  # [N, n_pmts, 4]
        end_overlap   = np.minimum(bounds_plus[..., None], wire_end)     # [N, n_pmts, 4]
        overlap       = np.maximum(0.0, end_overlap - start_overlap)     # [N, n_pmts, 4]

        return np.sum(overlap, axis=-1, keepdims=True) / (2 * anode_r)   # [N, n_pmts, 1]

    return compute_n_perp_wires_in_way


def make_anode_mesh_input(pmt_pos, pmt_r) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns compute_anode_mesh_input(events: [N, 2]) -> [N, n_pmts, 2],
    with outputs = [wire obstruction fraction, normalized angle to mesh]
    """
    angle = np.float32(-np.pi / 3)  # mesh angle
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ], dtype=np.float32).T  # [2, 2]

    pmt_pos = np.asarray(pmt_pos, dtype=np.float32)         # [n_pmts, 2]
    pmt_rot = pmt_pos @ rot_matrix                            # [n_pmts, 2]

    pmt_r = np.asarray(pmt_r, dtype=np.float32)             # [n_pmts]
    anode_r = np.float32(0.216 / 2)
    h = np.float32(0.027)
    H = np.float32(68.58 / 10)
    dz = h + H
    wire_pitch = np.float32(0.5)

    def compute_anode_mesh_input(events):
        # events: [N, 2]
        events_rot = events @ rot_matrix                      # [N, 2]

        # Wire obstruction component
        diff = np.abs(2 * (pmt_rot[:, 0] - events_rot[:, 0, None]) * h / dz) / wire_pitch  # [N, n_pmts]
        n_wires = diff[..., None] / np.float32(0.025)                                    # [N, n_pmts, 1]

        # Angle component
        event_dir = np.arctan2(events[:, 1], events[:, 0])                                # [N]
        relative_angle = np.mod(event_dir - angle, np.pi)                                # [N]
        folded = np.where(relative_angle > (np.pi / 2),
                           np.pi - relative_angle,
                           relative_angle)                                                 # [N]
        norm_angle = (folded / (np.pi / 2)).reshape(-1, 1, 1)                             # [N, 1, 1]
        norm_angle = np.repeat(norm_angle, pmt_pos.shape[0], axis=1)                     # [N, n_pmts, 1]

        return np.concatenate([n_wires, norm_angle], axis=-1)                             # [N, n_pmts, 2]

    return compute_anode_mesh_input

def make_normalization_layer() -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function that normalizes input X across axis=1 (per row sum),
    with stability clamp to avoid division by zero.
    """
    eps = np.float32(1e-7)

    def normalize(X: np.ndarray) -> np.ndarray:
        norm = np.maximum(np.sum(X, axis=1, keepdims=True), eps)  # [batch_size, 1]
        return X / norm

    return normalize

# --------------------------------------------------------------------------------
# ------------------- Dense Layer Implementation ---------------------------------
# --------------------------------------------------------------------------------
def make_dense_layer(
    kernel: np.ndarray,     # [input_dim, units]
    bias: np.ndarray,       # [units]
    activation_fn: Callable[[np.ndarray], np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Applies a monotonic dense layer independently on the last axis of input.

    Inputs:
        kernel: [D, U]
        bias:   [U]
        X:      [N, n_pmts, D]

    Output:
        [N, n_pmts, U]
    """
    def layer(X: np.ndarray) -> np.ndarray:
        # To ensure it operates on the last dimension only 
        h = np.einsum("...d,du->...u", X, kernel) + bias  # [N, n_pmts, U]
        return activation_fn(h)

    return layer

def make_mono_activations(base_activation: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a hardcoded activation function:
    output = (7 * f(x) + 7 * g(x) + 2 * h(x)) / 16
    where:
        f(x) = convex
        g(x) = concave (as -f(-x))
        h(x) = saturated
    """
    w_convex = np.float32(7.0)
    w_concave = np.float32(7.0)
    w_saturated = np.float32(2.0)
    total = np.float32(16.0)

    base_activation = base_activation.lower()

    def np_sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def np_relu(x):
        return np.maximum(x, 0.0)

    if base_activation == 'tanh':
        f = np.tanh
        g = lambda x: -np.tanh(-x)
        h = np_sigmoid

    elif base_activation == 'exponential':
        f = np.exp
        g = lambda x: -np.exp(-x)
        h = np_sigmoid

    elif base_activation == 'relu':
        f = np_relu
        g = lambda x: -np_relu(-x)
        h = np_sigmoid

    elif base_activation == 'sigmoid':
        f = np_sigmoid
        g = lambda x: -np_sigmoid(-x)
        h = np_sigmoid

    else:
        raise ValueError(f"Unsupported base activation: {base_activation}")

    def activation_fn(x: np.ndarray) -> np.ndarray:
        return (
            w_convex * f(x) +
            w_concave * g(x) +
            w_saturated * h(x)
        ) / total

    return activation_fn


# --------------------------------------------------------------------------------
# ------------------------------ Special Layers  ---------------------------------
# --------------------------------------------------------------------------------
def make_I0_layer(i0: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Constructs a function that multiplies input X by per-PMT gain factors `i0`.

    Args:
        i0: array of shape [n_pmts], must be non-negative (enforced externally)

    Returns:
        Function X -> X * i0, with X of shape [batch_size, n_pmts]
    """
    i0 = np.asarray(i0, dtype=np.float32)  # [n_pmts]

    def apply_i0(X: np.ndarray) -> np.ndarray:
        return X * i0  # broadcasting over batch dim

    return apply_i0

def make_radial_lce_layer(params: Sequence[float]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Constructs a radial LCE layer function:
    Inputs:
        params: [p, d, a, b] — scalar coefficients
    Returns:
        Function X -> LCE(X) of shape [batch_size, n_pmts]
    """
    p, d, a, b = map(np.float32, params)  # ensure constants

    def radial_lce(X: np.ndarray) -> np.ndarray:
        # X: [batch_size, n_pmts], interpreted as rho
        return (
            (1.0 - b) / np.power(1.0 + np.square(X / d), p)
            + a * X
            + b
        )  # shape [batch_size, n_pmts]

    return radial_lce



# --------------------------------------------------------------------------------
# ---------------------------- Likelihood Function -------------------------------
# --------------------------------------------------------------------------------
# First the LUT table generator functions
# Then the exact likelihood function
# Then the LUT lookup functions
# And lastly a unified wrapper that fully configures the loss function

def make_lut_table_with_std(
        n_pe_domain: np.ndarray,
        n_ph_domain: np.ndarray,
        x_domain: np.ndarray,
        sigma_domain: np.ndarray,
        switching_signal: float,
        p_dpe: float,
        reduce_ph: bool = False,
        reduce_pe: bool = False
        ) -> np.ndarray:

    n_mu = int(switching_signal) + 1
    n_x = x_domain.shape[0]
    n_sigma = sigma_domain.shape[0]
    n_ph = n_ph_domain.shape[0]
    n_pe = n_pe_domain.shape[0]

    def compute_binomial_factors():
        k = n_pe_domain[None, :] - n_ph_domain[:, None]
        mask = (k >= 0) & (k <= n_ph_domain[:, None])
        k_safe = np.clip(k, 0, n_ph_domain[:, None])
        log_binom = scipy.special.gammaln(n_ph_domain[:, None] + 1) \
                  - scipy.special.gammaln(k_safe + 1) \
                  - scipy.special.gammaln(n_ph_domain[:, None] - k_safe + 1)
        log_prob = log_binom + k_safe * np.log(p_dpe) + (n_ph_domain[:, None] - k_safe) * np.log(1 - p_dpe)
        return np.where(mask, np.exp(log_prob), 0.0)  # shape [n_ph, n_pe]

    def compute_poisson_factors():
        mu_vals = np.arange(n_mu, dtype=np.float32)  # shape [n_mu]
        mu = mu_vals[:, None]  # shape [n_mu, 1]
        k = n_ph_domain[None, :]  # shape [1, n_ph]
        log_p = k * np.log(np.maximum(mu, 1e-8)) - mu - scipy.special.gammaln(k + 1.0)
        return np.where(mu > 0, np.exp(log_p), np.where(k == 0, 1.0, 0.0))  # shape [n_mu, n_ph]

    def compute_gaussian_factors():
        x = x_domain[None, :, None, None, None]  # [1, n_x, 1, 1, 1]
        sigma = sigma_domain[None, None, :, None, None]  # [1, 1, n_sigma, 1, 1]
        npe = n_pe_domain[None, None, None, None, :]  # [1, 1, 1, 1, n_pe]
        mean = npe
        std = sigma * np.sqrt(npe + 1e-12)
        norm = scipy.stats.norm.pdf(x, loc=mean, scale=std + 1e-12)
        return norm  # shape [1, n_x, n_sigma, 1, n_pe]

    def compute_chunked_lut(ph_slice, pe_slice):
        b = binom_factors[ph_slice][:, pe_slice]  # [n_ph_chunk, n_pe_chunk]
        c = pois_factors[:, ph_slice]  # [n_mu, n_ph_chunk]
        a = gauss_factors[..., pe_slice]  # [1, n_x, n_sigma, 1, n_pe_chunk]
        b_bcast = b[None, None, None, :, :]  # [1, 1, 1, n_ph_chunk, n_pe_chunk]
        c_bcast = c[:, None, None, :, None]  # [n_mu, 1, 1, n_ph_chunk, 1]
        return np.sum(a * b_bcast * c_bcast, axis=(-2, -1))  # [n_mu, n_x, n_sigma]

    binom_factors = compute_binomial_factors()  # [n_ph, n_pe]
    pois_factors = compute_poisson_factors()    # [n_mu, n_ph]
    gauss_factors = compute_gaussian_factors()  # [1, n_x, n_sigma, 1, n_pe]

    if reduce_ph or reduce_pe:
        result = np.zeros((n_mu, n_x, n_sigma), dtype=np.float32)
        ph_splits = np.array_split(np.arange(n_ph), 4 if reduce_ph else 1)
        pe_splits = np.array_split(np.arange(n_pe), 4 if reduce_pe else 1)
        for ph_slice in ph_splits:
            for pe_slice in pe_splits:
                result += compute_chunked_lut(ph_slice, pe_slice)
        return result
    else:
        return compute_chunked_lut(np.arange(n_ph), np.arange(n_pe))

def make_lut_table_fixed_std(
        n_pe_domain: np.ndarray,
        n_ph_domain: np.ndarray,
        x_domain: np.ndarray,
        stds: np.ndarray,
        switching_signal: float,
        p_dpe: float,
        reduce_ph: bool = False,
        reduce_pe: bool = False
        ) -> np.ndarray:
    n_mu = int(switching_signal) + 1
    n_x = x_domain.shape[0]
    n_sigma = stds.shape[0]
    n_ph = n_ph_domain.shape[0]
    n_pe = n_pe_domain.shape[0]

    def compute_binomial_factors():
        k = n_pe_domain[None, :] - n_ph_domain[:, None]
        mask = (k >= 0) & (k <= n_ph_domain[:, None])
        k_safe = np.clip(k, 0, n_ph_domain[:, None])
        log_binom = scipy.special.gammaln(n_ph_domain[:, None] + 1) \
                  - scipy.special.gammaln(k_safe + 1) \
                  - scipy.special.gammaln(n_ph_domain[:, None] - k_safe + 1)
        log_prob = log_binom + k_safe * np.log(p_dpe) + (n_ph_domain[:, None] - k_safe) * np.log(1 - p_dpe)
        return np.where(mask, np.exp(log_prob), 0.0)  # shape [n_ph, n_pe]

    def compute_poisson_factors():
        mu_vals = np.arange(n_mu, dtype=np.float32)  # shape [n_mu]
        mu = mu_vals[:, None]  # shape [n_mu, 1]
        k = n_ph_domain[None, :]  # shape [1, n_ph]
        log_p = k * np.log(np.maximum(mu, 1e-8)) - mu - scipy.special.gammaln(k + 1.0)
        return np.where(mu > 0, np.exp(log_p), np.where(k == 0, 1.0, 0.0))  # shape [n_mu, n_ph]

    def compute_gaussian_factors():
        x = x_domain[None, :, None, None, None]  # [1, n_x, 1, 1, 1]
        sigma = stds[None, None, :, None, None]  # [1, 1, n_sigma, 1, 1]
        npe = n_pe_domain[None, None, None, None, :]  # [1, 1, 1, 1, n_pe]
        mean = npe
        std = sigma * np.sqrt(npe + 1e-12)
        norm = scipy.stats.norm.pdf(x, loc=mean, scale=std + 1e-12)
        return norm  # shape [1, n_x, n_sigma, 1, n_pe]

    def compute_chunked_lut(ph_slice, pe_slice):
        b = binom_factors[ph_slice][:, pe_slice]  # [n_ph_chunk, n_pe_chunk]
        c = pois_factors[:, ph_slice]  # [n_mu, n_ph_chunk]
        a = gauss_factors[..., pe_slice]  # [1, n_x, n_sigma, 1, n_pe_chunk]
        b_bcast = b[None, None, None, :, :]  # [1, 1, 1, n_ph_chunk, n_pe_chunk]
        c_bcast = c[:, None, None, :, None]  # [n_mu, 1, 1, n_ph_chunk, 1]
        return np.sum(a * b_bcast * c_bcast, axis=(-2, -1))  # [n_mu, n_x, n_sigma]

    binom_factors = compute_binomial_factors()  # [n_ph, n_pe]
    pois_factors = compute_poisson_factors()    # [n_mu, n_ph]
    gauss_factors = compute_gaussian_factors()  # [1, n_x, n_sigma, 1, n_pe]

    if reduce_ph or reduce_pe:
        result = np.zeros((n_mu, n_x, n_sigma), dtype=np.float32)
        ph_splits = np.array_split(np.arange(n_ph), 4 if reduce_ph else 1)
        pe_splits = np.array_split(np.arange(n_pe), 4 if reduce_pe else 1)
        for ph_slice in ph_splits:
            for pe_slice in pe_splits:
                result += compute_chunked_lut(ph_slice, pe_slice)
        return result
    else:
        return compute_chunked_lut(np.arange(n_ph), np.arange(n_pe))


def make_exact_lr(
        n_pe_domain: np.ndarray,
        n_ph_domain: np.ndarray,
        p_dpe: float,
        switching_signal: float,
        nan_safe: bool,
        nan_safe_value: float
        ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def exact_likelihood(x, mu, std):
        npe_grid = n_pe_domain[None, :]  # [1, N_pe]
        nph_grid = n_ph_domain[:, None]  # [N_ph, 1]

        # Gaussian factor
        gaussian_std = std[:, None] * np.sqrt(npe_grid + 1e-12)  # [B, N_pe]
        gaussian_prob = scipy.stats.norm.pdf(x[:, None], loc=npe_grid, scale=gaussian_std + 1e-12)  # [B, N_pe]

        # Binomial factor
        k = npe_grid[None, :, None] - nph_grid.T[None, None, :]  # [1, N_pe, N_ph]
        k = np.clip(k, 0, nph_grid.T)
        log_binom = scipy.special.gammaln(nph_grid.T + 1) \
                    - scipy.special.gammaln(k + 1) \
                    - scipy.special.gammaln(nph_grid.T - k + 1)
        log_prob = log_binom + k * np.log(p_dpe) + (nph_grid.T - k) * np.log(1 - p_dpe)
        binom_prob = np.where((k >= 0) & (k <= nph_grid.T), np.exp(log_prob), 0.0)  # [1, N_pe, N_ph]

        # Poisson factor
        mu_exp = mu[:, None, None]
        log_p = nph_grid[None, None, :] * np.log(np.maximum(mu_exp, 1e-8)) \
                - mu_exp - scipy.special.gammaln(nph_grid[None, None, :] + 1.0)
        pois_prob = np.where(mu_exp > 0, np.exp(log_p), np.where(nph_grid[None, None, :] == 0, 1.0, 0.0))  # [B, 1, N_ph]

        result = gaussian_prob[:, :, None] * binom_prob[None, :, :] * pois_prob  # [B, N_pe, N_ph]
        return np.sum(result, axis=(1, 2))

    def gaussian_likelihood(x, mu, std):
        npe_mean = mu * (1 + p_dpe)
        npe_var = mu * (1 + p_dpe)**2 + mu * p_dpe * (1 - p_dpe) + np.abs(npe_mean) * std**2
        return -0.5 * ((x - npe_mean)**2 / (npe_var + 1e-6) + np.log(2 * np.pi * npe_var + 1e-6))

    def likelihood_fn(pred, observed, std):
        x   = observed.reshape(-1)
        mu  = pred.reshape(-1)
        std = std.reshape(-1)

        bmap   = (x > switching_signal) | (mu > switching_signal)
        result = np.full_like(x, nan_safe_value)

        def safe(mask, values):
            # if nan_safe, mask out bad entries; otherwise return values directly
            return np.where(nan_safe, np.where(mask, values, nan_safe_value), values)

        # Gaussian‐branch entries
        gauss_vals = -2 * (
            gaussian_likelihood(x[bmap], mu[bmap], std[bmap])
            - gaussian_likelihood(mu[bmap], mu[bmap], std[bmap])
        )
        result[bmap] = safe(bmap[bmap], gauss_vals)

        # Exact‐branch entries
        exact_vals = -2 * (
            np.log(exact_likelihood(x[~bmap], mu[~bmap], std[~bmap]) + 1e-10)
            - np.log(exact_likelihood(mu[~bmap], mu[~bmap], std[~bmap]) + 1e-10)
        )
        result[~bmap] = safe(~bmap[~bmap], exact_vals)

        return result.reshape(pred.shape)

    return likelihood_fn

def make_lut_trainable_std_lr(
        lut_table: np.ndarray,
        x_domain: np.ndarray,
        sigma_domain: np.ndarray,
        switching_signal: float,
        mle_estimator: Callable[[np.ndarray], np.ndarray],
        return_ratio: bool,
        nan_safe: bool,
        nan_safe_value: float
        ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def nearest_idx(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        diffs = np.abs(grid[None, :] - values[:, None])
        return np.argmin(diffs, axis=1)

    def lookup_likelihood(x, mu, std):
        mu_idx = np.clip(np.round(mu), 0, lut_table.shape[0] - 1).astype(np.int32)
        x_idx = nearest_idx(x_domain, x)
        sigma_idx = nearest_idx(sigma_domain, std)
        return lut_table[mu_idx, x_idx, sigma_idx]

    def loss_fn(pred, observed, std):
        x   = observed.reshape(-1)
        mu  = pred.reshape(-1)
        std = std.reshape(-1)

        bmap   = (x > switching_signal) | (mu > switching_signal)
        result = np.full_like(x, nan_safe_value)

        def safe(mask, values):
            return np.where(nan_safe, np.where(mask, values, nan_safe_value), values)

        # compute the raw values once
        if return_ratio:
            mle     = mle_estimator(x)
            log_l1  = np.log(lookup_likelihood(x,   mu,  std) + 1e-10)
            log_l2  = np.log(lookup_likelihood(x,   mle, std) + 1e-10)
            values  = -2 * (log_l1 - log_l2)
        else:
            values  = -np.log(lookup_likelihood(x, mu, std) + 1e-10)

        # assign only the entries where ~bmap
        masked_vals = safe(~bmap, values[~bmap])
        result[~bmap] = masked_vals

        return result.reshape(pred.shape)
    
    return loss_fn


def make_lut_fixed_std_lr(
        lut_table: np.ndarray,
        x_domain: np.ndarray,
        switching_signal: float,
        mle_estimator: Callable[[np.ndarray], np.ndarray],
        return_ratio: bool,
        nan_safe: bool,
        nan_safe_value: float
        ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    def nearest_idx(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
        diffs = np.abs(grid[None, :] - values[:, None])
        return np.argmin(diffs, axis=1)

    def lookup_likelihood(x, mu, sigma_idx):
        mu_idx = np.clip(np.round(mu), 0, lut_table.shape[0] - 1).astype(np.int32)
        x_idx = nearest_idx(x_domain, x)
        sigma_idx = np.clip(sigma_idx.astype(np.int32), 0, lut_table.shape[2] - 1)
        return lut_table[mu_idx, x_idx, sigma_idx]

    def loss_fn(pred, observed, std_idx):
        x      = observed.reshape(-1)
        mu     = pred.reshape(-1)
        idx    = std_idx.reshape(-1)

        bmap   = (x > switching_signal) | (mu > switching_signal)
        result = np.full_like(x, nan_safe_value)

        def safe(mask, values):
            return np.where(nan_safe, np.where(mask, values, nan_safe_value), values)

        # Compute raw values for every index
        if return_ratio:
            mle      = mle_estimator(x)
            log_l1   = np.log(lookup_likelihood(x,   mu,  idx) + 1e-10)
            log_l2   = np.log(lookup_likelihood(x,   mle, idx) + 1e-10)
            values   = -2 * (log_l1 - log_l2)
        else:
            values   = -np.log(lookup_likelihood(x, mu, idx) + 1e-10)

        # Apply NaN‐safety and mask‐assign only for the exact‐branch (~bmap)
        masked_vals = safe(~bmap, values[~bmap])
        result[~bmap] = masked_vals

        return result.reshape(pred.shape)

    return loss_fn

def make_likelihood_fn(
        mode: str,
        return_ratio: bool = False,
        nan_safe: bool = True,
        nan_safe_value: float = 1e5,
        switching_signal: float = 40.0,
        mle_estimator: Callable[[np.ndarray], np.ndarray] = lambda x: x / 1.2,
        std: np.ndarray = np.ones((246,), dtype=np.float32) * 0.5,
        lut_table: Optional[np.ndarray] = None,
        x_domain: Optional[np.ndarray] = None,
        sigma_domain: Optional[np.ndarray] = None,
        p_dpe: Optional[float] = 0.2,
        m: Optional[int] = 5,
        z: Optional[int] = 20
        ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    # Compute domains from switching signal
    max_x = int(np.ceil(switching_signal))
    n_pe_domain = np.arange(0.0, switching_signal + 5 * np.sqrt(switching_signal) + 2)
    n_ph_domain = np.arange(0.0, switching_signal / (1 + p_dpe) + 5 * np.sqrt(switching_signal / (1 + p_dpe)) + 2)

    if mode == "exact":
        assert p_dpe is not None
        loss_fn = make_exact_lr(
            n_pe_domain=n_pe_domain,
            n_ph_domain=n_ph_domain,
            p_dpe=p_dpe,
            switching_signal=switching_signal,
            nan_safe=nan_safe,
            nan_safe_value=nan_safe_value,
        )
    elif mode == "LUT_trainable_std":
        assert lut_table is not None and x_domain is not None and sigma_domain is not None
        loss_fn = make_lut_trainable_std_lr(
            lut_table=lut_table,
            x_domain=x_domain,
            sigma_domain=sigma_domain,
            switching_signal=switching_signal,
            mle_estimator=mle_estimator,
            return_ratio=return_ratio,
            nan_safe=nan_safe,
            nan_safe_value=nan_safe_value,
        )
    elif mode == "LUT_fixed_std":
        assert lut_table is not None and x_domain is not None
        loss_fn = make_lut_fixed_std_lr(
            lut_table=lut_table,
            x_domain=x_domain,
            switching_signal=switching_signal,
            mle_estimator=mle_estimator,
            return_ratio=return_ratio,
            nan_safe=nan_safe,
            nan_safe_value=nan_safe_value,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}', must be one of 'exact', 'LUT_trainable_std', 'LUT_fixed_std'")

    def wrapped_fn(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        std_tiled = (
            np.tile(std[None, :], (batch_size, 1)) if mode != "LUT_fixed_std"
            else np.tile(np.arange(std.shape[0])[None, :], (batch_size, 1))
        )
        return loss_fn(mu, x, std_tiled)

    return wrapped_fn



# --------------------------------------------------------------------------------
# ---------------------------- Miscellaneous -------------------------------------
# --------------------------------------------------------------------------------

# Start with loading weights
def load_weights_dict(path: str) -> dict[str, dict[str, np.ndarray]]:
    """
    Reads your JSON weight file and returns a nested dict:
      { layer_name: { var_name: array, ... }, ... }
    """
    with open(path) as f:
        raw = json.load(f)

    by_layer = defaultdict(dict)
    for full_key, val in raw.items():
        # full_key looks like  "LayerName/var_name:0"
        layer, var = full_key.split("/", 1)
        var = var.split(":", 1)[0]                # drop the ":0"
        arr = np.array(val, dtype=np.float32)   # convert list → array
        by_layer[layer][var] = arr

    return by_layer


# Generate the full fully configured model 
def gen_np_model(
            pmt_pos_top: np.ndarray,
            weights_path: str,
            *, # Enforces keyword only after this point
            include_wall: bool = False,
            include_perp: bool = False,
            include_anode: bool = False,
            multiplication_layers: bool = False,
            radialLCE: bool = False
            ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    # Load weights from JSON
    weights = load_weights_dict(weights_path)

    # Unpack input functions
    (
        compute_radial_distance,
        compute_event_wall_dist,
        compute_n_perp_wires_in_way,
        compute_anode_mesh_input,
    ) = get_input_functions(
        pmt_pos_top,
        tpc_r=66.4,
        wall_dist_tolerance=0.1,
        wall_dist_step=0.1,
        pmt_r=7.62 / 2,
    )

    # Branch: Direct detect vs. Radial LCE
    if radialLCE:
        grp = weights['radial_lce']['lce_params'][0]
        radial_lce = make_radial_lce_layer(tuple(grp))
        def x_branch(xy):
            d = compute_radial_distance(xy)
            return radial_lce(d)
    else:
        def build_dense(name: str, act: str):
            k = weights[name]['kernel']
            b = weights[name]['bias']
            return make_dense_layer(
                kernel=k, bias=b,
                activation_fn=make_mono_activations(act)
            )
        dd1 = build_dense('PMT_dist_dense_1', 'tanh')
        dd2 = build_dense('PMT_dist_dense_2', 'tanh')
        dd3 = build_dense('PMT_dist_dense_3', 'exponential')
        def x_branch(xy):
            d = compute_radial_distance(xy)
            x = d[..., None]
            x = dd1(x); x = dd2(x); x = dd3(x)
            return x[..., 0]

    # Optional contributions
    if include_wall:
        wd1 = build_dense('PMT_wall_dense_1', 'tanh')
        wd2 = build_dense('PMT_wall_dense_2', 'tanh')
        wd3 = build_dense('PMT_wall_dense_3', 'exponential')
    if include_perp:
        pd1 = build_dense('perp_dense_1', 'relu')
        pd2 = build_dense('perp_dense_2', 'relu')
        pd3 = build_dense('perp_dense_3', 'sigmoid')
    if include_anode:
        ad1 = build_dense('anode_dense_1', 'relu')
        ad2 = build_dense('anode_dense_2', 'relu')
        ad3 = build_dense('anode_dense_3', 'sigmoid')

    # I0 & normalization
    i0_arr = weights['i0']['I_0']
    apply_i0 = make_I0_layer(i0_arr)
    normalize = make_normalization_layer()

    # Likelihood (exact mode)
    stds = weights['LossLayer']['GaussianStandardDeviation']
    lut = weights.get('LossLayer/L_table')
    x_dom = weights.get('LossLayer/x_domain')
    s_dom = weights.get('LossLayer/sigma_domain')
    loss_fn = make_likelihood_fn(
        mode='LUT_fixed_std', return_ratio=True,
        nan_safe=False, nan_safe_value=1e5,
        switching_signal=40.0,
        mle_estimator=lambda x: x/1.2,
        std=stds, lut_table=lut,
        x_domain=x_dom, sigma_domain=s_dom,
        p_dpe=0.2
    )

    def np_model(xy: np.ndarray, obs: np.ndarray) -> np.ndarray:
        out = x_branch(xy)
        if include_wall:
            w = compute_event_wall_dist(xy)
            w = wd1(w); w = wd2(w); w = wd3(w)
            out = out + w[..., 0]
        if include_perp:
            p = compute_n_perp_wires_in_way(xy)
            p = pd1(p); p = pd2(p); p = pd3(p)
            out = (out * p[..., 0]) if multiplication_layers else (out + p[..., 0])
        if include_anode:
            a = compute_anode_mesh_input(xy)
            a = ad1(a); a = ad2(a); a = ad3(a)
            out = (out * a[..., 0]) if multiplication_layers else (out + a[..., 0])
        out = apply_i0(out)
        out = normalize(out)
        return loss_fn(out, obs)

    return np_model
