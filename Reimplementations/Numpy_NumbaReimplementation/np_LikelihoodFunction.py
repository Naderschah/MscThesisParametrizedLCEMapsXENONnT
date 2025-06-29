
from . type_names import *

import numpy as np
from scipy.special import betaln 
from scipy.special import gammaln 

def gen_pe_ph_domain(
    switching_signal: float,
    n_sigma: int,
    p_dpe: float,
    float_type: Type[np.floating] = np.float32
) -> tuple[
    Annotated[np.ndarray, (None,), float],
    Annotated[np.ndarray, (None,), float]
]:
    """
    Generate the photon and photoelectron domain for the likelihood function (NumPy only).
    """
    n_pe_domain = np.arange(
        0.0,
        switching_signal + n_sigma * np.sqrt(switching_signal) + 2,
        dtype=float_type
    )
    n_ph_domain = np.arange(
        0.0,
        switching_signal / (1 + p_dpe) + n_sigma * np.sqrt(switching_signal / (1 + p_dpe)) + 2,
        dtype=float_type
    )
    return n_pe_domain, n_ph_domain


# --- exact_likelihood_generator (NumPy only) ---
def exact_likelihood_generator(
    n_pe_domain: Annotated[np.ndarray, (None,), float],
    n_ph_domain: Annotated[np.ndarray, (None,), float],
    p: float,
    float_type: Type[np.floating] = np.float32
) -> Callable:
    """
    Generate callable for the exact likelihood function (NumPy only).
    """
    # Gaussian helper
    def gaussian_log_likelihood(
        x: Annotated[np.ndarray, (None, 1, 1), float],
        loc: Annotated[np.ndarray, (1, 1, None), float],
        scale: Annotated[np.ndarray, (None, 1, None), float]
    ) -> Annotated[np.ndarray, (None, 1, None), float]:
        return (
            -float_type(0.5) * ((x - loc) / scale) ** float_type(2)
            - np.log(scale)
            - float_type(0.5) * np.log(float_type(2.0) * np.pi)
        )

    # Binomial helper using scipy
    from scipy.special import betaln as spbetaln, gammaln as spgammaln

    def get_valid_np(n, k, float_type):
        return ((k <= n) & (k >= 0.)).astype(dtype=float_type)

    def binomial_pmf(
        n: Annotated[np.ndarray, (1, None, 1), float],
        p: float,
        k: Annotated[np.ndarray, (1, None, None), float],
        float_type: Type[np.floating]
    ) -> Annotated[np.ndarray, (1, None, None), float]:
        k_safe = np.clip(k, 0, n)
        valid = get_valid_np(n, k, float_type)
        one_dtype = float_type(1.0)
        log_unnormalized = np.log(p) * k_safe + np.log1p(-p) * (n - k_safe)
        log_normalization = spbetaln(one_dtype + k_safe, one_dtype + n - k_safe) + np.log(n + one_dtype)
        return valid * np.exp(log_unnormalized - log_normalization)

    def gammaln_np(x: np.ndarray) -> np.ndarray:
        return spgammaln(x)

    def poisson_pmf(
        n: Annotated[np.ndarray, (1, None, 1), float],
        mu: Annotated[np.ndarray, (None, 1, 1), float],
        float_type: Type[np.floating] = np.float32
    ) -> Annotated[np.ndarray, (None, None, 1), float]:
        mask = (mu > float_type(0.0))
        valid = mask.astype(float_type)
        one_dtype = float_type(1.0)
        safe_mu = np.where(mask, mu, one_dtype)
        log_p = n * np.log(safe_mu) - mu - gammaln_np(n + one_dtype)
        return valid * np.exp(log_p)

    # Build the grids
    n_ph_grid: Annotated[np.ndarray, (1, None, 1), float] = np.reshape(n_ph_domain, (1, -1, 1))
    n_pe_grid: Annotated[np.ndarray, (1, 1, None), float] = np.reshape(n_pe_domain, (1, 1, -1))

    def exact_likelihood(
        x: Annotated[np.ndarray, (None,), float],
        mu: Annotated[np.ndarray, (None,), float],
        sigma: Annotated[np.ndarray, (None,), float]
    ) -> Annotated[np.ndarray, (None,), float]:
        x = np.reshape(x, (-1, 1, 1))
        mu = np.reshape(mu, (-1, 1, 1))
        sigma = np.reshape(sigma, (-1, 1, 1))

        a: Annotated[np.ndarray, (None, 1, None), float] = np.exp(gaussian_log_likelihood(x, n_pe_grid, np.maximum(sigma * np.sqrt(n_pe_grid), float_type(1e-10))))
        b: Annotated[np.ndarray, (1, None, None), float] = binomial_pmf(n_ph_grid, p, n_pe_grid - n_ph_grid, float_type)
        c: Annotated[np.ndarray, (None, None, 1), float] = poisson_pmf(n_ph_grid, mu)
        res: Annotated[np.ndarray, (None,), float] = np.sum(a * b * c, axis=(1, 2))
        return res

    return exact_likelihood


# --- exact_likelihood_MLE_generator (NumPy only) ---
def exact_likelihood_MLE_generator(
    usenumba: bool = False
):
    raise NotImplementedError("Exact likelihood MLE generator needs to be implemented here")


# --- generate_LUT (NumPy only) ---
def generate_LUT(
    m: int,
    p: float,
    switching_signal: float,
    gaussian_stds: Annotated[np.ndarray, (None,), float],
    n_sigma: int = 5,
    obs_min: float = -3.0,
    float_type: Type[np.floating] = np.float32
) -> tuple[
    Annotated[np.ndarray, (None, None, None), float],
    Annotated[np.ndarray, (None, None), float],
    Annotated[np.ndarray, (None,), float],
    Annotated[np.ndarray, (None,), float]
]:
    """
    Computes the Look up table from the exact method (NumPy only).
    """
    # Build x, mu and sigma domain
    i_vals = np.arange(obs_min, switching_signal, dtype=float_type)
    sub_vals = np.arange(float(m), dtype=float_type) / m
    x_domain = np.concatenate(
        [(i_vals[:, None] + sub_vals[None, :]).ravel(), np.array([switching_signal], dtype=float_type)]
    )
    nx = x_domain.shape[0]

    i_vals = np.arange(0.0, switching_signal, dtype=float_type)
    sub_vals = np.arange(float(m), dtype=float_type) / m
    mu_domain = np.concatenate(
        [(i_vals[:, None] + sub_vals[None, :]).ravel(), np.array([switching_signal], dtype=float_type)]
    )
    nmu = mu_domain.shape[0]

    sigma_domain = np.array(gaussian_stds, dtype=float_type)
    n_pmts = sigma_domain.shape[0]

    # Generate evaluation domain and exact likelihood function
    n_pe_domain, n_ph_domain = gen_pe_ph_domain(
        switching_signal=switching_signal,
        n_sigma=n_sigma,
        p_dpe=p,
        float_type=float_type
    )
    exact_likelihood_function = exact_likelihood_generator(
        n_pe_domain=n_pe_domain,
        n_ph_domain=n_ph_domain,
        p=p,
        float_type=float_type
    )

    # Generate empty LUT
    LUT = np.zeros((nmu, nx, n_pmts), dtype=float_type)
    for idx in range(nmu):
        x_grid = np.repeat(x_domain[:, None], n_pmts, axis=1).ravel()
        mu_grid = np.full_like(x_grid, mu_domain[idx])
        std_grid = np.tile(sigma_domain[None, :], (nx, 1)).ravel()
        probs = exact_likelihood_function(x_grid, mu_grid, std_grid)
        LUT[idx] = probs.reshape(nx, n_pmts)

    # Find per x MLE mu Likelihoods
    LUT_MLE = np.max(LUT, axis=0)
    return LUT, LUT_MLE, x_domain, mu_domain


# --- gen_nearest_index_1d (NumPy only) ---
def gen_nearest_index_1d(usenumba: bool = False) -> Callable[
    [Annotated[Array, (None,), float], Annotated[Array, (None,), float]],
    Annotated[Array, (None,), int]
]:
    """
    Pure‐NumPy “nearest‐index‐in‐1D” function factory.
    """
    def nearest_index_1d(
        grid: Annotated[Array, (None,), float],
        values: Annotated[Array, (None,), float],
    ) -> Annotated[Array, (None,), int]:
        idx_float = np.searchsorted(grid, values, side='left')
        idx_below = np.clip(idx_float - 1, 0, grid.size - 1)
        idx_above = np.clip(idx_float,     0, grid.size - 1)

        below_vals = grid[idx_below]
        above_vals = grid[idx_above]

        dist_below = np.abs(values - below_vals)
        dist_above = np.abs(values - above_vals)

        idx_nearest = np.where(dist_above < dist_below, idx_above, idx_below)
        return idx_nearest.astype(np.int32)

    return nearest_index_1d


# --- get_likelihood_from_LUT_generator (NumPy only) ---
def get_likelihood_from_LUT_generator(
    x_domain: Annotated[Array, (None,), float],
    mu_domain: Annotated[Array, (None,), float],
    LUT: Annotated[Array, (None, None, None), float],
    usenumba: bool = False
) -> Callable[
    [Annotated[Array, (None,), float],
     Annotated[Array, (None,), float],
     Annotated[Array, (None,), int]],
    Annotated[Array, (None,), float]
]:
    """
    Pure‐NumPy LUT lookup: x_idx = nearest_index_1d(x_domain, x);
                         mu_idx = nearest_index_1d(mu_domain, mu);
                         return LUT[mu_idx, x_idx, sigma_idx]
    """
    nearest_index_1d_py = gen_nearest_index_1d(usenumba=False)

    def get_likelihood_from_LUT(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        sigma_idx: Annotated[Array, (None,), int],
    ) -> Annotated[Array, (None,), float]:
        x_idx  = nearest_index_1d_py(x_domain, x)
        mu_idx = nearest_index_1d_py(mu_domain, mu)
        coords = np.stack((mu_idx, x_idx, sigma_idx), axis=-1)
        return LUT[tuple(coords.T.astype(np.int32))]

    return get_likelihood_from_LUT


# --- get_MLEfrom_LUT_generator (NumPy only) ---
def get_MLEfrom_LUT_generator(
    x_domain: Annotated[Array, (None,), float],
    LUT_MLE: Annotated[Array, (None, None), float],
    usenumba: bool = False
) -> Callable[
    [Annotated[Array, (None,), float],
     Annotated[Array, (None,), int]],
    Annotated[Array, (None,), float]
]:
    """
    Pure‐NumPy MLE LUT lookup: x_idx = nearest_index_1d(x_domain, x);
                               return LUT_MLE[x_idx, sigma_idx]
    """
    nearest_index_1d_py = gen_nearest_index_1d(usenumba=False)

    def get_MLE_from_LUT(
        x: Annotated[Array, (None,), float],
        sigma_idx: Annotated[Array, (None,), int],
    ) -> Annotated[Array, (None,), float]:
        x_idx = nearest_index_1d_py(x_domain, x)
        coords = np.stack((x_idx, sigma_idx), axis=-1)
        return LUT_MLE[tuple(coords.T.astype(np.int32))]

    return get_MLE_from_LUT


# --- lossFuncGenerator (NumPy only) ---
def lossFuncGenerator(
    likelihoodFunction: Callable,
    ratio: bool = True,
    likelihoodMLEFunction: Optional[Callable] = None,
    log: bool = False,
    usenumba: bool = False,
    float_type: Type[np.floating] = np.float32
) -> Callable:
    """
    Generate negative‐log‐likelihood or -2 log‐likelihood ratio (NumPy only).
    """
    min_val = np.finfo(float_type).tiny

    def neg_log_likelihood(
        x: Annotated[np.ndarray, (None,), float],
        mu: Annotated[np.ndarray, (None,), float],
        sigma: Annotated[np.ndarray, (None,), float],
    ) -> Annotated[np.ndarray, (None,), float]:
        if log:
            return -likelihoodFunction(x, mu, sigma)
        else:
            return -np.log(np.maximum(likelihoodFunction(x, mu, sigma), min_val))

    def neg_log_likelihood_ratio(
        x: Annotated[np.ndarray, (None,), float],
        mu: Annotated[np.ndarray, (None,), float],
        sigma: Annotated[np.ndarray, (None,), float]
    ) -> Annotated[np.ndarray, (None,), float]:
        if log:
            logL = likelihoodFunction(x, mu, sigma)
            logL_mle = likelihoodMLEFunction(x, sigma)
            return -2.0 * (logL - logL_mle)
        else:
            L = np.maximum(likelihoodFunction(x, mu, sigma), min_val)
            L_mle = np.maximum(likelihoodMLEFunction(x, sigma), min_val)
            return -2.0 * (np.log(L) - np.log(L_mle))

    if ratio:
        if likelihoodMLEFunction is None:
            raise ValueError(
                "likelihoodMLEFunction must be provided when ratio=True."
            )
        return neg_log_likelihood_ratio
    else:
        return neg_log_likelihood


# --- compute_common_std_gen (NumPy only) ---
def compute_common_std_gen(
    usenumba: bool = False,
    float_type: Type[np.floating] = np.float32
) -> Callable[
    [Annotated[np.ndarray, (None,), float], Annotated[np.ndarray, (None,), float], float],
    Annotated[np.ndarray, (None,), float]
]:
    def compute_common_std(
        mu: Annotated[np.ndarray, (None,), float],
        std: Annotated[np.ndarray, (None,), float],
        p_dpe: float
    ) -> Annotated[np.ndarray, (None,), float]:
        one_dtype = float_type(1.0)
        npe_mean = mu * (one_dtype + p_dpe)
        combined = (
            mu * (one_dtype + p_dpe) ** 2 +
            mu * p_dpe * (1 - p_dpe) +
            (np.sqrt(np.abs(npe_mean)) * std ** 2)
        )
        return np.sqrt(np.maximum(np.abs(combined), float_type(1e-6)))
    return compute_common_std


# --- gaussian_approx (NumPy only) ---
def gaussian_approx(
    p_dpe: float,
    float_type: Type[np.floating] = np.float32
) -> Callable[
    [Annotated[np.ndarray, (None,), float], Annotated[np.ndarray, (None,), float], Annotated[np.ndarray, (None,), float]],
    Annotated[np.ndarray, (None,), float]
]:
    """
    Returns a callable that computes the Gaussian log probability (NumPy only).
    log N(mu*(1+p_dpe), combined_std).
    """
    compute_common_std = compute_common_std_gen(usenumba=False, float_type=float_type)

    def gaussian_approximation(
        x: Annotated[np.ndarray, (None,), float],
        mu: Annotated[np.ndarray, (None,), float],
        std: Annotated[np.ndarray, (None,), float]
    ) -> Annotated[np.ndarray, (None,), float]:
        one_dtype = float_type(1.0)
        npe_mean = mu * (one_dtype + p_dpe)
        combined_std = compute_common_std(mu, std, p_dpe)
        # Evaluate log‐pdf
        return (
            -0.5 * ((x - npe_mean) / combined_std) ** 2
            - np.log(combined_std)
            - 0.5 * np.log(2.0 * np.pi)
        )

    return gaussian_approximation


# --- gaussian_approx_MLE (NumPy only) ---
def gaussian_approx_MLE(
    p_dpe: float,
    float_type: Type[np.floating] = np.float32
) -> Callable:
    """
    Returns a callable that computes the Gaussian log likelihood at MLE (mu=x) (NumPy only).
    """
    gaussian_log_likelihood = gaussian_approx(p_dpe=p_dpe, usenumba=False, float_type=float_type)
    compute_common_std = compute_common_std_gen(usenumba=False, float_type=float_type)

    def gaussian_likelihood_MLE(
        x: Annotated[np.ndarray, (None,), float],
        std: Annotated[np.ndarray, (None,), float]
    ) -> Annotated[np.ndarray, (None,), float]:
        return gaussian_log_likelihood(x, x, std)

    return gaussian_likelihood_MLE


# --- parent_gen (NumPy only) ---
def parent_gen(
    gaussian_stds: Annotated[np.ndarray, (None,), float],
    method: Literal["LUT", "Exact"] = "LUT",
    return_ratio: bool = True,
    switching_signal: float = 40.0,
    p_dpe: float = 0.2,
    n_sigma: int = 5,
    m: int = 5,
    usenumba: bool = False,
    float_type: Type[np.floating] = np.float32
) -> Callable[
    [Annotated[np.ndarray, (None, None), float],
     Annotated[np.ndarray, (None, None), float]],
    Annotated[np.ndarray, (None, None), float]
]:
    """
    NumPy‐only full loss function generator.
    """
    # scalar casts (no effect if usenumba=False)
    p_dpe = float_type(p_dpe)
    switching_signal = float_type(switching_signal)

    gaussian_stds = np.asarray(gaussian_stds, dtype=float_type)

    # 1) Build "Exact" or "LUT" likelihoodFunction_exact and likelihoodMLEFunction_exact
    if method.lower() == "exact":
        n_pe_domain, n_ph_domain = gen_pe_ph_domain(
            switching_signal=switching_signal,
            n_sigma=n_sigma,
            p_dpe=p_dpe,
            float_type=float_type
        )
        likelihoodFunction_exact = exact_likelihood_generator(
            n_pe_domain=n_pe_domain,
            n_ph_domain=n_ph_domain,
            p=p_dpe,
            float_type=float_type
        )
        if return_ratio:
            likelihoodMLEFunction_exact = exact_likelihood_MLE_generator(usenumba=False)
        else:
            likelihoodMLEFunction_exact = None
    else:  # method == "lut"
        LUT, LUT_MLE, x_domain, mu_domain = generate_LUT(
            m=m,
            p=p_dpe,
            switching_signal=switching_signal,
            gaussian_stds=gaussian_stds,
            n_sigma=n_sigma,
            float_type=float_type
        )
        likelihoodFunction_exact = get_likelihood_from_LUT_generator(
            x_domain=x_domain,
            mu_domain=mu_domain,
            LUT=LUT,
            usenumba=False
        )
        if return_ratio:
            likelihoodMLEFunction_exact = get_MLEfrom_LUT_generator(
                x_domain=x_domain,
                LUT_MLE=LUT_MLE,
                usenumba=False
            )
        else:
            likelihoodMLEFunction_exact = None

    # 2) Build the Gaussian‐approx likelihood and its MLE version
    likelihoodFunction_Gauss = gaussian_approx(
        p_dpe=p_dpe,
        usenumba=False,
        float_type=float_type
    )
    if return_ratio:
        likelihoodMLEFunction_Gauss = gaussian_approx_MLE(
            p_dpe=p_dpe,
            usenumba=False,
            float_type=float_type
        )
    else:
        likelihoodMLEFunction_Gauss = None

    # 3) Build the negative‐log‐likelihood wrappers
    lossfunc_exact = lossFuncGenerator(
        likelihoodFunction=likelihoodFunction_exact,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunction_exact,
        usenumba=False,
        float_type=float_type
    )
    lossfunc_Gauss = lossFuncGenerator(
        likelihoodFunction=likelihoodFunction_Gauss,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunction_Gauss,
        log=True,
        usenumba=False,
        float_type=float_type
    )

    # 4) compute_scaling (NumPy)
    def compute_scaling(x: Annotated[np.ndarray, (None, None), float]) -> np.ndarray:
        return np.sum(np.where(x > 0, x, 0), axis=-1, keepdims=True)

    # 5) NumPy lossfunc
    def lossfunc_numpy(
        *,
        x: Annotated[np.ndarray, (None, None), float],
        mu: Annotated[np.ndarray, (None, None), float]
    ) -> Annotated[np.ndarray, (None, None), float]:
        B, N = x.shape

        std_idx_grid = np.broadcast_to(
            np.arange(gaussian_stds.shape[0], dtype=np.int32),
            x.shape
        )
        std_grid = np.broadcast_to(gaussian_stds, x.shape)

        scaling = compute_scaling(x)
        mu_scaled = mu * scaling

        result = np.empty(x.shape, dtype=float_type)
        mask = (mu_scaled > switching_signal) | (x > switching_signal)

        # Gaussian only
        if mask.any():
            xi_g   = x[mask]
            mui_g  = mu_scaled[mask]
            std_g  = std_grid[mask]
            res_g  = lossfunc_Gauss(xi_g, mui_g, std_g)
            result[mask] = res_g

        # Exact / LUT
        inv_mask = ~mask
        if inv_mask.any():
            xi_e  = x[inv_mask]
            mui_e = mu_scaled[inv_mask]
            if method.lower() == "exact":
                std_e = std_grid[inv_mask]
                res_e = lossfunc_exact(xi_e, mui_e, std_e)
            else:
                sigma_idx_e = std_idx_grid[inv_mask]
                res_e = lossfunc_exact(xi_e, mui_e, sigma_idx_e)
            result[inv_mask] = res_e

        return result

    return lossfunc_numpy