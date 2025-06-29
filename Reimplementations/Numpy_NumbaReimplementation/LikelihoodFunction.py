'''

Numpy/Numba implementation of the full Likelihood function

Note that the typing is not written for any static solver but only for the reader to make it more
understandable

from . type_names import *

import numpy as np
from scipy.special import betaln as spbetaln
from scipy.special import gammaln as spgammaln


epsilon = 1e-10


def gaussian_log_likelihood_gen(usenumba: bool = False,
                                float_type: Type[float] = np.float32) -> Callable[
    [Annotated[Array, (), float], Annotated[Array, (), float], Annotated[Array, (), float]],
    Annotated[Array, (), float]]:
    def gaussian_log_likelihood(
        x: Annotated[Array, (), float],
        loc: Annotated[Array, (), float],
        scale: Annotated[Array, (), float]
    ) -> Annotated[Array, (), float]:
        """
        Returns log of N(x | loc, scale^2):
        log pdf = −0.5 * ((x−loc)/scale)**2 − log(scale) − 0.5*log(2π)
        """
        return (
            -float_type(0.5) * ((x - loc) / scale) ** float_type(2)
            - np.log(scale)
            - float_type(0.5) * np.log(float_type(2.0) * np.pi)
        )
    if usenumba:
        from numba import njit
        gaussian_log_likelihood = njit(gaussian_log_likelihood, boundscheck=False)
        return gaussian_log_likelihood
    else:
        return gaussian_log_likelihood


def gen_pe_ph_domain(
    switching_signal: Int,
    n_sigma: Int,
    p_dpe: Float,
    float_type: Type[float] = np.float32
) -> tuple[
    Annotated[Array, (None,), float],
    Annotated[Array, (None,), float]
]:
    """
    Generate the photon and photoelectron domain for the likelihood function
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


def exact_likelihood_generator(
    n_pe_domain: Annotated[Array, (None,), float],
    n_ph_domain: Annotated[Array, (None,), float],
    p: float,
    usenumba: bool = False,
    float_type: Type[float] = np.float32
) -> Callable:
    """
    Generate callable for the exact likelihood function
    """
    if usenumba:
        from numba import njit
    gaussian_log_likelihood = gaussian_log_likelihood_gen(usenumba=usenumba)

    n_ph_grid: Annotated[Array, (1, None, 1), float] = np.reshape(n_ph_domain, (1, -1, 1))
    n_pe_grid: Annotated[Array, (1, 1, None), float] = np.reshape(n_pe_domain, (1, 1, -1))

    def exact_gaussian(
        x: Annotated[Array, (None, 1, 1), float],
        sigma: Annotated[Array, (None, 1, 1), float]
    ) -> Annotated[Array, (None, 1, None), float]:
        """
        Computes the gaussian component over the allowed photoelectron grid
        x: Observed Signal
        sigma: Gaussian Variance
        """
        scale: Annotated[Array, (None, 1, None), float] = np.maximum(sigma * np.sqrt(n_pe_grid), float_type(epsilon))
        mean: Annotated[Array, (1, 1, None), float] = n_pe_grid
        res: Annotated[Array, (None, 1, None), float] = gaussian_log_likelihood(x, mean, scale)
        return res
    if usenumba:
        exact_gaussian   = njit(exact_gaussian, boundscheck=False, cache=True)

    if usenumba:
        from numba import prange
        @njit(boundscheck=False, cache=True)
        def betaln_scalar(a, b):
            if a <= 0.0 and math.floor(a) == a:
                return float_type(math.inf)
            if b <= 0.0 and math.floor(b) == b:
                return float_type(math.inf)
            return float_type(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
        
        @njit(boundscheck=False)
        def betaln(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            Apply `betaln_scalar` elementwise to two equally‐shaped float32 arrays.
            Returns a new array of the same shape & dtype.
            """
            # allocate output array (same shape and dtype as a)
            out = np.empty(a.shape, dtype=a.dtype)

            # flatten everything
            flat_a = a.ravel()
            flat_b = b.ravel()
            flat_out = out.ravel()
            n = flat_a.size
            # elementwise loop
            for i in prange(n):
                flat_out[i] = betaln_scalar(flat_a[i], flat_b[i])

            return out
    else:
        betaln = spbetaln

    
    
    def get_valid_np(n,k):
        return ((k <= n) & (k >= 0.)).astype(dtype=float_type)

    if usenumba:
        def get_valid_nb(n, k):
            N = n.shape[1]
            M = k.shape[2]
            valid = np.empty((1, N, M), dtype=float_type)
            for i in prange(N):
                n_val = n[0, i, 0]
                for j in prange(M):
                    kv = k[0, i, j]
                    if (kv >= 0.0) and (kv <= n_val):
                        valid[0, i, j] = float_type(1.0)
                    else:
                        valid[0, i, j] = float_type(0.0)

            
            return valid
        get_valid = njit(get_valid_nb, boundscheck=False)
    else:
        get_valid = get_valid_np

    def binomial_pmf(
        n: Annotated[Array, (1, None, 1), float],
        p: float,
        k: Annotated[Array, (1, None, None), float]
    ) -> Annotated[Array, (1, None, None), float]:
        """
        Computes the binomial probability mass function
        Handles nonsense k (<0) by placing null probability
        """
        k_safe = np.clip(k, 0, n)
        valid = get_valid(n, k)

        one_dtype = float_type(1.0)
        log_unnormalized = np.log(p) * k_safe + np.log1p(-p) * (n - k_safe)
        log_normalization = betaln(one_dtype + k_safe, one_dtype + n - k_safe) + np.log(n +one_dtype)

        return valid * np.exp(log_unnormalized - log_normalization)
    
    if usenumba:
        binomial_pmf     = njit(binomial_pmf, boundscheck=False)


    if usenumba:
        import math
        from numba import prange
        @njit(boundscheck=False)
        def gammaln(x: np.ndarray) -> np.ndarray:
            """
            Apply gammaln_scalar to every element of x (any shape) and return an array
            of the same shape.
            """
            # Allocate output array of the same shape & dtype as x
            out = np.empty(x.shape, x.dtype)
            # Flatten both arrays into 1D views
            flat_x = x.ravel()
            flat_out = out.ravel()
            n = flat_x.size
            # Loop over all indices in flattened view
            for i in prange(n):
                flat_out[i] = math.lgamma(flat_x[i])
            return out

    else:
        gammaln = spgammaln

    def poisson_pmf(
        n: Annotated[Array, (1, None, 1), float],
        mu: Annotated[Array, (None, 1, 1), float],
    ) -> Annotated[Array, (None, None, 1), float]:
        """
        Computes the Poisson probability mass function
        Handles mu (= 0) by placing null probability
        """
        mask = (mu > float_type(0.0))
        valid = mask.astype(float_type)
        one_dtype = float_type(1.0)
        safe_mu = np.where(mask, mu, one_dtype)
        log_p = n * np.log(safe_mu) - mu - gammaln(n + one_dtype)
        return valid * np.exp(log_p)
    
    if usenumba:
        poisson_pmf      = njit(poisson_pmf, boundscheck=False)

    """Numba doesn support axis args in sum"""
    def _prob_sum_np(x):
        return np.sum(x, axis=(1, 2))
    
    

    if usenumba:
        from numba import prange
        def _prob_sum_nb(x):
            n, M, K = x.shape
            out = np.empty(n, dtype=x.dtype)
            for i in prange(n):
                total = 0.0
                for j in prange(M):
                    for k in prange(K):
                        total += x[i, j, k]
                out[i] = total
            return out
        _prob_sum = njit(_prob_sum_nb, boundscheck=False)
    else:
        _prob_sum = _prob_sum_np

    def exact_likelihood(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        sigma: Annotated[Array, (None,), float]
    ) -> Annotated[Array, (None,), float]:
        """
        x : Observed Signal  : [n]
        mu : Expected Signal  : [n]
        sigma : Gaussian Variance : [n]
        """
        x: Annotated[Array, (None, 1, 1), float] = np.reshape(x, (-1, 1, 1))
        mu: Annotated[Array, (None, 1, 1), float] = np.reshape(mu, (-1, 1, 1))
        sigma: Annotated[Array, (None, 1, 1), float] = np.reshape(sigma, (-1, 1, 1))
        # Gaussian component
        a: Annotated[Array, (None, 1, None), float] = np.exp(exact_gaussian(x, sigma))
        # Binomial component
        b: Annotated[Array, (1, None, None), float] = binomial_pmf(n_ph_grid, p, n_pe_grid - n_ph_grid)
        # Poisson component
        c: Annotated[Array, (None, None, 1), float] = poisson_pmf(n_ph_grid, mu)
        # Double Sum
        res: Annotated[Array, (None,), float] = _prob_sum(a * b * c)
        return res
    
    if usenumba:
        exact_likelihood = njit(exact_likelihood, boundscheck=False)

    return exact_likelihood


def exact_likelihood_MLE_generator():
    return NotImplementedError("Exact likelihood MLE generator needs to be implemented here")


def generate_LUT(
    m: int,
    p: float,
    switching_signal: float,
    gaussian_stds: Annotated[Array, (None,), float],
    n_sigma: int = 5,
    obs_min: float = -3.0,
    usenumba: bool = False,
    float_type: Type[float] = np.float32
) -> tuple[
    Annotated[Array, (None, None, None), float],
    Annotated[Array, (None, None), float],
    Annotated[Array, (None,), float],
    Annotated[Array, (None,), float],
]:
    """
    Computes the Look up table from the exact method

    return LUT and LUT MLE

    dimensions contain (mu, x, sigma)
    where mu and x must be searched, and sigma corresponds to the PMT index or rather index in which stds are passed
    """
    # Build x, mu and sigma domain
    i_vals: Annotated[Array, (None,)] = np.arange(obs_min, switching_signal, dtype=float_type)
    sub_vals: Annotated[Array, (None,)] = np.arange(float(m), dtype=float_type) / m
    x_domain: Annotated[Array, (None,)] = np.concatenate(
        [(i_vals[:, None] + sub_vals[None, :]).ravel(), np.array([switching_signal], dtype=float_type)]
    )
    nx: int = x_domain.shape[0]

    i_vals: Annotated[Array, (None,)] = np.arange(0.0, switching_signal, dtype=float_type)
    sub_vals: Annotated[Array, (None,)] = np.arange(float(m), dtype=float_type) / m
    mu_domain: Annotated[Array, (None,)] = np.concatenate(
        [(i_vals[:, None] + sub_vals[None, :]).ravel(), np.array([switching_signal], dtype=float_type)]
    )
    nmu: int = mu_domain.shape[0]

    sigma_domain = np.array(gaussian_stds, dtype=float_type)
    n_pmts: int = sigma_domain.shape[0]

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
        usenumba=usenumba,
        float_type=float_type
    )

    # Generate empty LUT
    LUT: Annotated[Array, (nmu, nx, n_pmts), float] = np.zeros((nmu, nx, n_pmts), dtype=float_type)
    for idx in range(len(LUT)):
        x_grid = np.repeat(x_domain[:, None], n_pmts, axis=1).ravel()
        mu_grid = np.full_like(x_grid, mu_domain[idx])
        std_grid = np.tile(sigma_domain[None, :], (nx, 1)).ravel()
        probs = exact_likelihood_function(x_grid, mu_grid, std_grid)
        LUT[idx] = probs.reshape(nx, n_pmts)

    # Find per x MLE mu Likelihoods : L(x|mu_MLE)
    LUT_MLE: Annotated[Array, (nx, n_pmts), float] = np.max(LUT, axis=0)

    return LUT, LUT_MLE, x_domain, mu_domain

def gen_nearest_index_1d(usenumba: bool = False) -> Callable[
    [Annotated[Array, (None,), float], Annotated[Array, (None,), float]],
    Annotated[Array, (None,), int]
]:
    """
    Factory for a “nearest‐index‐in‐1D” function.
    If usenumba=False, returns a pure‐NumPy version.
    If usenumba=True, returns an njit‐compiled version.
    """
    def nearest_index_1d(
        grid: Annotated[Array, (None,), float],
        values: Annotated[Array, (None,), float],
    ) -> Annotated[Array, (None,), int]:
        """
        Pure‐NumPy: For each entry in `values`, find the index in `grid` whose
        element is closest.
        """
        idx_float = np.searchsorted(grid, values, side='left')  # shape = values.shape
        idx_below = np.clip(idx_float - 1, 0, grid.size - 1)
        idx_above = np.clip(idx_float,     0, grid.size - 1)

        below_vals = grid[idx_below]
        above_vals = grid[idx_above]

        dist_below = np.abs(values - below_vals)
        dist_above = np.abs(values - above_vals)

        idx_nearest = np.where(dist_above < dist_below, idx_above, idx_below)
        return idx_nearest.astype(np.int32)

    if usenumba:
        from numba import njit
        nearest_index_1d = njit(nearest_index_1d, boundscheck=False)

    return nearest_index_1d


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
    Returns a function f(x, mu, sigma_idx) that does:
      1) x_idx  = nearest_index_1d(x_domain, x)
      2) mu_idx = nearest_index_1d(mu_domain, mu)
      3) return LUT[mu_idx, x_idx, sigma_idx]

    Pure‐NumPy version uses advanced indexing.
    Numba version (usenumba=True) uses an explicit loop + binary search.
    """
    # Pure‐NumPy nearest‐index
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

    if not usenumba:
        return get_likelihood_from_LUT

    # Numba‐compiled version (no tuple indexing)
    from numba import njit

    @njit(boundscheck=False)
    def get_likelihood_from_LUT_numba(
        x:          np.ndarray,
        mu:         np.ndarray,
        sigma_idx:  np.ndarray,
    ) -> np.ndarray:
        n = x.shape[0]
        out = np.empty(n, dtype=LUT.dtype)
        nx = x_domain.size
        nm = mu_domain.size

        for i in range(n):
            # 1) locate x_idx by binary search on x_domain
            xi = x[i]
            xpos = np.searchsorted(x_domain, xi)
            if xpos == 0:
                x_idx = 0
            elif xpos == nx:
                x_idx = nx - 1
            else:
                left  = x_domain[xpos - 1]
                right = x_domain[xpos]
                if (xi - left) <= (right - xi):
                    x_idx = xpos - 1
                else:
                    x_idx = xpos

            # 2) locate mu_idx by binary search on mu_domain
            mui = mu[i]
            mpos = np.searchsorted(mu_domain, mui)
            if mpos == 0:
                mu_idx = 0
            elif mpos == nm:
                mu_idx = nm - 1
            else:
                leftm  = mu_domain[mpos - 1]
                rightm = mu_domain[mpos]
                if (mui - leftm) <= (rightm - mui):
                    mu_idx = mpos - 1
                else:
                    mu_idx = mpos

            # 3) lookup
            sid = int(sigma_idx[i])
            out[i] = LUT[mu_idx, x_idx, sid]

        return out

    return get_likelihood_from_LUT_numba


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
    Returns a function g(x, sigma_idx) that does:
      1) x_idx = nearest_index_1d(x_domain, x)
      2) return LUT_MLE[x_idx, sigma_idx]

    Pure‐NumPy version uses advanced indexing.
    Numba version (usenumba=True) uses an explicit loop + binary search.
    """
    # Pure‐NumPy nearest‐index
    nearest_index_1d_py = gen_nearest_index_1d(usenumba=False)

    def get_MLE_from_LUT(
        x: Annotated[Array, (None,), float],
        sigma_idx: Annotated[Array, (None,), int],
    ) -> Annotated[Array, (None,), float]:
        x_idx = nearest_index_1d_py(x_domain, x)
        coords = np.stack((x_idx, sigma_idx), axis=-1)
        return LUT_MLE[tuple(coords.T.astype(np.int32))]

    if not usenumba:
        return get_MLE_from_LUT

    # Numba‐compiled version
    from numba import njit

    @njit(boundscheck=False)
    def get_MLE_from_LUT_numba(
        x:          np.ndarray,
        sigma_idx:  np.ndarray
    ) -> np.ndarray:
        n = x.shape[0]
        out = np.empty(n, dtype=LUT_MLE.dtype)
        nx = x_domain.size

        for i in range(n):
            # find x_idx via binary search
            xi = x[i]
            xpos = np.searchsorted(x_domain, xi)
            if xpos == 0:
                x_idx = 0
            elif xpos == nx:
                x_idx = nx - 1
            else:
                left  = x_domain[xpos - 1]
                right = x_domain[xpos]
                if (xi - left) <= (right - xi):
                    x_idx = xpos - 1
                else:
                    x_idx = xpos

            sid = int(sigma_idx[i])
            out[i] = LUT_MLE[x_idx, sid]

        return out

    return get_MLE_from_LUT_numba


def lossFuncGenerator(
    likelihoodFunction: Callable,
    ratio: bool = True,
    likelihoodMLEFunction: Optional[Callable] = None,
    log: bool = False,
    usenumba: bool = False,
    float_type: Type[float] = np.float32
) -> Callable:
    """
    ratio : bool
        If False, return the negative log-likelihood.
        If True, return the -2 log-likelihood ratio (χ²).
    likelihoodFunction : Callable
        If log=False, this should return L(x|μ,σ).
        If log=True, this should return log L(x|μ,σ).
    likelihoodMLEFunction : Optional[Callable]
        Required if ratio=True. If log=False, this returns L(x|μ_MLE,σ);
        if log=True, it returns log L(x|μ_MLE,σ).
    log : bool
        If False, treat likelihoodFunction as returning a probability (L).
        If True, treat likelihoodFunction as returning log L directly.
    """
    min_val = np.finfo(float_type).tiny

    def neg_log_likelihood(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        sigma: Annotated[Array, (None,), Union[float, int]],
    ) -> Annotated[Array, (None,), float]:
        if log:
            return -likelihoodFunction(x, mu, sigma)
        else:
            return -np.log(np.maximum(likelihoodFunction(x, mu, sigma), min_val))
        
    if usenumba:
        from numba import njit
        neg_log_likelihood = njit(neg_log_likelihood, boundscheck=False)

    def neg_log_likelihood_ratio(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        sigma: Annotated[Array, (None,), Union[float, int]]
    ) -> Annotated[Array, (None,), float]:
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
                "likelihoodMLEFunction must be provided when ratio=True. "
                "If log=False, it should return L(x|μ_MLE,σ). "
                "If log=True, return log L(x|μ_MLE,σ)."
            )
        if usenumba:
            from numba import njit
            neg_log_likelihood_ratio = njit(neg_log_likelihood_ratio, boundscheck=False)

        return neg_log_likelihood_ratio
    else:
        return neg_log_likelihood


def compute_common_std_gen(usenumba: bool = False,
                           float_type: Type[float] = np.float32) -> Callable[
    [Annotated[Array, (None,), float], Annotated[Array, (None,), float], float],
    Annotated[Array, (None,), float]
    ]:
    def compute_common_std(
        mu: Annotated[Array, (None,), float],
        std: Annotated[Array, (None,), float],
        p_dpe: float
    ) -> Annotated[Array, (None,), float]:
        """
        Computes an effective standard deviation combining Poisson, Binomial, and Gaussian contributions.
        Args:
            mu: expected signal (np array)
            std: Gaussian std component (np array)
            p_dpe: double photoelectron probability (scalar)
        Returns:
            combined standard deviation (np array)
        """
        one_dtype = float_type(1.0)
        npe_mean = mu * (one_dtype + p_dpe)
        combined = (
            mu * (one_dtype + p_dpe) ** 2 +             # Poisson component
            mu * p_dpe * (1 - p_dpe) +          # Binomial component
            (np.sqrt(np.abs(npe_mean)) * std ** 2)  # Gaussian component
        )
        return np.sqrt(np.maximum(np.abs(combined), float_type(1e-6)))  # avoid underflow
    if usenumba:
        from numba import njit
        compute_common_std = njit(compute_common_std, boundscheck=False)
    return compute_common_std

def gaussian_approx(
    p_dpe: float,
    usenumba: bool = False,
    float_type: Type[float] = np.float32
) -> Callable[
    [Annotated[Array, (None,), float], Annotated[Array, (None,), float], Annotated[Array, (None,), float]],
    Annotated[Array, (None,), float]
]:
    """
    Returns a callable function that computes the Gaussian log probability:
    N(mu * (1 + p_dpe), compute_common_std(mu, std))

    Args:
        p_dpe: Double photoelectron probability (scalar)
    Returns:
        Callable that takes (x, mu, std) and returns log probability [n]
    """
    compute_common_std = compute_common_std_gen(usenumba=usenumba, float_type=float_type)
    gaussian_log_likelihood = gaussian_log_likelihood_gen(usenumba=usenumba, float_type=float_type)

    def gaussian_approximation(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        std: Annotated[Array, (None,), float]
    ) -> Annotated[Array, (None,), float]:
        one_dtype = float_type(1.0)
        npe_mean: Annotated[Array, (None,), float] = mu * (one_dtype + p_dpe)
        combined_std: Annotated[Array, (None,), float] = compute_common_std(mu, std, p_dpe)

        res: Annotated[Array, (None,), float] = gaussian_log_likelihood(x, npe_mean, combined_std)
        return res
    if usenumba:
        from numba import njit
        gaussian_approximation = njit(gaussian_approximation, boundscheck=False)

    return gaussian_approximation


def gaussian_approx_MLE(
    p_dpe: float,
    usenumba: bool = False,
    float_type=np.float32
) -> Callable:
    """
    Returns a callable that computes the Gaussian log‐likelihood at the MLE (mu = x):
      log N(x | x*(1+p_dpe), compute_common_std(x, std)^2)

    Args:
      p_dpe: Double photoelectron probability (scalar)
    Returns:
      A function taking (x, std) and returning log‐likelihood [n]
    """
    gaussian_log_likelihood = gaussian_log_likelihood_gen(usenumba=usenumba, float_type=float_type)
    compute_common_std = compute_common_std_gen(usenumba=usenumba, float_type=float_type)
    def gaussian_likelihood_MLE(
        x: Annotated[Array, (None,), float],
        std: Annotated[Array, (None,), float]
    ) -> Annotated[Array, (None,), float]:
        npe_mean = x * (1 + p_dpe)
        combined_std = compute_common_std(x, std, p_dpe)  # shape [n]

        return gaussian_log_likelihood(x, npe_mean, combined_std)

    if usenumba:
        from numba import njit
        gaussian_likelihood_MLE = njit(gaussian_likelihood_MLE, boundscheck=False)

    return gaussian_likelihood_MLE


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
    Generates the full loss function (NumPy or Numba) based on the chosen method.
    """

    if usenumba:
        from numba import njit, prange
        p_dpe = float_type(p_dpe)
        switching_signal = float_type(switching_signal)

    gaussian_stds = np.asarray(gaussian_stds, dtype=float_type)

    # 1) Build "Exact" or "LUT" likelihoodFunction and likelihoodMLEFunction
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
            usenumba=usenumba,
            float_type=float_type
        )
        if return_ratio:
            likelihoodMLEFunction_exact = exact_likelihood_MLE_generator(
                usenumba=usenumba
            )
        else:
            likelihoodMLEFunction_exact = None

    else:  # method == "lut"
        LUT, LUT_MLE, x_domain, mu_domain = generate_LUT(
            m=m,
            p=p_dpe,
            switching_signal=switching_signal,
            gaussian_stds=gaussian_stds,
            n_sigma=n_sigma,
            usenumba=usenumba,
            float_type=float_type
        )
        likelihoodFunction_exact = get_likelihood_from_LUT_generator(
            x_domain=x_domain,
            mu_domain=mu_domain,
            LUT=LUT,
            usenumba=usenumba
        )
        if return_ratio:
            likelihoodMLEFunction_exact = get_MLEfrom_LUT_generator(
                x_domain=x_domain,
                LUT_MLE=LUT_MLE,
                usenumba=usenumba
            )
        else:
            likelihoodMLEFunction_exact = None

    # 2) Build the Gaussian‐approx likelihood and its MLE version
    gaussian_approx_func = gaussian_approx(
        p_dpe=p_dpe,
        usenumba=usenumba,
        float_type=float_type
    )
    if return_ratio:
        likelihoodMLEFunction_Gauss = gaussian_approx_MLE(
            p_dpe=p_dpe,
            usenumba=usenumba,
            float_type=float_type
        )
    else:
        likelihoodMLEFunction_Gauss = None

    # 3) Build the negative‐log‐likelihood wrappers (we don’t call these directly
    # in the Gaussian‐only branch)
    lossfunc_exact = lossFuncGenerator(
        likelihoodFunction=likelihoodFunction_exact,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunction_exact,
        usenumba=usenumba,
        float_type=float_type
    )
    lossfunc_Gauss = lossFuncGenerator(
        likelihoodFunction=gaussian_approx_func,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunction_Gauss,
        log=True,
        usenumba=usenumba,
        float_type=float_type
    )

    # 4) Build compute_scaling for NumPy vs Numba
    def compute_scaling_np(x: Annotated[np.ndarray, (None, None), float]) -> np.ndarray:
        return np.sum(np.where(x > 0, x, 0), axis=-1, keepdims=True)

    compute_scaling = compute_scaling_np
    if usenumba:
        @njit(boundscheck=False)
        def compute_scaling_nb(x: np.ndarray) -> np.ndarray:
            B, N = x.shape
            out = np.empty((B, 1), dtype=x.dtype)
            for i in prange(B):
                s = 0.0
                for j in prange(N):
                    val = x[i, j]
                    if val > 0.0:
                        s += val
                out[i, 0] = s
            return out

        compute_scaling = compute_scaling_nb

    # 5) NumPy‐only lossfunc
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

        # Gaussian‐only branch
        if mask.any():
            xi_g   = x[mask]
            mui_g  = mu_scaled[mask]
            std_g  = std_grid[mask]
            res_g  = gaussian_approx_func(xi_g, mui_g, std_g)
            result[mask] = res_g

        # Exact / LUT branch
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

    # 6) Numba‐compiled lossfunc
    def _make_lossfunc_numba() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        from numba import njit, prange

        @njit(boundscheck=False)
        def lossfunc_numba(
            x: np.ndarray,
            mu: np.ndarray
        ) -> np.ndarray:
            B, N = x.shape

            scaling = compute_scaling_nb(x)
            mu_scaled = mu * scaling

            result = np.empty((B, N), dtype=float_type)

            # Pre‐allocate 1‐element buffers
            tmp_x   = np.empty((1,), dtype=float_type)
            tmp_mu  = np.empty((1,), dtype=float_type)
            tmp_s   = np.empty((1,), dtype=float_type)
            tmp_idx = np.empty((1,), dtype=np.int32)

            for i in prange(B):
                for j in prange(N):
                    xi  = x[i, j]
                    mui = mu_scaled[i, j]

                    tmp_x[0]   = xi
                    tmp_mu[0]  = mui
                    tmp_s[0]   = gaussian_stds[j]
                    tmp_idx[0] = j

                    # (A) Gaussian‐only branch: call the raw gaussian_approx function
                    if (mui > switching_signal) or (xi > switching_signal):
                        out_arr = gaussian_approx_func(tmp_x, tmp_mu, tmp_s)
                    else:
                        # (B) Exact / LUT branch
                        if method.lower() == "exact":
                            out_arr = lossfunc_exact(tmp_x, tmp_mu, tmp_s)
                        else:
                            out_arr = lossfunc_exact(tmp_x, tmp_mu, tmp_idx)

                    result[i, j] = out_arr[0]

            return result

        return lossfunc_numba

    if usenumba:
        lossfunc_nb = _make_lossfunc_numba()
        return njit(lossfunc_nb, boundscheck=False)
    else:
        return lossfunc_numpy'''

from . type_names import *

import numpy as np

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
    ) -> Callable:
    """Wrapper for nb vs np version"""
    if usenumba:
        from . nb_LikelihoodFunction import parent_gen
    else:
        from . np_LikelihoodFunction import parent_gen
    return parent_gen(
        gaussian_stds=gaussian_stds,
        method=method,
        return_ratio=return_ratio,
        switching_signal=switching_signal,
        p_dpe=p_dpe,
        n_sigma=n_sigma,
        m=m,
        float_type=float_type
    )