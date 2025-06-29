
from . type_names import *

import numpy as np
from numba import njit, prange
import math

# --- gen_pe_ph_domain (Numba only) ---
@njit(boundscheck=False)
def gen_pe_ph_domain(
    switching_signal: float,
    n_sigma: int,
    p_dpe: float,
    float_type: Type[np.floating] = np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate photon and photoelectron domain (Numba only).
    """
    n_pe_max = switching_signal + n_sigma * np.sqrt(switching_signal) + 2
    n_ph_max = switching_signal / (1 + p_dpe) + n_sigma * np.sqrt(switching_signal / (1 + p_dpe)) + 2

    # Compute sizes
    size_pe = int(np.floor(n_pe_max - 0.0)) + 1
    size_ph = int(np.floor(n_ph_max - 0.0)) + 1

    n_pe_domain = np.empty(size_pe, dtype=float_type)
    n_ph_domain = np.empty(size_ph, dtype=float_type)

    for i in range(size_pe):
        n_pe_domain[i] = float_type(i)

    for i in range(size_ph):
        n_ph_domain[i] = float_type(i)

    return n_pe_domain, n_ph_domain


# --- exact_likelihood_generator (Numba only) ---
def exact_likelihood_generator(
    n_pe_domain: Annotated[np.ndarray, (None,), float],
    n_ph_domain: Annotated[np.ndarray, (None,), float],
    p: float,
    float_type: Type[np.floating] = np.float32
) -> Callable:
    """
    Generate callable for the exact likelihood function (Numba only).
    """
    # Raw scalar lgamma for Numba
    inf_float = float_type(np.inf)
    p = float_type(p)
    @njit
    def betaln_scalar(a, b):
        if a <= 0.0 and np.floor(a) == a:
            return inf_float
        if b <= 0.0 and np.floor(b) == b:
            return inf_float
        return float_type(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

    # constant terms : Gaussian
    half = float_type(0.5)
    const = half * np.log(float_type(2.0) * float_type(np.pi))
    @njit
    def gaussian_log_likelihood_scalar(
        x: float,
        loc: float,
        scale: float,
    ) -> float:
        diff = (x - loc) / scale
        return -half * diff * diff - np.log(scale) - const

    one_dtype = float_type(1.0)
    @njit
    def binomial_pmf_nb_scalar(
        n: float,
        p: float,
        k: float,
    ) -> float:
        # only valid if k is in correct range
        if (k < 0) or (k > n): return 0.0

        # Valid‐indicator (assumes get_valid_nb works on scalars)
        # log [p^k * (1–p)^(n–k)]
        log_unnormalized = np.log(p) * k + np.log1p(-p) * (n - k)
        # log [Beta(k+1, n–k+1) * (n+1)]
        log_normalization = (
            betaln_scalar(one_dtype + k, one_dtype + n - k)
            + np.log(n + one_dtype)
        )
        return np.exp(log_unnormalized - log_normalization)

    zero_dtype = float_type(0.0)
    @njit
    def poisson_pmf_nb_scalar(n: float, np1lgamma: float, mu: float, logmu: float) -> float:
        # If μ≤0, the pmf is 0.0; skip all log/exp/lgamma work
        if mu <= zero_dtype:
            return zero_dtype
        # At this point, μ>0. Compute log‐PMF directly:
        #     log_p = n·ln(μ) − μ − ln(Γ(n+1))
        # (Note: the literal 1.0 is promoted to float32 or float64 by Numba.)
        log_p = n * logmu - mu - np1lgamma
        return np.exp(log_p)

    # Precompute grids
    K = n_pe_domain.shape[0]
    J = n_ph_domain.shape[0]
    b_grid = np.empty((J, K))
    @njit(boundscheck=False)
    def get_grid_b(n_pe_domain, n_ph_domain, b_grid, J, K):
        for j in range(J):
            for k in range(K):
                b_grid[j, k] = binomial_pmf_nb_scalar(
                    n_ph_domain[j],
                    p,
                    n_pe_domain[k] - n_ph_domain[j],
                )
        return b_grid
    
    b_grid = get_grid_b(n_pe_domain, n_ph_domain, b_grid, J, K)
    sqrt_pe = np.sqrt(n_pe_domain, dtype = float_type)
    tiny = float_type(1e-10)
    @njit(boundscheck=False)
    def exact_likelihood(
        x: float,
        mu: float,
        sigma: float
    ) -> float:

        s = zero_dtype
        logmu = np.log(mu)
        for j in range(J):
            nph = n_ph_domain[j]
            nphlgamma = math.lgamma(nph + one_dtype)
            for k in range(K):
                npe = n_pe_domain[k]
                scale = sigma * sqrt_pe[k]
                # Gaussian piece (scalar):
                if scale < tiny:
                    scale = tiny
                # gaussian_log_likelihood(xi, mean=npe, scale)
                a = np.exp(gaussian_log_likelihood_scalar(x, npe, scale))
                # b_grid[j,k] was precomputed
                bb = b_grid[j, k]
                # Poisson piece (scalar):
                cc = poisson_pmf_nb_scalar(nph, nphlgamma, mu, logmu)
                s += a * bb * cc
        #return out
        return s

    return exact_likelihood


# --- exact_likelihood_MLE_generator (Numba only) ---
def exact_likelihood_MLE_generator(
):
    raise NotImplementedError("Exact likelihood MLE generator needs to be implemented here")


# --- generate_LUT (Numba only) ---
def generate_LUT(
    m: int,
    p: float,
    switching_signal: float,
    gaussian_stds: np.ndarray,
    n_sigma: int = 5,
    obs_min: float = -3.0,
    float_type: Type[np.floating] = np.float32,
    parallel: bool = True
) -> tuple[
    np.ndarray,  # LUT
    np.ndarray,  # LUT_MLE
    np.ndarray,  # x_domain
    np.ndarray   # mu_domain
]:
    """
    Computes the Look‐Up Table using a Numba‐JIT inner loop.  All of the
    arrays and scalars needed by the JIT function are passed in as arguments.
    """

    # 1) Build x_domain
    x_list = np.arange(obs_min, switching_signal, dtype=float_type)
    sub_list = np.arange(float(m), dtype=float_type) / m
    x_domain = np.concatenate([
        (x_list[:, None] + sub_list[None, :]).ravel(),
        np.array([switching_signal], dtype=float_type)
    ])
    nx = x_domain.shape[0]

    # 2) Build mu_domain
    mu_list = np.arange(0.0, switching_signal, dtype=float_type)
    sub_list = np.arange(float(m), dtype=float_type) / m
    mu_domain = np.concatenate([
        (mu_list[:, None] + sub_list[None, :]).ravel(),
        np.array([switching_signal], dtype=float_type)
    ])
    nmu = mu_domain.shape[0]

    # 3) sigma_domain
    sigma_domain = np.array(gaussian_stds, dtype=float_type)
    n_pmts = sigma_domain.shape[0]

    # 4) Build the “exact likelihood” function (Numba‐compiled) outside.
    n_pe_domain, n_ph_domain = gen_pe_ph_domain(switching_signal, n_sigma, p, float_type)
    exact_likelihood_function = exact_likelihood_generator(
        n_pe_domain=n_pe_domain,
        n_ph_domain=n_ph_domain,
        p=p,
        float_type=float_type
    )
    # At this point, exact_likelihood_function(...) is a Numba‐compiled function
    # with signature (x_scalar, mu_scalar, sigma_scalar) -> float.

    # 6) Now define a top‐level (module‐scope) Numba JIT.  It must explicitly
    #    take every array and scalar it needs—no closures allowed.
    @njit(boundscheck=False, parallel=parallel)
    def gen_LUT_jit(
        x_domain:      np.ndarray,  # shape (nx,)
        mu_domain:     np.ndarray,  # shape (nmu,)
        sigma_domain:  np.ndarray,  # shape (n_pmts,)
        nmu:           int,
        nx:            int,
        n_pmts:        int,
        exact_like_fn,          # function: (float, float, float) -> float
        LUT: np.ndarray,  # shape (nmu, nx, n_pmts)
        LUT_MLE: np.ndarray,  # shape (nx, n_pmts)
        float_type: Type[np.floating] = np.float32
    ) -> tuple[np.ndarray, np.ndarray]:

        # Fill LUT[idx, i, j] = exact_like_fn(x_domain[i], mu_domain[idx], sigma_domain[j])
        for idx in prange(nmu):
            mu_val = mu_domain[idx]
            for i in range(nx):
                x_val = x_domain[i]
                for j in range(n_pmts):
                    std_val = sigma_domain[j]
                    val = exact_like_fn(x_val, mu_val, std_val)
                    LUT[idx, i, j] = val
                    # update max in one go:
                    if val > LUT_MLE[i, j]:
                        LUT_MLE[i, j] = val

        return LUT, LUT_MLE

    # 7) Call the JIT function with all arguments explicitly
    # Allocate LUT (nmu × nx × n_pmts) and LUT_MLE (nx × n_pmts)
    LUT     = np.zeros((nmu, nx, n_pmts), dtype=float_type)
    LUT_MLE = np.zeros((nx,    n_pmts),    dtype=float_type)
    LUT, LUT_MLE = gen_LUT_jit(
        x_domain,
        mu_domain,
        sigma_domain,
        nmu,
        nx,
        n_pmts,
        exact_likelihood_function,
        LUT,
        LUT_MLE,
        float_type
    )

    return LUT, LUT_MLE, x_domain, mu_domain


# --- gen_nearest_index_1d (Numba only) ---
@njit(boundscheck=False)
def gen_nearest_index_1d(
    grid: np.ndarray,
    values: np.ndarray
) -> np.ndarray:
    """
    Numba nopython version: find nearest index for each value.
    """
    n = values.shape[0]
    out = np.empty(n, dtype=np.int32)
    N = grid.shape[0]

    for i in range(n):
        xi = values[i]
        pos = np.searchsorted(grid, xi)
        if pos == 0:
            idx = 0
        elif pos == N:
            idx = N - 1
        else:
            left = grid[pos - 1]
            right = grid[pos]
            if (xi - left) <= (right - xi):
                idx = pos - 1
            else:
                idx = pos
        out[i] = idx

    return out


# --- get_likelihood_from_LUT_generator (Numba only) ---
def get_likelihood_from_LUT_generator(
    x_domain:  np.ndarray,   # 1D float32 array
    mu_domain: np.ndarray,   # 1D float32 array
    LUT:        np.ndarray    # 3D float32 array of shape (nmu, nx, n_pmts)
) -> Callable[[np.float32, np.float32, np.int32], np.float32]:
    """
    Returns a scalar JIT function f(x: float32, mu: float32, sigma_idx: int32) -> float32
    which finds the nearest indices in x_domain & mu_domain, then returns LUT[mu_idx, x_idx, sigma_idx].
    """
    nx  = x_domain.size
    nmu = mu_domain.size

    @njit(boundscheck=False)
    def get_likelihood_from_LUT_numba(
        x:         np.float32,
        mu:        np.float32,
        sigma_idx: np.int32
    ) -> np.float32:
        # 1) find x‐index (nearest neighbor in x_domain)
        xpos = np.searchsorted(x_domain, x)
        if xpos == 0:
            x_idx = 0
        elif xpos == nx:
            x_idx = nx - 1
        else:
            left  = x_domain[xpos - 1]
            right = x_domain[xpos]
            if (x - left) <= (right - x):
                x_idx = xpos - 1
            else:
                x_idx = xpos

        # 2) find mu‐index (nearest neighbor in mu_domain)
        mpos = np.searchsorted(mu_domain, mu)
        if mpos == 0:
            mu_idx = 0
        elif mpos == nmu:
            mu_idx = nmu - 1
        else:
            leftm  = mu_domain[mpos - 1]
            rightm = mu_domain[mpos]
            if (mu - leftm) <= (rightm - mu):
                mu_idx = mpos - 1
            else:
                mu_idx = mpos

        # 3) sigma_idx is already int32, so perform a direct integer index
        return LUT[mu_idx, x_idx, sigma_idx]

    return get_likelihood_from_LUT_numba



# --- get_MLEfrom_LUT_generator (Numba only, scalar) ---
def get_MLEfrom_LUT_generator(
    x_domain:  np.ndarray,   # 1D float32 array
    LUT_MLE:   np.ndarray    # 2D float32 array of shape (nx, n_pmts)
) -> Callable[[np.float32, np.int32], np.float32]:
    """
    Returns a scalar JIT function f(x: float32, sigma_idx: int32) -> float32
    which finds the nearest index in x_domain, then returns LUT_MLE[x_idx, sigma_idx].
    """
    nx = x_domain.size

    @njit(boundscheck=False)
    def get_MLE_from_LUT_numba(
        x:         np.float32,
        sigma_idx: np.int32
    ) -> np.float32:
        # 1) find x‐index
        xpos = np.searchsorted(x_domain, x)
        if xpos == 0:
            x_idx = 0
        elif xpos == nx:
            x_idx = nx - 1
        else:
            left  = x_domain[xpos - 1]
            right = x_domain[xpos]
            if (x - left) <= (right - x):
                x_idx = xpos - 1
            else:
                x_idx = xpos

        # 2) perform integer lookup
        return LUT_MLE[x_idx, sigma_idx]

    return get_MLE_from_LUT_numba


# --- lossFuncGenerator (Numba only) ---
def lossFuncGenerator(
    likelihoodFunction: Callable,
    ratio: bool = True,
    likelihoodMLEFunction: Optional[Callable] = None,
    log: bool = False,
    float_type: Type[np.floating] = np.float32
) -> Callable:
    """
    Generate negative‐log‐likelihood or -2 log‐likelihood ratio (Numba only).
    """
    min_val = np.finfo(float_type).tiny

    @njit(boundscheck=False)
    def neg_log_likelihood(
        x: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        if log:
            return -likelihoodFunction(x, mu, sigma)
        else:
            return -np.log(np.maximum(likelihoodFunction(x, mu, sigma), min_val))

    @njit(boundscheck=False)
    def neg_log_likelihood_ratio(
        x: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        if log:
            logL = likelihoodFunction(x, mu, sigma)
            logL_mle = likelihoodMLEFunction(x, sigma)
            return -2.0 * (logL - logL_mle)
        else:
            L = np.maximum(likelihoodFunction(x, mu, sigma), min_val)
            L_mle = np.maximum(likelihoodMLEFunction(x, sigma), min_val)
            out = -2.0 * (np.log(L) - np.log(L_mle))
            return out

    if ratio:
        if likelihoodMLEFunction is None:
            raise ValueError(
                "likelihoodMLEFunction must be provided when ratio=True."
            )
        return neg_log_likelihood_ratio
    else:
        return neg_log_likelihood


# --- compute_common_std_gen (Numba only) ---
def compute_common_std_gen(
    float_type: Type[np.floating] = np.float32
) -> Callable[
    [Annotated[np.ndarray, (None,), float], Annotated[np.ndarray, (None,), float], float],
    Annotated[np.ndarray, (None,), float]
]:
    @njit(boundscheck=False)
    def compute_common_std(
        mu: np.ndarray,
        std: np.ndarray,
        p_dpe: float
    ) -> np.ndarray:
        one_dtype = float_type(1.0)
        npe_mean = mu * (one_dtype + p_dpe)
        combined = (
            mu * (one_dtype + p_dpe) ** 2 +
            mu * p_dpe * (1 - p_dpe) +
            (np.sqrt(np.abs(npe_mean)) * std ** 2)
        )
        return np.sqrt(np.maximum(np.abs(combined), float_type(1e-6)))

    return compute_common_std


# --- gaussian_approx (Numba only) ---
def gaussian_approx(
    p_dpe: float,
    float_type: Type[np.floating] = np.float32
) -> Callable[
    [Annotated[np.ndarray, (None,), float], Annotated[np.ndarray, (None,), float], Annotated[np.ndarray, (None,), float]],
    Annotated[np.ndarray, (None,), float]
]:
    """
    Returns a callable that computes the Gaussian log probability (Numba only).
    """
    compute_common_std = compute_common_std_gen(float_type=float_type)
    half = float_type(0.5)
    const = half * np.log(float_type(2.0) * float_type(np.pi))
    @njit(boundscheck=False)
    def gaussian_approximation(
        x: np.ndarray,
        mu: np.ndarray,
        std: np.ndarray
    ) -> np.ndarray:
        one_dtype = float_type(1.0)
        npe_mean = mu * (one_dtype + p_dpe)
        combined_std = compute_common_std(mu, std, p_dpe)
        return (
            -half * ((x - npe_mean) / combined_std) ** 2
            - np.log(combined_std)
            - const
        )

    return gaussian_approximation


# --- gaussian_approx_MLE (Numba only) ---
def gaussian_approx_MLE(
    p_dpe: float,
    float_type: Type[np.floating] = np.float32
) -> Callable:
    """
    Returns a callable that computes the Gaussian log likelihood at MLE (mu = x) (Numba only).
    """
    gaussian_log_likelihood = gaussian_approx(p_dpe=p_dpe, float_type=float_type)

    @njit(boundscheck=False)
    def gaussian_likelihood_MLE(
        x: np.ndarray,
        std: np.ndarray
    ) -> np.ndarray:
        return gaussian_log_likelihood(x, x, std)

    return gaussian_likelihood_MLE


# --- parent_gen (Numba only) ---
def parent_gen(
    gaussian_stds: Annotated[np.ndarray, (None,), float],
    method: Literal["LUT", "Exact"] = "LUT",
    return_ratio: bool = True,
    switching_signal: float = 40.0,
    p_dpe: float = 0.2,
    n_sigma: int = 5,
    m: int = 5,
    float_type: Type[np.floating] = np.float32,
    lut_Things: Optional[tuple] = None,
    parallel: bool = True
) -> Callable[
    [Annotated[np.ndarray, (None, None), float],
     Annotated[np.ndarray, (None, None), float]],
    Annotated[np.ndarray, (None, None), float]
]:
    """
    Numba‐only full loss function generator.
    """
    p_dpe = float_type(p_dpe)
    switching_signal = float_type(switching_signal)
    gaussian_stds = np.asarray(gaussian_stds, dtype=float_type)

    # 1) Build "Exact" or "LUT"
    if method.lower() == "exact":
        n_pe_domain, n_ph_domain = gen_pe_ph_domain(
            switching_signal, n_sigma, p_dpe, float_type
        )
        likelihoodFunction_exact = exact_likelihood_generator(
            n_pe_domain=n_pe_domain,
            n_ph_domain=n_ph_domain,
            p=p_dpe,
            float_type=float_type
        )
        if return_ratio:
            likelihoodMLEFunction_exact = exact_likelihood_MLE_generator()
        else:
            likelihoodMLEFunction_exact = None

    else:  # method == "lut"
        if lut_Things is not None:
            # Unpack the precomputed LUT things
            LUT, LUT_MLE, x_domain, mu_domain = lut_Things
        else:
            LUT, LUT_MLE, x_domain, mu_domain = generate_LUT(
                m=m,
                p=p_dpe,
                switching_signal=switching_signal,
                gaussian_stds=gaussian_stds,
                n_sigma=n_sigma,
                float_type=float_type,
                parallel = parallel
            )
        likelihoodFunction_exact = get_likelihood_from_LUT_generator(
            x_domain=x_domain,
            mu_domain=mu_domain,
            LUT=LUT,
        )
        if return_ratio:
            likelihoodMLEFunction_exact = get_MLEfrom_LUT_generator(
                x_domain=x_domain,
                LUT_MLE=LUT_MLE,
            )
        else:
            likelihoodMLEFunction_exact = None

    # 2) Gaussian‐approx
    gaussian_approx_func = gaussian_approx(
        p_dpe=p_dpe,
        float_type=float_type
    )
    if return_ratio:
        likelihoodMLEFunction_Gauss = gaussian_approx_MLE(
            p_dpe=p_dpe,
            float_type=float_type
        )
    else:
        likelihoodMLEFunction_Gauss = None

    # 3) Negative‐log wrappers
    lossfunc_exact = lossFuncGenerator(
        likelihoodFunction=likelihoodFunction_exact,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunction_exact,
        log = False,
        float_type=float_type
    )
    lossfunc_Gauss = lossFuncGenerator(
        likelihoodFunction=gaussian_approx_func,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunction_Gauss,
        log=True,
        float_type=float_type
    )

    # 4) compute_scaling (Numba)
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

    # 5) Numba lossfunc
    # 5) Numba lossfunc
    def _make_lossfunc_numba() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if method.lower() == "exact":
            @njit(boundscheck=False)
            def idx_or_value(xi, mui, sj, j):
                return lossfunc_exact(xi, mui, sj)
        else:
            @njit(boundscheck=False)
            def idx_or_value(xi, mui, sj, j):
                return lossfunc_exact(xi, mui, np.int32(j))
        
        @njit(boundscheck=False, parallel= parallel)
        def lossfunc_numba(
            x: np.ndarray,
            mu: np.ndarray
        ) -> np.ndarray:
            B, N = x.shape
            scaling = compute_scaling_nb(x)
            result = np.empty((B, N), dtype=float_type)
            for i in prange(B):
                scale_i = scaling[i, 0]
                for j in range(N):
                    # pull out scalar inputs
                    xi  = float_type(x[i, j])
                    mui = float_type(mu[i, j] * scale_i)
                    sj  = float_type(gaussian_stds[j])

                    if (mui > switching_signal) or (xi > switching_signal):
                        # Gaussian branch takes (x:float, mu:float, sigma:float)
                        out_arr = lossfunc_Gauss(xi, mui, sj)
                    else:
                        out_arr = idx_or_value(xi, mui, sj, j)

                    result[i, j] = out_arr

            return result

        return lossfunc_numba

    lossfunc_nb = _make_lossfunc_numba()
    return lossfunc_nb
