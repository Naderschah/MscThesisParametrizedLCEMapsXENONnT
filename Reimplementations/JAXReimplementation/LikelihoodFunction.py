


"""
Order of functions is:

exact likelihood implementation

LUT implementation

negloglikelihood and ratio callers

Gaussian Exact methods

Parent Call

"""

from . type_names import *


import jax.numpy as jnp
import jax



epsilon = 1e-10
float_type = jnp.float32


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
        -0.5 * ((x - loc) / scale) ** 2
        - jnp.log(scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )


def gen_pe_ph_domain(switching_signal   : Int, 
                     n_sigma            : Int, 
                     p_dpe              : Float
                     ) -> tuple[
                        Annotated[Array, (None,), float],
                        Annotated[Array, (None,), float]
                    ]:
    """
    Generate the photon and photoelectron domain for the likelihood function
    """
    n_pe_domain = jnp.arange(0., switching_signal + n_sigma*jnp.sqrt(switching_signal)+2)
    n_ph_domain = jnp.arange(0., switching_signal/(1+p_dpe) + n_sigma*jnp.sqrt(switching_signal/(1+p_dpe))+2)
    return n_pe_domain, n_ph_domain


def exact_likelihood_generator(n_pe_domain : Annotated[Array, (None,), float], 
                               n_ph_domain : Annotated[Array, (None,), float], 
                               p : float,
                               ) -> Callable:
    """
    Generate callable for the exact likelihood function
    """
    n_ph_grid : Annotated[Array, (1, None, 1), float] = jnp.reshape(n_ph_domain, (1, -1, 1))
    n_pe_grid : Annotated[Array, (1, 1, None), float] = jnp.reshape(n_pe_domain, (1, 1, -1))

    def exact_gaussian(x        : Annotated[Array, (None, 1, 1), float], 
                       sigma    : Annotated[Array, (None, 1, 1), float]
                       )        -> Annotated[Array, (None, 1, None), float]:
        """
        Computes the gaussian component over the allowed photoelectron grid
        x: Observed Signal
        sigma: Gaussian Variance 
        """
        scale : Annotated[Array, (None, 1, None), float] = jnp.maximum(sigma * jnp.sqrt(n_pe_grid), epsilon)
        mean  : Annotated[Array, (1, 1, None), float]    = n_pe_grid
        res   : Annotated[Array, (None, 1, None), float] =  gaussian_log_likelihood(x, mean, scale)
        return res
    
    
    def binomial_pmf(n : Annotated[Array, (1, None, 1), float],
                     p : float, 
                     k : Annotated[Array, (1, None, None), float]
                     ) -> Annotated[Array, (1, None, None), float]:
        """
        Computes the binomial probability mass function
        Handles nonsense k (<0) by placing null probability
        """
        k_safe = jnp.clip(k, 0, n)
        valid = ((k <= n) & (k >= 0)).astype(dtype=float_type)
        
        log_unnormalized = jnp.log(p) * k_safe + jnp.log1p(-p) * (n - k_safe)
        log_normalization = jax.scipy.special.betaln(1. + k_safe, 1. + n - k_safe) + jnp.log(n + 1)

        return valid * jnp.exp(log_unnormalized - log_normalization)

    def poisson_pmf(n   : Annotated[Array, (1, None, 1), float], 
                    mu  : Annotated[Array, (None, 1, 1), float]
                    )  -> Annotated[Array, (None, None, 1), float]:
        """
        Computes the Poisson probability mass function
        Handles mu (= 0) by placing null probability
        """
        mask = (mu > 0.0)
        valid = mask.astype(float_type)
        safe_mu = jnp.where(mask, mu, 1.0)
        log_p = n * jnp.log(safe_mu) - mu - jax.scipy.special.gammaln(n + 1.0)
        return valid * jnp.exp(log_p)

    def exact_likelihood(x      : Annotated[Array, (None,), float], 
                         mu     : Annotated[Array, (None,), float], 
                         sigma  : Annotated[Array, (None,), float]
                         )      -> Annotated[Array, (None,), float]:
        """
        x : Observed Signal  : [n]
        mu : Expected Signal  : [n]
        sigma : Gaussian Variance : [n]
        """
        x       : Annotated[Array, (None, 1, 1), float] = jnp.reshape(x, (-1, 1, 1))
        mu      : Annotated[Array, (None, 1, 1), float] = jnp.reshape(mu, (-1, 1, 1))
        sigma   : Annotated[Array, (None, 1, 1), float] = jnp.reshape(sigma, (-1, 1, 1))
        # Gaussian component 
        a       : Annotated[Array, (None, 1, None), float] = jnp.exp(exact_gaussian(x, sigma))
        # Binomial component
        b       : Annotated[Array, (1, None, None), float] = binomial_pmf(n_ph_grid, p, n_pe_grid - n_ph_grid)
        # Poisson Component
        c       : Annotated[Array, (None, None, 1), float] = poisson_pmf(n_ph_grid, mu)
        # Double Sum
        res     : Annotated[Array, (None,), float] = jnp.sum(a * b * c, axis=[1,2])
        return res
    
    return exact_likelihood

def exact_likelihood_MLE_generator():
    return NotImplementedError("Exact likelihood MLE generator needs to be implemented here")

def generate_LUT(m : int, 
                 p : float, 
                 switching_signal : float, 
                 gaussian_stds : Annotated[Array, (None,), float],
                 n_sigma : int = 5,
                 obs_min : float = -3.,
                 ) -> tuple[Annotated[Array, (None, None, None), float], Annotated[Array, (None, None), float], Annotated[Array, (None,), float], Annotated[Array, (None,), float]]:
    """
    Computes the Look up table from the exact method 

    return LUT and LUT MLE

    dimensions contain (mu, x, sigma)
    where mu and x must be searched, and sigma corresponds to the PMT index or rather index in which stds are passed
    """
    # Build x, mu and sigma domain
    i_vals      : Annotated[Array, (None, )] = jnp.arange(obs_min, switching_signal, dtype=float_type)
    sub_vals    : Annotated[Array, (None, )] = jnp.arange(float(m), dtype=float_type) / m
    x_domain    : Annotated[Array, (None, )] = jnp.concatenate([(i_vals[:, None] + sub_vals[None, :]).ravel(), jnp.array([switching_signal], dtype=float_type)])
    nx          : int = x_domain.shape[0]

    i_vals      : Annotated[Array, (None, )] = jnp.arange(0., switching_signal, dtype=float_type)
    sub_vals    : Annotated[Array, (None, )] = jnp.arange(float(m), dtype=float_type) / m
    mu_domain   : Annotated[Array, (None, )] = jnp.concatenate([(i_vals[:, None] + sub_vals[None, :]).ravel(), jnp.array([switching_signal], dtype=float_type)])
    nmu         : int = mu_domain.shape[0]

    sigma_domain = jnp.array(gaussian_stds, dtype=float_type)
    n_pmts      : int = sigma_domain.shape[0]

    # Generate evaluation domain and exact likelihood function
    n_pe_domain, n_ph_domain = gen_pe_ph_domain(switching_signal = switching_signal,
                                                        n_sigma = n_sigma,
                                                        p_dpe = p
                                                        )

    exact_likelihood_function = exact_likelihood_generator(
                                                        n_pe_domain=n_pe_domain,
                                                        n_ph_domain=n_ph_domain,
                                                        p=p
                                                    )

    # Iterate over mu (obscene amounts of ram otherwise since in mem shape before sum is (nmu, nx, n_pmts, n_pe_domain, n_ph_domain))
    
    # Generate empty LUT
    LUT     : Annotated[Array, (nmu, nx, n_pmts), float] = jnp.zeros((nmu, nx, n_pmts), dtype=float_type)
    for idx in range(len(LUT)):
        x_grid   = jnp.repeat(x_domain[:, None], n_pmts, axis=1).ravel()
        mu_grid   = jnp.full_like(x_grid, mu_domain[idx])
        std_grid = jnp.tile(sigma_domain[None, :], (nx, 1)).ravel()
        probs = exact_likelihood_function(x_grid, mu_grid, std_grid)
        LUT = LUT.at[idx].set(probs.reshape(nx, n_pmts))
    # Find per x MLE mu Likelihoods : L(x|mu_MLE)
    LUT_MLE : Annotated[Array, (nx, n_pmts), float] = jnp.max(LUT, axis=0)

    return LUT, LUT_MLE, x_domain, mu_domain


def neaerest_index_1d(grid   : Annotated[Array, (None,), float],
                          values : Annotated[Array, (None,), float],
                            )    -> Annotated[Array, (None,), int]:
    """
    Find the index of the nearest value in a 1D array

    Used in LUT lookups
    """
    idx_float = jnp.searchsorted(grid, values, side='left')  # shape = values.shape

    idx_below = jnp.clip(idx_float - 1, 0, grid.size - 1)
    idx_above = jnp.clip(idx_float, 0, grid.size - 1)

    below_vals = grid[idx_below]
    above_vals = grid[idx_above]

    dist_below = jnp.abs(values - below_vals)
    dist_above = jnp.abs(values - above_vals)

    idx_nearest = jnp.where(dist_above < dist_below, idx_above, idx_below)
    return idx_nearest.astype(jnp.int32)

def get_likelihood_from_LUT_generator(x_domain : Annotated[Array, (None,), float],
                                        mu_domain : Annotated[Array, (None,), float], 
                                        LUT : Annotated[Array, (None, None, None), float], 
                                        ) -> Callable:
    """Returns lookup method
    domains and LUT baked into the function directly
    """

    def get_likelihood_from_LUT(
                                x : Annotated[Array, (None,), float],
                                mu : Annotated[Array, (None,), float],
                                sigma_idx : Annotated[Array, (None,), int],
                                ) -> Annotated[Array, (None,), float]:
        """
        Get the likelihood from the LUT

        mu : Expected Signal  : [n]
        x : Observed Signal  : [n]
        sigma_idx : PMT index : [n]
        """
        x_idx  : Annotated[Array, (None,), int] = neaerest_index_1d(x_domain, x)
        mu_idx : Annotated[Array, (None,), int] = neaerest_index_1d(mu_domain, mu)

        coords = jnp.stack((mu_idx, x_idx, sigma_idx), axis=-1)
        return LUT[tuple(coords.T.astype(jnp.int32))]
    return get_likelihood_from_LUT

def get_MLEfrom_LUT_generator(
                                        x_domain : Annotated[Array, (None,), float], 
                                        LUT_MLE : Annotated[Array, (None, None), float], 
                                        ) -> Callable:
    """Returns lookup method
    domain and LUT MLE baked into the function directly
    """

    def get_MLE_from_LUT(
                        x : Annotated[Array, (None,), float],
                        sigma_idx : Annotated[Array, (None,), int],
                        ) -> Annotated[Array, (None,), float]:
        """
        Get the likelihood from the LUT

        mu : Expected Signal  : [n]
        x : Observed Signal  : [n]
        sigma_idx : PMT index : [n]
        """
        x_idx : Annotated[Array, (None,), int] = neaerest_index_1d(x_domain, x)

        coords = jnp.stack((x_idx, sigma_idx), axis=-1)
        # Each axes gets its own array so we have [3, batch] with tupple becomes the same as (batch, batch, batch)
        return LUT_MLE[tuple(coords.T.astype(jnp.int32))]
    
    return get_MLE_from_LUT


def lossFuncGenerator(
    likelihoodFunction: Callable,
    ratio: bool = True,
    likelihoodMLEFunction: Optional[Callable] = None,
    log: bool = False,
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
    min_val = jnp.finfo(float_type).tiny
    # Negative log-likelihood when not taking a ratio
    def neg_log_likelihood(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        sigma: Annotated[Array, (None,), Union[float, int]],
    ) -> Annotated[Array, (None,), float]:
        if log:
            # likelihoodFunction returns log L already
            # so - log L → negative log-likelihood
            return -likelihoodFunction(x, mu, sigma)
        else:
            # likelihoodFunction returns L; we do -log (max(L, ε))
            return -jnp.log(jnp.maximum(likelihoodFunction(x, mu, sigma), min_val))

    # Negative 2 log-likelihood ratio
    def neg_log_likelihood_ratio(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        sigma: Annotated[Array, (None,), Union[float, int]]
    ) -> Annotated[Array, (None,), float]:
        if log:
            # likelihoodFunction returns log L, same for MLE
            logL = likelihoodFunction(x, mu, sigma)
            logL_mle = likelihoodMLEFunction(x, sigma)
            return -2.0 * (logL - logL_mle)
        else:
            # likelihoodFunction returns L; take log then ratio
            L = jnp.maximum(likelihoodFunction(x, mu, sigma), min_val)
            L_mle = jnp.maximum(likelihoodMLEFunction(x, sigma), min_val)
            return -2.0 * (jnp.log(L) - jnp.log(L_mle))

    if ratio:
        if likelihoodMLEFunction is None:
            raise ValueError(
                "likelihoodMLEFunction must be provided when ratio=True. "
                "If log=False, it should return L(x|μ_MLE,σ). "
                "If log=True, return log L(x|μ_MLE,σ)."
            )
        return neg_log_likelihood_ratio
    else:
        return neg_log_likelihood


def compute_common_std(mu   : Annotated[Array, (None,), float], 
                       std  : Annotated[Array, (None,), float], 
                       p_dpe: float) -> Annotated[Array, (None,), float]: 
    """
    Computes an effective standard deviation combining Poisson, Binomial, and Gaussian contributions.
    Args:
        mu: expected signal (jnp array)
        std: Gaussian std component (jnp array)
        p_dpe: double photoelectron probability (scalar)
    Returns:
        combined standard deviation (jnp array)
    """
    npe_mean = mu * (1 + p_dpe)
    combined = (
        mu * (1 + p_dpe)**2 +                # Poisson component
        mu * p_dpe * (1 - p_dpe) +           # Binomial component
        (jnp.sqrt(jnp.abs(npe_mean)) * std**2)  # Gaussian component
    )
    return jnp.sqrt(jnp.maximum(jnp.abs(combined), 1e-6))  # avoid underflow

def gaussian_approx(
                p_dpe: float,
            ) -> Callable[
                [Annotated[Array, (None,), float], Annotated[Array, (None,), float], Annotated[Array, (None,), float]],
                Annotated[Array, (None,), float]
            ]:
    """
    Returns a callable function that computes the Gaussian log probability:
    N(mu * (1 + p_dpe), compute_common_std(mu, std))

    Args:
        p_dpe: Double photoelectron probability (scalar)
        epsilon: Small value to avoid log(0)
    Returns:
        Callable that takes (x, mu, std) and returns log probability [n]
    """
    
    def gaussian_approximation(
        x: Annotated[Array, (None,), float],
        mu: Annotated[Array, (None,), float],
        std: Annotated[Array, (None,), float]
    ) -> Annotated[Array, (None,), float]:
        npe_mean: Annotated[Array, (None,), float] = mu * (1 + p_dpe)
        combined_std: Annotated[Array, (None,), float] = compute_common_std(mu, std, p_dpe)

        res : Annotated[Array, (None,), float] = gaussian_log_likelihood(x, npe_mean, combined_std)

        return res

    return gaussian_approximation

def gaussian_approx_MLE(
        p_dpe: float,
    ) -> Callable:
    """
    Returns a callable that computes the Gaussian log‐likelihood at the MLE (mu = x):
      log N(x | x*(1+p_dpe), compute_common_std(x, std)^2)

    Args:
      p_dpe: Double photoelectron probability (scalar)
      epsilon: Small floor to avoid log(0)
    Returns:
      A function taking (x, std) and returning log‐likelihood [n]
    """
    def gaussian_likelihood_MLE(
            x: Annotated[Array, (None,), float],
            std: Annotated[Array, (None,), float]
        ) -> Annotated[Array, (None,), float]:

        npe_mean = x * (1 + p_dpe)
        combined_std = compute_common_std(x, std, p_dpe)   # shape [n]

        # log‐likelihood = −0.5*((x – loc)/scale)^2 − log(scale) − 0.5*log(2π)
        return gaussian_log_likelihood(x, npe_mean, combined_std)

    return gaussian_likelihood_MLE

def parent_gen(
        gaussian_stds : Annotated[Array, (None,), float],
        method : Literal["LUT", "Exact"] = "LUT",
        return_ratio: bool = True,
        switching_signal : float = 40.,
        p_dpe : float = 0.2,
        n_sigma : int = 5,
        m : int = 5,
    ):
    """
    Generates the full call method including all generations that are required
    
    """
    gaussian_stds = jnp.asarray(gaussian_stds, dtype = float_type) 
    # Generate call methods
    if method.lower() == "exact":
        n_pe_domain, n_ph_domain = gen_pe_ph_domain(switching_signal=switching_signal,
                                                    n_sigma=n_sigma,
                                                    p_dpe=p_dpe
                                                    )
        likelihoodFunction = exact_likelihood_generator(
                                                    n_pe_domain=n_pe_domain, 
                                                    n_ph_domain=n_ph_domain,
                                                    p=p_dpe
                                                )
        if return_ratio:
            # FIXME : Not Implemented - throws exception
            likelihoodMLEFunction = exact_likelihood_MLE_generator()
        else:
            likelihoodMLEFunction = None

    elif method.lower() == "lut":
        # Generate the LUT
        LUT, LUT_MLE, x_domain, mu_domain = generate_LUT(m=m, 
                                                        p=p_dpe, 
                                                        switching_signal=switching_signal, 
                                                        gaussian_stds=gaussian_stds, 
                                                        n_sigma=n_sigma)
        
        # Generate the lookup methods
        likelihoodFunction = get_likelihood_from_LUT_generator(
                                                x_domain=x_domain,
                                                mu_domain=mu_domain,
                                                LUT=LUT
                                            )
        if return_ratio:
            likelihoodMLEFunction = get_MLEfrom_LUT_generator(
                                                    x_domain=x_domain,
                                                    LUT_MLE=LUT_MLE
                                                )
        else:
            likelihoodMLEFunction = None
    
    # Generate the call function 
    lossfunc_exact = lossFuncGenerator(
        likelihoodFunction=likelihoodFunction,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunction,
    )

    likelihoodFunctionGauss = gaussian_approx(
                                            p_dpe = p_dpe,
                                            )

    if return_ratio:
        likelihoodMLEFunctionGauss = gaussian_approx_MLE(
                                            p_dpe = p_dpe,
                                            )
    else:
        likelihoodMLEFunctionGauss = None

    lossfunc_Gauss = lossFuncGenerator(
        likelihoodFunction=likelihoodFunctionGauss,
        ratio=return_ratio,
        likelihoodMLEFunction=likelihoodMLEFunctionGauss,
        log = True,
    )

    def lossfunc(*,
            x: Annotated[Array, (None,None,), float],
            mu: Annotated[Array, (None,None,), float],
        ) -> Annotated[Array, (None,None,), float]:
        """
        Per-element partitioned evaluation using lax.cond.

        Note keyword argument only > Too many times did i mess up the argument order and lost hours
        """
        # Generate the indeces
        std_idx_grid = jnp.broadcast_to(               # indices 0..n_pmts-1
            jnp.arange(gaussian_stds.shape[0], dtype=jnp.int32), x.shape
        )
        std_grid = jnp.broadcast_to(               # physical σ values
            gaussian_stds, x.shape
        )
        # Scale Prediciton
        scaling = jnp.sum(jnp.where(x > 0, x, 0), axis=-1, keepdims=True)  # (B,1)
        mu_scaled      = mu * scaling

        def per_element(xi, mui, sigma_val, sigma_idx):

            use_gauss = jnp.logical_or(mui > switching_signal, xi > switching_signal)
            # Outer conditional use to select between exact and Gaussian methods
            # Inner coditional select idx or val of std for LUT vs exact methods 
            return jax.lax.cond(
                use_gauss,
                lambda *_: lossfunc_Gauss(xi[None], mui[None], sigma_val[None])[0],
                lambda *_: jax.lax.cond( method.lower() == 'exact', 
                                        lambda *_: lossfunc_exact(xi[None], mui[None], sigma_val[None])[0],
                                        lambda *_: lossfunc_exact(xi[None], mui[None], sigma_idx[None])[0],
                                        ),
                operand=None,
                )
        # Apply vmap to per_element functions
        result = jax.vmap(per_element)(x.ravel(), mu_scaled.ravel(), std_grid.ravel(), std_idx_grid.ravel())
        #result = lossfunc_Gauss(x.ravel(), mu_scaled.ravel(), std_grid.ravel())
        return result.reshape(x.shape)

    
    return jax.jit(lossfunc)