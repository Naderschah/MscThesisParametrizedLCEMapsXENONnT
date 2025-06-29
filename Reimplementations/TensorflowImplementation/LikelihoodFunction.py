

"""

The Likelihood function is implemented in this file (single class)

It is implemented seperately to allow seperating of functional documentation and explanation of the function


The Likelihood is comprised of three elements 

L_{full}(x|\mu) = \sum^\infty_{N=0}L_P(N|\mu) \sum^N_{k=0} L_B(k|N,p) L_{G}(x|N+k ,\sigma\sqrt{N+k})

Where x is the observed signal, and mu is the number of photons expected as predicted by the model

N is the number of photoelectrons liberated in the photocathode
k the number of double photoelectron events 
N+k is then the total number of photoelectrons inside the PMT
and this signal is then passed through the gaussian to represent miscellaneous noise sources

The double sum arrises as a number of N and k combinations are possible that allow for some x given some mu

The implementation has to deal with a few things
- The infinite sum (assume poisson goes gaussian and use tis sigma to define a 5 sigma interval)
- same thing for the second sum 
- this takes very long to execute -> hence Look Up Table 
- The gaussian approximation past the switching signal, the mean for photons is x/1.2 due to double photoelectron and variance is constructed from the gaussian limit of each distribution

The function can be configured to run in:
- exact mode : Compute everything explicitly
- LUT_trainable_std : Compute a look up table for the defined standard deviations when set_mode is called and use this to compute the likelihood
- LUT_untrainable_std : Compute a look up table for pre segmented standard deviations (number of stds computed controlled by z parameter)

Each has two subtypes:
- Compute likelihood
- COmpute chi^2 

For chi^2 the MLE in the LUT implementations is directly taken from the LUT 
In the exact implementation a trianed neural net is used (only works for sigma >0.3 below errors are massive and naive approximateion x_MLE = x/ 1.2 is better)


TODO: Mle_estimator path shopuld eb predefined to point to trained model 
Import minmaxconstraint
Double Check dockstrings
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np 
import tensorflow_probability as tfp

from .custom_constrains import MinMaxConstraint


class LikelihoodRatio(keras.layers.Layer):
    """
    LikelihoodRatio Layer

    FIXME: 
    - Need to Implement MLE arguments - Likely will be NN trained for this so should provide a method for training and saving this from the LUT or exact also should implement callable that is supplied to get MLE 
    - For now likelihood ratio does not work
    - Added the return_ratio argument - Completely undocumented all docs higher level than computational need to be updated 


    Overview:
        This class implements a likelihood evaluator with several evaluation modes that 
        balance exact, computationally expensive calculations with fast approximate 
        lookup table (LUT) based evaluations. The layer supports both Gaussian-based 
        approximations and full exact evaluations. Furthermore, it offers variants that 
        either use trainable standard deviations (variable std) or fixed standard 
        deviations (via direct index mapping) in the LUT. NaN safety can be enabled 
        so that any NaN inputs are replaced with a large, user-defined loss value 
        (useful for position refinement tasks).

    Physics Explanation:
        Here we compute the likelihood ratio or likelihood depending on the mode, this considers the
        phenomena of Poisson photon detection variance, Binomial DPE emission, and Gaussian Noise in 
        components past the PMT photocathode, the Gaussian component is a simplification, and technically
        this implementation is laking a description of: PMT vacuum contamination, dynode noise, electronic noise
        also it is assuming a flat 20% DPE emission probability, which is likely not the case and should be done per PMT

    Call Methods:
        The following evaluation modes can be selected using the set_call_mode() method:
        
        - "exact":
            • Computes the exact likelihood ratio using full evaluation of the underlying 
            probability distributions (Binomial, Poisson, and truncated Gaussian). 
            • This mode is computationally expensive.
            
        - "LUT_trainable_std":
            • Precomputes a lookup table over m subdivisions in the x-domain and z subdivisions 
            over the sigma range (from sigma_min to sigma_max). The lookup uses trainable standard 
            deviation weights. 
            • Returns the negative log likelihood (suitable for training the std values).
            
        - "LUT_untrainable_std":
            • Precomputes a lookup table where the sigma dimension corresponds directly to the 
            pretrained standard deviation (std) indices.
            • Returns the negative log likelihood ratio (i.e. -2*log(L(x|mu)/L(mu|mu))).

        These modes are set by calling set_call_mode() before any layer invocation. Once set, 
        the layer's call() method dispatches to parent_call(), which in turn delegates to one of:
            • gaussian_call (for Gaussian/LUT-based approximations)
            • nongaussian_call (for exact or LUT-based non-Gaussian evaluation)

    Self Attributes - Variables:
        - n_pmts (tf.Variable, int32):
            Number of PMTs; defines dimensions for the trainable weights (std) and for LUT construction.
        - switching_signal (tf.Variable, float32):
            Threshold value used to decide whether to use the exact evaluation or the LUT/approximation.
        - n_sigma (tf.Variable, float32):
            Number of sigma values used for exact computation.
        - p_dpe (tf.Variable, float32):
            Probability used in the Binomial model to account for double photoelectron (DPE) emission.
        - nan_safe (tf.Variable, bool):
            Flag that determines if NaN inputs should be masked with a large loss value.
        - nan_safe_value (tf.Variable, float32):
            The loss value to assign when inputs are NaN in nan_safe mode.
        - tile_std_indices (tf.Variable, bool):
            Flag that indicates whether standard deviation values should be treated as indices 
            (used in LUT_untrainable_std mode).
        - m (tf.Variable, int32):
            Number of subdivisions for the x-domain, used when precomputing LUTs.
        - z (tf.Variable, int32):
            Number of subdivisions in the sigma range (for LUT_trainable_std mode).
        - sigma_min (tf.Variable, float32):
            Minimum sigma value for LUT precomputation (variable std).
        - sigma_max (tf.Variable, float32):
            Maximum sigma value for LUT precomputation (variable std).
        - n_pe_domain (tf.Tensor):
            Domain for n_pe values (created from a NumPy range and reshaped with extra axes); 
            used in computing the Binomial and Gaussian factors.
        - n_ph_domain (tf.Tensor):
            Domain for n_ph values (created from a NumPy range and reshaped with extra axes);
            used in the Poisson factor computation.

    Self Attributes - Standard Attributes:
        - std (tf.Variable):
            Trainable standard deviation weights (one per PMT), created in build(). Used in LUT 
            computations and Gaussian likelihood approximations.
        - L_table:
            The precomputed lookup table (LUT) for likelihood evaluations; built by either 
            precompute_LUTable_with_std or precompute_LUTable_without_variable_std.
        - gaussian_call:
            A callable (wrapped with tf.function) that computes the Gaussian (or LUT-based) component 
            of the likelihood evaluation.
        - nongaussian_call:
            A callable (wrapped with tf.function) that computes the non-Gaussian (exact or LUT-based) 
            likelihood evaluation.
        - call:
            The overridden Keras layer call method, set to point to parent_call() after configuration via set_call_mode().

    Methods:
        - __init__(n_pmts, switching_signal, n_sigma, sigma_min, sigma_max, p_dpe, nan_safe, nan_safe_value, m, z, **kwargs):
            Initializes the layer variables and domains used for likelihood computation.
        - build(input_shape):
            Builds the layer by creating the trainable std weight.
        - set_call_mode(call_mode):
            Sets the evaluation mode for the layer. Valid modes are "exact", "LUT_trainable_std", 
            and "LUT_untrainable_std". This method also precomputes LUTs as needed and sets 
            gaussian_call, nongaussian_call, and the final call method.
        - parent_call(pred, observed):
            The main call method that preprocesses inputs (scaling, reshaping), handles NaN masking, 
            and dispatches evaluations to gaussian_call or nongaussian_call based on input thresholds.
        - exact_likelihood(x, mu, std):
            Computes the exact likelihood by combining the truncated Gaussian, Binomial, and Poisson factors.
        - exact_gaussian(x, std):
            Computes the Gaussian probability; used within exact_likelihood.
        - gaussian_likelihood_ratio(x, mu, std):
            Computes the -2 log likelihood ratio using the Gaussian approximation.
        - gaussian_likelihood(x, mu, std):
            Computes the log probability from a Gaussian approximation.
        - compute_common_std(mu, std):
            Computes an effective standard deviation that combines Poisson, Binomial, and Gaussian contributions.
        - precompute_LUTable_with_std(m, z):
            Precomputes the LUT for a variable-std mode over m subdivisions in x and z subdivisions in sigma.
        - lookup_negloglikelihood_with_std(x, mu, std):
            Looks up the negative log likelihood from the LUT with variable std.
        - precompute_LUTable_without_variable_std(m):
            Precomputes the LUT for the fixed standard deviation case, using self.std as the sigma domain.
        - lookup_likelihood_without_variable_std(x, mu, sigma_idx):
            Looks up likelihood values from the fixed std LUT using sigma_idx.
        - loglikelihoodratio_without_variable_std(x, mu, sigma_idx):
            Computes the log likelihood ratio using the LUT with fixed std.
        - _nearest_idx_1d(grid, values):
            Helper function that returns the nearest indices in a sorted 1D grid for the given values.
        - differentiable_sigma_idx(std):
            Provides a differentiable mapping to obtain the sigma index from the sigma domain.

    Usage:
        1. Instantiate the layer and call set_call_mode(<mode>) with one of the following strings:
                - "exact"
                - "LUT_trainable_std"
                - "LUT_untrainable_std"
        2. Invoke the layer on predictions and observations; the call() method will process the inputs 
        appropriately (including scaling, reshaping, and NaN handling) and dispatch to the proper 
        evaluation routines based on the selected mode.

    Notes:
        - Ensure that any external attributes (such as self.min_pe used in exact_gaussian) are defined.
        - The physics explanation (to be filled in) should detail the derivation and rationale behind 
        the combined use of Poisson, Binomial, and Gaussian distributions in this likelihood model.

    Possible Class Extensions:
        - More complete Likelihood function
        - Per PMT DPE probability - should be fittable but complex to implement (needs another dimension on look up table)
    """
    epsilon = 1e-10
    log_epsilon = tf.experimental.numpy.finfo(tf.float32).tiny
    def __init__(self, n_pmts, 
                        return_ratio = False, 
                        switching_signal=40, n_sigma=5,
                        sigma_min=0.05, sigma_max=1, 
                        p_dpe=0.2, 
                        nan_safe=True, nan_safe_value = 1e5, 
                        m=5, z=20, 
                        mle_estimator = None, 
                        sigma_default = None,
                        **kwargs):
        """
        All inputs here are tensorflow variables to assure that these are saved reliably

        n_pmts - Int32 - Number of PMTs
        return_ratio - Bool - Wheather to return the likelihood ratio or only the likelihood
        switching_signal - Float32 - Switching signal for the likelihood functions
        n_sigma - Int32 - Number of sigma values to use for pre and/or exact computation
        sigma_min - Float32 - Minimum sigma value for std weights - only needed when training Gaussian std
        sigma_max - Float32 - Maximum sigma value for std weights - only needed when training Gaussian std
        p_dpe - Float32 - Probability of double photoelectron emission
        nan_safe - Bool - Whether to use NaN safety for inputs (ie if Nan goes into the function does a NaN come out)
        nan_safe_value - Float32 - Value to use for NaN safety
        m - Int32 - Number of subdivisions for x-domain in LUT
        z - Int32 - Number of subdivisions for sigma range in LUT
        """
        super().__init__(**kwargs)
        self.n_pmts = tf.Variable(n_pmts, trainable=False, dtype=tf.int32, name="n_pmts_LikelihoodFunction")
        self.switching_signal = tf.Variable(switching_signal, trainable=False, dtype=tf.float32, name="switching_signal_LikelihoodFunction")
        self.return_ratio = tf.Variable(return_ratio, trainable=False, dtype=tf.bool, name="return_ratio_LikelihoodFunction")
        self.n_sigma = tf.Variable(n_sigma, trainable=False, dtype=tf.float32, name="n_sigma_LikelihoodFunction")
        self.p_dpe = tf.Variable(p_dpe, trainable=False, dtype=tf.float32, name="p_dpe_LikelihoodFunction")
        self.nan_safe = tf.Variable(nan_safe, trainable=False, dtype=tf.bool, name="nan_safe_LikelihoodFunction") 
        self.nan_safe_value = tf.Variable(nan_safe_value if nan_safe else float("nan"), trainable=False, dtype=tf.float32, name="nan_safe_value_LikelihoodFunction")
        # tile_std_indices only required for exact std LUT 
        self.tile_std_indices = tf.Variable(False, trainable=False, dtype=tf.bool, name="tile_std_indices_LikelihoodFunction") # In eval can return likelihood or likelihood ratio
        # Number of integer divisions for x and mu
        self.m = tf.Variable(m, trainable=False, dtype=tf.int32, name="m_LikelihoodFunction")
        # Number of std divisions in sigma range
        self.z = tf.Variable(z, trainable=False, dtype=tf.int32, name="z_LikelihoodFunction")
        self.sigma_min = tf.Variable(sigma_min, trainable=False, dtype=tf.float32, name="sigma_min_LikelihoodFunction")
        self.sigma_max = tf.Variable(sigma_max, trainable=False, dtype=tf.float32, name="sigma_max_LikelihoodFunction")
        # Minimum x in lookup table
        self.obs_min = tf.Variable(-3, trainable=False, dtype=tf.float32, name="sigma_max_LikelihoodFunction")
        # For the exact call set integer evaluation points for Binomial and Poisson
        self.n_pe_domain = np.arange(0., switching_signal + 5*np.sqrt(switching_signal)+2)
        self.n_ph_domain = np.arange(0., switching_signal/(1+p_dpe) + 5*np.sqrt(switching_signal/(1+p_dpe))+2)
        self.n_pe_domain = tf.constant(self.n_pe_domain, dtype=tf.float32, name="n_pe_domain_LikelihoodFunction")
        self.n_ph_domain = tf.constant(self.n_ph_domain, dtype=tf.float32, name="n_ph_domain_LikelihoodFunction")
        
        if type(mle_estimator) == str: # TODO I dont think this serializes
            print("Loading MLE estimator from file: ", mle_estimator)
            mlp = tf.keras.models.load_model(mle_estimator)
            def mle_estimator(x, σ):
                # x,σ are tf tensors of shape [N]
                w=0.5
                bump = tf.exp(-(x**2)/(2*w*w))
                inp  = tf.stack([x, σ, bump], axis=1)    # now [N,3]
                return tf.squeeze(mlp(inp), axis=1)
            self.mle_estimator = mle_estimator
        elif mle_estimator is None:
            def mle(x, sigma):
                return tf.clip_by_value(x/(1+self.p_dpe), clip_value_min=1.0, clip_value_max=self.switching_signal)
            self.mle_estimator = tf.function(mle)
        else:
            self.mle_estimator = tf.function(mle_estimator)

        # Create std weight matrix
        if sigma_default is None:
            init_vals = 0.5
        else:
            # make sure sigma_default is a float32 array of shape [n_pmts]
            init_vals = tf.convert_to_tensor(sigma_default, dtype=tf.float32)

        self.std = self.add_weight(
            shape=[int(self.n_pmts)], # Convert to plaine python type
            initializer=tf.constant_initializer(init_vals),
            trainable=True,
            name="GaussianStandardDeviation", 
            constraint=MinMaxConstraint(sigma_min, sigma_max, name="LossFuncMinMax"),
        )

        return

    def set_call_mode(self, call_mode = "LUT_untrainable_std", return_ratio = None):
        docstring="""
        Configure the evaluation mode for the likelihood layer.

        This method selects and sets up the evaluation strategy based on the specified mode.
        It configures internal callables (gaussian_call and nongaussian_call) that define how the
        likelihood is computed during layer invocation. Depending on the mode, it may precompute
        lookup tables (LUTs) and adjust the layer's trainability (e.g. for learning the standard
        deviation weights).

        Available modes:
            - "exact":
                • Uses full, exact evaluation of the likelihood.
                • Sets:
                        - self.gaussian_call to a tf.function wrapping self.gaussian_likelihood_ratio.
                        - self.nongaussian_call to a tf.function wrapping self.exact_likelihood.
                • Marks the layer as non-trainable.
            - "LUT_trainable_std":
                • Precomputes a LUT over the x-domain (with 'm' subdivisions) and the sigma range 
                    (with 'z' subdivisions) using the trainable std weights.
                • The LUT-based evaluation returns the negative log likelihood.
                • Sets:
                        - Precomputes self.L_table via precompute_LUTable_with_std(self.m, self.z).
                        - self.gaussian_call to a tf.function wrapping a negative likelihood variant
                        of self.gaussian_likelihood.
                        - self.nongaussian_call to a tf.function wrapping self.lookup_negloglikelihood_with_std.
                • Marks the layer as trainable (to allow updating the std weights).
            - "LUT_untrainable_std":
                • Precomputes a LUT where the sigma dimension directly corresponds to fixed standard 
                    deviation indices (i.e., self.std is used directly as the sigma domain).
                • Sets:
                        - Precomputes self.L_table via precompute_LUTable_without_variable_std(self.m).
                        - Updates self.tile_std_indices to True to indicate that std values are handled 
                        as indices.
                        - self.gaussian_call to a tf.function wrapping self.gaussian_likelihood_ratio.
                        - self.nongaussian_call to a tf.function wrapping self.loglikelihoodratio_without_variable_std.
                • Marks the layer as non-trainable.

        After configuring the mode, the layer's 'call' method is recompiled and set to a
        tf.function-wrapped version of parent_call() to ensure optimized performance.

        Parameters:
            call_mode (str): The evaluation mode, one of:
                            "exact", "LUT_trainable_std", or "LUT_untrainable_std".
            return_ratio (bool): If True, the layer will return the likelihood ratio instead of the
                            likelihood. If None uses whatever was passed in init (default False)

        Raises:
            ValueError: If call_mode is not one of the supported strings.
        
        Side Effects:
            - Sets self.gaussian_call and self.nongaussian_call to wrapped functions.
            - Possibly precomputes the lookup table (self.L_table) for LUT modes.
            - Updates self.trainable flag based on the selected mode.
            - Reassigns self.call to a compiled version of parent_call().
        """
        # Function input signature declaration
        _kwargs = dict(
                input_signature=[
                tf.TensorSpec([None], tf.float32),
                tf.TensorSpec([None], tf.float32),
                tf.TensorSpec([None], tf.float32),
                ],
                reduce_retracing=True
                )


        if return_ratio is not None:
            self.return_ratio.assign(return_ratio)
        
        if self.return_ratio:
            self.gaussian_call = tf.function(
                                                func=self.gaussian_log_likelihood_ratio,
                                                **_kwargs
                                            )
        else:
            self.gaussian_call = tf.function(
                                                func=self.gaussian_neg_log_likelihood,
                                                **_kwargs
                                            )
        if call_mode == "exact":
            if self.return_ratio:
                self.nongaussian_call = tf.function(
                                            func=self.exact_log_likelihood_ratio,
                                            **_kwargs
                                        )
            else:
                self.nongaussian_call = tf.function(
                                            func=self.exact_neg_log_likelihood,
                                            **_kwargs
                                        )
        elif call_mode == "LUT_trainable_std":
            # Populate self.L_table with precomputed values
            self.precompute_LUTable_with_std(self.m, self.z)
            self.tile_std_indices.assign(False)
            if self.return_ratio:
                self.nongaussian_call = tf.function(
                                            func=self.lookup_log_likelihood_ratio_with_std,
                                            **_kwargs
                                        )
            else:
                self.nongaussian_call = tf.function(
                                            func=self.lookup_neg_log_likelihood_with_std,
                                            **_kwargs
                                        )
        elif call_mode == "LUT_untrainable_std":
            # Uses indices directly rather than looking them up
            self.precompute_LUTable_without_variable_std(self.m)
            self.tile_std_indices.assign(True)
            if self.return_ratio:
                self.nongaussian_call = tf.function(
                                            func=self.lookup_log_likelihood_ratio_without_std,
                                            **_kwargs
                                        )
            else:
                self.nongaussian_call = tf.function(
                                            func=self.lookup_neg_log_likelihood_without_std,
                                            **_kwargs
                                        )
        else:
            print(docstring)  
            raise ValueError("Invalid call mode. Choose 'exact', 'LUT_trainable_std', or 'LUT_untrainable_std'.")
        # Recompile the parent_call
        self.call = self.parent_call
        self.trainable = False
        return
    
    def parent_call(self, pred, observed, training=None):
        # Scaling treat neg as Gaussian noise 
        scaling = tf.reduce_sum(tf.where(observed > 0, observed, 0), axis=-1, keepdims=True)
        pred = pred * scaling

        # 2) Flatten tensors
        batch_size = tf.shape(pred)[0]
        mu  = tf.reshape(pred,    [-1])  # [batch_size * n_pmts]
        x   = tf.reshape(observed, [-1])  # [batch_size * n_pmts]

        # Tile indices or lookup std per pmt and broadcast to correct array shape  
        base = tf.cond(
            self.tile_std_indices,
            lambda: tf.reshape(tf.cast(tf.range(tf.cast(self.n_pmts, tf.int32), dtype=tf.int32), tf.float32), [1, -1]),
            lambda: tf.reshape(self.std, [1, self.n_pmts])
        )
        std = tf.reshape(tf.tile(base, [batch_size, 1]), [-1]) # [batch_size * n_pmts]

        # 3) Branch masks
        finite_mask  = tf.math.is_finite(x) & tf.math.is_finite(mu)
        bmap = tf.logical_or(x > self.switching_signal, mu > self.switching_signal)

        mask_exact   = finite_mask & tf.logical_not(bmap)
        mask_gauss   = finite_mask & bmap

        # partitioning 
        pid = tf.where(mask_gauss, 1,
              tf.where(mask_exact, 0,
                    2))
        pid = tf.cast(pid, tf.int32)
        idx = tf.range(tf.shape(x)[0])
        parts_x   = tf.dynamic_partition(x,   pid, 3)
        parts_mu  = tf.dynamic_partition(mu,  pid, 3)
        parts_std = tf.dynamic_partition(std, pid, 3)
        parts_idx = tf.dynamic_partition(idx, pid, 3)
        
        # Compute partitions 
        out_exact = self.nongaussian_call(
            parts_x[0], parts_mu[0], parts_std[0]
        )
        # In case std is now indeces we need to undo this for the gaussian call
        std_gauss = tf.cond(
            self.tile_std_indices,
            lambda: tf.gather(self.std,
                              tf.raw_ops.Cast(x=parts_std[1], DstT=tf.int32)),
            lambda: parts_std[1]
        )
        out_gauss = self.gaussian_call(
            parts_x[1], parts_mu[1], std_gauss
        )
        out_invalid = tf.fill(
            [tf.shape(parts_idx[2])[0]],
            self.nan_safe_value
        )
        # stitch back together
        result = tf.dynamic_stitch(
            [parts_idx[0], parts_idx[1], parts_idx[2]],
            [out_exact,  out_gauss,  out_invalid]
        )
        # 5) Reshape to [batch_size, n_pmts], mean over PMTs → per-event loss func and return full
        loss = tf.reshape(result, [batch_size, self.n_pmts])
        per_event = tf.reduce_mean(loss, axis=1)
        self.add_loss(per_event) # [batch_size]
        return loss  # [batch_size, n_pmts]


    ######################## Exact Methods
    @tf.function(
        input_signature=[
          tf.TensorSpec([None], tf.float32),
          tf.TensorSpec([None], tf.float32),
          tf.TensorSpec([None], tf.float32),
        ],
        reduce_retracing=True
    )
    def exact_likelihood(self, x, mu, std):
        """
        Computes the exact likelihood by combining Gaussian, Binomial, and Poisson factors.
        Reshapes x and mu to [n, 1, 1] for broadcast compatibility with self.n_ph_domain and self.n_pe_domain.
        Uses self.exact_gaussian(x, std) for SPE fluctuations, Binomial for DPE, and Poisson for mu.
        Returns the sum over axes [1, 2].

        The expectation input to Binomial (expectation and test val) is integer only
        For the Gaussian only the observation (x) is non integer - as it is relative to the number of PE + DPE produced by the signal and fractional signals can not be produced (represent count of photoelectrons)
        The Poisson may take non integer expectation by default - Gamma is only used as its more stable - no non integer counts are allowed to be observed (represent n photons)
        """
        x = tf.expand_dims(x,axis=-1)  # shape: [batch*n_pmts, 1, 1]
        mu = tf.expand_dims(tf.expand_dims(mu, axis=-1), axis=-1)   # shape: [batch*n_pmts, 1, 1]
        
        # Construct 2D grids from the precomputed 1D domains.
        n_ph_grid = tf.expand_dims(tf.expand_dims(self.n_ph_domain, axis=0), axis=-1) # shape: [1,N_ph, 1]
        n_pe_grid = tf.expand_dims(tf.expand_dims(self.n_pe_domain, axis=0), axis=0) # shape: [1,1, N_pe]
        
        # SPE fluctuation (truncated Gaussian)
        a = self.exact_gaussian(x, std)  # shape: [batch*n_pmts, N_pe]
        a = tf.expand_dims(a, axis=1)
        
        # Binomial factor: total_count from n_ph_grid, evaluated at k = n_pe_grid - n_ph_grid
        k = (n_pe_grid - n_ph_grid) # We need to mask all entries < 0 
        valid = (k >= 0) & (k <= n_ph_grid) # Mask les than zero and more successes than trials (latter should not necessarily be needed but to be sure)
        dtype = tf.raw_ops.Cast(x=valid, DstT=tf.float32, name="valid_mask_cast_binomial_Input")
        k_safe = tf.clip_by_value(k, 0, tf.cast(n_ph_grid, k.dtype))
        b = tfp.distributions.Binomial(total_count=n_ph_grid, probs=self.p_dpe).prob(k_safe) * dtype  
        
        # Poisson factor: evaluated on self.n_ph_domain for each sample (shape: [batch*n_pmts, N_ph])
        """
        Poisson has a safety feature for out of support values assigning -inf if out of support. However,
        I use tf.debug.enable_check_numerics() to find whne nan's or inf's appear
        It seems like the compiler recognizes the safety branch as an inf even if it is never triggered 
        so we manually write the pmf
        """
        #c = tfp.distributions.Poisson(rate=tf.maximum(mu, 1e-8)).prob((n_ph_grid))# shape: [batch*n_pmts, N_ph, 1]
        mask   = mu > 0.0  # We do not compute for mu = 0 but replace c entries with 0 
        # - Trained model will never supply a negative signal  - we mask out in log and in c assignment
        logp = n_ph_grid * tf.math.log(tf.where(mask, mu, tf.ones_like(mu)) ) - mu - tf.math.lgamma(n_ph_grid + 1.0)
        c      = tf.where(mask, tf.exp(logp), tf.zeros_like(logp))
        
        product = a * b * c
        return tf.reduce_sum(product, axis=[1,2])

    def exact_gaussian(self, x, std):
        """
        Computes the SPE fluctuation likelihood using a normal distribution.
        Scales std by √(self.n_pe_domain) and sets self.n_pe_domain as the mean.
        Evaluates the truncated normal probability density at x.
        Returns the resulting probability.
        """
        std = tf.expand_dims(std,axis=1) * tf.sqrt(tf.expand_dims(self.n_pe_domain, axis=0))
        mean = tf.expand_dims(self.n_pe_domain, axis=0)
        return tfp.distributions.Normal(
                loc=mean, 
                scale=tf.maximum(std,self.epsilon), 
            ).prob(x)
    def exact_log_likelihood_ratio(self, x, mu, std):
        res1 = tf.math.log(tf.maximum(self.exact_likelihood(x, mu, std),self.log_epsilon))
        res2 = tf.math.log(tf.maximum(self.exact_likelihood(x, self.mle_estimator(x, std), std),self.log_epsilon))
        return -2 * (res1 - res2)
    def exact_neg_log_likelihood(self, x, mu, std):
        return -tf.math.log(tf.maximum(self.exact_likelihood(x, mu, std), self.log_epsilon))

    ########################    Gaussian Call Methods ( above switching signal )
    def gaussian_likelihood(self, x, mu, std):
        """
        Computes the Gaussian log probability of x.
        Uses a Normal distribution with mean = mu*(1+p_dpe) and standard deviation from compute_common_std.
        """
        npe_mean = mu * (1 + self.p_dpe)
        return tfp.distributions.Normal(
            loc=npe_mean,
            scale=self.compute_common_std(mu,std)
        ).log_prob(x)
    def compute_common_std(self, mu,std):
        """
        Computes an effective standard deviation combining Poisson, Binomial, and Gaussian contributions.
        """
        npe_mean = mu * (1 + self.p_dpe)
        return tf.sqrt(tf.maximum(tf.abs(
                mu * (1 + self.p_dpe)**2 +          #Poisson 
                mu * self.p_dpe * (1-self.p_dpe) +  # Binom 
                (tf.sqrt(tf.abs(npe_mean))* (std)**2 ) # Gaussian
                ), 1e-6)) # 1e-6 to avoid underflow -> Shouldnt be needed
    def gaussian_log_likelihood_ratio(self, x, mu, std):
        res1 = self.gaussian_likelihood(x, mu, std)
        res2 = self.gaussian_likelihood(x, x, std)
        return - 2 * (res1 - res2)
    def gaussian_neg_log_likelihood(self, x, mu, std): 
        return -self.gaussian_likelihood(x, mu, std)
    
    ######################## Precompute LUT methods
    def precompute_LUTable_without_variable_std(self, m=5):
        """
        Precompute a fixed-σ LUT by directly calling exact_likelihood(x, μ, σ).

        Arguments
        ---------
        m : int
            Number of subdivisions per unit in x (so NX = switching_signal*m*40 + 1).

        After running, self.L_table has shape
        (switching_signal+1, NX, n_pmts)
        where
        - axis 0 = μ = 0,1,…,switching_signal
        - axis 1 = x_grid points
        - axis 2 = PMT index (σ from self.std)
        """
        # 1) build x and mu domains
        x_vals = []
        for i in range(int(self.obs_min), int(self.switching_signal)):
            for sub in range(int(m)):
                x_vals.append(i + sub / float(m))
        x_vals.append(float(self.switching_signal))
        x_domain = np.array(x_vals, dtype=np.float32)         # shape [NX]
        self.x_domain  = tf.constant(x_domain,  dtype=tf.float32)
        nx = x_domain.size

        mu_vals = []
        for i in range(int(self.switching_signal)):
            for sub in range(int(m)):
                mu_vals.append(i + sub/float(m))
        mu_vals.append(float(self.switching_signal))
        self.mu_domain = tf.constant(mu_vals, dtype=tf.float32)

        # 2) sigma_domain is trainable std array
        sigma_domain = self.std.numpy() if tf.is_tensor(self.std) else np.array(self.std, dtype=np.float32)
        self.sigma_domain = tf.constant(sigma_domain, dtype=tf.float32)
        n_pmts = sigma_domain.size

        # 3) prepare an empty LUT
        mu_vals = self.mu_domain.numpy()
        S = len(mu_vals)
        L = np.zeros((S, nx, n_pmts), dtype=np.float32)

        for idx, mu_val in enumerate(mu_vals):
            x_grid   = np.repeat(x_domain[:, None], n_pmts, axis=1).ravel()
            mu_grid  = np.full_like(x_grid, fill_value=mu_val, dtype=np.float32)
            std_grid = np.tile(sigma_domain[None, :],          (nx, 1)).ravel()
            # call your exact function (returns array of shape [NX*n_pmts])
            probs = self.exact_likelihood(x_grid, mu_grid, std_grid)
            L[idx, :, :] = tf.reshape(probs, [nx, n_pmts])

        # 5) stash it as a TF tensor
        self.L_table   = tf.constant(L)  # shape [S, NX, n_pmts]

        # Generate MLE Table 
        self.L_mle_table = tf.reduce_max(
            self.L_table,
            axis=0
        )  # shape [NX, n_pmts]
        return
    
    def precompute_LUTable_with_std(self,
                                m=5,   # Subdivisions in x, mu
                                z=20,  # Subdivisions in sigma
                                ):
        # 1) build x and mu domains
        x_vals = []
        for i in range(int(self.obs_min), int(self.switching_signal)):
            for sub in range(int(m)):
                x_vals.append(i + sub / float(m))
        x_vals.append(float(self.switching_signal))
        x_domain = np.array(x_vals, dtype=np.float32)         # shape [NX]
        self.x_domain  = tf.constant(x_domain,  dtype=tf.float32)
        nx = x_domain.size

        mu_vals = []
        for i in range(int(self.switching_signal)):
            for sub in range(int(m)):
                mu_vals.append(i + sub/float(m))
        mu_vals.append(float(self.switching_signal))
        self.mu_domain = tf.constant(mu_vals, dtype=tf.float32)

        # Build Sigma domain
        self.sigma_domain = tf.linspace(self.sigma_min, self.sigma_max, z)  # [z]

        mu_vals = self.mu_domain.numpy()
        S = len(mu_vals)
        L = np.zeros((S, nx, int(z)), dtype=np.float32)

        for idx, mu_val in enumerate(mu_vals):
            x_grid   = np.repeat(x_domain[:, None], z, axis=1).ravel()
            mu_grid  = np.full_like(x_grid, fill_value=mu_val, dtype=np.float32)
            std_grid = np.tile(self.sigma_domain[None, :],          (nx, 1)).ravel()
            # call your exact function (returns array of shape [NX*n_pmts])
            probs = self.exact_likelihood(x_grid, mu_grid, std_grid)
            L[idx, :, :] = tf.reshape(probs, [nx, z])

        self.L_table   = tf.constant(L)  # shape [S, NX, n_pmts]

        # Generate MLE Table 
        self.L_mle_table = tf.reduce_max(
            self.L_table,
            axis=0
        )  # shape [NX, n_pmts]
        return
        

    def old(self, m=5, z=20):
        """
        Precomputes a 3D LUT (self.L_table) for likelihood evaluation with variable std,
        and a 2D MLE‐based LUT (self.L_mle_table).

        Subdivides the x-domain into 40*m + 1 points and the sigma range into z steps
        (using self.sigma_min, self.sigma_max).  Computes:
        - Binomial factor 'b' from self.n_pe_domain, self.n_ph_domain, self.p_dpe.
        - Poisson factor 'c' for mu = 0 ... switching_signal.
        - Gaussian factor from the sigma domain.
        Then for each μ in [0..switching_signal] it calls self.exact_likelihood(x,μ,σ)
        over the (x,σ) grid (vectorized per‐slice) to build self.L_table of shape
        [switching_signal+1, nx, z], and similarly builds self.L_mle_table of shape [nx, z].
        """
        # --- 1) Binomial factor b_full: [n_ph, n_pe] ---
        n_pe = self.n_pe_domain.shape[0]
        n_ph = self.n_ph_domain.shape[0]

        k = self.n_pe_domain[tf.newaxis, :] - self.n_ph_domain[..., tf.newaxis]
        total_counts_matrix = tf.reshape(self.n_ph_domain, [n_ph, 1])
        valid_mask = tf.logical_and(k >= 0, k <= total_counts_matrix)
        valid_indices = tf.where(valid_mask)
        valid_k = tf.gather_nd(k, valid_indices)
        total_counts = tf.gather(self.n_ph_domain, valid_indices[:, 0])
        p_valid = tfp.distributions.Binomial(
            total_count=total_counts, probs=self.p_dpe
        ).prob(valid_k)
        b_full = tf.zeros([n_ph, n_pe], dtype=tf.float32)
        self.b = tf.tensor_scatter_nd_update(b_full, valid_indices, p_valid)

        # --- 2) Build x_domain and mu_domain ---
        x_vals = []
        for i in range(int(self.obs_min), int(self.switching_signal)):
            for sub in range(int(m)):
                x_vals.append(i + sub / float(m))
        x_vals.append(float(self.switching_signal))
        self.x_domain = tf.constant(x_vals, dtype=tf.float32)  # [nx]

        mu_vals = []
        for i in range(int(self.switching_signal)):
            for sub in range(int(m)):
                mu_vals.append(i + sub/float(m))
        mu_vals.append(float(self.switching_signal))
        self.mu_domain = tf.constant(mu_vals, dtype=tf.float32)

        # --- 3) Poisson factor c: [s0, n_ph] ---
        c_list = []
        mu_vals_np = self.mu_domain.numpy()
        for mu in mu_vals_np:
            if mu == 0:
                c_list.append( tf.where(
                    tf.equal(self.n_ph_domain, 0.0),
                    tf.ones_like(self.n_ph_domain),
                    tf.zeros_like(self.n_ph_domain),
                ))
            else:
                c_list.append(
                    tfp.distributions.Poisson(rate=float(mu))
                    .prob(self.n_ph_domain)
                )
        self.c = tf.stack(c_list, axis=0)  # [Sm, n_ph]

        """ From Pre mu indexing 
        c_list = []
        for mu_val in range(int(self.switching_signal) + 1):
            if mu_val == 0:
                c_val = tf.where(
                    tf.equal(self.n_ph_domain, 0.0),
                    tf.ones_like(self.n_ph_domain),
                    tf.zeros_like(self.n_ph_domain),
                )
            else:
                c_val = tfp.distributions.Poisson(
                    rate=float(mu_val)
                ).prob(self.n_ph_domain)
            c_list.append(c_val)
        self.c = tf.stack(c_list, axis=0)  # [s0, n_ph]
        """
        # --- 4) Build sigma_domain ---
        self.sigma_domain = tf.linspace(self.sigma_min, self.sigma_max, z)  # [z]

        # -------------------------------------------------------------------------
        # From here on, replace the big broadcast‐sum with vectorized exact_likelihood
        # -------------------------------------------------------------------------

        # 5) Grab domains as NumPy arrays to use for tf.constant / tf.fill
        x_vals_np     = self.x_domain.numpy()     # shape [nx]
        sigma_vals_np = self.sigma_domain.numpy() # shape [z]
        S  = len(mu_vals_np)#int(self.switching_signal) + 1
        nx = x_vals_np.size
        Z  = sigma_vals_np.size

        # 6) Precompute flat (x, sigma) grid once
        xg, sg = np.meshgrid(x_vals_np, sigma_vals_np, indexing='ij')  # [nx, Z]
        xf = tf.constant(xg.ravel(), dtype=tf.float32)                # [nx*Z]
        sf = tf.constant(sg.ravel(), dtype=tf.float32)                # [nx*Z]

        # 7) Prepare TensorArray to collect S slices of shape [nx,Z]
        ta = tf.TensorArray(dtype=tf.float32, size=S)

        # 8) Loop over mu_idx only
        #for mu_idx in range(S): When mu wasnt a lsit this was correct
        for idx in range(S):
            mf = tf.fill(tf.shape(xf), mu_vals_np[idx])               # [nx*Z]
            pf = self.exact_likelihood(xf, mf, sf)                  # [nx*Z]
            ta = ta.write(idx, tf.reshape(pf, [nx, Z]))

        # 9) Stack into [S, nx, Z] and assign
        self.L_table = ta.stack()                                   # [S,nx,Z]

        # -------------------------------------------------------------------------
        # Build MLE table 
        # -------------------------------------------------------------------------

        self.L_mle_table = tf.reduce_max(
            self.L_table,
            axis=0
        )  # shape [nx, Z]

        return


    ####################### Indexing Methods 
    def _nearest_idx_1d(self, grid, values):
        """
        Returns the nearest indices in a sorted 1D tensor 'grid' for each value in 'values'.
        Uses tf.searchsorted and neighbor comparison to determine the index with minimal distance.
        Inputs:
        grid: Tensor of shape [N].
        values: Tensor of arbitrary shape.
        Output: Tensor of indices, same shape as 'values'.
        """
        # searchsorted gives the insertion index. Then we clamp & decide which neighbor is closer.
        idx_float = tf.searchsorted(grid, values, side='left')  # shape = same as values
        # idx_float is in [0..N], clamp
        idx_below = tf.clip_by_value(idx_float - 1, 0, tf.size(grid)-1)
        idx_above = tf.clip_by_value(idx_float,     0, tf.size(grid)-1)

        # pick whichever is closer to 'values'
        below_vals = tf.gather(grid, idx_below)
        above_vals = tf.gather(grid, idx_above)

        dist_below = tf.abs(values - below_vals)
        dist_above = tf.abs(values - above_vals)

        choice = tf.cast(dist_above < dist_below, tf.int32)  # 1 if 'above' is nearer
        # final index
        idx_nearest = idx_below + choice  # shape same as values
        return idx_nearest
    
    @tf.custom_gradient
    def differentiable_sigma_idx(self,std):
        """
        Returns the nearest index in self.sigma_domain for each value in 'std'.
        The custom gradient passes the incoming gradient through unchanged.
        """
        sigma_idx = self._nearest_idx_1d(self.sigma_domain, std)
        def grad(dy):
            return dy 
        return sigma_idx, grad
    
    def get_x_mu_idx(self, x, mu):
        """Quick Helper to get the coordinates"""
        x_idx = self._nearest_idx_1d(self.x_domain, x)
        mu_idx     = self._nearest_idx_1d(self.mu_domain, mu)
        return x_idx, mu_idx
    
    def gather_from_L_table(self, x_idx, mu_idx, sigma_idx):
        """Quick helper to gather values from table"""
        coords = tf.stack([mu_idx, x_idx, sigma_idx], axis=-1)  # shape (batch_size, 3)
        return tf.gather_nd(self.L_table, coords)     

    ################### Lookup Methods 

    def lookup_neg_log_likelihood_with_std(self, x, mu, std):
        """
        Lookup with std domain search 
        """
        x_idx, mu_idx = self.get_x_mu_idx(x, mu)
        sigma_idx = self.differentiable_sigma_idx(std) # sigma_idx - needs to be found 
        return -tf.math.log(tf.maximum(self.gather_from_L_table(x_idx, mu_idx, sigma_idx), self.log_epsilon))  # Avoid log(0) = -inf
    
    def lookup_neg_log_likelihood_without_std(self, x, mu, sigma_idx):
        """
        lookup without std domain search (idx corresponds to pmt number)
        """
        x_idx, mu_idx = self.get_x_mu_idx(x, mu)
        sigma_idx = tf.cast(sigma_idx, tf.int32) # sigma_idx - passed directly 
        return -tf.math.log(tf.maximum(self.gather_from_L_table(x_idx, mu_idx, sigma_idx), self.log_epsilon))  # Avoid log(0) = -inf
    
    def lookup_likelihood_with_variable_std(self, x, mu, std):
        sigma_idx = self.differentiable_sigma_idx(std) # sigma_idx - needs to be found 
        x_idx, mu_idx = self.get_x_mu_idx(x, mu)
        # Build the indices and perform the LUT lookup.
        return self.gather_from_L_table(x_idx, mu_idx, sigma_idx)

    def lookup_likelihood_without_variable_std(self, x, mu, sigma_idx):
        sigma_idx = tf.cast(sigma_idx, tf.int32)  
        x_idx, mu_idx = self.get_x_mu_idx(x, mu)
        # Build the indices and perform the LUT lookup.
        return self.gather_from_L_table(x_idx, mu_idx, sigma_idx)

    def lookup_likelihood_mle_with_variable_std(self, x, std):
        sigma_idx = self.differentiable_sigma_idx(std) # sigma_idx - needs to be found 
        x_idx = self._nearest_idx_1d(self.x_domain, x)
        # Gather from the 2D MLE table [nx, n_pmts]
        return tf.gather_nd(self.L_mle_table, tf.stack([x_idx, sigma_idx], axis=-1))

    def lookup_likelihood_mle_without_variable_std(self, x, sigma_idx):
        sigma_idx = tf.cast(sigma_idx, tf.int32)
        x_idx = self._nearest_idx_1d(self.x_domain, x)
        # Gather from the 2D MLE table [nx, n_pmts]
        return tf.gather_nd(self.L_mle_table, tf.stack([x_idx, sigma_idx], axis=-1))
    
    def lookup_log_likelihood_ratio_with_std(self, x, mu, std):
        res1 = tf.math.log(tf.maximum(self.lookup_likelihood_with_variable_std(x, mu, std),self.log_epsilon))
        res2 = tf.math.log(tf.maximum(self.lookup_likelihood_mle_with_variable_std(x, std),self.log_epsilon))
        return -2 * (res1 - res2)
    
    def lookup_log_likelihood_ratio_without_std(self, x, mu, sigma_idx):
        res1 = tf.math.log(tf.maximum(self.lookup_likelihood_without_variable_std(x, mu, sigma_idx),self.log_epsilon))
        res2 = tf.math.log(tf.maximum(self.lookup_likelihood_mle_without_variable_std(x, sigma_idx),self.log_epsilon))
        return -2 * (res1 - res2)

    
    ###################### Lookup ratios 
    def loglikelihoodratio_without_variable_std(self, x, mu, sigma_idx):
        """
        Computes the log-likelihood ratio using the fixed-std LUT.
        Looks up likelihoods for x (given mu) and for mu (as reference) using lookup_likelihood_without_variable_std.
        Returns -2 * (log(L_x) - log(L_mu)), ensuring numerical stability.
        """
        # Get the likelihoods for x and mu
        L_x = self.lookup_likelihood_without_variable_std(x, mu, sigma_idx)
        L_mu = self.lookup_likelihood_without_variable_std(mu, mu/(1+self.p_dpe), sigma_idx)

        # Compute the log-likelihood ratio
        return -2 * (tf.math.log(tf.maximum(L_x, self.log_epsilon)) - tf.math.log(tf.maximum(L_mu, self.log_epsilon)))
    
    ############################## Serialization 
    def get_config(self):
        """Returns the configuration of the layer for serialization.

        This method serializes all parameters provided at initialization as well as
        those that are essential for reconstructing the layer. The trainable standard
        deviation weights (self.std) are saved automatically by the base layer.

        Returns:
            dict: A dictionary containing the configuration of the layer.
        """
        config = super().get_config()
        config.update({
            "n_pmts": int(self.n_pmts),
            "switching_signal": float(self.switching_signal),
            "n_sigma": int(self.n_sigma),
            "sigma_min": float(self.sigma_min),
            "sigma_max": float(self.sigma_max),
            "p_dpe": float(self.p_dpe),
            "nan_safe": bool(self.nan_safe),
            "nan_safe_value": float(self.nan_safe_value),
            "m": int(self.m),
            "z": int(self.z)
        })
        return config
    @classmethod
    def from_config(cls, config):
        """Creates a layer from its configuration.

        Args:
            config (dict): A configuration dictionary.

        Returns:
            LikelihoodRatio: A new layer instance.
        """
        return cls(**config)

