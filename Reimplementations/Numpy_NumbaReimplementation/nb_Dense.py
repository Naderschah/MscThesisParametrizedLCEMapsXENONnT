"""
Here the numba dense network is implemented
With variable Generators and the full model generator
"""

import numpy as np
from numba import njit, prange, float32


# Activation Functions

@njit
def relu(x):
    return x if x > 0 else 0.0

@njit
def sigmoid(x):
    # fastmath=True lets us use exp safely
    return 1.0 / (1.0 + np.exp(-x))


def make_dense_forward(W: np.ndarray,
                       b: np.ndarray,
                       activation,
                       *,
                       parallel: bool = True,
                       fastmath: bool = True):
    """
    Returns an njit‐compiled forward(x) for the dense layer
        y = activation(W @ x + b)

    activation must itself be a @njit’d (or nopython) function taking
    a scalar and returning a scalar.

    parallel: whether to use prange over output neurons
    fastmath: whether to enable fastmath optimizations

    All inputs must be reshaped to .reshape(-1, size[-1]) this creates a view not a copy

    Fastmath should be fine to use for Dense Networks
    All in and out should be float32 - functions are specialized for this 
    boundschecks are off by default -- But the generator will complain if dims are wrong
    the gil is of course also released 
    """
    # Validate shapes
    out_dim, in_dim = W.shape
    assert b.shape[0]    == out_dim

    # Choose decorator kwargs
    njit_kwargs = {'parallel': parallel, 'fastmath': fastmath, 'boundscheck':False, 'nogil':True, }

    @njit(float32[:, :](float32[:, :], float32[:, :]), **njit_kwargs)
    def forward(x2d, out2d):
        # x2d: (N, in_dim), out2d: (N, out_dim)
        N = x2d.shape[0]
        # local references for speed
        w = W
        bias = b
        act = activation
        od = out_dim
        idim = in_dim

        for i in prange(N):
            for j in range(od):
                acc = bias[j]
                # dot-product W[j,:] · x2d[i,:]
                for k in range(idim):
                    acc += w[j, k] * x2d[i, k]
                out2d[i, j] = act(acc)
        return out2d

    return forward


def inputs_generator(pmt_pos_top, parallel = True, fastmath = True):
    """
    Takes flattened input 
    Has to create a new array 
    """
    TODO
    raise Exception("Not Implemented Error")
    in_dim = 8?

    njit_kwargs = {'parallel': parallel, 'fastmath': fastmath, 'boundscheck':False, 'nogil':True, }
    @njit(float32[:,:](float32[:,:]), **njit_kwargs) 
    def input_layer(x):
        for i in prange:
            inputs = COMPUTE THEM
        return inputs.reshape(-1, in_dim)


def model_generator(pmt_pos_top, weights, parallel = True, fastmath = True):


    inputs_ = inputs_generator(parallel = parallel, fastmath = fastmath)

    # TODO Logic for assigning weights to dense layers
    dense1 = make_dense_forward(W, b , activation , ...)
    dense2 = make_dense_forward(W, b , activation , ...)
    
    n_pmts = len(pmt_pos_top)

    assert # TODO Verify final bias is the same size as n_pmts
    njit_kwargs = {'parallel': parallel, 'fastmath': fastmath, 'boundscheck':False, 'nogil':True, }
    @njit(float32[:,:](float32[:,:], float32[:,:], float32[:,:]), **njit_kwargs)
    def model(inputs, out_in, out_dense1, out_dense2):
        """
        Handles actual dense application batched or full size
        """
        x = inputs_(inputs, out_in) # Shape (n_events * n_pmts, in_features)
        x1 = dense1(x, out_dense1) 
        out = dense2(x1, out_dense2) 
        return out.reshape(-1, n_pmts)
    
    def model_zero_shot(inputs):
        """
        inputs : np.ndarray, shape (n_events, 2)
        returns : np.ndarray, shape (n_events, n_pmts)

        TODO Array sizes
        """
        n_events = inputs.shape[0]
        
        # 1) allocate the intermediate buffers
        #    shape (n_events * n_pmts, in_features)
        out_in     = np.empty((n_events * n_pmts, in_features), dtype=np.float32)
        #    shape (n_events * n_pmts, dim1)
        out_dense1 = np.empty((n_events * n_pmts, dim1),      dtype=np.float32)
        #    shape (n_events * n_pmts, 1)
        out_dense2 = np.empty((n_events * n_pmts, 1),         dtype=np.float32)
        
        # 2) call your JIT’d single‐step model
        #    model(inputs, out_in, out_dense1, out_dense2)
        flat_out = model(inputs, out_in, out_dense1, out_dense2)
        
        # 3) reshape back to (n_events, n_pmts)
        return flat_out
    def model_batched(inputs, batch_size):
        """
        inputs     : np.ndarray, shape (n_events, 2)
        batch_size : int

        Returns out : np.ndarray, shape (n_events, n_pmts)
        """
        n_events = inputs.shape[0]
        # final output
        output = np.empty((n_events, n_pmts), dtype=np.float32)

        # preallocate maximum‐size temp buffers once
        max_rows   = batch_size * n_pmts
        buf_in     = np.empty((max_rows, in_features), dtype=np.float32)
        buf_d1     = np.empty((max_rows,      dim1), dtype=np.float32)
        buf_d2     = np.empty((max_rows,         1), dtype=np.float32)

        # loop over full dataset in chunks
        for start in range(0, n_events, batch_size):
            stop = min(start + batch_size, n_events)
            bs   = stop - start

            # slice raw inputs for this mini‐batch
            sub_inputs = inputs[start:stop]  # shape (bs, 2)

            # slice views of each buffer to exactly bs*n_pmts rows
            rows = bs * n_pmts
            out_in     = buf_in[:rows, :]
            out_dense1 = buf_d1[:rows, :]
            out_dense2 = buf_d2[:rows, :]

            # run the single‐step JIT model
            flat = model(sub_inputs, out_in, out_dense1, out_dense2)
            # flat is shape (rows, 1) or (rows,) depending on your signature;
            # we reshape to (bs, n_pmts)
            output[start:stop] = flat.reshape(bs, n_pmts)

        return output
        
    
