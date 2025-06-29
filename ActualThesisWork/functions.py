"""
Here live (almost) all relevant functions for data processing and plotting
-----------
First Functions
Then Plotting
"""
import scipy as sp
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow
import scipy
import keras
from IPython.display import clear_output
from functools import partial
import h5py
import time
import numba
import matplotlib.gridspec as gridspec
from matplotlib.patches import Wedge
## Because I got annoyed
_scalar_types = (int,float,complex,bool,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,np.float16,np.float32,np.float64,np.complex64,np.complex128,np.bool_)

##---------------------------------- Utility -----------------------------------
def split_data(positions, patterns, train_slice, val_slice):
    x_train = tensorflow.gather(positions, train_slice, axis=0)
    y_train = tensorflow.gather(patterns, train_slice, axis=0)
    val = (tensorflow.gather(positions, val_slice, axis=0), tensorflow.gather(patterns, val_slice, axis=0))
    return x_train, y_train, val


## Loss funcs
def logl_loss_generator(pmt_positions, B, not_dead_pmts = None, cap=float('inf'), allow_njit=True, median_loss = False, check_nan = False, loss_poisson=True):
    """
    Generates a loss function for the specified backend (keras/tensorflow or numpy)
    Reasoning being that we dont want any conditionals in here, but a ready to execute
    functions.
    ------------------------------------------
    pmt_positions : Top array pmt positions (n_pmts, 2) in order x,y
    B : Backend to use, use numpy (module) for numpy and anything else for tf
    not_dead_pmts : Boolean list of alive pmts for filtering
    cap : Maximum loss value
    allow_njit: JIT compiles all functions, using njit for numpy and tf.function (XLA) for tf 
    - No Point for numpy, testing is better but actually using is slower

    - Median_loss -> Absolutely pointless for the small changes im looking at

    # TODO : Refactor generator as conditional prefix wrappers, i.e. median func takes all of loss func as global
    Should allow to get rid of empty function calls
    """

    if not_dead_pmts is not None:
        n_dof = len(not_dead_pmts) 
        not_dead_pmts_tf = tensorflow.constant(not_dead_pmts, dtype=tensorflow.int32)
    else:
        n_dof = len(pmt_positions)

    # Removing dead pmts -> Assign function to use
    def rem_pmts_tf(ao, ae):
        ao = tensorflow.gather(ao, not_dead_pmts_tf, axis=1)
        ae = tensorflow.gather(ae, not_dead_pmts_tf, axis=1)
        return ao, ae
    def _rem_pmts_np(ao, ae):
        return ao[:, not_dead_pmts], ae[:,not_dead_pmts]
    def no_pmts_missing(ao, ae):
        return ao, ae
    
    if not_dead_pmts is not None:
        if B == np:
            if allow_njit:
                rem_pmts = numba.njit(_rem_pmts_np) 
            else:
                rem_pmts = _rem_pmts_np
        else:
            if allow_njit:
                rem_pmts = tensorflow.function(rem_pmts_tf)
            else:
                rem_pmts = rem_pmts_tf
    else:
        rem_pmts = no_pmts_missing

    def tf_rem_nan(loss):
        # Get rid of nan
        indices = tensorflow.where(tensorflow.math.is_nan(loss))
        return tensorflow.tensor_scatter_nd_update(
            loss,
            indices,
            tensorflow.ones((tensorflow.shape(indices)[0]))*cap
        )
    def np_rem_nan(loss):
        return np.nan_to_num(loss, nan=cap)
    if check_nan:
        np_nan_handle = np_rem_nan
        tf_nan_handle = tf_rem_nan
    else:
        np_nan_handle = lambda X: X
        tf_nan_handle = lambda X: X

    def _loss_np(patterns_true, prediction):
        # Scale prediciton
        patterns_pred = np.reshape(np.sum(patterns_true, axis=1), (-1, 1)) * prediction
        ae = np.clip(patterns_pred, 1e-10, cap) 
        ao = np.clip(patterns_true, 1e-10, cap) 
        ao, ae = rem_pmts(ao, ae)
        loss = np.clip(np.sum((((ao - ae)**2) / ae), axis=1) / n_dof, 0, cap)
        return np_nan_handle(loss)
    def _loss_tensor_standard(patterns_true, prediction):
        patterns_true, prediction = rem_pmts(patterns_true, prediction)
        patterns_pred = tensorflow.reshape(keras.backend.sum(patterns_true, axis=1), (-1, 1)) * prediction / tensorflow.reshape(keras.backend.sum(prediction, axis=1), (-1, 1))
        ae = keras.backend.clip(patterns_pred, 1e-10, cap) 
        ao = keras.backend.clip(patterns_true, 1e-10, cap) 
        loss = tensorflow.clip_by_value(keras.backend.sum((((ao - ae)**2) / ae), axis=1) / n_dof, 0, cap)
        return tf_nan_handle(loss)
    
    def _loss_tensor_poisson(patterns_true, prediction):
        patterns_pred = tensorflow.reshape(keras.backend.sum(patterns_true, axis=1), (-1, 1)) * prediction
        ae = keras.backend.clip(patterns_pred, 1e-10, cap) 
        ao = keras.backend.clip(patterns_true, 1e-10, cap) 
        ao, ae = rem_pmts(ao, ae)
        loss = tensorflow.clip_by_value(2*keras.backend.sum(ao*tensorflow.math.log(ao/ae) - (ao - ae), axis=1) / n_dof, 0, cap)
        return tf_nan_handle(loss)
    
    _loss_tensor = _loss_tensor_poisson if loss_poisson else _loss_tensor_standard
    
    if allow_njit:
        loss_np = numba.njit(_loss_np)
        loss_tf = tensorflow.function(_loss_tensor)
    else:
        loss_np = _loss_np
        loss_tf = _loss_tensor

    # Define which loss function to use
    return loss_np if B == np else loss_tf


## ------------------------ For LCE, mainly geometry ----------------------------
def norm_angle(theta): #  -pi to pi
    """Truncates angle: -pi to pi"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

def sym_angle(theta, B=tensorflow):
    mi = -B.abs(norm_angle(np.pi + theta))
    ma = B.abs(norm_angle(np.pi - theta))
    if B == tensorflow:  return tensorflow.clip_by_value(norm_angle(theta), mi, ma)
    else:                return np.clip(norm_angle(theta), mi, ma)

def positions_to_norm_coords(X, pmt_positions):
    """Convert x, y coordinates to alteranate representations
    Taken from Jelle
    TODO: Returning absolute coordinates
    """
    r_event_tpc = tensorflow.math.sqrt(tensorflow.math.square(X[:, :, 0]) + tensorflow.math.square(X[:, :, 1]))
    
    #X -= pmt_positions
    x, y = X[:, :, 0] - pmt_positions[:,0], X[:, :, 1] - pmt_positions[:,1]
    r_event_pmt = get_r_event_pmt_minimal(x, y)
    q_event_pmt = norm_angle(tensorflow.atan2(y, x) 
                             + np.pi/2
                             - tensorflow.atan2(pmt_positions[:,1], pmt_positions[:,0]))
    
    return keras.backend.stack([
        r_event_pmt,
        # Phi from event to PMT, symmetrized, and suppressed near origin
        sym_angle(q_event_pmt) * r_event_pmt**0.5,
        sym_angle(q_event_pmt + np.pi/2) * r_event_pmt**0.5,
        # Absolute event position
        X[:, :, 0], 
        X[:, :, 1],
        r_event_tpc,
        ], axis=2)

@tensorflow.function
def get_r_event_pmt_minimal(x, y):
    """Get Distance from curr PMT for above func"""
    return tensorflow.math.sqrt(tensorflow.math.square(x) + tensorflow.math.square(y))

@tensorflow.function
def get_r_event_pmt(X, pmt_positions):
    """Get Distance from curr PMT"""
    return tensorflow.math.sqrt(tensorflow.math.square(X[:, :, 0] - pmt_positions[:,0]) + tensorflow.math.square(X[:, :, 1] - pmt_positions[:,1]))




## ------------------------------  Ring Groups ---------------------------------

def get_ring_group_with_edge():
    return [
        [126], 
        [109, 110, 125, 127, 142, 143],
        [ 93, 108, 92,111, 141, 144,128, 159,124, 160, 158, 94],
        [76, 77, 91, 95, 107, 112, 140, 145, 157, 161, 175, 129, 176, 177, 78, 75, 123, 174],
        [60, 61, 62, 74, 79, 90, 96, 106, 113, 139, 146, 156, 162, 173, 178, 190, 191, 192, 193, 189, 59, 63, 130, 122],
        [45, 46, 47, 48, 58, 64, 73, 80, 89, 97, 105, 114, 138, 147, 155, 163, 172, 179, 188, 194, 204, 205, 206, 207, 203, 121, 44, 49, 131, 208],
        [31, 32, 33, 34, 35, 43, 50, 57, 65, 72, 81, 88, 98, 104, 115, 137, 148, 154, 164, 171, 180, 187, 195, 202, 209, 217, 218, 219, 220, 221, 216, 222, 120, 36, 132, 30],
        [18, 19, 20, 21, 22, 23, 29, 37, 42, 51, 56, 66, 71, 82, 87, 99, 103, 116, 136, 149, 153, 165, 170, 181, 186, 196, 201, 210, 215, 223, 229, 230, 231, 232, 233, 234, 228, 235, 119, 17, 24, 133],
        [7, 8, 9, 10, 11, 12, 13, 16, 25, 28, 38, 41, 52, 55, 67, 70, 83, 86, 100, 102, 117, 135, 150, 152, 166, 169, 182, 185, 197, 200, 211, 214, 224, 227, 236, 239, 240, 241, 242, 243, 244, 245, 238, 118, 246, 6, 14, 134],
        [0, 1, 2, 3, 4, 5, 15, 26, 27, 39, 40, 53, 54, 68, 69, 84, 85, 101, 151, 167, 168, 183, 184, 198, 199, 212, 213, 225, 226, 237, 247, 248, 249, 250, 251, 252]
    ]
    
def get_ring_group_without_edge():
    return [
        [126],
        [109, 110, 125, 127, 142, 143],
        [93, 108, 111, 141, 144, 159],
        [76, 77, 91, 92, 94, 95, 107, 112, 124, 128, 140, 145, 157, 158, 160, 161, 175, 176],
        [60, 61, 62, 74, 75, 78, 79, 90, 96, 106, 113, 123, 129, 139, 146, 156, 162, 173, 174, 177, 178, 190, 191, 192],
        [45, 46, 47, 48, 58, 59, 63, 64, 73, 80, 89, 97, 105, 114, 122, 130, 138, 147, 155, 163, 172, 179, 188, 189, 193, 194, 204, 205, 206, 207],
        [31, 32, 33, 34, 35, 43, 44, 49, 50, 57, 65, 72, 81, 88, 98, 104, 115, 121, 131, 137, 148, 154, 164, 171, 180, 187, 195, 202, 203, 208, 209, 217, 218, 219, 220, 221],
        [18, 19, 20, 21, 22, 23, 29, 30, 36, 37, 42, 51, 56, 66, 71, 82, 87, 99, 103, 116, 120, 132, 136, 149, 153, 165, 170, 181, 186, 196, 201, 210, 215, 216, 222, 223, 229, 230, 231, 232, 233, 234],
        [7, 8, 9, 10, 11, 12, 13, 16, 17, 24, 25, 28, 38, 41, 52, 55, 67, 70, 83, 86, 100, 102, 117, 119, 133, 135, 150, 152, 166, 169, 182, 185, 197, 200, 211, 214, 224, 227, 228, 235, 236, 239, 240, 241, 242, 243, 244, 245],
        [0, 1, 2, 3, 4, 5, 6, 14, 15, 26, 27, 39, 40, 53, 54, 68, 69, 84, 85, 101, 118, 134, 151, 167, 168, 183, 184, 198, 199, 212, 213, 225, 226, 237, 238, 246, 247, 248, 249, 250, 251, 252],
    ]

def man_group_4_ring():
    indiv_pmt_groups = get_ring_group_without_edge()
    pmt_groups = []
    pmt_groups.append(indiv_pmt_groups[0] + indiv_pmt_groups[1] + indiv_pmt_groups[2] + indiv_pmt_groups[3]) # Second Red is idx 3
    pmt_groups.append(indiv_pmt_groups[4] + indiv_pmt_groups[5]) # Second blue is idx 5
    pmt_groups.append(indiv_pmt_groups[6] + indiv_pmt_groups[7]) # third green is idx 7
    pmt_groups.append(indiv_pmt_groups[8] + indiv_pmt_groups[9]) # outre ring is idx 9
    return pmt_groups



## ----------------------  Layer to hold per PMT multiplier --------------------

class GetRadius(keras.layers.Layer):
    """
    Computes Radius from input x, y
    """
    def __init__(self, output_shape, pmt_positions, **kwargs):
        super().__init__(**kwargs)
        self._output_shape = output_shape
        self.pmt_positions = tensorflow.constant(pmt_positions, dtype=tensorflow.float32, name="pmt_pos_Get_Radius")
    
    def build(self, input_shape):
        super().build(input_shape)

    @tensorflow.function
    def call(self, X):
        return get_r_event_pmt(X, self.pmt_positions)
    
    def compute_output_shape(self, input_shape):
        return self._output_shape
    
class GetWallDist(keras.layers.Layer):
    """
    Computes Wall Distance from input x, y
    """
    def __init__(self, output_shape, tpc_radius, **kwargs):
        super().__init__(**kwargs)
        self._output_shape = output_shape
        self.tpc_radius = tensorflow.constant(tpc_radius, dtype=tensorflow.float32, name="tpc_radius")
    
    def build(self, input_shape):
        super().build(input_shape)

    @tensorflow.function
    def call(self, X):
        return self.tpc_radius-tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(X), axis=-1))
    
    def compute_output_shape(self, input_shape):
        return self._output_shape
    
class GetFullCoordinateSystem(keras.layers.Layer):
    """
    Computes Radius from input x, y TODO 
    """
    def __init__(self, output_shape, pmt_positions, **kwargs):
        super().__init__(**kwargs)
        self._output_shape = output_shape
        self.pmt_positions =  tensorflow.constant(pmt_positions, dtype=tensorflow.float32, name="pmt_positions_Full_CoordSys")
        self.trainable = False
    
    def build(self, input_shape):
        super().build(input_shape)

    @tensorflow.function
    def call(self, X):
        return positions_to_norm_coords(X, self.pmt_positions)
    
    def compute_output_shape(self, input_shape):
        return self._output_shape

class I0Layer(keras.layers.Layer):
    """Correct response for I0, for each PMT"""
    def __init__(self, n_pmts, init_val=1/77, **kwargs):
        super().__init__(**kwargs)
        self.n_pmts = tensorflow.constant(n_pmts, dtype=tensorflow.float32, name="n_pmts_i0_Layer")
        self.init_val = init_val 
        # Tensorflwo scalars are really annoying
        if isinstance(init_val,_scalar_types):
            self.scalar_init = True
        elif isinstance(init_val, tensorflow.Tensor):
            if init_val.shape == ():
                self.scalar_init = True
            else:
                self.scalar_init = False
        else:
            self.scalar_init = False

    def build(self, input_shape):
        self.i0 = self.add_weight(
            name='I_0', 
            shape=(self.n_pmts,),
            initializer=keras.initializers.Constant(value=self.init_val if self.scalar_init else 1),
            constraint=keras.constraints.NonNeg(),
            trainable=True)
        super().build(input_shape)
        # Init with pretrained weights
        if ~self.scalar_init: 
            self.i0.weights = self.init_val
    
    @tensorflow.function
    def call(self, X):
        return X * self.i0
    
class NormalizationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, X):
        X /= keras.backend.reshape(keras.backend.sum(X, axis=1), (-1, 1))
        return X
    
    def compute_output_shape(self, input_shape):
        return input_shape
    

class RadialLCELayer(keras.layers.Layer):
    """Compute radial LCE
    
    Wall terms keep failing the fit 

    Input (n_patterns, n_pmts), rho
    Output (n_patterns, n_pmts) with LCE/PMT
    """
    def __init__(self, guess, n_groups, group_slices,**kwargs):
        super().__init__(**kwargs)
        # Decay + fudge
        self.guess = guess
        self.n_groups = n_groups
        self.group_slices = tensorflow.ragged.constant(group_slices, dtype=tensorflow.int32)
        if n_groups == 1:
            # Debug leftover, but should be quicker
            self.call = self.single_group_call
        else:
            self.call = self.multi_group_call
        
        self.call = tensorflow.function(self.call)
        return 
    
    
    def lce_map(self,rho, p, d, a, b):
        # Last parameter filled role of linear addition and multiplier -> Pointless removed
        # Not sure using tensorflow funcs actually makes thsi faster or just less legible
        # also I came when i removed c to test something I+1 was easier than finding all the invocations
        return (
             tensorflow.math.divide((1 - b), tensorflow.math.pow((1 + tensorflow.math.square(rho/d)), p) )
            + tensorflow.math.multiply(a,  rho )
            + b
        )
    
    def build(self, input_shape):
        self.params = self.add_weight(
            name='lce_params', 
            shape=(self.n_groups, len(self.guess)),
            initializer=keras.initializers.Constant(
                np.array(self.guess * self.n_groups).reshape(self.n_groups, len(self.guess))),
                trainable=True)
        super().build(input_shape)

    def multi_group_call(self, X, training=None):        
        """Something here is broken """
        def process_group(group_idx, shape):
            # Select group
            group_indices = self.group_slices[group_idx]
            group_X = tensorflow.gather(X, group_indices, axis=1)
            # Need to pass group parameters explicitly 
            gp = self.params[group_idx, :len(self.guess)]
            # Apply Op
            res = self.lce_map(group_X, gp[0],gp[1],gp[2],gp[3])
            # We now need to generate the parameters to return a sparse matrix 
            batch_size = tensorflow.shape(res)[0]
            n_indices = len(group_indices)  
            # Generate one row per batch, with the batch idx (so for batch isze 256 contains 256 [i,i,i] with i in 0 255 )
            batch_indices = tensorflow.tile(tensorflow.range(batch_size)[:, tensorflow.newaxis], [1, n_indices])
            # Generate a list with first dime batch size and second n_indcs with the indcs as elements
            column_indices = tensorflow.tile(group_indices[tensorflow.newaxis, :], [batch_size, 1]) 
            # We stack the two together to get shape (batch_size, n_indeces, 2) < - last dim fully specifies pos within batch and which batch
            # This si since tf uses the flattened versions internaly
            indices = tensorflow.stack([batch_indices, column_indices], axis=-1)
            return tensorflow.scatter_nd(indices, res, shape)
        def process_all_groups(X):
            combined_result = tensorflow.zeros_like(X)
            # Assign for each group idx 
            shape = tensorflow.shape(combined_result)
            group_indeces = tensorflow.range(self.n_groups)  
            def per_group(group_idx):
                return process_group(group_idx, shape)
            # Cant do for loop with tf.function
            group_result = tensorflow.map_fn(per_group, group_indeces, dtype=X.dtype)
            # We have a 1 D higher dimensional sparse matrix
            combined_result = tensorflow.reduce_sum(group_result,axis=0)
            #for group_idx in tensorflow.range(tensorflow.shape(self.group_slices)[0]):
            #    combined_result += process_group(group_idx,shape)
            return combined_result
        return process_all_groups(X)
    
    def single_group_call(self, X, training=None):
        """
        Debug function as loss's are on order e16
        """
        gp = self.params[0, :len(self.guess)]
        return self.lce_map(X, gp[0],gp[1],gp[2],gp[3])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

## ----------------------- Plotting Helper -------------------------------------

def bin_events_to_pmt(events, pmt_pos, do_not_discard = False, radius = 3  * 2.54 / 2, batch_size = None):
    """
    Bins events to PMT positions

    base behavior : Discard events between positions (filter by within radius)
    TODO: Below looks slow
    alternative behavior : Assign eventt positions to closest pmt available
    

    returns:
    
    binn_mask = boolean array of shape n_events, n_pmts with true marking the closest per event
    binn_counts = sum over first axis of binn_mask -> Binn counts per pmt
    """
    n_pmts = len(pmt_pos)
    n_events = len(events)
    if batch_size is None:
        batch_size = n_events
    # Keep track of number of binned events per pmt
    binned_counts = np.zeros(n_pmts, dtype=int)
    # mask to track bin counts
    binn_mask = np.zeros((n_events, n_pmts), dtype=bool)

    for idx in range(0, n_events, batch_size):
        end = min(idx + batch_size, n_events)
        subset = events[idx:end]
        # Compute distances for event subset
        event_matrix = subset[:, np.newaxis, :] - pmt_pos[np.newaxis,:,:]
        d = np.linalg.norm(event_matrix, axis=2)
        if do_not_discard:
            closest_pmts = np.argmin(d, axis=1) # TODO: This looks to slow - optim when act using it
            for i, pmt_idx in enumerate(closest_pmts):
                binn_mask[idx + i, pmt_idx] = True
        else:
            binn_mask[idx:end, :] = d <= radius

    return binn_mask, np.sum(binn_mask, axis=0)


## ------------------------------- Plotting ------------------------------------
def do_density_plot(ax, pos,title='',bins=100, norm=None):
    radius = np.abs(np.max(pos))
    x_edges = y_edges = np.linspace(-radius, radius, bins)
    hist, xedges, yedges = np.histogram2d(pos[:, 0], pos[:, 1], bins=[x_edges, y_edges])
    im = ax.imshow(hist.T, origin='lower', extent=[-radius, radius, -radius, radius], cmap='viridis', norm=norm)
    plt.colorbar(im, label='Density', ax=ax)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    return ax

def plot_pmt_response(pmt_pos, response, ax=None, title=None, vmin=None, vmax=None, logscale=False, dead_pmts=None, mark_spot=None, secondary_response=None, text=None, cmap=plt.cm.viridis):
    # Misc
    radius = 3 * 2.54 / 2
    if isinstance(pmt_pos, pd.DataFrame):
        _pmt_pos = pmt_pos.to_numpy()
    else:
        _pmt_pos = pmt_pos

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Primary response normalization
    if not logscale:
        primary_normalizer = plt.Normalize
        _response = response
    else:
        primary_normalizer = matplotlib.colors.LogNorm
        _response = response.copy()
        _response[_response < 1] = vmin if vmin is not None else 1
    
    # Secondary response normalization
    if secondary_response is not None:
        if not logscale:
            secondary_normalizer = plt.Normalize
            _secondary_response = secondary_response
        else:
            secondary_normalizer = matplotlib.colors.LogNorm
            _secondary_response = secondary_response.copy()
            _secondary_response[_secondary_response < 1] = vmin if vmin is not None else 1
    # Plot dead PMTs and remove them from the list
    if dead_pmts is not None:
        for i in dead_pmts:
            circle = plt.Circle((_pmt_pos[i, -2], _pmt_pos[i, -1]), radius, color='grey', fill=True)
            ax.add_artist(circle)
        mask = np.ones(pmt_pos.shape[0], dtype=bool)
        mask[dead_pmts] = False
        _pmt_pos = _pmt_pos[mask]
        _response = _response[mask]
        if secondary_response is not None: _secondary_response = _secondary_response[mask]

    primary_norm = primary_normalizer(vmin=_response.min() if vmin is None else vmin,
                                       vmax=_response.max() if vmax is None else vmax)
    if secondary_response is not None:
        secondary_norm = secondary_normalizer(vmin=_secondary_response.min() if vmin is None else vmin,
                                              vmax=_secondary_response.max() if vmax is None else vmax)
    # Plot circles with response
    for i in range(len(_pmt_pos)):
        # Primary response color
        primary_response_color = cmap(primary_norm(_response[i]))
        if secondary_response is not None:
            # Wedge for primary response (half)
            primary_wedge = Wedge((_pmt_pos[i, -2], _pmt_pos[i, -1]), radius, 45, 225, color=primary_response_color, fill=True)
            ax.add_artist(primary_wedge)
            # Secondary response color
            secondary_response_color = cmap(secondary_norm(_secondary_response[i]))
            # Wedge for secondary response (half)
            secondary_wedge = Wedge((_pmt_pos[i, -2], _pmt_pos[i, -1]), radius, 225, 405, color=secondary_response_color, fill=True)
            ax.add_artist(secondary_wedge)
        else:
            # Full circle for primary response
            circle = plt.Circle((_pmt_pos[i, -2], _pmt_pos[i, -1]), radius, color=primary_response_color, fill=True)
            ax.add_artist(circle)

    # Colorbar for primary response
    sm_primary = plt.cm.ScalarMappable(cmap=cmap, norm=primary_norm)
    sm_primary.set_array([])  # ScalarMappable requires a dummy array
    cbar_primary = plt.colorbar(sm_primary, ax=ax)

    # Colorbar for secondary response
    if secondary_response is not None:
        sm_secondary = plt.cm.ScalarMappable(cmap=cmap, norm=secondary_norm)
        sm_secondary.set_array([])  # ScalarMappable requires a dummy array
        cbar_secondary = plt.colorbar(sm_secondary, ax=ax)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X position (cm)')
    ax.set_ylabel('Y position (cm)')
    ax.set_xlim([_pmt_pos[:, -2].min() - 10, _pmt_pos[:, -2].max() + 10])
    ax.set_ylim([_pmt_pos[:, -1].min() - 10, _pmt_pos[:, -1].max() + 10])
    if mark_spot is not None:
        ax.scatter(mark_spot[0], mark_spot[1], c='r', marker='x')
    if title is not None:
        ax.set_title(title)

    if text is not None:
        for key in text:
            add_scalable_text(ax, text[key], location="bottom_right")

    return ax


def add_scalable_text(ax, text, location="bottom_right"):
    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height
    font_size = min(width, height) * 0.04
    # Set text position based on specified location
    if location == "bottom_right":
        x, y = 0.8, 0.025  # Adjusted for bottom right
    elif location == "bottom_left":
        x, y = 0.01, 0.025
    elif location == "top_right":
        x, y = 0.8, 0.875
    elif location == "top_left":
        x, y = 0.01, 0.875

    # Add text to the plot, scaling font size with figure size
    ax.text(x, y, text, ha="center", va="center",
            fontsize=font_size,  # Scales with figure size
            transform=ax.transAxes, wrap=True)


def plot_pmt_indeces(pmt_pos, ax=None, fontsize=10, dead_pmts = None, color='black'):
    """pmt_pos as returned by the get pmt function
    dead_pmts : indices of dead pmts
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    if type(pmt_pos) == pd.core.frame.DataFrame: _pmt_pos = pmt_pos.to_numpy()
    else: _pmt_pos = pmt_pos
    radius = 3  * 2.54 / 2
    for i in range(len(_pmt_pos)):
        circle = plt.Circle((_pmt_pos[i,-2], _pmt_pos[i,-1]), radius, color=color, fill=False)
        ax.add_artist(circle)
        ax.text(_pmt_pos[i, -2], _pmt_pos[i, -1], str(i), color=color, 
                fontsize=fontsize, ha='center', va='center', fontweight='bold')
    if dead_pmts is not None:
        for i in dead_pmts:
            circle = plt.Circle((_pmt_pos[i,-2], _pmt_pos[i,-1]), radius, color='grey', fill=True)
            ax.add_artist(circle)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X position (cm)')
    ax.set_ylabel('Y position (cm)')

    ax.set_xlim([_pmt_pos[:, -2].min() - 10, _pmt_pos[:, -2].max() + 10])
    ax.set_ylim([_pmt_pos[:, -1].min() - 10, _pmt_pos[:, -1].max() + 10])
    return ax

def plot_pmt_circles(ax, pmt_pos, dead_pmts=None, alpha=1, linestyle='-', color='black'):
    """
    Utility function to mark pmt's
    """
    if type(pmt_pos) == pd.core.frame.DataFrame: _pmt_pos = pmt_pos.to_numpy()
    else: _pmt_pos = pmt_pos
    radius = 3  * 2.54 / 2
    for i in range(len(_pmt_pos)):
        circle = plt.Circle((_pmt_pos[i,-2], _pmt_pos[i,-1]), radius, color=color, fill=False, alpha=alpha, linestyle=linestyle)
        ax.add_artist(circle)
    if dead_pmts is not None:
        for i in dead_pmts:
            circle = plt.Circle((_pmt_pos[i,-2], _pmt_pos[i,-1]), radius, color='grey', fill=True, alpha=alpha, linestyle=linestyle)
            ax.add_artist(circle)
    return ax

def plot_ring_group(pmt_groups,pmt_pos_top, ax=None, fontsize=6):
    clrs = ['red', 'green', 'blue']
    cntr = 0
    ax = plot_pmt_indeces(pmt_pos_top,fontsize=fontsize, ax =ax)
    for i in pmt_groups:
        for j in i:
            circle = plt.Circle((pmt_pos_top[j,-2], pmt_pos_top[j,-1]), 3  * 2.54 / 2, color=clrs[cntr%3],alpha=0.7, fill=True)
            ax.add_artist(circle)
        cntr += 1
    return ax

def plot_loss_histogram(model, dat, val, loss_func, bins=50, ax=None, title = 'Histogram of Loss for Validation and Testing Datasets'):
    """GPT made this
    Plots a histogram of the loss values for both validation and testing data.
    
    Parameters:
    - model: Trained Keras model used for predictions.
    - validation_slice: A slice or index array for selecting the validation set.
    - testing_slice: A slice or index array for selecting the testing set.
    - x_data: The input dataset (features).
    - y_data: The ground truth dataset (targets).
    - bins: The number of bins for the histogram.
    """
    
    # Get validation and testing data
    x_val, y_val = val
    x_test, y_test = dat

    # Predict using the model for validation and test data
    y_val_pred = model.predict(x_val, verbose=0) 
    y_test_pred = model.predict(x_test, verbose=0)

    # Calculate loss (using mean squared error as an example)
    val_losses = loss_func(y_val, y_val_pred)
    test_losses = loss_func(y_test, y_test_pred)

    # Create the histogram plot
    if ax is None:
        plt.figure(figsize=(10, 6))
    else:
        plt.sca(ax)

    # Plot for validation loss
    hist, bins_ = np.histogram(val_losses, bins=bins)
    logbins = np.logspace(np.log10(bins_[0]),np.log10(bins_[-1]),len(bins_))
    plt.hist(val_losses, bins=logbins, alpha=0.5, label='Validation Loss', color='blue')

    # Plot for testing loss
    hist, bins_ = np.histogram(test_losses, bins=bins)
    logbins = np.logspace(np.log10(bins_[0]),np.log10(bins_[-1]),len(bins_))
    plt.hist(test_losses, bins=logbins, alpha=0.5, label='Training Loss', color='orange')
    plt.axvline(1, c='r', linestyle='dotted')
    med = np.sort(np.concatenate([val_losses, test_losses]))[(len(val_losses) + len(test_losses)) //2]
    plt.axvline(med, c='g', label='median {:.3}'.format(med))
    plt.xscale('log')
    plt.yscale('log')
    # Add labels and title
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title(title)
    #plt.xlim(left=0.1)
    # Add a legend
    plt.legend(loc='upper right')
    # Show the plot
    if ax is None:
        plt.show()
    return val_losses, test_losses


def plot_max_min_loss(trivial_lce, train_loss, val_loss, x_train, y_train, val, dead_pmts, pmt_pos_top,vmin=None, ax=None):
    max_loss = tensorflow.argmax(train_loss).numpy()
    min_loss = tensorflow.argmin(train_loss).numpy()

    not_dead = np.array([i not in dead_pmts for i in range(len(pmt_pos_top))])

    # Creating subplots
    if ax is None:
        fig, ax = plt.subplots(4, 2, figsize=(12, 20))
    # Maximum Loss pattern (convert to NumPy if necessary for the plotting function)
    plot_pmt_response(pmt_pos_top, y_train[max_loss].numpy(), ax=ax[0,0], 
                    title="Maximum Loss pattern train", mark_spot=x_train[max_loss].numpy(), 
                    dead_pmts=dead_pmts)
    # Minimum Loss pattern
    plot_pmt_response(pmt_pos_top, y_train[min_loss].numpy(), ax=ax[0,1], 
                    title="Minimum Loss pattern train", mark_spot=x_train[min_loss].numpy(), 
                    dead_pmts=dead_pmts)
    # Maximum Loss prediction (sum the pattern using TensorFlow and apply the prediction)
    pred_max = trivial_lce.predict(tensorflow.expand_dims(x_train[max_loss], axis=0), verbose=0)[0]
    pred_max = (np.abs(np.sum(y_train[max_loss].numpy())) / np.abs(np.sum(pred_max))) * pred_max
    plot_pmt_response(pmt_pos_top,pred_max , 
                    ax=ax[1,0], title="Maximum Loss prediction Train", 
                    mark_spot=x_train[max_loss].numpy(), dead_pmts=dead_pmts, vmin=np.min(pred_max[not_dead]), vmax = np.max(pred_max[not_dead]))
    # Minimum Loss prediction
    pred_min = trivial_lce.predict(tensorflow.expand_dims(x_train[min_loss], axis=0), verbose=0)[0]
    pred_min = (np.abs(np.sum(y_train[min_loss].numpy())) / np.abs(np.sum(pred_min))) * pred_min
    plot_pmt_response(pmt_pos_top, pred_min,  
                    ax=ax[1,1], title="Minimum Loss prediction Train", 
                    mark_spot=x_train[min_loss].numpy(), dead_pmts=dead_pmts, vmin=np.min(pred_min[not_dead]), vmax = np.max(pred_min[not_dead]))
    max_loss = tensorflow.argmax(val_loss).numpy()
    min_loss = tensorflow.argmin(val_loss).numpy()

    plot_pmt_response(pmt_pos_top, val[1][max_loss].numpy(), ax=ax[2,0], 
                    title="Maximum Loss pattern val", mark_spot=val[0][max_loss].numpy(), 
                    dead_pmts=dead_pmts)
    # Minimum Loss pattern
    plot_pmt_response(pmt_pos_top, val[1][min_loss].numpy(), ax=ax[2,1], 
                    title="Minimum Loss pattern val", mark_spot=val[0][min_loss].numpy(), 
                    dead_pmts=dead_pmts)
    # Maximum Loss prediction (sum the pattern using TensorFlow and apply the prediction)
    pred_max = trivial_lce.predict(tensorflow.expand_dims(val[0][max_loss], axis=0), verbose=0)[0]
    pred_max = (np.abs(np.sum(val[1][max_loss].numpy())) / np.abs(np.sum(pred_max))) * pred_max
    plot_pmt_response(pmt_pos_top,pred_max , 
                    ax=ax[3,0], title="Maximum Loss prediction Val", 
                    mark_spot=val[0][max_loss].numpy(), dead_pmts=dead_pmts, vmin=np.min(pred_max[not_dead]), vmax = np.max(pred_max[not_dead]))
    # Minimum Loss prediction
    pred_min = trivial_lce.predict(tensorflow.expand_dims(val[0][min_loss], axis=0), verbose=0)[0]
    pred_min = (np.abs(np.sum(val[1][min_loss].numpy())) / np.abs(np.sum(pred_min))) * pred_min
    plot_pmt_response(pmt_pos_top, pred_min,  
                    ax=ax[3,1], title="Minimum Loss prediction Val", 
                    mark_spot=val[0][min_loss].numpy(), dead_pmts=dead_pmts, vmin=np.min(pred_min[not_dead]), vmax = np.max(pred_min[not_dead]))
    print("Overall Median Value: {}".format(np.sort(np.concatenate([val_loss, train_loss]))[(len(val_loss) + len(train_loss)) //2]))
    if ax is None:
        plt.show()
    return pred_max, pred_min


def plot_train_results(model, dat, val, loss_func, dead_pmts,pmt_pos_top,figsize=(12, 25), vmin = None, bin = 1000):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 1])  # Equal height for all rows

    # Create the first plot spanning both columns in the first row
    ax1 = fig.add_subplot(gs[0, :])  # Row 0, span both columns
    # We make this current active so we dont have to rewrite the function
    #plt.sca(ax1) 
    val_loss, train_loss = plot_loss_histogram(model, dat, val, loss_func=loss_func,ax=ax1, bins=1000)

    # Create the remaining 2-column by 4-row plots
    ax = np.zeros((4,2), dtype=object)
    ax[0,0] = fig.add_subplot(gs[1, 0])  # Row 1, Col 0
    ax[0,1] = fig.add_subplot(gs[1, 1])  # Row 1, Col 1
    ax[1,0] = fig.add_subplot(gs[2, 0])  # Row 2, Col 0
    ax[1,1] = fig.add_subplot(gs[2, 1])  # Row 2, Col 1
    ax[2,0] = fig.add_subplot(gs[3, 0])  # Row 3, Col 0
    ax[2,1] = fig.add_subplot(gs[3, 1])  # Row 3, Col 1
    ax[3,0] = fig.add_subplot(gs[4, 0])  # Row 4, Col 0
    ax[3,1] = fig.add_subplot(gs[4, 1])  # Row 4, Col 1

    plot_max_min_loss(model, train_loss, val_loss, dat[0], dat[1], val, dead_pmts, pmt_pos_top, ax = ax, vmin=None)
    
    fig.show()

def plot_anode_gate(max_r, ax=None, s=0.1, n= 500):
    def gate_wires(x, op):
        # op = +-
        return op(np.sqrt(3)*x*10, 263) / 10
    def anode_wires(x, op, which):
        # op = +-
        return op(np.sqrt(3)*x*10, 318 if which == 0 else 283) /10
    x_vals = np.linspace(-max_r, max_r, n)
    for i in (lambda X, Y : X+Y, lambda X, Y : X-Y):
        y_gate = gate_wires(x_vals, i)
        mask_gate = (x_vals**2 + y_gate**2 )<= max_r**2
        ax.scatter(x_vals[mask_gate], y_gate[mask_gate], label='Gate Wires', color='blue',s=s)
    for i in (lambda X, Y : X+Y, lambda X, Y : X-Y):
        for j in (0, 1):
            y_gate = anode_wires(x_vals, i, j)
            mask_gate = (x_vals**2 + y_gate**2 )<= max_r**2
            ax.scatter(x_vals[mask_gate], y_gate[mask_gate], label='Anode Wires', color='red',s=s)


def get_circular_grid(center, radius, n_points):
    """This will not generate the exact points requested, but good enough"""
    n = int(np.sqrt((4 / np.pi) * n_points))
    x_values = np.linspace(center[0]-radius, center[0]+radius, n)
    y_values = np.linspace(center[1]-radius, center[1]+radius, n)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    grid = np.vstack((x_flat, y_flat)).T
    grid = grid[((grid[:,0]-center[0])**2 + (grid[:,1]-center[1])**2) <= radius**2]
    return grid

def plot_interpolated_cmap(pos, loss,center,radius, ax=None, title = "Interpolated Loss Colormap", mark_spot= None, max_n_per_axes=2000,mark_min=True):
    if ax is None:
        plt.figure(figsize=(10,10)) 
    else:
        plt.sca(ax)

    x_min, y_min = np.min(pos, axis=0)
    x_max, y_max = np.max(pos, axis=0)
    min_d = np.min([np.abs(x_max-x_min)/max_n_per_axes, np.abs(y_max-y_min/max_n_per_axes)])
    grid_x, grid_y = np.mgrid[x_min:x_max:min_d, y_min:y_max:min_d]

    # Grid mask 
    distance_from_center = np.sqrt((grid_x - center[0])**2 + (grid_y - center[1])**2)
    circular_mask = distance_from_center <= radius
    # Interpolate
    grid_z = sp.interpolate.griddata(pos, loss, (grid_x, grid_y), method='nearest')
    pref = 'Loss'

    grid_z[~circular_mask] = np.nan
    plt.imshow(grid_z.T, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis')
    plt.colorbar(label=pref)
    #plt.scatter(positions[:, 0], positions[:, 1], c='r', s=10, label='Original Positions')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if mark_spot is not None:
        plt.scatter(mark_spot[0], mark_spot[1], c='r', marker='x', label="Event Pos")
    if mark_min:
        minloss = np.argmin(loss)
        plt.scatter(pos[minloss,0],pos[minloss,1],c='orange',marker='x',label='Minimum Loss')
        return pos[minloss], loss[minloss]




def plot_mesh(tpc_radius, wire_pitch, num_wires, rotation_angle, ax, offset=0, linewidth=0.2, color='blue'):
    """With parameters from MC"""
    def rotate_point(x, y, angle):
        """Rotate a point (x, y) by a given angle."""
        x_rotated = x * np.cos(angle) - y * np.sin(angle)
        y_rotated = x * np.sin(angle) + y * np.cos(angle)
        return x_rotated, y_rotated
    # Plot the TPC boundary
    circle = plt.Circle((0, 0), tpc_radius, color='black', fill=False)
    ax.add_artist(circle)

    # Generate main anode wires
    for i in range(num_wires):
        # Calculate the x position for each wire
        x = (i - num_wires // 2) * wire_pitch
        # Calculate the corresponding y-values using the circle equation
        y_top = np.sqrt(tpc_radius**2 - x**2) if abs(x) <= tpc_radius else 0
        y_bottom = -y_top
        x += offset
        
        # Rotate the wire endpoints
        x_top_rotated, y_top_rotated = rotate_point(x, y_top, rotation_angle)
        x_bottom_rotated, y_bottom_rotated = rotate_point(x, y_bottom, rotation_angle)

        # Plot the rotated wire line
        ax.plot([x_bottom_rotated, x_top_rotated], [y_bottom_rotated, y_top_rotated], color=color, linewidth=0.2)



def pos_diff_plot(old_pos, new_pos, pmt_pos, dead_pmts,max_r, vmax=2, s=0.15, dpi=200, save=None, xlim=None, ylim=None):
    from scipy.stats import binned_statistic_2d

    def bindif_2d(p1, p2, bins=30):
        bsx = binned_statistic_2d(p1[:,0], p1[:,1], (p2-p1)[:,0], bins=bins, statistic='median')
        bsy = binned_statistic_2d(p1[:,0], p1[:,1], (p2-p1)[:,1], bins=bins, statistic='median')
        xcenters = (bsx.x_edge[1:] + bsx.x_edge[:-1])/2
        ycenters = (bsx.y_edge[1:] + bsx.y_edge[:-1])/2
        return xcenters, ycenters, bsx.statistic, bsy.statistic
    dead_pmts_positions = pmt_pos.to_numpy()[dead_pmts,2:]
    pmt_positions = pmt_pos.to_numpy()[:,2:]
    def plot_pmts():
        opts = dict(s=10, marker='o', zorder=5, alpha=0.5, edgecolors='none')
        plt.scatter(pmt_positions[:,0] * max_r, pmt_positions[:,1] * max_r, c='k', **opts)
        plt.scatter(dead_pmts_positions[:,0] * max_r, dead_pmts_positions[:,1] * max_r, c='r', **opts)
    difs = np.linalg.norm(new_pos - old_pos, axis=1) 
    f, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    for i, (x, title) in enumerate(
            ([old_pos, 'Old positions'], [new_pos, 'New positions'])):
        plt.sca(axes[i])
        plot_pmts()
        plt.scatter(
            x[:, 0],
            x[:, 1],
            c=difs,
            s=s,
            vmin=0, vmax=vmax,
            edgecolors='none', 
            cmap=plt.cm.jet)
        if i == 1:
            plt.colorbar(label='Distance moved [cm]', ax=axes, extend='max')
            
        x, y, dx, dy = bindif_2d(old_pos , new_pos , bins=50)
        # Have to give indexing='ij' here, if you just give x, y quiver will
        # meshgrid without it.... an
        xx, yy = np.meshgrid(x, y, indexing='ij')
        plt.quiver(xx, yy, dx, dy,
                   angles='xy', scale_units='xy', scale=1)
            
        ax = plt.gca()
        ax.set_aspect(1)
        #plt.set_aspect(1)
        ax.add_artist(plt.Circle((0, 0), max_r, color='k',
                                 linewidth=0.5, alpha=0.5, fill=False,
                                 zorder=10))

        plt.title(title)
        q = 1.1
        plt.xlim(-max_r * q, max_r * q)
        if xlim is not None: plt.xlim(xlim)
        plt.xlabel("X [cm]")
        plt.ylim(-max_r * q, max_r * q)
        if ylim is not None: plt.ylim(ylim)
        plt.xlabel("Y [cm]")
    if save:
        plt.savefig(save, dpi=dpi)
    plt.show()