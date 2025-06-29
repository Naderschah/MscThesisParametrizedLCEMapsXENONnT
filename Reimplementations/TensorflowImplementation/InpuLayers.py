

"""
Within this file all import layers are presented 


Note that while searlization is always implemented it doesnt actually work for all layers


Classes: 
- GetRadius - Input for Direct Detection Network 
- GetWallDistPredictive - Applies the grid search to find the shortest wall reflection path - input to wall reflection model 
- PerpendicularWire_Parametrization - Computes the amount of perpendicular wire in the path of the instantaneous event site. 
- Anode_Parametrization - Computes average amount of anode mesh in the light cone 


TODOs:
GetWallDistPredictive : Change Name 
Anode_SHadowinBias : Or whatever ends up being the implementation 
AdvCoordTransfrom 

All functions Implemented : False
"""
import tensorflow as tf
from tensorflow import keras 
import numpy as np 

class GetRadius(keras.layers.Layer):
    """
    Computes distance of event site to PMT center
    """
    def __init__(self, output_shape, pmt_positions, **kwargs):
        super().__init__(**kwargs)
        self._output_shape_hint = output_shape
        self.pmt_positions = tf.constant(pmt_positions, dtype=tf.float32, name="pmt_positions_GetRadius")
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, X):
        # Norm is not undeflow safe
        rad = tf.sqrt(tf.reduce_sum(tf.square(X - self.pmt_positions),axis=-1) + 1.0e-12)
        # Enforce Maximum Distance for pos refit
        return tf.minimum(rad, 2.2*66.4)
    
    def compute_output_shape(self, input_shape):
        return self._output_shape

    ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_shape": self._output_shape_hint,
            # Convert the tensor to a list for serialization.
            "pmt_positions": self.pmt_positions.numpy().tolist()
        })
        return config
    @classmethod
    def from_config(cls, config):
        # No conversion needed here since __init__ will cast pmt_positions to a tensor.
        return cls(**config)
    

class GetWallDistPredictive(tf.keras.layers.Layer):
    def __init__(self, angle_model=None, pmt_positions=None, tpc_r=None,
                 max_iter=50, tol=1e-4, **kwargs):
        super().__init__(**kwargs)

        self.angle_model = angle_model
        self.angle_model.trainable = False

        self.pmt_positions = tf.constant(pmt_positions, dtype=tf.float32)
        self.tpc_r = tf.constant(tpc_r, dtype=tf.float32)
        self.n_pmts = pmt_positions.shape[0]

        r_pmt = tf.norm(self.pmt_positions, axis=1, keepdims=True)
        theta_pmt = tf.atan2(self.pmt_positions[:, 1:2], self.pmt_positions[:, 0:1])
        self.pmt_polar = tf.concat([r_pmt, theta_pmt], axis=1)

        self.max_iter = max_iter
        self.tol = tol

    def _predict_with_model(self, X, training = None):
        batch_size = tf.shape(X)[0]
        r_event = tf.norm(X, axis=1, keepdims=True)
        theta_event = tf.atan2(X[:, 1:2], X[:, 0:1])
        r_event_tiled = tf.repeat(r_event, self.n_pmts, axis=0)
        theta_event_tiled = tf.repeat(theta_event, self.n_pmts, axis=0)
        r_pmt_tiled = tf.tile(self.pmt_polar[:, 0:1], [batch_size, 1])
        theta_pmt_tiled = tf.tile(self.pmt_polar[:, 1:2], [batch_size, 1])
        model_input = tf.concat([r_event_tiled, theta_event_tiled, r_pmt_tiled, theta_pmt_tiled], axis=1)

        pred_vec = self.angle_model(model_input)
        W = self.tpc_r * pred_vec

        X_tiled = tf.repeat(X, self.n_pmts, axis=0)
        pmts_tiled = tf.tile(self.pmt_positions, [batch_size, 1])

        d_event = tf.norm(W - X_tiled, axis=1, keepdims=True)
        d_pmt = tf.norm(W - pmts_tiled, axis=1, keepdims=True)

        return tf.reshape(tf.concat([d_event, d_pmt], axis=1), (batch_size, self.n_pmts, 2))

    @tf.function
    def call(self, X, training = None):
        batch_size = tf.shape(X)[0]
        event_exp = tf.repeat(X, self.n_pmts, axis=0)
        pmts_exp = tf.tile(self.pmt_positions, [batch_size, 1])

        theta_event = tf.atan2(event_exp[:, 1], event_exp[:, 0])
        theta_pmt = tf.atan2(pmts_exp[:, 1], pmts_exp[:, 0])
        #theta = tf.math.atan2(tf.sin(theta_event + theta_pmt), tf.cos(theta_event + theta_pmt)) / 2.0
        delta = tf.atan2(tf.sin(theta_pmt - theta_event),
                 tf.cos(theta_pmt - theta_event))
        theta = theta_event + 0.5 * delta

        step = tf.constant(0.1, dtype=tf.float32)

        def compute_loss(th):
            W = self.tpc_r * tf.stack([tf.cos(th), tf.sin(th)], axis=1)
            return tf.norm(W - event_exp, axis=1) + tf.norm(W - pmts_exp, axis=1)

        def cond(i, theta, step):
            """Greater equal since otherwise with default tolerance doesnt actually do any optimizing"""
            return tf.less(i, self.max_iter) & tf.greater_equal(step, self.tol)

        def body(i, theta, step):
            loss = compute_loss(theta)
            loss_plus = compute_loss(theta + step)
            loss_minus = compute_loss(theta - step)

            better_plus = loss_plus < loss
            better_minus = loss_minus < loss

            move = tf.where(better_plus, step, tf.where(better_minus, -step, 0.0))
            theta_new = theta + move
            step_new = tf.where(tf.reduce_any(move != 0.0), step, step * 0.5)

            return i + 1, theta_new, step_new

        i = tf.constant(0)
        _, theta_final, _ = tf.while_loop(cond, body, [i, theta, step])

        # Handle trivial coord = 0,0 case -> Otherwise the sign for low tolerance biases the result
        is_center = tf.less(tf.norm(event_exp, axis=1), 1e-8)          
        theta_final = tf.where(is_center, theta_pmt, theta_final)

        W_final = self.tpc_r * tf.stack([tf.cos(theta_final), tf.sin(theta_final)], axis=1)
        d_event = tf.norm(W_final - event_exp, axis=1)
        d_pmt = tf.norm(W_final - pmts_exp, axis=1)
        return tf.reshape(tf.stack([d_event, d_pmt], axis=1), (batch_size, self.n_pmts, 2))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_pmts, 2)

    def get_config(self):
        return {
            **super().get_config(),
            "pmt_positions": self.pmt_positions.numpy().tolist(),
            "tpc_r": float(self.tpc_r.numpy()),
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    @classmethod
    def from_config(cls, config):
        angle_model = keras.saving.deserialize_keras_object(config.pop("angle_model"))
        return cls(angle_model=angle_model, **config)



class PerpendicularWire_Parametrization(tf.keras.layers.Layer): 
    """
    Output 0-1 for total amount of wire in the way of Event To PMT
    """
    def __init__(self, pmt_pos, pmt_r, **kwargs):
        super().__init__(**kwargs)
        self.angle = tf.constant(-(np.pi / 3) + (np.pi / 2), dtype=tf.float32, name="angle_PerpendicularWire")
        self.rot_matrix = tf.transpose(tf.stack([[tf.cos(self.angle ), -tf.sin(self.angle )],[tf.sin(self.angle ),  tf.cos(self.angle )]]))
        self.rot_matrix = tf.constant(self.rot_matrix, dtype=tf.float32, name="rot_matrix_PerpendicularWire")
        self.pmt_pos = tf.constant(pmt_pos, dtype=tf.float32, name="pmt_pos_PerpendicularWire") # (n_pmts, 2)
        self.pmt_rot = tf.constant(tf.linalg.matmul(self.pmt_pos, self.rot_matrix), dtype=tf.float32, name="pmt_rot_PerpendicularWire") # (n_pmts, 2)

        # Constant
        self.pmt_r = tf.constant(pmt_r, dtype=tf.float32, name="pmt_r_PerpendicularWire")
        self.tpc_r = tf.constant(66.4, dtype=tf.float32, name="tpc_r_PerpendicularWire")
        self.h = tf.constant(0.027, dtype=tf.float32, name="h_PerpendicularWire")
        self.H = tf.constant(68.58 / 10, dtype=tf.float32, name="H_PerpendicularWire")
        self.anode_r = tf.constant(0.0304 /2, dtype=tf.float32, name="anode_r_PerpendicularWire")
        self.dz = tf.constant(self.h + self.H, dtype=tf.float32, name="dz_PerpendicularWire")

        self.wire_start = tf.reshape(tf.constant([31.8, -31.8, 28.3, -28.3]) - self.anode_r, ( 1, 4))
        self.wire_end = tf.reshape(tf.constant([31.8, -31.8, 28.3, -28.3]) + self.anode_r, (1, 4))

        return

    def build(self, input_shape):
        super().build(input_shape)

    def n_wires_in_way(self, events):
        """
        The logic of this really simple problem keeps confusing me
        So we compute the gradient of the hypotenuse (triangle event site pmt center pmt edge)
        And hten just multiply by height and add / subtract
        
        """
        # (batch, 2)
        events_rot = tf.linalg.matmul(events, self.rot_matrix)
        # (batch, n_pmts)
        grad_minus = tf.math.abs(((self.pmt_rot[:,0]-self.pmt_r) - events_rot[:,0, None]) / self.dz)
        grad_plus = tf.math.abs(((self.pmt_rot[:,0]+self.pmt_r) - events_rot[:,0, None]) / self.dz)
        # Compute bounds on anode mesh height
        bounds_minus = events_rot[:,0, None] - grad_minus * self.h
        bounds_plus = events_rot[:,0, None] + grad_plus * self.h
        """
        Overlap = min( 0, (min(x_1end, x_2end) - max(x_1start, x_2start)) )
        """
        start_overlap = tf.maximum(tf.expand_dims(bounds_minus,axis=-1), self.wire_start)    # (batch, n_pmts, 4)
        end_overlap   = tf.minimum(tf.expand_dims(bounds_plus,axis=-1), self.wire_end)       # (batch, n_pmts, 4)
        overlap       = tf.maximum(0.0, end_overlap - start_overlap) # (batch, n_pmts, 4)

        # Return Fraction of Wire in the way 
        return tf.math.reduce_sum(overlap, axis=-1, keepdims=True) / (2 * self.anode_r)

    def call(self, x):
        return self.n_wires_in_way(x)

    ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "pmt_pos": self.pmt_pos.numpy().tolist() if hasattr(self.pmt_pos, "numpy") else self.pmt_pos,
            "pmt_r": float(self.pmt_r.numpy()) if hasattr(self.pmt_r, "numpy") else self.pmt_r
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    


class Anode_Parametrization(tf.keras.layers.Layer):
    """
    Implements the average number of wires in the way of the light cone of an event site to a PMT 
    """
    def __init__(self, pmt_pos,pmt_r, **kwargs):
        super().__init__(**kwargs)
        self.angle = tf.constant( -np.pi / 3, dtype=tf.float32, name="angle_AnodeParametrization")
        self.rot_matrix = tf.constant(tf.transpose(tf.stack([[tf.cos(self.angle ), -tf.sin(self.angle )],[tf.sin(self.angle ),  tf.cos(self.angle )]])), dtype=tf.float32, name="rot_matrix_AnodeParametrization")
        self.pmt_pos = tf.constant(pmt_pos, dtype='float32', name="pmt_pos_anode_parametrization") # (n_pmts, 2)
        # (n_pmts, 2)
        self.pmt_rot = tf.constant(tf.linalg.matmul(self.pmt_pos, self.rot_matrix), dtype='float32', name="pmt_rot_anode_parametrization") 

        self.pmt_r = tf.constant(pmt_r, dtype=tf.float32, name="pmt_r_AnodeParametrization")
        self.anode_r = tf.constant(0.216 / 2, dtype=tf.float32, name="anode_r_AnodeParametrization")
        self.h = tf.constant(0.027, dtype=tf.float32, name="h_AnodeParametrization")
        self.H = tf.constant(68.58 / 10, dtype=tf.float32, name="H_AnodeParametrization")
        self.dz = tf.constant(self.h + self.H, dtype=tf.float32, name="dz_AnodeParametrization")
        self.wire_pitch = tf.constant(0.5, dtype=tf.float32, name="wire_pitch_AnodeParametrization")

        self.pmt_mesh_angle = tf.math.atan2(self.pmt_pos[:, 1], self.pmt_pos[:, 0])  # angle of (x, y)
        # pmt_mesh_angle is shape (n_pmts,)
        self.pmt_mesh_angle = tf.math.mod(self.pmt_mesh_angle - self.angle, np.pi)  # bring into [0, pi]

    def build(self, input_shape):
        super().build(input_shape)
    
    def n_wires_in_way(self, events):
        # (batch, 2)
        events_rot = tf.linalg.matmul(events, self.rot_matrix)
        # (batch, n_pmts)
        diff = tf.math.abs(2*(self.pmt_rot[:,0] - events_rot[:,0, None]) * self.h / self.dz) / self.wire_pitch
        return tf.expand_dims(diff, axis=-1) # Output range is 0 to 2 
    
    def call(self, x, training):
        n_wires = self.n_wires_in_way(x)  # (batch, n_pmts, 1)
        return n_wires

    ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "pmt_pos": self.pmt_pos.numpy().tolist() if hasattr(self.pmt_pos, "numpy") else self.pmt_pos,
            "pmt_r": float(self.pmt_r.numpy()) if hasattr(self.pmt_r, "numpy") else self.pmt_r
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)