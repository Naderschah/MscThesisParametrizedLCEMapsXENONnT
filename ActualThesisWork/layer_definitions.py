import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import numpy as np 
import math 
from tensorflow.keras.constraints import NonNeg 

#### All  of the Layers

# Network architecture to fit a layer in Radius a layer in wall distance and a layer in Anode mesh distance 

#### ---------------------------------------------- Saving and Loading Models -------------------------------

"""
Why:
- Serialization does not work consistently, and I keep running into a string not callable error allthough this is impossible 
    - Allthough it is definitely related MonoDense Layers (alongside my own) but I was unable to fix the monodense implementation
- Getting weights directly uses indices to assign weights, so under model change it will not work 
- This method uses the layer names and subpaths within the model rather than the list indices
- Allows removing and adding components fairly easily skipping the intermediary code for weight assignment 
"""
import json, os

# Grabbed from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)
    
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

def generate_weight_dict(model):
    """Generates Fully qualified name trees so that weights are assigned back correctly assuming layer names did not change"""
    weights = {}

    def collect_weights(layer, prefix=""):
        name = layer.name
        if isinstance(layer, tf.keras.Model) and prefix == "":
            # Skip root model name
            prefix = ""
        else:
            prefix = f"{prefix}{name}/"

        for var in layer.weights:
            weights[f"{prefix}{var.name}"] = var.numpy()

        for sublayer in getattr(layer, 'layers', []):
            collect_weights(sublayer, prefix)

    collect_weights(model)
    return weights

    
def save_weights_to_dict(model, path):
    """
    Saves weights to a json file with layer names as keys so that model is invariant to order changes

    Mainly made this function to include the numpy encoder
    """
    weights = convert_numpy(generate_weight_dict(model))
    with open(path, 'w') as f:
        json.dump(weights, f, cls=NumpyEncoder)

def load_weights_from_dict(model, weight_dict, silent = False):
    """
    Loads weights from a dict (or from a filepath) into a model, using fully qualified paths.
    """
    if isinstance(weight_dict, str):
        with open(weight_dict, 'r') as f:
            weight_dict = json.load(f)
    
    if not isinstance(weight_dict, dict):
        raise ValueError("weight_dict must be a dictionary after loading")
    
    def assign_weights(layer, prefix=""):
        name = layer.name
        if isinstance(layer, tf.keras.Model) and prefix == "":
            prefix = ""
        else:
            prefix = f"{prefix}{name}/"

        for var in layer.weights:
            key = f"{prefix}{var.name}"
            if key in weight_dict:
                var.assign(weight_dict[key])
            else:
                if not silent:
                    print(f"Warning: {key} not found in weight_dict")

        for sublayer in getattr(layer, 'layers', []):
            assign_weights(sublayer, prefix)

    assign_weights(model)
    return model

#### ----------------------------------------------   Compute Input Variables ---------------------------------------------- ####
# Compute Radial Distance
class GetRadius(keras.layers.Layer):
    """
    Computes Radius from input x, y
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

class GetX2Y2(keras.layers.Layer):
    """
    Rotates both event positions and PMT positions by a fixed angle,
    then returns squared (x, y) distances in that rotated frame:
    output shape = (batch_size, n_pmts, 2).
    """
    def __init__(self, pmt_positions, **kwargs):
        super().__init__(**kwargs)
        angle=-np.pi/3
        # build rotation matrix R for angle θ
        c = np.cos(angle).astype(np.float32)
        s = np.sin(angle).astype(np.float32)
        R = np.array([[ c, -s],
                      [ s,  c]], dtype=np.float32)  # shape (2,2)

        # store as a constant
        self.R = tf.constant(R, name="rotation_matrix")

        # rotate PMT positions once: (n_pmts,2) @ (2,2).T → (n_pmts,2)
        ppos = tf.constant(pmt_positions, dtype=tf.float32)
        self.pmt_positions = tf.matmul(ppos, tf.transpose(self.R))

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, X):
        # X: (batch,2). Rotate events: (batch,2) @ R^T → (batch,2)
        X_rot = tf.matmul(X, tf.transpose(self.R))
        # broadcast to (batch, n_pmts, 2) minus rotated PMT positions (1,n_pmts,2)
        d = X_rot - self.pmt_positions
        # return squared components
        return tf.square(d)  # shape (batch, n_pmts, 2)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "output_shape": self._output_shape_hint,
            "pmt_positions": self.pmt_positions.numpy().tolist(),
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)



class I0Layer(keras.layers.Layer):
    """Correct response for I0, for each PMT"""
    def __init__(self, n_pmts, init_val=1/77, **kwargs):
        super().__init__(**kwargs)
        self.n_pmts = n_pmts
        self.i0 = self.add_weight(
            name='I_0', 
            shape=(self.n_pmts,),
            initializer=keras.initializers.Constant(value=init_val),
            constraint=keras.constraints.NonNeg(),
            trainable=True) 
    def build(self, input_shape):
        super().build(input_shape)
    @tf.function
    def call(self, X):
        return X * self.i0

    ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_pmts": self.n_pmts,
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Compute Event Wall Distance
class GetWallDist(keras.layers.Layer):
    """
    Wall Dist

    Computes distance to projected point on the wall from event site - terrible approximation
    """
    def __init__(self, output_shape, pmt_positions, tpc_r, **kwargs):
        super().__init__(**kwargs)
        self._output_shape = output_shape
        self.pmt_positions = tf.constant(pmt_positions, dtype=tf.float32, name="pmt_positions_WallDist")
        self.tpc_r =tf.constant(tpc_r, dtype=tf.float32, name="tpc_r_WallDist")
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, X):
        # Position Vector Magnitude and scaling (For now we operate on last one as well)
        pos_mag = tf.sqrt(tf.reduce_sum(tf.square(X), axis=-1))
        # Calculate the scaling factors
        scaling = self.tpc_r / tf.maximum(pos_mag, 1e-7)
        # Project the positions
        projection = X * tf.expand_dims(scaling, axis=-1)
        # Calculate the event wall distance
        event_wall_dist = tf.abs(self.tpc_r - pos_mag)
        # Enforce Maximum Distance for pos refit
        event_wall_dist = tf.minimum(event_wall_dist, 2.2*66.4)
        # Calculate the PMT wall distance
        pmt_wall_dist = tf.norm(tf.expand_dims(projection, axis=1) - self.pmt_positions, axis=-1)
        # Enforce Maximum Distance for pos refit
        pmt_wall_dist = tf.minimum(pmt_wall_dist, 2.2*66.4)
        # Generates flattened array of pairs pmt_wall_dist, event_wall_dist
        #return tf.reshape(tf.concat([(tf.expand_dims(pmt_wall_dist, axis=-1)), tf.expand_dims(tf.repeat(tf.expand_dims(event_wall_dist, axis=-1), 246, axis=-1), axis=-1)], axis=-1), (-1,2))
        return tf.concat([(tf.expand_dims(pmt_wall_dist, axis=-1)), tf.expand_dims(tf.repeat(tf.expand_dims(event_wall_dist, axis=-1), 246, axis=-1), axis=-1)], axis=-1)
    
    def compute_output_shape(self, input_shape):
        return self._output_shape

    ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_shape": self._output_shape,
            # Serialize the PMT positions as a list
            "pmt_positions": self.pmt_positions.numpy().tolist(),
            # Serialize tpc_r as a float
            "tpc_r": float(self.tpc_r.numpy())
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

import tensorflow as tf
from tensorflow import keras


class GetWallDistPredictive(tf.keras.layers.Layer):
    # TODO Add method to overwrite calling the model in case someone finds a better way of doing this
    # TODO Also retrain with the optimzatin algo here, original training was poor resolution
    def __init__(self, angle_model=None, model_path=None, pmt_positions=None, tpc_r=None,
                 mode="optimize", max_iter=50, tol=1e-4, **kwargs):
        super().__init__(**kwargs)

        if (angle_model is None) and (model_path is None) and (mode == "model"):
            raise ValueError("Either 'angle_model' or 'model_path' must be provided to run with model mode.")
        if angle_model is None:
            angle_model = tf.keras.models.load_model(model_path, compile=False)

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
        self.set_mode(mode)

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

    def _predict_with_optimization(self, X, training = None):
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

    def set_mode(self, mode, max_iter=None, tol=None):
        if max_iter is not None:
            self.max_iter = max_iter
        if tol is not None:
            self.tol = tol
        if mode == "model":
            self.call = tf.function(self._predict_with_model)
        elif mode == "optimize":
            self.call = tf.function(self._predict_with_optimization)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_pmts, 2)

    def get_config(self):
        return {
            **super().get_config(),
            "angle_model": keras.saving.serialize_keras_object(self.angle_model),
            "pmt_positions": self.pmt_positions.numpy().tolist(),
            "tpc_r": float(self.tpc_r.numpy()),
            "mode": "model",  # Default for deserialization
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    @classmethod
    def from_config(cls, config):
        angle_model = keras.saving.deserialize_keras_object(config.pop("angle_model"))
        return cls(angle_model=angle_model, **config)


# Compute Anode Mesh distances - Currently Unused
class GetAnodeDist(keras.layers.Layer):
    """
    """
    def __init__(self, tpc_r=66.4, angle = -np.pi / 3, **kwargs):
        super().__init__(**kwargs)
        # Compute x positions in rotated frame where mesh has x coordinates and y<= R_tpc
        self.n_wires = tf.constant(265, dtype=tf.float32, name="n_wires_AnodeDist")
        self.angle = tf.constant(angle, dtype=tf.float32, name="angle_AnodeDist")
        self.tpc_r = tf.constant(tpc_r, dtype=tf.float32, name="tpc_r_AnodeDist")
        self.anode_rot_x = self.compute_x_coords()

    def compute_x_coords(self):
        x_coords = []
        wire_pitch = 5.0 / 10
        offset = 0
        for i in range(self.n_wires):
            # Calculate the x position for each wire
            x = (i - self.n_wires // 2) * wire_pitch
            x += offset
            x_coords.append(x)
        # Convert to tensorflow object
        return tf.constant(x_coords, dtype=tf.float32)
    
    def rotate_coords(self, X):
        # Rotate the coordinates to the anode mesh frame
        return X[:,0] * tf.cos(self.angle) + X[:,1] * tf.sin(self.angle)
    
    def build(self, input_shape):
        super().build(input_shape)
    @tf.function
    def call(self, X):
        rot_X = tf.expand_dims(self.rotate_coords(X), axis=-1)
        # Calculate the distance to each wire
        return (rot_X - self.anode_rot_x)
    
    def compute_output_shape(self, input_shape):
        return (self.input_shape[0], self.n_wires)
    ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "tpc_r": float(self.tpc_r.numpy()),
            "angle": float(self.angle.numpy())
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def norm_angle(theta): #  -pi to pi
    """Truncates angle: -pi to pi"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

# From Shenyang - Currenlty Unused
class AdvCoordTransform(tf.keras.layers.Layer):
    """Convert (batch, pos) positions to (batch, pmt, r) coordinates, where

    :param X: (m, n_pmts, 2), positions in detector's (x,y) cartesian coords / max_r.
    :returns: (m, n_pmts, 2), positions in (de, di)
    """
    def __init__(self, pmt_pos_top, not_dead_pmts, **kwargs):
        super().__init__(**kwargs)
        self.pmt_pos_top = pmt_pos_top
        self.not_dead_pmts = not_dead_pmts


    def build(self, input_shape):
        self.pmt_pos = tf.constant(
            self.pmt_pos_top[self.not_dead_pmts], dtype=tf.float32, name='pmt_pos')
        super().build(input_shape)

    @tf.function
    def call(self, x):
        max_r = 66.4
        # Event position
        xx_abs, yy_abs = x[:, :, 0], x[:, :, 1]
        xx, yy = xx_abs / max_r, yy_abs / max_r
        rr = (xx**2 + yy**2)**0.5
        ww = 1 - rr # to wall position
        
        zz = x - self.pmt_pos
        xp, yp = zz[:, :, 0] / max_r, zz[:, :, 1] / max_r
        rp = (xp**2 + yp**2)**0.5
        
        qp = norm_angle(tf.atan2(yp, xp) 
            + np.pi/2
            - tf.atan2(self.pmt_pos[:, 1], self.pmt_pos[:, 0]))

        # To supporting wire position
        wr1 = tf.math.abs(tf.math.sqrt(3.0) * xx_abs - yy_abs - 26.30) / tf.math.sqrt(1 + (tf.math.sqrt(3.0))**2) / max_r
        wr2 = tf.math.abs(tf.math.sqrt(3.0) * xx_abs - yy_abs + 26.30) / tf.math.sqrt(1 + (tf.math.sqrt(3.0))**2) / max_r
        
        # rotated to horizontal plane
        angle = -math.pi/3
        xx_rotated = (xx_abs*tf.math.cos(angle) - yy_abs*tf.math.sin(angle))
        yy_rotated = (yy_abs*tf.math.sin(angle) + yy_abs*tf.math.cos(angle))
        to_anode = tf.math.minimum(xx_rotated - (xx_rotated//0.5)*0.5, 0.5 - (xx_rotated - (xx_rotated//0.5)*0.5) ) / max_r
        
        xx_rotated = xx_rotated / max_r
        yy_rotated = yy_rotated/ max_r
        return tf.stack([
                        xx_rotated, # Rot X and Y 
                        yy_rotated,
                        to_anode, # Minimum Anode wire Distance
                        wr1, # Distance to supporting wires
                        wr2
                        ], axis=2)
        ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "pmt_pos_top": self.pmt_pos_top.tolist() if hasattr(self.pmt_pos_top, "tolist") else self.pmt_pos_top,
            "not_dead_pmts": self.not_dead_pmts.tolist() if hasattr(self.pmt_pos_top, "tolist") else self.not_dead_pmts,
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Currently unsuccessfull
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
    """Actually used version, above functions now as equation dump
    
    Verifying Anode shadow is a pain in the ass when plotting
    Getting matplotlib to show it is impossible best is to do whats done here with the numebr of 2d bins as the pixel size and then just check in feh 
    https://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
    """
    def __init__(self, pmt_pos,pmt_r,anode_no_angle, **kwargs):
        super().__init__(**kwargs)
        self.angle = tf.constant( -np.pi / 3, dtype=tf.float32, name="angle_AnodeParametrization")
        self.rot_matrix = tf.constant(tf.transpose(tf.stack([[tf.cos(self.angle ), -tf.sin(self.angle )],[tf.sin(self.angle ),  tf.cos(self.angle )]])), dtype=tf.float32, name="rot_matrix_AnodeParametrization")
        self.pmt_pos = tf.constant(pmt_pos, dtype='float32', name="pmt_pos_anode_parametrization") # (n_pmts, 2)
        # (n_pmts, 2)
        self.pmt_rot = tf.constant(tf.linalg.matmul(self.pmt_pos, self.rot_matrix), dtype='float32', name="pmt_rot_anode_parametrization") 
        self.anode_no_angle = anode_no_angle


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
        return tf.expand_dims(diff, axis=-1) # Output range is 0 to 2 (Need to reverify this )
    
    def angle_to_mesh(self, events):
        """
        Computes the angle between mesh and event
        """
        # events: (batch, 2)
        event_dir = tf.math.atan2(events[:, 1], events[:, 0])  # angle of (x, y)
        event_mesh_angle = event_dir  # relative to mesh
        event_mesh_angle = tf.math.mod(event_mesh_angle, np.pi)  # bring into [0, pi]
        event_mesh_angle = tf.where(event_mesh_angle > np.pi / 2, np.pi - event_mesh_angle, event_mesh_angle)  # fold to [0, pi/2]
        event_mesh_angle = tf.reshape(event_mesh_angle, (-1,1,1))  # (batch, 1, 1)

        # Now we find the angle relative to the mesh
        relative_angle = tf.reshape(self.pmt_mesh_angle - self.angle, (1,-1,1)) - event_mesh_angle # (batch, n_pmts, 1)
        relative_angle = tf.math.mod(relative_angle, np.pi)  # bring into [0, pi]
        relative_angle = tf.where(relative_angle > np.pi / 2, np.pi - relative_angle, relative_angle)  # fold to [0, pi/2]
        
        # normalize to [0,1]
        return relative_angle / (np.pi/2) # (batch, n_pmts, 1)

    def call(self, x, training):
        if self.anode_no_angle:
            return self.call_one(x, training=training)
        else:
            return self.call_both(x, training=training)

    def call_both(self, x, training):
        n_wires = self.n_wires_in_way(x)  # (batch, n_pmts, 1)
        angle = self.angle_to_mesh(x)  # (batch, n_pmts, 1)
        return tf.concat([n_wires, angle], axis=-1)  # (batch, n_pmts, 2)
    
    def call_one(self, x, training):
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

class BiasedDirectIn(tf.keras.layers.Layer):
    """Based on the anode shadow parametrization

    Returns rotated coordinates in range -1, 1
    And PMT event distance in rotated coordinates same range

    Meant to learn the shadowing bias -> No idea how to parametrize uniquely
    """
    def __init__(self, pmt_pos,pmt_r, **kwargs):
        """
        New Direct Detection layer that allows for shifted anode mesh contribution and x,y dependent detection
        
        """
        super().__init__(**kwargs)
        self.angle = tf.constant( -np.pi / 3, dtype=tf.float32, name="angle_AnodeParametrization")
        self.rot_matrix = tf.constant(tf.transpose(tf.stack([[tf.cos(self.angle ), -tf.sin(self.angle )],[tf.sin(self.angle ),  tf.cos(self.angle )]])), dtype=tf.float32, name="rot_matrix_AnodeParametrization")
        self.pmt_pos = tf.constant(pmt_pos, dtype='float32', name="pmt_pos_anode_parametrization") # (n_pmts, 2)
        # (n_pmts, 2)
        self.pmt_rot = tf.constant(tf.linalg.matmul(self.pmt_pos, self.rot_matrix), dtype='float32', name="pmt_rot_anode_parametrization") 

        self.pmt_r = tf.constant(pmt_r, dtype=tf.float32, name="pmt_r_AnodeParametrization")
        """
        Do PMT groups based on how they align in x (along mesh)

        THen they share the same offset for this coordinate only 
        """
        factor = 1e4
        unique_x, self.group_idx = tf.unique(tf.round(self.pmt_rot[:, 0] * factor) / factor)

        n_groups = tf.shape(unique_x)[0]
        self.group_offsets = self.add_weight(
            shape=(n_groups, 1),
            initializer='zeros',
            constraint=tf.keras.constraints.MaxNorm(max_value=self.pmt_r, axis=-1),
            trainable=True,
            name='group_offsets'
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training):
        """
        Compute rotated coordinates -> Get biasing relative to perpendicular wires 
        Compute PMT event dist in rot coord -> Get shadowing effect 

        Stack and normalize to -1 to 1
        """
        x_rot = tf.linalg.matmul(x, self.rot_matrix) # Shape (batch, 2)
        # Build full pmt_offset (n_pmts, 2) dynamically
        x_offsets = tf.gather(self.group_offsets[:, 0], self.group_idx)
        pmt_offset_full = tf.stack([x_offsets, tf.zeros_like(x_offsets)], axis=-1) 

        pmt_event_dist = x_rot[:, tf.newaxis, :] - (self.pmt_rot - pmt_offset_full)[tf.newaxis, :, :] # Shape (batch, n_pmts, 2)

        return pmt_event_dist**2
    
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


class Anode_ShadowingBias(tf.keras.layers.Layer):
    """Based on the anode shadow parametrization

    Returns rotated coordinates in range -1, 1
    And PMT event distance in rotated coordinates same range

    Meant to learn the shadowing bias -> No idea how to parametrize uniquely
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

    def call(self, x, training):
        """
        Compute rotated coordinates -> Get biasing relative to perpendicular wires 
        Compute PMT event dist in rot coord -> Get shadowing effect 

        Stack and normalize to -1 to 1
        """
        x_rot = tf.linalg.matmul(x, self.rot_matrix) # Shape (batch, 2)
        pmt_event_dist = x_rot[:, tf.newaxis, :] - self.pmt_rot[tf.newaxis, :, :] # Shape (batch, n_pmts, 2)

        # Stack along last dimension (batch,n_pmts, 4)
        vals = tf.concat([pmt_event_dist, tf.tile(x_rot[:, tf.newaxis, :], [1, tf.shape(self.pmt_rot)[0], 1])], axis=-1) # Shape (batch, n_pmts, 4)
        return vals / 66.4
    
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



# -------------------------------  Normalization + Combination --------------------------------- #
class NormalizationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, X):
        # Model likes to figure out that if it goes negative most loss funcs break
        # Add minimal number of last dimension if any is < 0
        #cond = tf.reduce_any(X < 0, axis=-1, keepdims=True)            # shape (...,1)
        #row_min = tf.reduce_min(X, axis=-1, keepdims=True)             # shape (...,1)
        """X = tf.where(cond,
                    X+ row_min,
                    X
                    )"""
        X /= (keras.backend.reshape(tf.maximum(keras.backend.sum(X, axis=1), 1e-7), (-1, 1)))
        return X
    def compute_output_shape(self, input_shape):
        return input_shape
    ### Serialization
    def get_config(self):
        config = super().get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class AdditionLayer(keras.layers.Layer):
    def __init__(self, output_shape, initializer= 'zeros', **kwargs):
        """
        To make deactivation of training easier split each component with individual addition layers
        
        """
        super().__init__(**kwargs)
        self.addition_weight = self.add_weight( 
                    shape=[1],
                    initializer = initializer,
                    trainable=True,
                    name="AdditionWeight",
                    constraint =MinMaxConstraint(0., 1.)
                )
        self._output_shape = output_shape
        return
    
    def build(self, input_shape):
        super().build(input_shape)
        return
    
    @tf.function
    def call(self, layers):
        """
        Base laer + some contribution * scaling parameter
        """
        return layers[0] + layers[1]*self.addition_weight[0]


class MultiplicationLayer(keras.layers.Layer):
    def __init__(self, output_shape, include_wall, include_perp, include_anode, **kwargs):
        super().__init__(**kwargs)
        # Wall Reflection contribution Weight - Always generate this as it makes passing weights logic easier
        if include_wall:
            self.wall_weight = self.add_weight( 
                        shape=[1],
                        initializer = 'ones',
                        trainable=True
                    )
        if include_perp:
            self.perp_weight = self.add_weight( 
                        shape=[1],
                        initializer = 'ones',
                        trainable=True
                    )
        if include_anode:
            self.anode_weight = self.add_weight( 
                        shape=[1],
                        initializer = 'ones',
                        trainable=True
            )
        self.gen_call_method(include_wall, include_perp, include_anode)
        self._output_shape = output_shape
        return
    
    def build(self, input_shape):
        super().build(input_shape)
        return
    
    def gen_call_method(self, include_wall, include_perp, include_anode):
        """
        Generate the call method based on which layers are included
        """
        if not include_wall:
            self._call = self.do_nothing
        if include_wall:
            self._call = self.with_wall
        if include_wall and include_perp:
            self._call = self.with_perp
        if include_wall and include_perp and include_anode:
            self._call = self.with_anode
        return

    def call(self, layers):
        """
        I forget why but I coudlnt just directly assign call to whatever method
        """
        return self._call(layers)

    def do_nothing(self, layers):
        """
        Only radial Component -> Return directly
        """
        return layers[0]

    def with_wall(self, layers):
        """
        Radial + Wall Reflection
        """
        return layers[0] + layers[1]*self.wall_weight[0]
    
    def with_perp(self, layers):
        """
        (Radial + Wall) * Perpendicular Wire Shadow
        """
        #return (layers[0] + layers[1]*self.weight[0]) * layers[2]
        return (layers[0] + layers[1]*self.wall_weight[0]) + layers[2]*self.perp_weight[0]

    def with_anode(self, layers):
        """
        (Radial + Wall) + Perpendicular Wire Shadow * Anode Wire Shadow
        """
        #return (layers[0] + layers[1]*self.weight[0]) * layers[2] * layers[3]
        return (layers[0] + layers[1]*self.wall_weight[0]) + layers[2]*self.perp_weight[0] + layers[3]*self.anode_weight[0]

    def compute_output_shape(self, input_shape):
        """Required, but I forget why"""
        return self._output_shape

        ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_shape": self._output_shape,
            "include_wall": (self._call == self.with_wall or self._call == self.with_perp or self._call == self.with_anode),
            "include_perp": (self._call == self.with_perp or self._call == self.with_anode),
            "include_anode": (self._call == self.with_anode)
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class FlattenArray(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return
    
    def build(self, input_shape):
        super().build(input_shape)
        return
    
    def call(self, X):
        return tf.reshape(X, (-1,1))

    ### Serialization
    def get_config(self):
        config = super().get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ReshapeArray(keras.layers.Layer):
    def __init__(self, n_alive_pmts, **kwargs):
        super().__init__(**kwargs)
        self.n_alive_pmts = n_alive_pmts
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, X):
        return tf.reshape(X, (-1, self.n_alive_pmts))
    
    ### Serialization
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_alive_pmts": self.n_alive_pmts
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class ReshapeWallDist(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, X):
        pmt_dist = X[:, :-1] # n,m 
        event_wall_dist =  X[:, -1] #n, 1 
        event_wall_dist =tf.tile(tf.expand_dims(event_wall_dist, axis=-1), [1, tf.shape(pmt_dist)[1]]) #n,m 
        combined = tf.concat([tf.expand_dims(pmt_dist, axis=-1), tf.expand_dims(event_wall_dist, axis=-1)], axis=-1) #n,m,2
        return tf.reshape(combined, (-1,2))
    
    ### Serialization
    def get_config(self):
        config = super().get_config()
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

class MinMaxConstraint(tf.keras.constraints.Constraint):
    """constrain model weights between x_min and x_max"""
    def __init__(self, x_min=0.0, x_max=1.0, name = None):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        if (x_min >= x_max):
            raise ValueError("x_min must be less than x_max")
        if name is not None:
            self.name = name
    
    def __call__(self, w):
        w_min = tf.minimum(tf.math.reduce_min(w), self.x_min)
        w_max = tf.maximum(tf.math.reduce_max(w), self.x_max)
        scale = (self.x_max - self.x_min) / (w_max - w_min)
        m = self.x_min - w_min * scale
        w = w * scale
        return w + m




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


#############################       Unused Code     #############################


class LikelihoodRatioLUT(keras.layers.Layer):
    """
    Fully precompute the entire likelihood for all integer mu in [0..40], <- Each mu required in each computation
    a subdivided x in [0..40], and sigma in [0.05..1]. <- Look Up Table in x mu and sigma 
    We then do a simple nearest-index lookup in call().

    Around 2 orders of magnitude faster than recomputing every time
    """
    def __init__(self, 
                 n_pmts, 
                 m=5,      # Subdivisions per integer in x
                 z=20,     # Steps for sigma
                 switching_signal = 40, 
                 **kwargs):
        super().__init__(**kwargs)

        self.train_variances = tf.Variable(False, trainable=False, dtype=tf.bool, name = "trainVariance") # In eval can return likelihood or likelihood ratio
        self.nan_safe = tf.Variable(True, trainable=False, dtype=tf.bool, name = "NanSafe") # Replaces Nans for training purposes

        ############################################################
        # 1) Trainable parameter for the "log of the base std dev"
        #    (one per PMT).
        ############################################################
        self.std = self.add_weight(
            shape=[n_pmts],
            initializer=tf.constant_initializer(0.5),
            trainable=True,
            name="GaussianStandardDeviation",
            constraint=MinMaxConstraint(0.05, 1.0),
        )

        ############################################################
        # 2) Constants
        ############################################################
        self.p_dpe   = tf.constant(0.2, dtype=tf.float32, name="p_dpe_LossLayer")
        if switching_signal > 40.: raise Exception("Not implemented for PE > 40, manually modify the class definition for larger range")
        self.switching_signal = switching_signal
        # Domain for Poisson (n_ph) and Binomial+Gaussian (n_pe):
        # shapes used later for the sum
        # Make sure these are 1D Tensors of the correct shape:
        self.n_ph_domain = tf.range(0., 58., dtype=tf.float32)   # shape [57]
        n_ph = self.n_ph_domain.shape[0]
        self.n_pe_domain = tf.range(1., 67., dtype=tf.float32)   # shape [66]
        n_pe = self.n_pe_domain.shape[0]

        ############################################################
        # 3) Precompute b: the Binomial factor across (n_ph, n_pe)
        #
        # b.shape => (n_ph, n_pe)
        # We'll reshape in the big broadcast later.
        # Implementation returns NaN for invalid input (k < 0)
        # But I dont want this becuase tf.debugging.enable_check_numerics will flag it so i need to get rid of invalid before rather than after
        ############################################################
        k = self.n_pe_domain[tf.newaxis, :] - self.n_ph_domain[..., tf.newaxis]  

        # Create a mask for valid k:
        # Valid when k >= 0 and k <= n_ph (i.e. the total_count for that row).
        # We broadcast self.n_ph_domain as a column vector.
        total_counts_matrix = tf.reshape(self.n_ph_domain, [n_ph, 1])  # shape [n_ph, 1]
        valid_mask = tf.logical_and(k >= 0, k <= total_counts_matrix)

        # Get the indices for valid entries.
        valid_indices = tf.where(valid_mask)    # shape [num_valid, 2]

        # Gather the corresponding valid k values.
        valid_k = tf.gather_nd(k, valid_indices)

        # Also, for each valid index, get the corresponding total count.
        # The row index (first element of valid_indices) corresponds to the proper n_ph.
        total_counts = tf.gather(self.n_ph_domain, valid_indices[:, 0])

        # Now create a Binomial distribution for each valid (n_ph, k) pair.
        # Note: Both total_counts and valid_k are integers, as required.
        p_valid = tfp.distributions.Binomial(total_count=total_counts, probs=self.p_dpe).prob(valid_k)

        # Create a full tensor (shape [n_ph, n_pe]) filled with zeros (float32).
        b_full = tf.zeros([n_ph, n_pe], dtype=tf.float32)
        # Scatter the computed probabilities back into b_full.
        self.b = tf.tensor_scatter_nd_update(b_full, valid_indices, p_valid)

        ############################################################
        # 4) Precompute c: the Poisson factor for mu=0..40
        #
        # c.shape => (41, n_ph)
        # Implementation not defined for expectation 0
        ############################################################
        c_list = []
        for mu_val in range(41):
            if mu_val == 0:
                # For mu==0, define: P(0)=1, and P(k)=0 for k>0.
                c_val = tf.where(tf.equal(self.n_ph_domain, 0.0),
                                tf.ones_like(self.n_ph_domain),
                                tf.zeros_like(self.n_ph_domain))
            else:
                c_val = tfp.distributions.Poisson(rate=float(mu_val)).prob(self.n_ph_domain)
            c_list.append(c_val)
        self.c = tf.stack(c_list, axis=0)

        ############################################################
        # 5) Build the subdivided x_domain.  
        # If m=3, then for each integer i in [0..39], we produce i + [0,1/3,2/3],
        # plus the integer 40 as the final endpoint => total 40*m + 1 points.
        ############################################################
        x_vals = []
        for i in range(40):
            for sub in range(m):
                x_vals.append(i + sub / float(m))
        x_vals.append(40.0)  # final endpoint
        self.x_domain = tf.constant(x_vals, dtype=tf.float32)
        # shape = [nx], where nx = 40*m + 1

        ############################################################
        # 6) Build the sigma domain from 0.05..1 in z steps
        ############################################################
        self.sigma_domain = tf.linspace(0.05, 1.0, z)  # shape [z]

        ############################################################
        # 7) Build the 3D LUT: L_table[mu_idx, x_idx, sigma_idx].
        # We'll do a big broadcast multiply: a(x,sigma) * b(n_ph,n_pe) * c(mu,n_ph)
        # Then sum over n_ph,n_pe.
        #
        # Final shape = [41, nx, z].
        ############################################################

        # 7a) Reshape for broadcast:
        # c   => shape [41,   57] => turn into [41, 1, 1, 57, 1]
        c_broadcast = tf.reshape(self.c, [41, 1, 1, n_ph, 1])

        # b   => shape [57, 66]  => turn into [1, 1, 1, 57, 66]
        b_broadcast = tf.reshape(self.b, [1, 1, 1, n_ph, n_pe])

        # x_domain => shape [nx] => turn into [1, nx, 1, 1, 1]
        nx = tf.size(self.x_domain)
        x_broadcast = tf.reshape(self.x_domain, [1, nx, 1, 1, 1])

        # sigma_domain => shape [z] => turn into [1, 1, z, 1, 1]
        z_ = tf.size(self.sigma_domain)
        sigma_broadcast = tf.reshape(self.sigma_domain, [1, 1, z_, 1, 1])

        # n_pe_domain => shape [66] => [1, 1, 1, 1, 66]
        n_pe_broadcast = tf.reshape(self.n_pe_domain, [1, 1, 1, 1, n_pe])

        # n_ph_domain => shape [57] => [1, 1, 1, 57, 1]
        # (We already used this in c_broadcast).
        # We'll just reuse c_broadcast's shape for the n_ph dimension.

        # 7b) Build the Gaussian factor:
        # mean = n_pe_domain
        # std  = sigma * sqrt(n_pe_domain)
        # shape after broadcast => [1, nx, z, 1, 66]
        mean = n_pe_broadcast
        std  = sigma_broadcast * tf.sqrt(n_pe_broadcast + 1e-12)

        normal = tfp.distributions.Normal(loc=mean, scale=std + 1e-12)
        # a(x, sigma) => normal.prob(x)
        # shape => [1, nx, z, 1, 66]
        a_broadcast = normal.prob(x_broadcast)

        # 7c) Multiply: [41, nx, z, 57, 66] after broadcast
        # We'll do step by step.
        # shape( a ) = [1,   nx, z, 1,   66]
        # shape( b ) = [1,   1,   1, 57, 66]
        # shape( c ) = [41, 1,   1, 57, 1 ]
        # result => [41, nx, z, 57, 66]
        tmp = a_broadcast * b_broadcast  # => [1, nx, z, 57, 66]
        tmp = tmp * c_broadcast          # => [41, nx, z, 57, 66]

        # 7d) Sum over [57, 66] => axis -2, -1 => shape [41, nx, z]
        L_full = tf.reduce_sum(tmp, axis=[3,4])

        # 7e) Store as self.L_table: [41, nx, z]
        self.L_table = L_full  # big 3D table

    def build(self, input_shape):
        super().build(input_shape)


    ############################################################
    # Nearest-grid lookup
    ############################################################
    @tf.custom_gradient
    def differentiable_sigma_idx(self,std):
        """Differentiable std lookup for LUT"""
        sigma_idx = self._nearest_idx_1d(self.sigma_domain, std)
        def grad(dy):
            return dy 
        return sigma_idx, grad

    def lookup_likelihood(self, x, mu, std):
        """
        For each (x_i, mu_i, std_i), find the nearest index in x_domain, mu in [0..40],
        and sigma_domain, then return L_table[mu_idx, x_idx, sigma_idx].
        """
        # 1) mu_idx
        # Clip mu to [0,40], cast to int
        mu_clipped = tf.clip_by_value(mu, 0.0, 40.0)
        mu_idx = tf.cast(tf.round(mu_clipped), tf.int32)  # shape = ?

        # 2) x_idx
        # We assume x in [0..40]. We find the nearest index in self.x_domain.
        # Easiest way: scale x by 'm', or do a full search. We'll do a piecewise approach:
        #    x_idx ~ round( (x * (m)) ), but that only works if we constructed x_domain linearly.
        # But let's do a quick approximate method: we know x_domain is sorted,
        # we can do "tf.searchsorted(self.x_domain, x, side='left')" and clamp to edges.
        # Then decide if we need to step back by 1. 
        # For simplicity, let's do a function nearest_idx_1d(...).
        x_idx = self._nearest_idx_1d(self.x_domain, x)

        # 3) sigma_idx
        sigma_idx = self.differentiable_sigma_idx(std)

        # 4) Gather from L_table. We want shape (batch_size,).
        # We'll use tf.gather_nd. We need to build indices of shape [batch_size, 3].
        # Each row is [mu_idx[i], x_idx[i], sigma_idx[i]].
        coords = tf.stack([mu_idx, x_idx, sigma_idx], axis=-1)  # shape (batch_size, 3)
        vals = tf.gather_nd(self.L_table, coords)               # shape (batch_size,)
        return vals

    def _nearest_idx_1d(self, grid, values):
        """
        Return nearest indices in 'grid' for each 'values'.
        grid: shape [N], sorted
        values: shape [batch_size], or any shape
        Output: same shape as 'values', containing integer indices in [0, N-1].
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
        
    ############################################################
    # -2 log-likelihood ratio
    ############################################################
    def true_likelihoodRatio(self, x, mu, std):
        """
        -2 * [ log L(x| mu, std) - log L(mu | mu, std) ]
        with a small floor to keep logs stable.

        Lower bound for MLE given by x/1.2
        """
        like_x = tf.maximum(self.lookup_likelihood(x, mu, std),    1e-9)
        like_mu= tf.maximum(self.lookup_likelihood(x, x/1.2, std), 1e-9)

        return -2.0 * (tf.math.log(like_x) - tf.math.log(like_mu))
    
    def compute_common_std(self, mu,std):
        """
        Compute Gaussian Limit equivalent standard deviation
        mu is (n_pmts) 
        """
        npe_mean = mu * (1 + self.p_dpe)
        computed_std= tf.sqrt(tf.abs(
                mu * (1 + self.p_dpe)**2 +          #Poisson 
                mu * self.p_dpe * (1-self.p_dpe) +  # Binom 
                (tf.abs(npe_mean)* (std+1e-12)**2 ) # Gaussian
                )) 
        # Floor the standard deviation to avoid division by zero.
        return tf.maximum(computed_std, 1e-6)
    

    @tf.custom_gradient
    def differentiable_lookup_std(self, std):
        """
        Define gradients relative to function input, lookup is not differnetiable 
        Used only in Gaussian Approx
        """
        def grad(dy):
            return dy  
        return tf.gather(self.sigma_domain, self._nearest_idx_1d(self.sigma_domain, std)), grad

    def gaussian_approx(self, x, mu,std):
        """Gaussian Approximation Likelihood"""
        npe_mean = mu * (1 + self.p_dpe)
        return tfp.distributions.Normal(
            loc=npe_mean,
            scale=self.compute_common_std(mu, # Compute closest std so there is some sense of consistency between the two 
                                          self.differentiable_lookup_std(std)
                                          )
        ).log_prob(x)
    
    def approximate_likelihoodRatio(self, x, mu, std):
        """Evaluates In gaussian Limit

        MLE is x | x in gaussian limit
        """
        return -2 * (self.gaussian_approx(x, mu, std)
                   - self.gaussian_approx(x, x, std))


    @tf.function
    def _call(self, pred, observed):
        """
        pred: shape [batch, n_pmts]   (floating in [0..some range])
        observed: shape [batch, n_pmts]
        We do scaling, reshape, then do the ratio across all PMTs, and reduce_mean.
        """

        # 1) Scale the predicted fraction by sum of observed (like in your code).
        scaling = tf.reduce_sum(tf.where(observed > 0, observed, 0), axis=-1, keepdims=True)
        pred = pred * scaling

        # 2) Flatten out - much easier to work with 
        batch_size = tf.shape(pred)[0]
        mu   = tf.reshape(pred,    [-1])  # shape [batch_size * n_pmts]
        x    = tf.reshape(observed,[-1])
        std = tf.reshape(self.std, [1, -1])             # shape [1, n_pmts]
        std = tf.tile(std, [batch_size, 1])             # shape [batch_size, n_pmts]
        std = tf.reshape(std, [-1])                     # shape [batch_size * n_pmts]

        # Check where to do lookup and where Gaussian 
        bmap = tf.logical_or(x > self.switching_signal, mu > self.switching_signal)
        if self.nan_safe:
            nan_mask =  tf.logical_or(tf.math.is_nan(x), tf.math.is_nan(mu))
            true_indices =tf.where(bmap & ~nan_mask)
        else:
            true_indices =tf.where(bmap)

        # 3) Evaluate likelihood ratio (per PMT)
        #   shape => [batch_size * n_pmts]
        
        if not self.train_variances:
            ratio = tf.zeros_like(x, dtype='float32')
            ratio = tf.tensor_scatter_nd_update(ratio,true_indices, self.approximate_likelihoodRatio(
                                                                        tf.gather_nd(x, true_indices), # Obs is x model predicts mu 
                                                                        tf.gather_nd(mu, true_indices), 
                                                                        tf.gather_nd(std, true_indices), 
                                                                        ))
            if self.nan_safe:
                false_indices =tf.where((~bmap & ~nan_mask))
            else:
                false_indices =tf.where((~bmap))

            ratio = tf.tensor_scatter_nd_update(ratio, false_indices, self.true_likelihoodRatio(
                                                                        tf.gather_nd(x, false_indices), 
                                                                        tf.gather_nd(mu, false_indices), 
                                                                        tf.gather_nd(std, false_indices), 
                                                                        ))
            # Fill all NaN they occure in pos ref assuming the model inputs mu as nan 
            if self.nan_safe:
                nan_indices = tf.where(nan_mask)
                updates = tf.fill(tf.shape(tf.gather_nd(x, nan_indices)), 100.0)
                ratio = tf.tensor_scatter_nd_update(ratio, nan_indices, updates)
            # 4) Reshape and Average over PMTs => shape [batch_size]
            ratio_2d = tf.reshape(ratio, (batch_size, -1))
            loss = tf.reduce_mean(ratio_2d, axis=-1)

        else:
            """Here we compute the negative log likelihood such that the variances become part of the loss function
            And we attempt to only compute this for the gaussian case 

            TODO Dont think this will work, change it to use the normal form
            """
            gaussian_losses = -self.gaussian_approx(
                tf.gather_nd(x, true_indices),
                tf.gather_nd(mu, true_indices),
                tf.gather_nd(std, true_indices)
            )
            # Directly take the average over all those elements.
            loss = tf.reduce_mean(gaussian_losses)
        return loss
    
    @tf.function
    def call(self, pred, observed):
        """
        pred: shape [batch, n_pmts]   (floating in [0..some range])
        observed: shape [batch, n_pmts]
        Performs scaling, reshaping, then computes the ratio across all PMTs and reduces by mean.
        """
        # 1) Scale the predicted fraction by the sum of observed values.
        scaling = tf.reduce_sum(tf.where(observed > 0, observed, 0), axis=-1, keepdims=True)
        pred = pred * scaling

        # 2) Flatten out. This makes subsequent operations easier.
        batch_size = tf.shape(pred)[0]
        mu = tf.reshape(pred, [-1])           # shape: [batch_size * n_pmts]
        x = tf.reshape(observed, [-1])
        std = tf.reshape(self.std, [1, -1])     # shape: [1, n_pmts]
        std = tf.tile(std, [batch_size, 1])     # shape: [batch_size, n_pmts]
        std = tf.reshape(std, [-1])             # shape: [batch_size * n_pmts]

        # 3) Determine where to do the lookup.
        bmap = tf.logical_or(x > self.switching_signal, mu > self.switching_signal)

        # Convert flags (these are tf.Variables now) to tensors for dynamic control flow.
        train_variances_flag = self.train_variances  # assumed to be a tf.Variable(dtype=tf.bool)
        nan_safe_flag = self.nan_safe                # assumed to be a tf.Variable(dtype=tf.bool)

        # Helper: compute true_indices (and nan_mask if needed) dynamically.
        def compute_indices():
            nan_mask_local = tf.logical_or(tf.math.is_nan(x), tf.math.is_nan(mu))
            true_indices_local = tf.cast(tf.where(bmap & ~nan_mask_local), tf.int32)
            return true_indices_local, nan_mask_local

        def compute_indices_no_nan():
            true_indices_local = tf.cast(tf.where(bmap), tf.int32)
            # Create a dummy nan_mask of the same shape as x (all False).
            return true_indices_local, tf.zeros_like(x, dtype=tf.bool)

        true_indices, nan_mask = tf.cond(nan_safe_flag, compute_indices, compute_indices_no_nan)

        # Helper to compute loss in the non-training branch.
        def compute_non_train_loss():
            # Start with an empty ratio (one value per PMT).
            ratio = tf.zeros_like(x, dtype=tf.float32)
            # Update for "true" indices (where lookup is used) with approximate likelihood.
            approx = self.approximate_likelihoodRatio(
                tf.gather_nd(x, true_indices),  # observed values
                tf.gather_nd(mu, true_indices), # predicted values
                tf.gather_nd(std, true_indices) # trainable std values
            )
            ratio_updated = tf.tensor_scatter_nd_update(ratio, true_indices, approx)

            # Determine the "false" indices.
            def compute_false_true():
                false_idx = tf.cast(tf.where((~bmap) & ~nan_mask), tf.int32)
                return false_idx

            def compute_false_false():
                false_idx = tf.cast(tf.where(~bmap), tf.int32)
                return false_idx

            false_indices = tf.cond(nan_safe_flag, compute_false_true, compute_false_false)

            ratio_updated2 = tf.tensor_scatter_nd_update(
                ratio_updated,
                false_indices,
                self.true_likelihoodRatio(
                    tf.gather_nd(x, false_indices),
                    tf.gather_nd(mu, false_indices),
                    tf.gather_nd(std, false_indices)
                )
            )

            # If nan_safe, fill indices corresponding to NaN with a fixed value.
            def fill_nan():
                nan_indices = tf.cast(tf.where(nan_mask), tf.int32)
                updates = tf.fill(tf.shape(tf.gather_nd(x, nan_indices)), 100.0)
                return tf.tensor_scatter_nd_update(ratio_updated2, nan_indices, updates)
            def no_fill_nan():
                return ratio_updated2

            ratio_final = tf.cond(nan_safe_flag, fill_nan, no_fill_nan)

            # Reshape and average over the PMTs.
            ratio_2d = tf.reshape(ratio_final, (batch_size, -1))
            return tf.reduce_mean(ratio_2d, axis=-1)

        # Helper to compute loss in the training branch (where variances are trained).
        def compute_train_loss():
            # Create a zero tensor with the same shape as x.
            loss_tensor = tf.zeros_like(x, dtype=tf.float32)
            # Compute Gaussian losses for the true indices.
            gaussian_losses = -self.gaussian_approx(
                tf.gather_nd(x, true_indices),
                tf.gather_nd(mu, true_indices),
                tf.gather_nd(std, true_indices)
            )
            # Scatter the computed losses into the loss tensor.
            loss_tensor = tf.tensor_scatter_nd_update(loss_tensor, true_indices, gaussian_losses)

            # Create a mask marking which indices were updated with a valid loss.
            mask_tensor = tf.zeros_like(x, dtype=tf.float32)
            ones = tf.ones_like(gaussian_losses, dtype=tf.float32)
            mask_tensor = tf.tensor_scatter_nd_update(mask_tensor, true_indices, ones)

            # Reshape both tensors to [batch_size, -1] so that each row corresponds to one sample.
            loss_tensor_2d = tf.reshape(loss_tensor, (batch_size, -1))
            mask_tensor_2d = tf.reshape(mask_tensor, (batch_size, -1))

            # Compute the sum of losses per sample.
            loss_sum = tf.reduce_sum(loss_tensor_2d, axis=-1)
            # Count the valid (non-null) entries per sample.
            valid_count = tf.reduce_sum(mask_tensor_2d, axis=-1)

            # Compute the mean loss per sample, ignoring null entries.
            # tf.math.divide_no_nan returns 0 when valid_count is 0.
            sample_loss_mean = tf.math.divide_no_nan(loss_sum, valid_count)
            return sample_loss_mean

        # Select the branch dynamically based on the train_variances_flag.
        loss = tf.cond(train_variances_flag, compute_train_loss, compute_non_train_loss)
        return loss

    ### Serialization
    def get_config(self):
        config = super().get_config()
        n_pmts = int(self.std.shape[0])
        x_domain_size = int(self.x_domain.shape[0])
        m = (x_domain_size - 1) // 4
        z = int(self.sigma_domain.shape[0])
        config.update({
            "n_pmts": n_pmts,
            "m": m,
            "z": z,
            "switching_signal": self.switching_signal
        })
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LossLayer(keras.layers.Layer):
    """
    Computes Likelihood space to max_sigma
    switches to gaussian approximation when signal is > self.switching_signal
    This is since it becomes increasingly costly to evaluate exactly
    
    """
    def __init__ (self, n_pmts, switching_signal,**kwargs):
        super().__init__(**kwargs)
        self.log_standard_deviations = self.add_weight( 
                        shape=[n_pmts],
                        initializer = tf.constant_initializer(0.5),
                        trainable=True,
                        name="GaussianStandardDeviation"
                      )
        self.p_dpe = 0.2 
        self.p_dpe = tf.constant(0.2, dtype=tf.float32, name="p_dpe_LossLayer")
        self.min_pe = tf.constant(0, dtype=tf.float32, name="min_pe_LossLayer")
        self.max_sigma = tf.constant(5, dtype=tf.float32, name="maxSigma_LossLayer")
        self.switching_signal = tf.constant(switching_signal, dtype=tf.float32, name="switching_signal_LossLayer")
        
        # Exact evaluation domain
        self.n_pe_domain = tf.range(1., 25.)[tf.newaxis,tf.newaxis,:]
        self.n_ph_domain = tf.range(1., 20.)[tf.newaxis,:,tf.newaxis]
            
    def build(self, input_shape):
        super().build(input_shape)

    def compute_common_std(self, mu,std):
        """
        Compute Gaussian Limit equivalent standard deviation
        mu is (n_pmts) 

        Nan fighting : tf.abs on npe_mean -> Will be high loss anyway
        """
        npe_mean = mu * (1 + self.p_dpe)
        return tf.sqrt(tf.abs(
                mu * (1 + self.p_dpe)**2 +          #Poisson 
                mu * self.p_dpe * (1-self.p_dpe) +  # Binom 
                (tf.abs(npe_mean)* (std+1e-12)**2 ) # Gaussian
                )) #Std made abs earlier + avoid 0 std
    
    def truncated_gaussian(self, x,std):
        std = std * tf.sqrt(self.n_pe_domain)
        mean = self.n_pe_domain
        return tfp.distributions.TruncatedNormal(
                loc=mean, 
                scale=(std+1e-12), # Avoid 0 std and std abs on orig call
                low=self.min_pe, 
                high=float('inf')
            ).prob(x)
    
    def gaussian_approx(self, x, mu,std):
        """Gaussian Approximation Likelihood"""
        npe_mean = mu * (1 + self.p_dpe)
        return tfp.distributions.Normal(
            loc=npe_mean,
            scale=self.compute_common_std(mu,std)
        ).log_prob(x)
    
    def true_likelihood(self, x, mu,std):
        """True Likelihood"""
        x = x[:,tf.newaxis,tf.newaxis]
        mu = mu[:,tf.newaxis,tf.newaxis]
        
        # SPE fluctuation > parse_version('1.9.9'
        # This has to be computed every time
        a = self.truncated_gaussian(x,std)
        
        # DPE emission
        # This can be entirely precomputed 
        b = tfp.distributions.Binomial(
                total_count=self.n_ph_domain, 
                probs=self.p_dpe
            ).prob(self.n_pe_domain - self.n_ph_domain)
        
        # Photon detection
        # I can probably precompute this as well
        # Mu can only be between 1 and 40 before I switch so we have a say 40x40 arrray 
        c = tfp.distributions.Poisson(
                rate=mu).prob(self.n_ph_domain)

        return tf.reduce_sum(a * b * c, axis=[1,2])

    def true_likelihoodRatio(self,x, mu,std):
        """Evaluates the true likelihood ratio"""
        return -2 *(tf.math.log(tf.maximum(self.true_likelihood(x, mu,std),1e-9)) - tf.maximum(tf.math.log(self.true_likelihood(mu, mu,std)),1e-10))

    def approximate_likelihoodRatio(self, x, mu, std):
        """Evaluates In gaussian Limit
        
        x is MLE for Gauss
        """
        return -2 * (self.gaussian_approx(x, mu, std)
                   - self.gaussian_approx(mu, mu, std))
    
    def do_the_thing(self,  pred, obs, std): 
        """Inputs are tensors of shape [1] at least until it works
        
        """
        obs= tf.maximum(obs, 1e-10)
        if (pred[0] > self.switching_signal) or (obs[0] > self.switching_signal):
            result = self.approximate_likelihoodRatio(pred, obs, std)
            if tf.math.is_inf(result):
                return 1e9
            return result
        else:
            result = self.true_likelihoodRatio(pred, obs, std)
            if tf.math.is_inf(result):
                return 1e9
            return result

    @tf.function()
    def call(self, pred, observed):
        """We need to evaluate per PMT per batch element
        We no longer evaluate the likelihood of observing PE but the likelihood of observing Ph
        So we scale according to most likely number of photons as observed from data and scale according to that
        rather than full PE response
        """
        # TODO How to deal with negative predictions in this case - Remove truncnorm
        # But how do I deal with sum scale
        #observed_min = tf.math.reduce_min(observed, axis=-1, keepdims=True) 
        #observed = tf.where(observed_min <0, observed+observed_min, observed)
        # Sum scale
        #scaling = tf.math.reduce_sum(observed, axis=-1, keepdims=True)

        scaling = tf.math.reduce_sum(tf.where(observed > 0, observed, 0), axis=-1, keepdims=True)
        # Eval Per PMT
        pred = pred * scaling
        """
        Computation is inconsistent on matrix sizewe need to compute
        this per element rather than all at once for ram and sanity purposes 
        """
        batch_size = tf.shape(pred)[0]
        one  = tf.reshape(pred, [-1,1])
        two =tf.reshape(observed, [-1,1])
        three = tf.reshape(tf.tile(tf.expand_dims(self.log_standard_deviations, axis=0), [batch_size, 1]), [-1,1])
        
        # Ensure tensors are fully evaluated before passing to map_fn
        # Because otherwise it passses a goddamn boolean for some reason 
        one = tf.ensure_shape(tf.identity(one), [None, 1])
        two = tf.ensure_shape(tf.identity(two), [None, 1])
        three = tf.ensure_shape(tf.identity(three), [None, 1])
        
        # If we do not train the weights we want the first
        #if not self.trainable:
        #loss = tf.map_fn(
        #    lambda x: self.do_the_thing(x[0], x[1], x[2]), 
        #    (one, two, three),
        #    fn_output_signature=tf.float32
        #)
        loss = self.do_the_thing(one, two, three)
        # Mean
        #loss = tf.reduce_mean(,axis=-1)
        mask = observed>20
        mask_f = tf.cast(mask, loss.dtype)
        # Sum only over masked elements
        masked_sum = tf.reduce_sum(tf.reshape(loss, (batch_size, -1)) * mask_f, axis=-1)  # shape [n]
        # Count number of valid elements per row
        valid_count = tf.reduce_sum(mask_f, axis=-1) 
        # Compute mean over valid elements; avoid division by zero
        loss = tf.math.divide_no_nan(masked_sum, valid_count)
        return loss
        """else:
            # For Fitting the Variances
            # Gaussian -2 log likeli is (x-mu)^2/sigma^2 + 2log(\sigma) so since we minimize it is encouraged to minimize sigma 
            mask = tf.greater(two, 20)
            loss = tf.map_fn(
                lambda x: -2*self.gaussian_approx(x[0], x[1], x[2]), 
                # Only pass elements satisfying since we do Gaussian Approx for variance fitting anyway
                (one, two, three),
                #(tf.boolean_mask(one, mask), tf.boolean_mask(two, mask), tf.boolean_mask(three, mask)),
                fn_output_signature=tf.float32
            )
            return tf.reduce_mean(loss)
        """

class LossLayerWDense(keras.layers.Layer):
    """DOES NOT WORK 
    Computes Likelihood space to max_sigma
    switches to gaussian approximation when signal is > self.switching_signal
    
    Same as above but uses a fitted likelihood below the switching signal, in this case 40PE 
    This is since it takes very long to evaluate the full model

    Problem is that I couldnt figure out a network that could reliably predict the likelihood
    """
    def __init__(self, model_path, n_pmts, switching_signal = 40, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.models.load_model(model_path)
        self.dense.trainable = False
        self.switching_signal = tf.constant(switching_signal, dtype=tf.float32, name="switching_signal_LossLayerWDense")
        self.p_dpe =tf.constant(0.2, dtype=tf.float32, name="p_dpe_LossLayerWDense")
        self.n_pmts = n_pmts

    def build(self, input_shape):
        self.stds = self.add_weight("stds",
                            shape=[self.n_pmts],
                            constraint=NonNeg(),
                            initializer=keras.initializers.Constant(value=0.5),)

    def compute_common_std(self, mu,std):
        """
        Compute Gaussian Limit equivalent standard deviation

        Nan fighting : tf.abs on npe_mean -> Will be high loss anyway
        """
        npe_mean = mu * (1 + self.p_dpe)
        return tf.sqrt(tf.abs(
                mu * (1 + self.p_dpe)**2 +          #Poisson 
                mu * self.p_dpe * (1-self.p_dpe) +  # Binom 
                (tf.abs(npe_mean)* (std+1e-12)**2 ) # Gaussian
                )) #Std made abs earlier + avoid 0 std

    def call_dense(self, obs, pred, std):
        xgmu = self.dense(tf.stack([obs, pred],axis=-1))
        mugmuhat = self.dense(tf.stack([pred, pred/1.2],axis=-1))
        return 2. * (xgmu - mugmuhat)

    def gaussian_approx(self, x, mu,std):
        """Gaussian Approximation Likelihood"""
        npe_mean = mu * (1 + self.p_dpe)
        return tf.math.log(tf.maximum(tfp.distributions.Normal(
            loc = npe_mean,
            scale  = tf.maximum(self.compute_common_std(mu, std), 1e-6)
        ).prob(x), 1e-12))

    def approximate_likelihoodRatio(self, x, mu, std):
        """Evaluates In gaussian Limit"""
        return -2. * (self.gaussian_approx(x, mu, std)
                   - self.gaussian_approx(mu, mu, std))

    def call(self, pred, observed, training = None):
        size = tf.shape(pred)
        pred = pred * tf.math.reduce_sum(observed, axis=-1, keepdims=True)
        one  = tf.reshape(pred, [-1,1])
        two = tf.reshape(observed, [-1,1])
        # Eval Per PMT
        bmap = tf.logical_or(one > self.switching_signal, two > self.switching_signal)
        # Do scatternd into final array of same size as either input
        loss = tf.zeros_like(one, dtype='float32')
        #stds = tf.fill(tf.shape(one), 0.5)
        # Tile stds to match size
        stds = tf.tile(tf.expand_dims(tf.minimum(self.stds, 1e-12), axis=0), [size[0], 1])
        true_indices =tf.where(bmap)
        loss = tf.tensor_scatter_nd_update(loss,true_indices, self.approximate_likelihoodRatio(
                                                                    tf.gather_nd(two, true_indices), # Obs is x model predicts mu 
                                                                    tf.gather_nd(one, true_indices), 
                                                                    tf.gather_nd(stds, true_indices), 
                                                                    ))
        false_indices =tf.where(~bmap)
        loss = tf.tensor_scatter_nd_update(loss, false_indices, tf.reshape(self.call_dense(
                                                                    tf.gather_nd(two, false_indices), 
                                                                    tf.gather_nd(one, false_indices), 
                                                                    tf.gather_nd(stds, false_indices), 
                                                                    ), [-1]))
        return tf.reshape(loss, size)
