
"""
Implements Layers that combine models or are their own model 

Classes
- I0Layer - applies the per PMT scaling 
- NormalizationLayer - normalizes the model prediction sum to 1 (negative inputs are not handled well)
- AdditionLayer - Weighted addition of secnd layer with weight constraint 0 to 1 

TODOs:
Import MinMaxConstraint from custom_constraints.py
MultiplicationLayer is that even still used or did i fully switch to iterative


All functions Implemented : False



"""
import tensorflow as tf
from tensorflow import keras
import numpy as nps


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
    
class NormalizationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, X):
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