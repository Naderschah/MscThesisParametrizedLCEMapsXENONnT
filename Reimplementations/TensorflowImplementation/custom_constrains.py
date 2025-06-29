
"""
Here custom constraints are defined


All functions Implemented : False

"""
import tensorflow as tf

class MinMaxConstraint(tf.keras.constraints.Constraint):
    """constrain model weights between x_min and x_max"""
    def __init__(self, x_min=0.0, x_max=1.0, name = None):
        super().__init__()
        self.x_min = tf.constant(x_min, name = "Constraint_Min")
        self.x_max = tf.constant(x_max, name = "Constraint_Max")
        if (x_min >= x_max):
            raise ValueError("x_min must be less than x_max")
        # Constraints dont support naming for some reason
        if name is not None:
            self.name = name
        return
    
    def __call__(self, w):
        w_min = tf.minimum(tf.math.reduce_min(w), self.x_min)
        w_max = tf.maximum(tf.math.reduce_max(w), self.x_max)
        scale = (self.x_max - self.x_min) / (w_max - w_min)
        m = self.x_min - w_min * scale
        w = w * scale
        return w + m