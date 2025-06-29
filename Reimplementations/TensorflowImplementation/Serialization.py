
"""
Here all custom serialization functions are defined.

Functions:
- generate_weight_dict(model): Generates a dictionary of weights with fully qualified names for each layer.
- save_weights_to_dict(model, path): Saves the generated weight dictionary to a JSON file.
- load_weights_from_dict(model, weight_dict): Loads weights from a dictionary or JSON file into the model using fully qualified paths.
- convert_numpy(obj): Converts numpy arrays and types to standard Python types for JSON serialization.

Classes:
- NumpyEncoder: A custom JSON encoder that handles numpy arrays and types, allowing them to be serialized to JSON. (Two seperate funtions as the Encoder did not always work)

Why 
- During training tf serialization was a pain, it would often fail on very basic things that I did not want to deal with while trying things
- It did not allow for iteratively adding contributions to the model, which is a requirement for the training strategy
- Pre and Post compillation some layers serialization behaved very problematically, specifically the Loss function due to the LUT table

The basic idea behind serialization is to simply use the full model path, imagine a tree of models and components
And generate file system links for each variable
So if the main model is called main its submodel called sub1 is adressed as main/sub1 and its attribute gamma1 with main/sub1/gamma1 etc. 
This provides a robust way to change everything about a model, removing layers adding layers etc. as long as within each path there are no duplicate names and names do not change

TODOs:
Make a proper model loader for different call modes since it doesnt quite work due to gaussian std


All functions Implemented : False
"""

import json, os
import numpy as np
import tensorflow as tf

# Grabbed from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    # Json doesnt support numpy arrays it turns out 
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