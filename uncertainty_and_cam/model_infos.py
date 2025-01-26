import tensorflow as tf

model_summary = [
    {"index": 0, "layer_type": "InputLayer", "output_shape": "(None, 64, 64, 3)", "param": 0},
    {"index": 1, "layer_type": "Conv2D", "output_shape": "(None, 64, 64, 16)", "param": 448},
    {"index": 2, "layer_type": "BatchNormalization", "output_shape": "(None, 64, 64, 16)", "param": 64},
    {"index": 3, "layer_type": "Activation", "output_shape": "(None, 64, 64, 16)", "param": 0},
    {"index": 4, "layer_type": "Conv2D", "output_shape": "(None, 64, 64, 128)", "param": 18560},
    {"index": 5, "layer_type": "BatchNormalization", "output_shape": "(None, 64, 64, 128)", "param": 512},
    {"index": 6, "layer_type": "Activation", "output_shape": "(None, 64, 64, 128)", "param": 0},
    {"index": 7, "layer_type": "Conv2D", "output_shape": "(None, 64, 64, 128)", "param": 147584},
    {"index": 8, "layer_type": "Conv2D", "output_shape": "(None, 64, 64, 128)", "param": 2176},
    {"index": 9, "layer_type": "Add", "output_shape": "(None, 64, 64, 128)", "param": 0},
    {"index": 10, "layer_type": "BatchNormalization", "output_shape": "(None, 64, 64, 128)", "param": 512},
    {"index": 11, "layer_type": "Activation", "output_shape": "(None, 64, 64, 128)", "param": 0},
    {"index": 12, "layer_type": "Conv2D", "output_shape": "(None, 64, 64, 128)", "param": 147584},
    {"index": 13, "layer_type": "BatchNormalization", "output_shape": "(None, 64, 64, 128)", "param": 512},
    {"index": 14, "layer_type": "Activation", "output_shape": "(None, 64, 64, 128)", "param": 0},
    {"index": 15, "layer_type": "Conv2D", "output_shape": "(None, 64, 64, 128)", "param": 147584},
    {"index": 16, "layer_type": "Add", "output_shape": "(None, 64, 64, 128)", "param": 0},
    {"index": 17, "layer_type": "BatchNormalization", "output_shape": "(None, 64, 64, 128)", "param": 512},
    {"index": 18, "layer_type": "Activation", "output_shape": "(None, 64, 64, 128)", "param": 0},
    {"index": 19, "layer_type": "Conv2D", "output_shape": "(None, 32, 32, 256)", "param": 295168},
    {"index": 20, "layer_type": "BatchNormalization", "output_shape": "(None, 32, 32, 256)", "param": 1024},
    {"index": 21, "layer_type": "Activation", "output_shape": "(None, 32, 32, 256)", "param": 0},
    {"index": 22, "layer_type": "Conv2D", "output_shape": "(None, 32, 32, 256)", "param": 590080},
    {"index": 23, "layer_type": "Conv2D", "output_shape": "(None, 32, 32, 256)", "param": 33024},
    {"index": 24, "layer_type": "Add", "output_shape": "(None, 32, 32, 256)", "param": 0},
    {"index": 25, "layer_type": "BatchNormalization", "output_shape": "(None, 32, 32, 256)", "param": 1024},
    {"index": 26, "layer_type": "Activation", "output_shape": "(None, 32, 32, 256)", "param": 0},
    {"index": 27, "layer_type": "Conv2D", "output_shape": "(None, 32, 32, 256)", "param": 590080},
    {"index": 28, "layer_type": "BatchNormalization", "output_shape": "(None, 32, 32, 256)", "param": 1024},
    {"index": 29, "layer_type": "Activation", "output_shape": "(None, 32, 32, 256)", "param": 0},
    {"index": 30, "layer_type": "Conv2D", "output_shape": "(None, 32, 32, 256)", "param": 590080},
    {"index": 31, "layer_type": "Add", "output_shape": "(None, 32, 32, 256)", "param": 0},
    {"index": 32, "layer_type": "BatchNormalization", "output_shape": "(None, 32, 32, 256)", "param": 1024},
    {"index": 33, "layer_type": "Activation", "output_shape": "(None, 32, 32, 256)", "param": 0},
    {"index": 34, "layer_type": "Conv2D", "output_shape": "(None, 16, 16, 512)", "param": 1180160},
    {"index": 35, "layer_type": "BatchNormalization", "output_shape": "(None, 16, 16, 512)", "param": 2048},
    {"index": 36, "layer_type": "Activation", "output_shape": "(None, 16, 16, 512)", "param": 0},
    {"index": 37, "layer_type": "Conv2D", "output_shape": "(None, 16, 16, 512)", "param": 2359808},
    {"index": 38, "layer_type": "Conv2D", "output_shape": "(None, 16, 16, 512)", "param": 131584},
    {"index": 39, "layer_type": "Add", "output_shape": "(None, 16, 16, 512)", "param": 0},
    {"index": 40, "layer_type": "BatchNormalization", "output_shape": "(None, 16, 16, 512)", "param": 2048},
    {"index": 41, "layer_type": "Activation", "output_shape": "(None, 16, 16, 512)", "param": 0},
    {"index": 42, "layer_type": "Conv2D", "output_shape": "(None, 16, 16, 512)", "param": 2359808},
    {"index": 43, "layer_type": "BatchNormalization", "output_shape": "(None, 16, 16, 512)", "param": 2048},
    {"index": 44, "layer_type": "Activation", "output_shape": "(None, 16, 16, 512)", "param": 0},
    {"index": 45, "layer_type": "Conv2D", "output_shape": "(None, 16, 16, 512)", "param": 2359808},
    {"index": 46, "layer_type": "Add", "output_shape": "(None, 16, 16, 512)", "param": 0},
    {"index": 47, "layer_type": "BatchNormalization", "output_shape": "(None, 16, 16, 512)", "param": 2048},
    {"index": 48, "layer_type": "Activation", "output_shape": "(None, 16, 16, 512)", "param": 0},
    {"index": 49, "layer_type": "AveragePooling2D", "output_shape": "(None, 2, 2, 512)", "param": 0},
    {"index": 50, "layer_type": "Flatten", "output_shape": "(None, 2048)", "param": 0},
    {"index": 51, "layer_type": "Dense", "output_shape": "(None, 72)", "param": 147528},
    {"index": 52, "layer_type": "Activation", "output_shape": "(None, 72)", "param": 0},
]

def generate_layer_summary(model_summary):
    summary_mapping = {}
    for layer in model_summary:
        layer_str = f"{layer['layer_type']}_{layer['output_shape']}" #_{layer['param']}"
        summary_mapping[layer["index"]] = layer_str
    return summary_mapping

def get_layer_description(layer_name, model):
    """
    Get a description of why a layer is visualized.

    Args:
        layer_name: Name of the layer.
        model: The trained Keras model.

    Returns:
        A string description for the layer.
    """
    layer = model.get_layer(name=layer_name)
    if isinstance(layer, tf.keras.layers.Conv2D):
        return "Convolutional layer: Detects spatial features."
    elif isinstance(layer, tf.keras.layers.Add):
        return "Residual connection: Combines features from shortcuts."
    elif isinstance(layer, tf.keras.layers.Activation):
        return "Activation layer: Applies non-linear transformation."
    elif isinstance(layer, tf.keras.layers.AveragePooling2D):
        return "Pooling layer: Downsamples feature maps."
    elif isinstance(layer, tf.keras.layers.Dense):
        return "Dense layer: Produces final output."
    else:
        return "Other layer type."