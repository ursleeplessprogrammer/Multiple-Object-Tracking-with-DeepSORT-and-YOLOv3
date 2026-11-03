import numpy as np


class WeightReader:
    def __init__(self, weight_file):
        self.weights = np.fromfile(weight_file, dtype="float32")
        self.offset = 5

    def load_weights(self, model):
        for layer in model.layers:
            if not layer.name.startswith("conv_"):
                continue
            layer_idx = int(layer.name.split("_")[-1])
            try:
                conv_layer = model.get_layer(f"conv_{layer_idx}")
                kernel_size, _, _, filters = conv_layer.get_weights()[0].shape
            except (ValueError, IndexError):
                continue
            if not conv_layer.use_bias:
                bn_layer = model.get_layer(f"bnorm_{layer_idx}")
                size = bn_layer.get_weights()[0].shape[0]
                bn_weights = self.weights[self.offset:self.offset + 4 * size].reshape((4, size))
                self.offset += 4 * size
                bn_weights = bn_weights[[1, 0, 2, 3]]
                conv_weights = self.weights[self.offset:self.offset + filters * kernel_size * kernel_size * conv_layer.kernel.shape[2]]
                self.offset += conv_weights.size
                conv_weights = conv_weights.reshape((filters, conv_layer.kernel.shape[2], kernel_size, kernel_size))
                conv_weights = conv_weights.transpose([2, 3, 1, 0])
                bn_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                bias = self.weights[self.offset:self.offset + filters]
                self.offset += filters
                conv_weights = self.weights[self.offset:self.offset + filters * kernel_size * kernel_size * conv_layer.kernel.shape[2]]
                self.offset += conv_weights.size
                conv_weights = conv_weights.reshape((filters, conv_layer.kernel.shape[2], kernel_size, kernel_size))
                conv_weights = conv_weights.transpose([2, 3, 1, 0])
                conv_layer.set_weights([conv_weights, bias])


def download_yolov3_weights(destination="yolov3.weights"):
    from urllib.request import urlretrieve
    urlretrieve("https://pjreddie.com/media/files/yolov3.weights", destination)
    return destination
