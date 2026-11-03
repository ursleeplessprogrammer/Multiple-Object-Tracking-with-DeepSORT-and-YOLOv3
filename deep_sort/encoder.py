import cv2
import numpy as np
import tensorflow.compat.v1 as tf


def _run_in_batches(function, data_dict, out, batch_size):
    data_len = len(out)
    for start in range(0, data_len, batch_size):
        end = min(start + batch_size, data_len)
        batch = {key: value[start:end] for key, value in data_dict.items()}
        out[start:end] = function(batch)


class ImageEncoder:
    def __init__(self, checkpoint_filename, input_name="images:0", output_name="features:0"):
        self.session = tf.Session()
        with tf.gfile.GFile(str(checkpoint_filename), "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        with self.session.graph.as_default():
            tf.import_graph_def(graph_def, name="")
        try:
            self.input_var = self.session.graph.get_tensor_by_name(input_name)
            self.output_var = self.session.graph.get_tensor_by_name(output_name)
        except KeyError:
            operations = self.session.graph.get_operations()
            self.input_var = self.session.graph.get_tensor_by_name(operations[0].name + ":0")
            self.output_var = self.session.graph.get_tensor_by_name(operations[-1].name + ":0")
        self.feature_dim = self.output_var.shape.as_list()[-1]
        self.image_shape = self.input_var.shape.as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        data_x = np.asarray(data_x)
        if len(data_x) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        out = np.zeros((len(data_x), self.feature_dim), dtype=np.float32)
        _run_in_batches(lambda x: self.session.run(self.output_var, feed_dict=x), {self.input_var: data_x}, out, batch_size)
        return out


def extract_image_patch(image, bbox, patch_shape):
    bbox = np.asarray(bbox, dtype=float).copy()
    if patch_shape is not None:
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(int)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    patch = image[sy:ey, sx:ex]
    if patch.size == 0:
        return None
    if patch_shape is not None:
        patch = cv2.resize(patch, tuple(patch_shape[::-1]))
    return patch


def create_box_encoder(model_filename, input_name="images:0", output_name="features:0", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                patch = np.zeros(image_shape, dtype=np.uint8)
            patches.append(patch)
        return image_encoder(np.asarray(patches), batch_size)

    return encoder
