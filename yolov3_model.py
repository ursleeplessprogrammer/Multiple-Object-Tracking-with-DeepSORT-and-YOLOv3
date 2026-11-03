from tensorflow.keras.layers import (
    Input, Conv2D, ZeroPadding2D, BatchNormalization, LeakyReLU,
    Add, UpSampling2D, Concatenate,
)
from tensorflow.keras.models import Model


def _conv_block(inp, convs, skip=True):
    x = inp
    skip_connection = None
    for count, conv in enumerate(convs):
        if count == len(convs) - 2 and skip:
            skip_connection = x
        if conv["stride"] > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = Conv2D(
            conv["filter"],
            conv["kernel"],
            strides=conv["stride"],
            padding="valid" if conv["stride"] > 1 else "same",
            name=f"conv_{conv['layer_idx']}",
            use_bias=not conv["bnorm"],
        )(x)
        if conv["bnorm"]:
            x = BatchNormalization(epsilon=0.001, name=f"bnorm_{conv['layer_idx']}")(x)
        if conv["leaky"]:
            x = LeakyReLU(alpha=0.1, name=f"leaky_{conv['layer_idx']}")(x)
    return Add()([skip_connection, x]) if skip else x


def _cfg(filters, kernel, stride, layer_idx, bnorm=True, leaky=True):
    return {
        "filter": filters,
        "kernel": kernel,
        "stride": stride,
        "bnorm": bnorm,
        "leaky": leaky,
        "layer_idx": layer_idx,
    }


def make_yolov3_model(num_classes=80):
    output_filters = 3 * (num_classes + 5)
    input_image = Input(shape=(None, None, 3))

    x = _conv_block(input_image, [
        _cfg(32, 3, 1, 0), _cfg(64, 3, 2, 1),
        _cfg(32, 1, 1, 2), _cfg(64, 3, 1, 3),
    ])
    x = _conv_block(x, [_cfg(128, 3, 2, 5), _cfg(64, 1, 1, 6), _cfg(128, 3, 1, 7)])
    x = _conv_block(x, [_cfg(64, 1, 1, 9), _cfg(128, 3, 1, 10)])
    x = _conv_block(x, [_cfg(256, 3, 2, 12), _cfg(128, 1, 1, 13), _cfg(256, 3, 1, 14)])
    for i in range(7):
        x = _conv_block(x, [_cfg(128, 1, 1, 16 + i * 3), _cfg(256, 3, 1, 17 + i * 3)])
    skip_36 = x

    x = _conv_block(x, [_cfg(512, 3, 2, 37), _cfg(256, 1, 1, 38), _cfg(512, 3, 1, 39)])
    for i in range(7):
        x = _conv_block(x, [_cfg(256, 1, 1, 41 + i * 3), _cfg(512, 3, 1, 42 + i * 3)])
    skip_61 = x

    x = _conv_block(x, [_cfg(1024, 3, 2, 62), _cfg(512, 1, 1, 63), _cfg(1024, 3, 1, 64)])
    for i in range(3):
        x = _conv_block(x, [_cfg(512, 1, 1, 66 + i * 3), _cfg(1024, 3, 1, 67 + i * 3)])
    x = _conv_block(x, [
        _cfg(512, 1, 1, 75), _cfg(1024, 3, 1, 76), _cfg(512, 1, 1, 77),
        _cfg(1024, 3, 1, 78), _cfg(512, 1, 1, 79),
    ], skip=False)

    yolo_82 = _conv_block(x, [_cfg(1024, 3, 1, 80), _cfg(output_filters, 1, 1, 81, False, False)], skip=False)
    x = _conv_block(x, [_cfg(256, 1, 1, 84)], skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_61])
    x = _conv_block(x, [
        _cfg(256, 1, 1, 87), _cfg(512, 3, 1, 88), _cfg(256, 1, 1, 89),
        _cfg(512, 3, 1, 90), _cfg(256, 1, 1, 91),
    ], skip=False)
    yolo_94 = _conv_block(x, [_cfg(512, 3, 1, 92), _cfg(output_filters, 1, 1, 93, False, False)], skip=False)

    x = _conv_block(x, [_cfg(128, 1, 1, 96)], skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_36])
    yolo_106 = _conv_block(x, [
        _cfg(128, 1, 1, 99), _cfg(256, 3, 1, 100), _cfg(128, 1, 1, 101),
        _cfg(256, 3, 1, 102), _cfg(128, 1, 1, 103), _cfg(256, 3, 1, 104),
        _cfg(output_filters, 1, 1, 105, False, False),
    ], skip=False)
    return Model(input_image, [yolo_82, yolo_94, yolo_106], name="yolov3")


def make_modified_yolov3_model(num_classes=80):
    output_filters = 3 * (num_classes + 5)
    input_image = Input(shape=(None, None, 3))

    x = _conv_block(input_image, [
        _cfg(32, 3, 1, 0), _cfg(64, 3, 2, 1), _cfg(32, 1, 1, 2), _cfg(64, 3, 1, 3),
    ])
    x = _conv_block(x, [_cfg(128, 3, 2, 5), _cfg(64, 1, 1, 6), _cfg(128, 3, 1, 7)])
    x = _conv_block(x, [_cfg(64, 1, 1, 9), _cfg(128, 3, 1, 10)])
    x = _conv_block(x, [_cfg(256, 3, 2, 12), _cfg(128, 1, 1, 13), _cfg(256, 3, 1, 14)])
    for i in range(7):
        x = _conv_block(x, [_cfg(128, 1, 1, 16 + i * 3), _cfg(256, 3, 1, 17 + i * 3)])
    skip_36 = x

    x = _conv_block(x, [_cfg(512, 3, 2, 37), _cfg(256, 1, 1, 38), _cfg(512, 3, 1, 39)])
    for i in range(7):
        x = _conv_block(x, [_cfg(256, 1, 1, 41 + i * 3), _cfg(512, 3, 1, 42 + i * 3)])
    skip_61 = x

    x = _conv_block(x, [_cfg(1024, 3, 2, 62), _cfg(512, 1, 1, 63), _cfg(1024, 3, 1, 64)])
    for i in range(3):
        x = _conv_block(x, [_cfg(512, 1, 1, 66 + i * 3), _cfg(1024, 3, 1, 67 + i * 3)])
    x = _conv_block(x, [
        _cfg(512, 1, 1, 75), _cfg(1024, 3, 1, 76), _cfg(512, 1, 1, 77),
        _cfg(1024, 3, 1, 78), _cfg(512, 1, 1, 79),
    ], skip=False)

    yolo_82 = _conv_block(x, [_cfg(1024, 3, 1, 80), _cfg(output_filters, 1, 1, 81, False, False)], skip=False)
    x = _conv_block(x, [_cfg(256, 1, 1, 84)], skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_61])
    x = _conv_block(x, [_cfg(256, 3, 1, 87), _cfg(512, 1, 1, 88), _cfg(256, 3, 1, 89)])
    x = _conv_block(x, [
        _cfg(256, 1, 1, 90), _cfg(512, 3, 1, 91), _cfg(256, 1, 1, 92),
        _cfg(512, 3, 1, 93), _cfg(256, 1, 1, 94),
    ], skip=False)
    yolo_97 = _conv_block(x, [_cfg(512, 3, 1, 95), _cfg(output_filters, 1, 1, 96, False, False)], skip=False)

    x = _conv_block(x, [_cfg(128, 1, 1, 99)], skip=False)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_36])
    yolo_109 = _conv_block(x, [
        _cfg(128, 1, 1, 102), _cfg(256, 3, 1, 103), _cfg(128, 1, 1, 104),
        _cfg(256, 3, 1, 105), _cfg(128, 1, 1, 106), _cfg(256, 3, 1, 107),
        _cfg(output_filters, 1, 1, 108, False, False),
    ], skip=False)
    return Model(input_image, [yolo_82, yolo_97, yolo_109], name="modified_yolov3")
