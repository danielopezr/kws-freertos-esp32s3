"""The three keyword spotting models (after Zhang et al., "Hello Edge", 2017).

All take a (49, 40, 1) map of MFCCs already normalised with the training-set
mean and standard deviation (see train.py / norm.json). Normalisation is kept
outside the graph so the INT8 model only uses ops TFLM supports; the firmware
applies (x - mean) / std and then the input scale and zero point.

  small  - DNN:    3 fully connected layers of 144 units (little compute, many weights)
  medium - CNN:    3 strided convolutions + global average pooling
  large  - DS-CNN: standard conv + 5 depthwise-separable blocks, 172 channels
                   (largest activations: stresses internal SRAM)
"""

import tensorflow as tf
from tensorflow.keras import layers

from . import features as F

INPUT_SHAPE = (F.N_FRAMES, F.N_MFCC, 1)
NAMES = ("dnn", "cnn", "dscnn")


def _stem():
    inp = layers.Input(INPUT_SHAPE, name="mfcc")
    return inp, inp


def _conv_bn_relu(x, filters, kernel, strides, name):
    x = layers.Conv2D(filters, kernel, strides=strides, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def dnn(n_classes, units=144, dropout=0.2):
    inp, x = _stem()
    x = layers.Flatten()(x)
    for i in range(3):
        x = layers.Dense(units, activation="relu", name=f"fc{i + 1}")(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, name="logits")(x)
    return tf.keras.Model(inp, out, name="dnn")


def cnn(n_classes, dropout=0.2):
    inp, x = _stem()
    x = _conv_bn_relu(x, 32, (10, 4), (2, 2), "c1")
    x = _conv_bn_relu(x, 48, (3, 3), (2, 2), "c2")
    x = _conv_bn_relu(x, 64, (3, 3), (2, 2), "c3")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, name="logits")(x)
    return tf.keras.Model(inp, out, name="cnn")


def dscnn(n_classes, channels=172, blocks=5, dropout=0.2):
    inp, x = _stem()
    x = _conv_bn_relu(x, channels, (10, 4), (2, 2), "c0")
    for i in range(blocks):
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, name=f"ds{i + 1}_dw")(x)
        x = layers.BatchNormalization(name=f"ds{i + 1}_dw_bn")(x)
        x = layers.ReLU(name=f"ds{i + 1}_dw_relu")(x)
        x = _conv_bn_relu(x, channels, (1, 1), (1, 1), f"ds{i + 1}_pw")
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, name="logits")(x)
    return tf.keras.Model(inp, out, name="dscnn")


def build(name, n_classes):
    return {"dnn": dnn, "cnn": cnn, "dscnn": dscnn}[name](n_classes)
