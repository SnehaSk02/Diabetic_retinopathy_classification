import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(img_array, model):

    # DenseNet backbone
    base_model = model.get_layer("densenet121")

    # Last convolution layer
    last_conv_layer = base_model.get_layer(
        "conv5_block16_concat"
    )

    # Rebuild classifier head manually
    x = base_model.output

    x = model.layers[1](x)  # GAP
    x = model.layers[2](x)  # BatchNorm
    x = model.layers[3](x, training=False)  # Dropout
    x = model.layers[4](x)  # Dense(64)
    x = model.layers[5](x, training=False)  # Dropout
    predictions = model.layers[6](x)  # Dense(5)

    # GradCAM model
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            predictions
        ]
    )

    # Forward pass + gradients
    with tf.GradientTape() as tape:

        conv_outputs, preds = grad_model(
            img_array,
            training=False
        )

        pred_index = tf.argmax(
            preds[0]
        )

        class_channel = preds[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Debug
    print("Conv Output Shape:", conv_outputs.shape)
    print("Prediction Shape:", preds.shape)
    print("Grad Shape:", None if grads is None else grads.shape)

    if grads is None:
        raise ValueError(
            "Gradients are None. Graph disconnected."
        )

    # Channel importance
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def overlay_heatmap(
    heatmap,
    original_img,
    alpha=0.4
):

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.resize(heatmap,
    (
        original_img.shape[1],
        original_img.shape[0]))

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    superimposed_img = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return superimposed_img