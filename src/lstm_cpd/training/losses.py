from __future__ import annotations

import math

import tensorflow as tf


ANNUALIZATION_FACTOR = math.sqrt(252.0)
LOSS_EPSILON = 1e-12


def squeeze_position_outputs(position_outputs: tf.Tensor | object) -> tf.Tensor:
    tensor = tf.convert_to_tensor(position_outputs, dtype=tf.float32)
    if tensor.shape.rank == 3:
        if tensor.shape[-1] != 1:
            raise ValueError(
                f"Expected trailing position dimension 1, got shape {tensor.shape}"
            )
        tensor = tf.squeeze(tensor, axis=-1)
    if tensor.shape.rank != 2:
        raise ValueError(
            f"Position outputs must have rank 2 or 3, got shape {tensor.shape}"
        )
    return tensor


def compute_realized_returns(
    position_outputs: tf.Tensor | object,
    target_scale: tf.Tensor | object,
) -> tf.Tensor:
    squeezed_positions = squeeze_position_outputs(position_outputs)
    target_scale_tensor = tf.convert_to_tensor(target_scale, dtype=tf.float32)
    if target_scale_tensor.shape.rank != 2:
        raise ValueError(
            f"target_scale must have rank 2, got shape {target_scale_tensor.shape}"
        )
    if squeezed_positions.shape != target_scale_tensor.shape:
        raise ValueError(
            "Position outputs and target_scale must share shape after squeezing: "
            f"{squeezed_positions.shape} != {target_scale_tensor.shape}"
        )
    return squeezed_positions * target_scale_tensor


def sharpe_loss_from_realized_returns(
    realized_returns: tf.Tensor | object,
    *,
    epsilon: float = LOSS_EPSILON,
) -> tf.Tensor:
    realized_tensor = tf.reshape(
        tf.convert_to_tensor(realized_returns, dtype=tf.float32),
        (-1,),
    )
    mean_return = tf.reduce_mean(realized_tensor)
    variance = tf.math.reduce_variance(realized_tensor)
    annualized_sharpe = (
        tf.cast(ANNUALIZATION_FACTOR, tf.float32)
        * mean_return
        / tf.sqrt(variance + tf.cast(epsilon, tf.float32))
    )
    return -annualized_sharpe


def sharpe_loss(
    position_outputs: tf.Tensor | object,
    target_scale: tf.Tensor | object,
    *,
    epsilon: float = LOSS_EPSILON,
) -> tf.Tensor:
    realized_returns = compute_realized_returns(position_outputs, target_scale)
    return sharpe_loss_from_realized_returns(realized_returns, epsilon=epsilon)


class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, *, epsilon: float = LOSS_EPSILON, name: str = "sharpe_loss"):
        super().__init__(name=name)
        self.epsilon = float(epsilon)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return sharpe_loss(y_pred, y_true, epsilon=self.epsilon)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config["epsilon"] = self.epsilon
        return config
