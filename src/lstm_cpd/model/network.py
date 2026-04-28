from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from lstm_cpd.datasets.join_and_split import MODEL_INPUT_COLUMNS
from lstm_cpd.datasets.sequences import SEQUENCE_LENGTH


FEATURE_COUNT = len(MODEL_INPUT_COLUMNS)
INPUT_DROPOUT_LAYER_NAME = "input_dropout"
LSTM_LAYER_NAME = "shared_lstm"
OUTPUT_DROPOUT_LAYER_NAME = "output_dropout"
TIME_DISTRIBUTED_LAYER_NAME = "time_distributed_head"
DENSE_HEAD_LAYER_NAME = "position_head_dense"


@dataclass(frozen=True)
class ModelSeedBundle:
    input_dropout_seed: int
    lstm_kernel_seed: int
    lstm_recurrent_seed: int
    output_dropout_seed: int
    dense_kernel_seed: int


@dataclass(frozen=True)
class ModelRuntimeConfig:
    dropout_rate: float
    hidden_size: int
    sequence_length: int = SEQUENCE_LENGTH
    feature_count: int = FEATURE_COUNT
    seeds: ModelSeedBundle | None = None


def derive_model_seed_bundle(candidate_seed: int) -> ModelSeedBundle:
    base_seed = int(candidate_seed)
    return ModelSeedBundle(
        input_dropout_seed=base_seed,
        lstm_kernel_seed=base_seed + 1,
        lstm_recurrent_seed=base_seed + 2,
        output_dropout_seed=base_seed + 3,
        dense_kernel_seed=base_seed + 4,
    )


def build_model_runtime(config: ModelRuntimeConfig) -> tf.keras.Model:
    if config.sequence_length != SEQUENCE_LENGTH:
        raise ValueError(
            f"Model runtime sequence_length must be {SEQUENCE_LENGTH}, got {config.sequence_length}"
        )
    if config.feature_count != FEATURE_COUNT:
        raise ValueError(
            f"Model runtime feature_count must be {FEATURE_COUNT}, got {config.feature_count}"
        )
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if not 0.0 <= config.dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0.0, 1.0)")

    seeds = config.seeds or derive_model_seed_bundle(20260421)
    inputs = tf.keras.Input(
        shape=(config.sequence_length, config.feature_count),
        dtype=tf.float32,
        name="sequence_inputs",
    )
    x = tf.keras.layers.Dropout(
        rate=config.dropout_rate,
        noise_shape=(None, 1, config.feature_count),
        seed=seeds.input_dropout_seed,
        name=INPUT_DROPOUT_LAYER_NAME,
    )(inputs)
    x = tf.keras.layers.LSTM(
        units=config.hidden_size,
        return_sequences=True,
        stateful=False,
        go_backwards=False,
        dropout=0.0,
        recurrent_dropout=0.0,
        kernel_initializer=tf.keras.initializers.GlorotUniform(
            seed=seeds.lstm_kernel_seed
        ),
        recurrent_initializer=tf.keras.initializers.Orthogonal(
            seed=seeds.lstm_recurrent_seed
        ),
        bias_initializer="zeros",
        name=LSTM_LAYER_NAME,
    )(x)
    x = tf.keras.layers.Dropout(
        rate=config.dropout_rate,
        noise_shape=(None, 1, config.hidden_size),
        seed=seeds.output_dropout_seed,
        name=OUTPUT_DROPOUT_LAYER_NAME,
    )(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(
            units=1,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=seeds.dense_kernel_seed
            ),
            bias_initializer="zeros",
            name=DENSE_HEAD_LAYER_NAME,
        ),
        name=TIME_DISTRIBUTED_LAYER_NAME,
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_cpd_dmn")


def get_single_lstm_layer(model: tf.keras.Model) -> tf.keras.layers.LSTM:
    lstm_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)
    ]
    if len(lstm_layers) != 1:
        raise ValueError(f"Expected exactly one LSTM layer, found {len(lstm_layers)}")
    return lstm_layers[0]


def get_dropout_layers(model: tf.keras.Model) -> tuple[tf.keras.layers.Dropout, ...]:
    return tuple(
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
    )
