from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lstm_cpd.model.network import (  # noqa: E402
    ModelRuntimeConfig,
    build_model_runtime,
    derive_model_seed_bundle,
    get_dropout_layers,
    get_single_lstm_layer,
)
from lstm_cpd.training.losses import (  # noqa: E402
    compute_realized_returns,
    sharpe_loss,
)


class T19ModelRuntimeTests(unittest.TestCase):
    def test_runtime_uses_one_stateless_unidirectional_lstm(self) -> None:
        model = build_model_runtime(
            ModelRuntimeConfig(
                dropout_rate=0.1,
                hidden_size=20,
                seeds=derive_model_seed_bundle(123),
            )
        )

        lstm_layer = get_single_lstm_layer(model)
        self.assertFalse(lstm_layer.stateful)
        self.assertFalse(lstm_layer.go_backwards)
        self.assertEqual(lstm_layer.recurrent_dropout, 0.0)
        self.assertEqual(model.input_shape, (None, 63, 10))
        self.assertEqual(model.output_shape, (None, 63, 1))

    def test_runtime_uses_sequence_shared_dropout_masks(self) -> None:
        model = build_model_runtime(
            ModelRuntimeConfig(
                dropout_rate=0.2,
                hidden_size=40,
                seeds=derive_model_seed_bundle(456),
            )
        )

        dropout_layers = get_dropout_layers(model)
        self.assertEqual([layer.name for layer in dropout_layers], ["input_dropout", "output_dropout"])
        self.assertEqual(dropout_layers[0].noise_shape, (None, 1, 10))
        self.assertEqual(dropout_layers[1].noise_shape, (None, 1, 40))

    def test_sharpe_loss_matches_manual_formula(self) -> None:
        positions = tf.constant([[[0.5], [1.0]], [[-0.5], [0.25]]], dtype=tf.float32)
        target_scale = tf.constant([[0.2, -0.1], [0.3, 0.4]], dtype=tf.float32)

        loss_value = float(sharpe_loss(positions, target_scale).numpy())
        realized_returns = np.asarray([0.1, -0.1, -0.15, 0.1], dtype=np.float32)
        expected = -math.sqrt(252.0) * realized_returns.mean() / math.sqrt(
            realized_returns.var() + 1e-12
        )
        self.assertAlmostEqual(loss_value, expected, places=6)

    def test_compute_realized_returns_accepts_rank_two_positions(self) -> None:
        positions = tf.constant([[0.5, -0.25]], dtype=tf.float32)
        target_scale = tf.constant([[0.2, 0.4]], dtype=tf.float32)

        realized = compute_realized_returns(positions, target_scale)

        self.assertEqual(tuple(realized.shape), (1, 2))
        self.assertAlmostEqual(float(realized[0, 0].numpy()), 0.1)
        self.assertAlmostEqual(float(realized[0, 1].numpy()), -0.1)


if __name__ == "__main__":
    unittest.main()
