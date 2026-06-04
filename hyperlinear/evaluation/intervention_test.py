"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import jax
import jax.numpy as jnp
from absl.testing import absltest

from hyperlinear import config_classes
from hyperlinear.evaluation.intervention import InterventionBatch
from hyperlinear.evaluation.intervention import InterventionRecorder
from hyperlinear.model import transformer


class InterventionTest(absltest.TestCase):
  def setUp(self) -> None:
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.seq_len = 13
    self.n_vocab = 17

    # Create dummy transformer config and model
    ffw_config = config_classes.FeedforwardConfig(
      type='sparsely_gated_linear',
      d_ffw=64,
      d_key=8,
      knn=2,
      n_channels=1,
      router_type='product_dense',
      activation='identity',
    )

    self.model = transformer.Transformer(
      n_vocab=self.n_vocab,
      d_model=16,
      d_heads=8,
      n_layers=3,
      feedforward_config=ffw_config,
      dtype=jnp.float32,
      remat=False,
    )

    # Initialize dummy parameters
    self.rng, init_key = jax.random.split(self.rng)
    dummy_input = jnp.zeros((1, self.seq_len), dtype=jnp.int32)
    self.params = self.model.init(init_key, dummy_input)

  def test_intervention_recorder(self) -> None:
    """Test the execution and outputs of the InterventionRecorder."""
    batch_size = 11
    rng1, rng2 = jax.random.split(self.rng)
    prefix_clean = jax.random.randint(rng1, (batch_size, self.seq_len), 0, self.n_vocab)
    prefix_counterfactual = jax.random.randint(
      rng2, (batch_size, self.seq_len), 0, self.n_vocab
    )
    answers_clean = jnp.array([[10]] * batch_size)
    answers_counterfactual = jnp.array([[12]] * batch_size)
    target_position = 3

    batch = InterventionBatch(
      prefix_clean=prefix_clean,
      prefix_counterfactual=prefix_counterfactual,
      answers_clean=answers_clean,
      answers_counterfactual=answers_counterfactual,
    )
    layers_to_patch = (0,)

    recorder = InterventionRecorder(
      model=self.model,
      params=self.params,
      layers_to_patch=layers_to_patch,
      position_to_patch=target_position,
      position_to_measure=target_position,
    )
    record = recorder(batch)

    # Input consistency
    self.assertTrue((record.router_clean.prefix == batch.prefix_clean).all())
    self.assertTrue(
      (record.router_counterfactual.prefix == batch.prefix_counterfactual).all()
    )

    # Check shapes for routing records
    expected_shape = (
      batch_size,
      self.seq_len,
      self.model.n_layers,
      self.model.feedforward_config.n_channels,
      self.model.feedforward_config.knn,
    )
    self.assertEqual(record.router_clean.score.shape, expected_shape)
    self.assertEqual(record.router_clean.neuron.shape, expected_shape)
    self.assertEqual(record.router_counterfactual.score.shape, expected_shape)
    self.assertEqual(record.router_counterfactual.neuron.shape, expected_shape)

    # Check metrics shapes
    self.assertEqual(record.metrics.logitdiff_clean.shape, (batch_size,))
    self.assertEqual(record.metrics.logitdiff_counterfactual.shape, (batch_size,))
    self.assertEqual(record.metrics.logitdiff_intervened.shape, (batch_size,))
    self.assertEqual(record.metrics.recovered_effect.shape, (batch_size,))
    self.assertEqual(record.metrics.rank_flip.shape, (batch_size,))
    self.assertEqual(record.metrics.correct_clean.shape, (batch_size,))
    self.assertEqual(record.metrics.correct_counterfactual.shape, (batch_size,))

    # Check if the intervention had an effect (i.e. intervened metrics differ from clean)
    self.assertFalse(
      jnp.allclose(record.metrics.logitdiff_clean, record.metrics.logitdiff_intervened)
    )
    self.assertFalse(
      jnp.allclose(
        record.metrics.logitdiff_counterfactual, record.metrics.logitdiff_intervened
      )
    )


if __name__ == '__main__':
  absltest.main()
