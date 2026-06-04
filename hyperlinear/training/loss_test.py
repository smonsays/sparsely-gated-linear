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
from absl.testing import parameterized

from hyperlinear.data_types import Batch
from hyperlinear.training.loss import AutoregressiveCrossEntropy
from hyperlinear.training.loss import shift_right


class LossTest(parameterized.TestCase):
  rng = jax.random.key(0)

  def test_autoregressive_cross_entropy(self) -> None:
    """Test behavior of AutoregressiveCrossEntropy with random predictions."""
    vocab_size = 10

    def random_apply_fn(params: dict, inputs: jax.Array, rngs: dict | None) -> jax.Array:
      """Create random logits."""
      batch_size, seq_len = inputs.shape
      return jax.random.normal(rngs['dropout'], (batch_size, seq_len, vocab_size))

    loss_fn = AutoregressiveCrossEntropy(
      apply_fn=random_apply_fn, pad_token_id=0, bos_token_id=8
    )
    batch = Batch(x=jnp.array([[1, 2, 3, 8], [2, 3, 4, 8]], dtype=jnp.int32))
    params = dict()
    loss, metrics = loss_fn(params, self.rng, batch)

    # For random predictions with vocab_size=10, loss should be around log(10) ≈ 2.3
    self.assertGreater(loss, 1.0)
    self.assertLess(loss, 3.0)
    self.assertGreaterEqual(metrics['acc'], 0.0)
    self.assertLessEqual(metrics['acc'], 1.0)
    self.assertTrue(jnp.isfinite(metrics['perplexity']))
    self.assertTrue(jnp.isfinite(metrics['perplexity_last_token']))

  def test_shift_right(self) -> None:
    """Test shift_right function shifts input correctly."""
    bos_id = 0
    x = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    shifted = shift_right(x, bos_id)
    expected = jnp.array([[bos_id, 1, 2, 3], [bos_id, 5, 6, 7]], dtype=jnp.int32)
    self.assertTrue(jnp.array_equal(shifted, expected))


if __name__ == '__main__':
  absltest.main()
