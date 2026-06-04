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

import chex
import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import lax

from hyperlinear.data_types import Batch
from hyperlinear.data_types import Metrics
from hyperlinear.training.experiment import ExperimentLoss


def shift_right(x: jax.Array, bos_token_id: int, axis: int = 1) -> jax.Array:
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(x, pad_widths, mode='constant', constant_values=bos_token_id)
  return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


@struct.dataclass
class AutoregressiveCrossEntropy(ExperimentLoss):
  bos_token_id: int
  pad_token_id: int

  def __call__(
    self, params: dict, rng: chex.PRNGKey, batch: Batch
  ) -> tuple[jax.Array, Metrics]:
    batch_size, seq_len = batch.x.shape

    inputs = shift_right(batch.x, bos_token_id=self.bos_token_id)
    targets = batch.x
    weights = jnp.where(targets != self.pad_token_id, 1, 0).astype(jnp.float32)
    denominator = jnp.sum(weights)

    logits = self.apply_fn(params, inputs, rngs={'dropout': rng})
    loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    loss = jnp.sum(loss_per_token * weights) / denominator
    acc = jnp.sum(jnp.equal(jnp.argmax(logits, axis=-1), targets) * weights) / denominator

    # Calculate perplexity for the last non-padding token in each sequence
    last_token_positions = jnp.argmax(weights * jnp.arange(seq_len)[None, :], axis=1)
    loss_last_token = loss_per_token[jnp.arange(batch_size), last_token_positions]
    perplexity_last_token = jnp.exp(jnp.mean(loss_last_token))

    metrics = dict(
      loss=loss,
      acc=acc,
      perplexity=jnp.exp(loss),
      perplexity_last_token=perplexity_last_token,
    )

    return loss, metrics
