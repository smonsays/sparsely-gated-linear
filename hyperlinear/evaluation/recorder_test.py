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

import dataclasses
from typing import Iterator

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from hyperlinear.data_types import Batch
from hyperlinear.evaluation.recorder import NeuronFeatureRecorder
from hyperlinear.model.feedforward import SparselyGatedLinear


class MockModel(nn.Module):
  """Mock model for testing purposes."""

  vocab_size: int
  d_model: int
  n_layers: int
  d_ffw: int
  d_key: int
  knn: int
  n_channels: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
    for layer in range(self.n_layers):
      x = SparselyGatedLinear(
        self.d_model,
        self.d_ffw,
        self.d_key,
        self.knn,
        self.n_channels,
        router_type='dense',
        activation='identity',
        dtype='bfloat16',
        name=f'layer_{layer}',
      )(x)
    x = nn.Dense(features=self.vocab_size)(x)
    return x


@dataclasses.dataclass
class MockDataLoader:
  batch_size: int = 4
  seq_len: int = 8
  vocab_size: int = 10
  num_batches: int = 5

  def __iter__(self) -> Iterator[Batch]:
    np.random.seed(0)
    for i in range(self.num_batches):
      x = np.random.randint(
        1, self.vocab_size, size=(self.batch_size, self.seq_len), dtype=np.int64
      )
      yield Batch(x=x, info={'batch_idx': np.full((self.batch_size,), i)})


class RecorderTest(parameterized.TestCase):
  rng: chex.PRNGKey = jax.random.key(0)

  def test_neuron_feature_recorder_basic(self) -> None:
    """Test NeuronFeatureRecorder basic functionality."""
    batch_size = 1
    n_batches = 1
    seq_len = 512
    vocab_size = 512
    d_model = 16
    n_layers = 4
    d_ffw = 1024
    d_key = 8
    knn = 8
    n_channels = 8

    model = MockModel(
      vocab_size=vocab_size,
      d_model=d_model,
      n_layers=n_layers,
      d_ffw=d_ffw,
      d_key=d_key,
      knn=knn,
      n_channels=n_channels,
    )

    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    params = model.init(self.rng, dummy_input)

    dataloader = MockDataLoader(
      batch_size=batch_size,
      seq_len=seq_len,
      vocab_size=vocab_size,
      num_batches=3,
    )

    # Create recorder
    recorder = NeuronFeatureRecorder(
      model=model,
      params=params,
      n_top_output_tokens=3,
      bos_token_id=0,
      seed=0,
    )

    records, infos = recorder.run(dataloader, n_batches=n_batches)
    self.assertEqual(records.sequence.shape[0], n_batches * batch_size)
    self.assertTrue(jnp.all(records.neuron < d_ffw))
    self.assertFalse(jnp.isinf(records.score).any())
    self.assertTrue(jnp.all(records.sequence > -1))

    self.assertIn('batch_idx', infos)
    self.assertEqual(infos['batch_idx'].shape, (n_batches * batch_size,))


if __name__ == '__main__':
  absltest.main()
