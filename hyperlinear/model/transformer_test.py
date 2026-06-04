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
from absl.testing import absltest
from absl.testing import parameterized

from hyperlinear import config_classes
from hyperlinear.model import transformer

jax.config.parse_flags_with_absl()
jax.config.update('jax_numpy_rank_promotion', 'raise')


class TransformerTest(parameterized.TestCase):
  rng: chex.PRNGKey = jax.random.key(42)

  def _create_config(self) -> config_classes.ModelConfig:
    """Helper to create a test configuration."""
    feedforward_config = config_classes.FeedforwardConfig(
      type='mlp',
      activation='gelu',
      d_ffw=64,
      n_channels=8,
      knn=4,
      d_key=32,
      router_type='dense',
    )
    return config_classes.ModelConfig(
      d_model=64,
      d_heads=8,
      n_layers=2,
      feedforward_config=feedforward_config,
      dtype='bfloat16',
      remat=False,
    )

  @chex.variants(with_jit=True, without_jit=True)
  def test_causal_attention(self) -> None:
    batch_size = 2
    context_size = 16
    cfg = self._create_config()

    model = transformer.CausalAttention(
      d_model=cfg.d_model, d_heads=cfg.d_heads, dtype=cfg.dtype
    )
    x_BxLxD = jax.random.normal(
      self.rng, (batch_size, context_size, cfg.d_model), dtype=cfg.dtype
    )
    params = model.init(self.rng, x_BxLxD)
    out_BxLxD = self.variant(model.apply)(params, x_BxLxD)

    chex.assert_shape(out_BxLxD, (batch_size, context_size, cfg.d_model))
    chex.assert_type(out_BxLxD, cfg.dtype)
    chex.assert_tree_all_finite(out_BxLxD)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(
    dict(ffw_type='mlp'),
    dict(ffw_type='sparsely_gated_linear'),
    dict(ffw_type='peer'),
    dict(ffw_type='moe'),
  )
  def test_transformer_block(self, ffw_type: str) -> None:
    """Test transformer block with different feedforward layer types."""
    batch_size = 2
    context_size = 16
    cfg = self._create_config()
    cfg.feedforward_config.type = ffw_type

    model = transformer.TransformerBlock(
      d_model=cfg.d_model,
      d_heads=cfg.d_heads,
      feedforward_config=cfg.feedforward_config,
      dtype=cfg.dtype,
    )
    x_BxLxD = jax.random.normal(
      self.rng, (batch_size, context_size, cfg.d_model), dtype=cfg.dtype
    )
    params = model.init(self.rng, x_BxLxD)
    out_BxLxD = self.variant(model.apply)(params, x_BxLxD)

    chex.assert_shape(out_BxLxD, (batch_size, context_size, cfg.d_model))
    chex.assert_type(out_BxLxD, cfg.dtype)
    chex.assert_tree_all_finite(out_BxLxD)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(
    dict(remat=False, dtype='float32'),
    dict(remat=True, dtype='float32'),
    dict(remat=False, dtype='bfloat16'),
    dict(remat=True, dtype='bfloat16'),
  )
  def test_transformer_do(self, remat: bool, dtype: str) -> None:
    batch_size = 2
    context_size = 16
    n_vocab = 32
    cfg = self._create_config()
    cfg.remat = remat
    cfg.dtype = dtype

    model = transformer.Transformer(
      n_vocab=n_vocab,
      d_model=cfg.d_model,
      d_heads=cfg.d_heads,
      n_layers=cfg.n_layers,
      feedforward_config=cfg.feedforward_config,
      dtype=cfg.dtype,
      remat=cfg.remat,
    )
    # Input should be token indices
    y_BxL = jax.random.randint(self.rng, (batch_size, context_size), 0, n_vocab)
    params = model.init(self.rng, y_BxL)
    logits_BxLxV = self.variant(model.apply)(params, y_BxL)

    chex.assert_shape(logits_BxLxV, (batch_size, context_size, n_vocab))
    chex.assert_type(logits_BxLxV, jnp.float32)  # logits are cast to float32
    chex.assert_tree_all_finite(logits_BxLxV)


if __name__ == '__main__':
  absltest.main()
