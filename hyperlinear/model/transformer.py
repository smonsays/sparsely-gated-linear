"""Transformer Decoder-only model.

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

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax import linen as nn

from hyperlinear import config_classes
from hyperlinear.model import feedforward
from hyperlinear.model import rotary


class Transformer(nn.Module):
  """Transformer decoder-only.

  Einsum notation:
    B: batch
    L: context_size
    D: d_model
    V: n_vocab
    H: n_heads
    Dh: d_heads
  """

  n_vocab: int
  d_model: int
  d_heads: int
  n_layers: int
  feedforward_config: config_classes.FeedforwardConfig
  dtype: str | jnp.dtype
  remat: bool

  def setup(self) -> None:
    self.embedder = nn.Embed(
      num_embeddings=self.n_vocab,
      features=self.d_model,
    )
    block = nn.remat(TransformerBlock) if self.remat else TransformerBlock
    self.blocks = [
      block(
        d_model=self.d_model,
        d_heads=self.d_heads,
        feedforward_config=self.feedforward_config,
        dtype=self.dtype,
      )
      for _ in range(self.n_layers)
    ]
    self.output_norm = nn.RMSNorm(dtype=self.dtype)

  def __call__(self, y_BxL: jt.Int[jt.Array, 'B L']) -> jt.Float[jt.Array, 'B L V']:
    y_BxLxD = self.embedder(y_BxL)
    for block in self.blocks:
      y_BxLxD = block(y_BxLxD)
    y_BxLxD = self.output_norm(y_BxLxD)
    logits_BxLxV = self.embedder.attend(y_BxLxD.astype(jnp.float32))
    return logits_BxLxV


class TransformerBlock(nn.Module):
  """Transformer Block."""

  d_model: int
  d_heads: int
  feedforward_config: config_classes.FeedforwardConfig
  dtype: str | jnp.dtype

  def setup(self) -> None:
    self.attention_norm = nn.RMSNorm(dtype=self.dtype)
    self.attention = CausalAttention(
      d_model=self.d_model,
      d_heads=self.d_heads,
      dtype=self.dtype,
    )
    self.feedforward_norm = nn.RMSNorm(dtype=self.dtype)

    match self.feedforward_config.type:
      case 'mlp':
        self.ffw = feedforward.MultilayerPerceptron(
          d_model=self.d_model,
          d_ffw=self.feedforward_config.d_ffw,
          activation=self.feedforward_config.activation,
          dtype=self.dtype,
        )
      case 'sparsely_gated_linear':
        self.ffw = feedforward.SparselyGatedLinear(
          d_model=self.d_model,
          n_experts=self.feedforward_config.d_ffw,
          d_key=self.feedforward_config.d_key,
          knn=self.feedforward_config.knn,
          n_channels=self.feedforward_config.n_channels,
          router_type=self.feedforward_config.router_type,
          activation=self.feedforward_config.activation,
          dtype=self.dtype,
        )
      case 'peer':
        self.ffw = feedforward.ParameterEfficientExpertRetrieval(
          d_model=self.d_model,
          n_experts=self.feedforward_config.d_ffw,
          d_key=self.feedforward_config.d_key,
          knn=self.feedforward_config.knn,
          n_channels=self.feedforward_config.n_channels,
          dtype=self.dtype,
        )
      case 'glu':
        self.ffw = feedforward.GatedLinearUnit(
          d_model=self.d_model,
          d_ffw=self.feedforward_config.d_ffw,
          activation=self.feedforward_config.activation,
          dtype=self.dtype,
        )
      case 'moe':
        self.ffw = feedforward.MixtureOfExperts(
          d_model=self.d_model,
          d_ffw=self.feedforward_config.d_ffw,
          n_channels=self.feedforward_config.n_channels,
          knn=self.feedforward_config.knn,
          activation=self.feedforward_config.activation,
          dtype=self.dtype,
        )
      case _:
        raise ValueError(f'Unknown feedforward type: {self.feedforward_config.type}')

  def __call__(
    self, input_BxLxD: jt.Float[jt.Array, 'B L D']
  ) -> jt.Float[jt.Array, 'B L D']:
    attn_BxLxD = self.attention_norm(input_BxLxD)
    attn_BxLxD = self.attention(attn_BxLxD)
    attn_BxLxD = input_BxLxD + attn_BxLxD

    ffw_BxLxD = self.feedforward_norm(attn_BxLxD)
    ffw_BxLxD = self.ffw(ffw_BxLxD)

    return attn_BxLxD + ffw_BxLxD


class CausalAttention(nn.Module):
  """Causal attention layer."""

  d_model: int
  d_heads: int
  dtype: str | jnp.dtype

  def setup(self) -> None:
    assert self.d_model % self.d_heads == 0
    n_heads = self.d_model // self.d_heads

    multilinear = partial(
      nn.DenseGeneral,
      axis=-1,
      features=(n_heads, self.d_heads),
      use_bias=False,
      dtype=self.dtype,
    )

    self.query_projection = multilinear(name='query_projection')
    self.key_projection = multilinear(name='key_projection')
    self.value_projection = multilinear(name='value_projection')

    self.query_norm = nn.RMSNorm(dtype=self.dtype)
    self.key_norm = nn.RMSNorm(dtype=self.dtype)

    self.output_projection = nn.DenseGeneral(
      features=self.d_model,
      name='output_projection',
      axis=(-2, -1),
      use_bias=False,
      dtype=self.dtype,
    )

  def __call__(self, x_BxLxD: jt.Float[jt.Array, 'B L D']) -> jt.Float[jt.Array, 'B L D']:
    query_BxLxHxDh = self.query_norm(self.query_projection(x_BxLxD))
    key_BxLxHxDh = self.key_norm(self.key_projection(x_BxLxD))
    value_BxLxHxDh = self.value_projection(x_BxLxD)

    seq_len = query_BxLxHxDh.shape[1]
    sin, cos = rotary.generate_fixed_pos_embedding(self.d_heads, seq_len)
    sin, cos = sin.astype(self.dtype), cos.astype(self.dtype)
    query_BxLxHxDh, key_BxLxHxDh = rotary.apply_rotary_embedding(
      query_BxLxHxDh, key_BxLxHxDh, cos, sin
    )

    query_BxLxHxDh /= self.d_heads**0.5
    attn_BxHxLxL = jnp.einsum('...qhd,...khd->...hqk', query_BxLxHxDh, key_BxLxHxDh)
    attn_BxHxLxL = attn_BxHxLxL.astype(jnp.float32)

    mask_1x1xLxL = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=jnp.bool_))
    _NEG_INF = jnp.finfo(jnp.float32).min
    attn_BxHxLxL = jnp.where(mask_1x1xLxL, attn_BxHxLxL, _NEG_INF)
    attn_BxHxLxL = jax.nn.softmax(attn_BxHxLxL, axis=-1).astype(self.dtype)

    out_BxLxHxDh = jnp.einsum('...hqk,...khd->...qhd', attn_BxHxLxL, value_BxLxHxDh)
    return self.output_projection(out_BxLxHxDh)
