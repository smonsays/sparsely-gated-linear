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

Einsum notation:
  B: batch
  T: context_size
  D: d_model
  F: d_ffw
  C: n_channels
  K: d_key
  S: knn
"""

import functools
import math

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax.linen import dtypes as nn_dtypes
from flax.linen import initializers as nn_initializers


def get_router(
  router_type: str,
  d_model: int,
  n_experts: int,
  d_key: int,
  knn: int,
  n_channels: int,
  dtype: str | jnp.dtype,
) -> nn.Module:
  match router_type:
    case 'dense':
      return DenseRouter(d_model, n_experts, d_key, knn, n_channels, dtype, name='router')
    case 'key_value':
      return KeyValueRouter(
        d_model, n_experts, d_key, knn, n_channels, dtype, name='router'
      )
    case 'product_dense':
      return ProductDenseRouter(
        d_model, n_experts, d_key, knn, n_channels, dtype, name='router'
      )
    case 'product_key_value':
      return ProductKeyValueRouter(
        d_model, n_experts, d_key, knn, n_channels, dtype, name='router'
      )
    case _:
      raise ValueError(f'Unknown router type: {router_type}')


def _experts_bincount(
  expert_indices: jt.Int[jt.Array, 'B T C K'],
  n_experts: int,
) -> jt.Int[jt.Array, 'B F']:
  """Count how often each expert occurs given a set of router indices."""
  indices_flat = einops.rearrange(expert_indices, 'b t c k -> b (t c k)')
  bincount_fn = functools.partial(jnp.bincount, length=n_experts)
  return jax.vmap(bincount_fn)(indices_flat)


class DenseRouter(nn.Module):
  d_model: int
  n_experts: int
  d_key: int
  knn: int
  n_channels: int
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(
    self, x_BxTxD: jt.Float[jt.Array, 'B T D']
  ) -> tuple[jt.Float[jt.Array, 'B T C S'], jt.Int[jt.Array, 'B T C S']]:
    router = nn.Sequential(
      (
        nn.Dense(features=self.d_key, use_bias=False, dtype=self.dtype),
        nn.DenseGeneral(
          (self.n_channels, self.n_experts), use_bias=False, dtype=self.dtype
        ),
      )
    )
    scores, indices = jax.lax.top_k(router(x_BxTxD), self.knn)
    self.sow('metrics', 'experts_used', _experts_bincount(indices, self.n_experts))
    self.sow('intermediates', 'indices', indices)
    self.sow('intermediates', 'scores', scores)

    return (scores, indices)


class KeyValueRouter(nn.Module):
  d_model: int
  n_experts: int
  d_key: int
  knn: int
  n_channels: int
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(
    self, x_BxTxD: jt.Float[jt.Array, 'B T D']
  ) -> tuple[jt.Float[jt.Array, 'B T C S'], jt.Int[jt.Array, 'B T C S']]:
    keys_init = nn_initializers.truncated_normal(stddev=1.0)
    keys = self.param('keys', keys_init, (self.n_channels, self.n_experts, self.d_key))
    (keys,) = nn_dtypes.promote_dtype(keys, dtype=self.dtype)

    queries = nn.Einsum(
      (self.n_channels, self.d_model, self.d_key),
      'btd,hdk->bthk',
      kernel_init=nn_initializers.variance_scaling(
        1.0, 'fan_in', 'truncated_normal', batch_axis=(0,)
      ),
      use_bias=False,
      dtype=self.dtype,
      name='queries',
    )(x_BxTxD)

    similarities = jnp.einsum('bthk,hfk->bthf', queries, keys) / math.sqrt(self.d_key)
    scores, indices = jax.lax.top_k(similarities, k=self.knn)
    self.sow('metrics', 'experts_used', _experts_bincount(indices, self.n_experts))
    self.sow('intermediates', 'indices', indices)
    self.sow('intermediates', 'scores', scores)

    return (scores, indices)


class ProductKeyValueRouter(nn.Module):
  d_model: int
  n_experts: int
  d_key: int
  knn: int
  n_channels: int
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(
    self, x_BxTxD: jt.Float[jt.Array, 'B T D']
  ) -> tuple[jt.Float[jt.Array, 'B T C S'], jt.Int[jt.Array, 'B T C S']]:
    n_keys = int(math.sqrt(self.n_experts))
    if n_keys**2 != self.n_experts:
      raise ValueError(
        'ProductKeyValueRouter assumes that n_experts is a perfect square.'
      )

    if self.knn >= n_keys:
      raise ValueError('ProductKeyValueRouter assumes that knn < n_keys=sqrt(n_experts).')

    keys_init = nn_initializers.truncated_normal(stddev=1.0)
    keys = self.param('keys', keys_init, (2, self.n_channels, n_keys, self.d_key))
    (keys,) = nn_dtypes.promote_dtype(keys, dtype=self.dtype)

    query_init = nn_initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal', batch_axis=(0, 1)
    )
    queries = nn.Einsum(
      shape=(2, self.n_channels, self.d_model, self.d_key),
      einsum_str='btd,phdk->pbthk',
      kernel_init=query_init,
      use_bias=False,
      dtype=self.dtype,
      name='queries',
    )(x_BxTxD)

    # The factor 2 in the normalization comes from the fact that the full similarity
    # scores for all virtual keys are the sum of p=2 parts
    similar = jnp.einsum('pbthk,phfk->pbthf', queries, keys) / math.sqrt(2 * self.d_key)
    scores_parts, indices_parts = jax.lax.top_k(similar, k=self.knn)

    scores_1, scores_2 = scores_parts  # bthk, bthk <- pbthk
    indices_1, indices_2 = indices_parts  # bthk, bthk <- pbthk

    # The similarity scores of the cartesian product of the top-k subkeys
    cartesian_scores = einops.rearrange(
      (
        einops.rearrange(scores_1, 'b t c k -> b t c k 1')
        + einops.rearrange(scores_2, 'b t c k -> b t c 1 k')
      ),
      'b t c k K -> b t c (k K)',
    )

    # The top-k indices of the virtual keys defined by the cartesian product of subkeys
    cartesian_indices = einops.rearrange(
      (
        einops.rearrange(indices_1, 'b t c k -> b t c k 1') * n_keys
        + einops.rearrange(indices_2, 'b t c k -> b t c 1 k')
      ),
      'b t c k K -> b t c (k K)',
    )

    scores, indices_rel_to_topk_scores = jax.lax.top_k(cartesian_scores, k=self.knn)
    indices = jnp.take_along_axis(cartesian_indices, indices_rel_to_topk_scores, axis=-1)
    self.sow('metrics', 'experts_used', _experts_bincount(indices, self.n_experts))
    self.sow('intermediates', 'indices', indices)
    self.sow('intermediates', 'scores', scores)

    return (scores, indices)


class ProductDenseRouter(nn.Module):
  d_model: int
  n_experts: int
  d_key: int
  knn: int
  n_channels: int
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(
    self, x_BxTxD: jt.Float[jt.Array, 'B T D']
  ) -> tuple[jt.Float[jt.Array, 'B T C S'], jt.Int[jt.Array, 'B T C S']]:
    n_keys = int(math.sqrt(self.n_experts))
    if n_keys**2 != self.n_experts:
      raise ValueError('ProductDenseRouter assumes that n_experts is a perfect square.')

    if self.knn >= n_keys:
      raise ValueError('ProductDenseRouter assumes that knn < n_keys=sqrt(n_experts).')

    similar = nn.Sequential(
      (
        nn.Dense(features=self.d_key, use_bias=False, dtype=self.dtype),
        nn.DenseGeneral((2, self.n_channels, n_keys), use_bias=False, dtype=self.dtype),
      )
    )(x_BxTxD)

    scores_parts, indices_parts = jax.lax.top_k(similar, k=self.knn)

    # bthk, bthk <- pbthk
    scores_1, scores_2 = scores_parts[:, :, 0], scores_parts[:, :, 1]
    indices_1, indices_2 = indices_parts[:, :, 0], indices_parts[:, :, 1]

    # The similarity scores of the cartesian product of the top-k subkeys
    cartesian_scores = einops.rearrange(
      (
        einops.rearrange(scores_1, 'b t c k -> b t c k 1')
        + einops.rearrange(scores_2, 'b t c k -> b t c 1 k')
      ),
      'b t c k K -> b t c (k K)',
    )

    # The top-k indices of the virtual keys defined by the cartesian product of subkeys
    cartesian_indices = einops.rearrange(
      (
        einops.rearrange(indices_1, 'b t c k -> b t c k 1') * n_keys
        + einops.rearrange(indices_2, 'b t c k -> b t c 1 k')
      ),
      'b t c k K -> b t c (k K)',
    )

    scores, indices_rel_to_topk_scores = jax.lax.top_k(cartesian_scores, k=self.knn)
    indices = jnp.take_along_axis(cartesian_indices, indices_rel_to_topk_scores, axis=-1)
    self.sow('metrics', 'experts_used', _experts_bincount(indices, self.n_experts))
    self.sow('intermediates', 'indices', indices)
    self.sow('intermediates', 'scores', scores)

    return (scores, indices)


class ParameterEfficientExpertRetrieval(nn.Module):
  d_model: int
  n_experts: int
  d_key: int
  knn: int
  n_channels: int
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(self, x_BxTxD: jt.Float[jt.Array, 'B T D']) -> jt.Float[jt.Array, 'B T D']:
    router = ProductKeyValueRouter(
      self.d_model, self.n_experts, self.d_key, self.knn, self.n_channels, self.dtype
    )
    scores, indices = router(x_BxTxD)

    experts_init = nn_initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal', batch_axis=0
    )
    experts = self.param('experts', experts_init, (2, self.n_experts, self.d_model))

    w_in, w_out = experts[:, indices]
    w_in, w_out = nn_dtypes.promote_dtype(w_in, w_out, dtype=self.dtype)

    z = jnp.einsum('btd,btckd->btck', x_BxTxD, w_in)
    z = jax.nn.gelu(z) * jax.nn.softmax(scores)
    z = jnp.einsum('btck,btckd->btd', z, w_out)

    return z


class SparselyGatedLinear(nn.Module):
  """Sparsely-gated linear layer."""

  d_model: int
  n_experts: int
  d_key: int
  knn: int
  n_channels: int
  router_type: str
  activation: str
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(self, x_BxTxD: jt.Float[jt.Array, 'B T D']) -> jt.Float[jt.Array, 'B T D']:
    router = get_router(
      self.router_type,
      self.d_model,
      self.n_experts,
      self.d_key,
      self.knn,
      self.n_channels,
      self.dtype,
    )
    scores, indices = router(x_BxTxD)

    experts_init = nn_initializers.variance_scaling(
      scale=1.0, mode='fan_in', distribution='truncated_normal', batch_axis=(0, 1)
    )
    # Each channel gets their own set of experts
    experts = self.param(
      'experts', experts_init, (2, self.n_channels, self.n_experts, self.d_model)
    )

    w_in, w_out = experts[:, jnp.arange(self.n_channels)[None, None, :, None], indices]
    w_in, w_out = nn_dtypes.promote_dtype(w_in, w_out, dtype=self.dtype)

    z = jnp.einsum('btd,btckd->btck', x_BxTxD, w_in)
    z = z * scores
    z = getattr(jax.nn, self.activation)(z)
    z = jnp.einsum('btck,btckd->btd', z, w_out)

    return z


class MultilayerPerceptron(nn.Module):
  """Multilayer perceptron."""

  d_model: int
  d_ffw: int
  activation: str
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(self, x_BxLxD: jt.Float[jt.Array, 'B C D']) -> jt.Float[jt.Array, 'B C D']:
    linear = functools.partial(nn.Dense, use_bias=False, dtype=self.dtype)
    x_BxLxF = linear(self.d_ffw)(x_BxLxD)
    x_BxLxF = getattr(jax.nn, self.activation)(x_BxLxF)
    x_BxLxD = linear(self.d_model)(x_BxLxF)

    return x_BxLxD


class GatedLinearUnit(nn.Module):
  """GPT-OSS-style GLU mlp with clamping and rescaling.

  See https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
  """

  d_model: int
  d_ffw: int
  activation: str
  dtype: str | jnp.dtype
  limit: float = 7.0
  alpha: float = 1.702

  @nn.compact
  def __call__(self, x_BxLxD: jt.Float[jt.Array, 'B C D']) -> jt.Float[jt.Array, 'B C D']:
    linear = functools.partial(nn.Dense, use_bias=True, dtype=self.dtype)

    z_BxLx2F = linear(2 * self.d_ffw)(x_BxLxD)

    # glu-gating
    gate_BxLxF, linear_BxLxF = jnp.split(z_BxLx2F, 2, axis=-1)
    gate_BxLxF = jnp.clip(gate_BxLxF, max=self.limit)
    linear_BxLxF = jnp.clip(linear_BxLxF, min=-self.limit, max=self.limit)
    glu_BxLxF = gate_BxLxF * getattr(jax.nn, self.activation)(self.alpha * gate_BxLxF)
    gated_BxLxF = glu_BxLxF * (linear_BxLxF + 1)

    out_BxLxD = linear(self.d_model)(gated_BxLxF)

    return out_BxLxD


def swiglu(
  x: jt.Float[jt.Array, '... F'],
  activation: str = 'sigmoid',
  alpha: float = 1.702,
  limit: float = 7.0,
) -> jt.Float[jt.Array, '...']:
  """SwiGLU activation function with clamping from GPT OSS.

  See https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
  """
  x_glu, x_linear = jnp.split(x, 2, axis=-1)
  x_glu = jnp.clip(x_glu, max=limit)
  x_linear = jnp.clip(x_linear, min=-limit, max=limit)
  out_glu = x_glu * getattr(jax.nn, activation)(alpha * x_glu)
  return out_glu * (x_linear + 1)


class MixtureOfExperts(nn.Module):
  """Mixture of experts feedforward layer as used in GPT OSS.

  See https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
  """

  d_model: int
  d_ffw: int
  n_channels: int  # corresponds to n_experts
  knn: int  # corresponds to experts_per_token
  activation: str
  dtype: str | jnp.dtype

  @nn.compact
  def __call__(self, x_BxTxD: jt.Float[jt.Array, 'B T D']) -> jt.Float[jt.Array, 'B T D']:
    gate_BxTxC = nn.Dense(
      features=self.n_channels,
      use_bias=False,
      dtype=self.dtype,
      name='gate',
    )(x_BxTxD)

    # Get top-k experts
    _, expert_indices = jax.lax.top_k(gate_BxTxC, k=self.knn)
    expert_weights = jax.nn.softmax(gate_BxTxC, axis=-1)

    # Run and mask all experts instead of slower fetching of token-specific experts
    out_BxTxD = jnp.zeros_like(x_BxTxD)
    for expert_idx in range(self.n_channels):
      mask = jnp.any(expert_indices == expert_idx, axis=-1, keepdims=True)
      expert = GatedLinearUnit(self.d_model, self.d_ffw, self.activation, self.dtype)
      out_BxTxD += expert(x_BxTxD) * mask * expert_weights[..., expert_idx, None]

    return out_BxTxD
