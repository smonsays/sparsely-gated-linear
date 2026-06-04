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

import math
from functools import partial

import jax
import jax.numpy as jnp
from flax import traverse_util as flax_traverse

from hyperlinear import config_classes
from hyperlinear.model import transformer


def _flops_attention(seq_len: int, d_model: int, d_heads: int) -> int:
  """FLOPS of a hyperlinear.transformer.CausalAttn layer"""
  n_heads = d_model // d_heads
  qkv_projection = 3 * seq_len * 2 * d_model * d_model
  attention_scores = n_heads * 2 * seq_len * seq_len * d_heads
  attention_output = n_heads * 2 * seq_len * seq_len * d_heads
  output_projection = seq_len * 2 * d_model * d_model

  return qkv_projection + attention_scores + attention_output + output_projection


def _flops_mlp(seq_len: int, d_model: int, d_ffw: int) -> int:
  """FLOPS of a hyperlinear.model.feedforward.MultilayerPerceptron layer"""
  return seq_len * 2 * 2 * d_model * d_ffw


def _flops_glu(seq_len: int, d_model: int, d_ffw: int) -> int:
  """FLOPS of a hyperlinear.model.feedforward.GatedLinearUnit layer"""
  return seq_len * 3 * 2 * d_model * d_ffw


def _flops_dense_router(
  seq_len: int, d_model: int, n_experts: int, d_key: int, knn: int, n_channels: int
) -> int:
  """FLOPS of a hyperlinear.model.feedforward.DenseRouter"""
  return (2 * d_model * d_key + 2 * d_key * n_experts * n_channels) * seq_len


def _flops_product_key_value_router(
  seq_len: int, d_model: int, n_experts: int, d_key: int, knn: int, n_channels: int
) -> int:
  """FLOPS of a hyperlinear.model.feedforward.ProductKeyValueRouter"""
  n_keys = int(math.sqrt(n_experts))
  return (
    2 * 2 * n_channels * d_key * d_model + 2 * 2 * n_channels * n_keys * d_key
  ) * seq_len


def _flops_product_dense_router(
  seq_len: int, d_model: int, n_experts: int, d_key: int, knn: int, n_channels: int
) -> int:
  """FLOPS of a hyperlinear.model.feedforward.ProductDenseRouter"""
  n_keys = int(math.sqrt(n_experts))
  return (2 * d_key * d_model + 2 * 2 * n_channels * n_keys * d_key) * seq_len


def _flops_router(
  router_type: str,
  seq_len: int,
  d_model: int,
  n_experts: int,
  d_key: int,
  knn: int,
  n_channels: int,
) -> int:
  match router_type:
    case 'product_key_value':
      return _flops_product_key_value_router(
        seq_len, d_model, n_experts, d_key, knn, n_channels
      )
    case 'product_dense':
      return _flops_product_dense_router(
        seq_len, d_model, n_experts, d_key, knn, n_channels
      )
    case 'dense':
      return _flops_dense_router(seq_len, d_model, n_experts, d_key, knn, n_channels)
    case _:
      raise NotImplementedError


def _flops_gated_linear(
  seq_len: int,
  d_model: int,
  n_experts: int,
  d_key: int,
  knn: int,
  n_channels: int,
  router_type: str,
) -> int:
  """FLOPS of a hyperlinear.model.feedforward.SparselyGatedLinear layer"""
  router = _flops_router(router_type, seq_len, d_model, n_experts, d_key, knn, n_channels)
  experts = seq_len * 2 * 2 * knn * n_channels * d_model
  return router + experts


def _flops_peer(
  seq_len: int,
  d_model: int,
  n_experts: int,
  d_key: int,
  knn: int,
  n_channels: int,
) -> int:
  """FLOPS of a hyperlinear.model.feedforward.ParameterEfficientExpertRetrieval layer"""
  router = _flops_router(
    'product_key_value', seq_len, d_model, n_experts, d_key, knn, n_channels
  )
  experts = seq_len * 2 * 2 * knn * n_channels * d_model
  return router + experts


def _flops_moe(seq_len: int, d_model: int, d_ffw: int, n_channels: int, knn: int) -> int:
  """FLOPS of a hyperlinear.model.feedforward.MixtureOfExperts layer"""
  gate = seq_len * 2 * d_model * n_channels
  w_in = seq_len * knn * 2 * d_model * (d_ffw * 2)
  w_out = seq_len * knn * 2 * d_ffw * d_model

  return gate + w_in + w_out


def count_forward_flops(model_config: config_classes.ModelConfig, seq_len: int) -> int:
  """FLOPS of a transformer forward pass (only counting matmuls)"""
  attn = _flops_attention(seq_len, model_config.d_model, model_config.d_heads)

  match model_config.feedforward_config.type:
    case 'mlp':
      ffw = _flops_mlp(
        seq_len, model_config.d_model, model_config.feedforward_config.d_ffw
      )
    case 'sparsely_gated_linear':
      ffw = _flops_gated_linear(
        seq_len,
        model_config.d_model,
        model_config.feedforward_config.d_ffw,
        model_config.feedforward_config.d_key,
        model_config.feedforward_config.knn,
        model_config.feedforward_config.n_channels,
        model_config.feedforward_config.router_type,
      )
    case 'peer':
      ffw = _flops_peer(
        seq_len,
        model_config.d_model,
        model_config.feedforward_config.d_ffw,
        model_config.feedforward_config.d_key,
        model_config.feedforward_config.knn,
        model_config.feedforward_config.n_channels,
      )
    case 'glu':
      ffw = _flops_glu(
        seq_len, model_config.d_model, model_config.feedforward_config.d_ffw
      )
    case 'moe':
      ffw = _flops_moe(
        seq_len,
        model_config.d_model,
        model_config.feedforward_config.d_ffw,
        model_config.feedforward_config.n_channels,
        model_config.feedforward_config.knn,
      )
    case _:
      raise ValueError(
        f'Unknown feedforward type: {model_config.feedforward_config.type}'
      )

  layer_flops = attn + ffw

  return layer_flops * model_config.n_layers


def count_parameters(model_config: config_classes.ModelConfig, vocab_size: int) -> int:
  with jax.default_device(jax.devices('cpu')[0]):
    sequence_model = transformer.Transformer(
      n_vocab=vocab_size,
      d_model=model_config.d_model,
      d_heads=model_config.d_heads,
      n_layers=model_config.n_layers,
      feedforward_config=model_config.feedforward_config,
      dtype=model_config.dtype,
      remat=model_config.remat,
    )
    rng_params, rng_dropout = jax.random.split(jax.random.key(0), 2)
    x = jnp.zeros((1, 1), dtype=jnp.int32)
    init_rngs = {'params': rng_params, 'dropout': rng_dropout, 'target': rng_params}
    params = jax.eval_shape(partial(sequence_model.init, mutable='params'), init_rngs, x)

    params_flat = flax_traverse.flatten_dict(params['params'])

    return sum(math.prod(p.shape) for p in params_flat.values())
