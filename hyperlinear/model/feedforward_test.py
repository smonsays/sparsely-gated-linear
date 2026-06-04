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

import functools
import math

import chex
import einops
import jax
import jax.numpy as jnp
from absl.testing import absltest

from hyperlinear.model import feedforward
from hyperlinear.utils import dict_filter

jax.config.parse_flags_with_absl()
jax.config.update('jax_numpy_rank_promotion', 'raise')


class FeedforwardTest(chex.TestCase):
  rng: chex.PRNGKey = jax.random.key(1)

  @chex.variants(with_jit=True, without_jit=True)
  def test_multilayer_perceptron(self) -> None:
    batch_size = 3
    context_size = 7
    d_model = 13
    d_ffw = 17
    activation = 'gelu'
    dtype = jnp.bfloat16

    model = feedforward.MultilayerPerceptron(d_model, d_ffw, activation, dtype)
    x_BxTxD = jnp.ones((batch_size, context_size, d_model), dtype=dtype)
    params = model.init(self.rng, x_BxTxD)
    out_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(out_BxTxD, (batch_size, context_size, d_model))
    chex.assert_type(out_BxTxD, dtype)
    chex.assert_tree_all_finite(out_BxTxD)

  @chex.variants(with_jit=True, without_jit=True)
  def test_swiglu(self) -> None:
    batch_size = 3
    context_size = 7
    d_model = 13
    d_ffw = 17
    activation = 'swish'
    dtype = jnp.bfloat16

    model = feedforward.GatedLinearUnit(d_model, d_ffw, activation, dtype)
    x_BxTxD = jnp.ones((batch_size, context_size, d_model), dtype=dtype)
    params = model.init(self.rng, x_BxTxD)
    out_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(out_BxTxD, (batch_size, context_size, d_model))
    chex.assert_type(out_BxTxD, dtype)
    chex.assert_tree_all_finite(out_BxTxD)

  @chex.variants(with_jit=True, without_jit=True)
  def test_dense_router(self) -> None:
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 9
    d_model = 13
    d_ffw = 17
    dtype = jnp.bfloat16

    model = feedforward.DenseRouter(d_model, d_ffw, d_key, knn, n_channels, dtype)
    x_BxTxD = jnp.ones((batch_size, context_size, d_model), dtype=dtype)
    params = model.init(self.rng, x_BxTxD)
    scores_BxTxD, indeces_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(scores_BxTxD, (batch_size, context_size, n_channels, knn))
    chex.assert_shape(indeces_BxTxD, (batch_size, context_size, n_channels, knn))

    chex.assert_type(scores_BxTxD, dtype)
    chex.assert_type(indeces_BxTxD, jnp.int32)
    chex.assert_tree_all_finite((scores_BxTxD, indeces_BxTxD))

  @chex.variants(with_jit=True, without_jit=True)
  def test_key_value_router(self) -> None:
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 11
    d_model = 128
    d_ffw = 17
    dtype = jnp.bfloat16

    rng_x, rng_params = jax.random.split(self.rng, 2)
    model = feedforward.KeyValueRouter(d_model, d_ffw, d_key, knn, n_channels, dtype)
    x_BxTxD = jax.random.normal(rng_x, (batch_size, context_size, d_model), dtype=dtype)
    params = model.init(rng_params, x_BxTxD)
    scores_BxTxD, indeces_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(scores_BxTxD, (batch_size, context_size, n_channels, knn))
    chex.assert_shape(indeces_BxTxD, (batch_size, context_size, n_channels, knn))

    chex.assert_type(scores_BxTxD, dtype)
    chex.assert_type(indeces_BxTxD, jnp.int32)
    chex.assert_tree_all_finite((scores_BxTxD, indeces_BxTxD))

  @chex.variants(with_jit=True, without_jit=True)
  def test_product_key_value_router(self) -> None:
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 11
    d_model = 128
    d_ffw = 16
    dtype = jnp.bfloat16

    rng_x, rng_params = jax.random.split(self.rng, 2)
    model = feedforward.ProductKeyValueRouter(
      d_model, d_ffw, d_key, knn, n_channels, dtype
    )
    x_BxTxD = jax.random.normal(rng_x, (batch_size, context_size, d_model), dtype=dtype)
    params = model.init(rng_params, x_BxTxD)
    scores_BxTxD, indeces_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(scores_BxTxD, (batch_size, context_size, n_channels, knn))
    chex.assert_shape(indeces_BxTxD, (batch_size, context_size, n_channels, knn))

    chex.assert_type(scores_BxTxD, dtype)
    chex.assert_type(indeces_BxTxD, jnp.int32)
    chex.assert_tree_all_finite((scores_BxTxD, indeces_BxTxD))

  @chex.variants(with_jit=True, without_jit=True)
  def test_product_dense_router(self) -> None:
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 11
    d_model = 128
    d_ffw = 16
    dtype = jnp.bfloat16

    rng_x, rng_params = jax.random.split(self.rng, 2)
    model = feedforward.ProductDenseRouter(d_model, d_ffw, d_key, knn, n_channels, dtype)
    x_BxTxD = jax.random.normal(rng_x, (batch_size, context_size, d_model), dtype=dtype)
    params = model.init(rng_params, x_BxTxD)
    scores_BxTxD, indeces_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(scores_BxTxD, (batch_size, context_size, n_channels, knn))
    chex.assert_shape(indeces_BxTxD, (batch_size, context_size, n_channels, knn))

    chex.assert_type(scores_BxTxD, dtype)
    chex.assert_type(indeces_BxTxD, jnp.int32)
    chex.assert_tree_all_finite((scores_BxTxD, indeces_BxTxD))

  def test_product_key_value_router_equals_key_value_router(self) -> None:
    """Test that product-key value router and key-value router give identical results.

    This equality holds for the case where the virtual keys defined by the cartesian
    product of the subkeys of the product-key router equal the full keys of the key-value
    router.
    """
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 11
    d_model = 128
    d_ffw = 16
    dtype = jnp.float32
    n_keys = int(math.sqrt(d_ffw))

    rng_x, rng_params = jax.random.split(self.rng, 2)
    pkv = feedforward.ProductKeyValueRouter(d_model, d_ffw, d_key, knn, n_channels, dtype)
    kv = feedforward.KeyValueRouter(d_model, d_ffw, 2 * d_key, knn, n_channels, dtype)

    x_BxTxD = jax.random.normal(rng_x, (batch_size, context_size, d_model), dtype=dtype)
    params_pkv = pkv.init(rng_params, x_BxTxD)

    keys_1, keys_2 = params_pkv['params']['keys']
    product_keys = jnp.concatenate(
      (
        einops.repeat(keys_1, 'h n k -> h n N k', N=n_keys),
        einops.repeat(keys_2, 'h n k -> h N n k', N=n_keys),
      ),
      axis=-1,
    )
    product_keys = einops.rearrange(product_keys, 'h N n K -> h (N n) K')
    queries_1, queries_2 = params_pkv['params']['queries']['kernel']
    queries_cat = jnp.concatenate((queries_1, queries_2), axis=-1)
    params_kv = dict(params=dict(keys=product_keys, queries=dict(kernel=queries_cat)))

    scores_BxTxD_pkv, indeces_BxTxD_pkv = pkv.apply(params_pkv, x_BxTxD)
    scores_BxTxD_kv, indeces_BxTxD_kv = kv.apply(params_kv, x_BxTxD)

    chex.assert_trees_all_close(scores_BxTxD_pkv, scores_BxTxD_kv, rtol=1e-5)
    chex.assert_trees_all_equal(indeces_BxTxD_pkv, indeces_BxTxD_kv)

  @chex.variants(with_jit=True, without_jit=True)
  def test_peer(self) -> None:
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 11
    d_model = 128
    d_ffw = 16
    dtype = jnp.bfloat16

    model = feedforward.ParameterEfficientExpertRetrieval(
      d_model, d_ffw, d_key, knn, n_channels, dtype
    )
    x_BxTxD = jnp.ones((batch_size, context_size, d_model), dtype=dtype)
    params = model.init(self.rng, x_BxTxD)
    out_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(out_BxTxD, (batch_size, context_size, d_model))
    chex.assert_type(out_BxTxD, dtype)
    chex.assert_tree_all_finite(out_BxTxD)

  @chex.variants(with_jit=True, without_jit=True)
  def test_gated_linear(self) -> None:
    batch_size = 4
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 11
    d_model = 128
    d_ffw = 16
    router_type = 'product_dense'
    activation = 'identity'
    dtype = jnp.bfloat16

    model = feedforward.SparselyGatedLinear(
      d_model=d_model,
      n_experts=d_ffw,
      d_key=d_key,
      knn=knn,
      n_channels=n_channels,
      router_type=router_type,
      activation=activation,
      dtype=dtype,
    )
    x_BxTxD = jnp.ones((batch_size, context_size, d_model), dtype=dtype)
    params = model.init(self.rng, x_BxTxD)
    model_apply = functools.partial(model.apply, mutable=['metrics'])
    out, state = self.variant(model_apply)(
      params,
      x_BxTxD,
    )

    chex.assert_shape(dict_filter(state, 'experts_used')[0], (batch_size, d_ffw))
    chex.assert_shape(out, (batch_size, context_size, d_model))
    chex.assert_type(out, dtype)
    chex.assert_tree_all_finite(out)

  def test_gated_linear_grads(self) -> None:
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_key = 11
    d_model = 128
    d_ffw = 16
    router_type = 'product_dense'
    activation = 'identity'
    dtype = jnp.bfloat16

    model = feedforward.SparselyGatedLinear(
      d_model=d_model,
      n_experts=d_ffw,
      d_key=d_key,
      knn=knn,
      n_channels=n_channels,
      router_type=router_type,
      activation=activation,
      dtype=dtype,
    )

    x_BxTxD = jnp.ones((batch_size, context_size, d_model), dtype=dtype)
    params = model.init(self.rng, x_BxTxD, mutable='params')

    grads = jax.grad(lambda p: jnp.sum(model.apply(p, x_BxTxD)))(params)
    chex.assert_trees_all_equal_shapes(params, grads)

    for g in jax.tree.flatten(grads)[0]:
      assert not jnp.allclose(g, 0)

  @chex.variants(with_jit=True, without_jit=True)
  def test_moe(self) -> None:
    batch_size = 1
    context_size = 7
    knn = 3
    n_channels = 5
    d_model = 128
    d_ffw = 16
    activation = 'sigmoid'
    dtype = jnp.bfloat16

    model = feedforward.MixtureOfExperts(
      d_model, d_ffw, n_channels, knn, activation, dtype
    )
    x_BxTxD = jnp.ones((batch_size, context_size, d_model), dtype=dtype)
    params = model.init(self.rng, x_BxTxD)
    out_BxTxD = self.variant(model.apply)(params, x_BxTxD)

    chex.assert_shape(out_BxTxD, (batch_size, context_size, d_model))
    chex.assert_type(out_BxTxD, dtype)
    chex.assert_tree_all_finite(out_BxTxD)


if __name__ == '__main__':
  absltest.main()
