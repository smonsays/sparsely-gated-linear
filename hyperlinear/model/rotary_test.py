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

from hyperlinear.model import rotary

jax.config.parse_flags_with_absl()
jax.config.update('jax_numpy_rank_promotion', 'raise')


class RotaryTest(chex.TestCase):
  rng: chex.PRNGKey = jax.random.key(42)

  def test_rotate_half(self) -> None:
    """Test that rotate_half correctly rotates the last dimension."""
    # Test with even dimension
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    result = rotary.rotate_half(x)
    expected = jnp.array([[-3.0, -4.0, 1.0, 2.0]])
    chex.assert_trees_all_close(result, expected)

    # Test with 3D tensor
    x_3d = jnp.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    result_3d = rotary.rotate_half(x_3d)
    expected_3d = jnp.array([[[-3.0, -4.0, 1.0, 2.0], [-7.0, -8.0, 5.0, 6.0]]])
    chex.assert_trees_all_close(result_3d, expected_3d)

  def test_generate_fixed_pos_embedding_shapes(self) -> None:
    """Test that generate_fixed_pos_embedding returns correct shapes."""
    features = 64
    length = 128

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(features, length)

    chex.assert_shape(sin_emb, (length, features))
    chex.assert_shape(cos_emb, (length, features))
    chex.assert_type(sin_emb, jnp.float32)
    chex.assert_type(cos_emb, jnp.float32)
    chex.assert_tree_all_finite((sin_emb, cos_emb))

  def test_generate_fixed_pos_embedding_properties(self) -> None:
    """Test mathematical properties of generated embeddings."""
    features = 32
    length = 64

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(features, length)

    # Test that sin^2 + cos^2 = 1 for each position and feature
    sin_cos_sum = sin_emb**2 + cos_emb**2
    expected_ones = jnp.ones_like(sin_cos_sum)
    chex.assert_trees_all_close(sin_cos_sum, expected_ones, rtol=1e-6)

    # Test that values are in valid range [-1, 1]
    assert jnp.all(sin_emb >= -1.0) and jnp.all(sin_emb <= 1.0)
    assert jnp.all(cos_emb >= -1.0) and jnp.all(cos_emb <= 1.0)

  def test_generate_fixed_pos_embedding_parameters(self) -> None:
    """Test generate_fixed_pos_embedding with different parameters."""
    features = 16
    length = 32
    min_timescale = 2.0
    max_timescale = 5000.0

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(
      features, length, min_timescale, max_timescale
    )

    chex.assert_shape(sin_emb, (length, features))
    chex.assert_shape(cos_emb, (length, features))
    chex.assert_tree_all_finite((sin_emb, cos_emb))

  @chex.variants(with_jit=True, without_jit=True)
  def test_apply_rotary_embedding_shapes(self) -> None:
    """Test that apply_rotary_embedding returns correct shapes."""
    batch_size = 2
    q_len = 5
    k_len = 7
    n_heads = 4
    d_model = 32
    max_len = 10

    # Generate test data
    rng_q, rng_k = jax.random.split(self.rng, 2)
    q = jax.random.normal(rng_q, (batch_size, q_len, n_heads, d_model))
    k = jax.random.normal(rng_k, (batch_size, k_len, n_heads, d_model))

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    out_q, out_k = self.variant(rotary.apply_rotary_embedding)(q, k, cos_emb, sin_emb)

    chex.assert_shape(out_q, (batch_size, q_len, n_heads, d_model))
    chex.assert_shape(out_k, (batch_size, k_len, n_heads, d_model))
    chex.assert_tree_all_finite((out_q, out_k))

  @chex.variants(with_jit=True, without_jit=True)
  def test_apply_rotary_embedding_multiquery(self) -> None:
    """Test apply_rotary_embedding with multi-query attention (3D k tensor)."""
    batch_size = 2
    q_len = 5
    k_len = 7
    n_heads = 4
    d_model = 32
    max_len = 10

    # Generate test data
    rng_q, rng_k = jax.random.split(self.rng, 2)
    q = jax.random.normal(rng_q, (batch_size, q_len, n_heads, d_model))
    k = jax.random.normal(rng_k, (batch_size, k_len, d_model))  # 3D for multiquery

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    out_q, out_k = self.variant(rotary.apply_rotary_embedding)(q, k, cos_emb, sin_emb)

    chex.assert_shape(out_q, (batch_size, q_len, n_heads, d_model))
    chex.assert_shape(out_k, (batch_size, k_len, d_model))  # Should remain 3D
    chex.assert_tree_all_finite((out_q, out_k))

  def test_apply_rotary_embedding_decode_mode_with_jit(self) -> None:
    """Test apply_rotary_embedding in decode mode with rotary_index (JIT)."""
    batch_size = 2
    q_len = 1  # Single token for decode
    k_len = 5
    n_heads = 4
    d_model = 32
    max_len = 10

    # Generate test data
    rng_q, rng_k, rng_idx = jax.random.split(self.rng, 3)
    q = jax.random.normal(rng_q, (batch_size, q_len, n_heads, d_model))
    k = jax.random.normal(rng_k, (batch_size, k_len, n_heads, d_model))
    rotary_index = jax.random.randint(rng_idx, (batch_size,), 0, max_len)

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    # JIT compile with static decode=True
    jitted_fn = jax.jit(rotary.apply_rotary_embedding, static_argnums=(4,))
    out_q, out_k = jitted_fn(q, k, cos_emb, sin_emb, True, rotary_index)

    chex.assert_shape(out_q, (batch_size, q_len, n_heads, d_model))
    chex.assert_shape(out_k, (batch_size, k_len, n_heads, d_model))
    chex.assert_tree_all_finite((out_q, out_k))

  def test_apply_rotary_embedding_decode_mode_without_jit(self) -> None:
    """Test apply_rotary_embedding in decode mode with rotary_index (no JIT)."""
    batch_size = 2
    q_len = 1  # Single token for decode
    k_len = 5
    n_heads = 4
    d_model = 32
    max_len = 10

    # Generate test data
    rng_q, rng_k, rng_idx = jax.random.split(self.rng, 3)
    q = jax.random.normal(rng_q, (batch_size, q_len, n_heads, d_model))
    k = jax.random.normal(rng_k, (batch_size, k_len, n_heads, d_model))
    rotary_index = jax.random.randint(rng_idx, (batch_size,), 0, max_len)

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    out_q, out_k = rotary.apply_rotary_embedding(
      q, k, cos_emb, sin_emb, decode=True, rotary_index=rotary_index
    )

    chex.assert_shape(out_q, (batch_size, q_len, n_heads, d_model))
    chex.assert_shape(out_k, (batch_size, k_len, n_heads, d_model))
    chex.assert_tree_all_finite((out_q, out_k))

  def test_apply_rotary_embedding_preserves_magnitude(self) -> None:
    """Test that rotary embedding preserves the magnitude of vectors."""
    batch_size = 1
    seq_len = 4
    n_heads = 2
    d_model = 8
    max_len = 10

    # Generate test data
    rng_q, rng_k = jax.random.split(self.rng, 2)
    q = jax.random.normal(rng_q, (batch_size, seq_len, n_heads, d_model))
    k = jax.random.normal(rng_k, (batch_size, seq_len, n_heads, d_model))

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    # Compute magnitudes before rotation
    q_mag_before = jnp.linalg.norm(q, axis=-1)
    k_mag_before = jnp.linalg.norm(k, axis=-1)

    out_q, out_k = rotary.apply_rotary_embedding(q, k, cos_emb, sin_emb)

    # Compute magnitudes after rotation
    q_mag_after = jnp.linalg.norm(out_q, axis=-1)
    k_mag_after = jnp.linalg.norm(out_k, axis=-1)

    # Magnitudes should be preserved (within numerical precision)
    chex.assert_trees_all_close(q_mag_before, q_mag_after, rtol=1e-6)
    chex.assert_trees_all_close(k_mag_before, k_mag_after, rtol=1e-6)

  def test_apply_rotary_embedding_dimension_mismatch_error(self) -> None:
    """Test that dimension mismatches raise appropriate errors."""
    batch_size = 2
    q_len = 5
    k_len = 7
    n_heads = 4
    d_model = 32
    max_len = 10

    rng_q, rng_k = jax.random.split(self.rng, 2)
    q = jax.random.normal(rng_q, (batch_size, q_len, n_heads, d_model))

    # Create k with different batch size
    k_wrong_batch = jax.random.normal(rng_k, (batch_size + 1, k_len, n_heads, d_model))

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    with self.assertRaises(AssertionError):
      rotary.apply_rotary_embedding(q, k_wrong_batch, cos_emb, sin_emb)

    # Create k with different d_model
    k_wrong_d = jax.random.normal(rng_k, (batch_size, k_len, n_heads, d_model + 1))

    with self.assertRaises(AssertionError):
      rotary.apply_rotary_embedding(q, k_wrong_d, cos_emb, sin_emb)

  def test_rotary_embedding_equivariance(self) -> None:
    """Test that rotary embeddings are equivariant to sequence shifts."""
    batch_size = 1
    seq_len = 6
    n_heads = 2
    d_model = 16
    max_len = 20
    shift = 3

    # Generate test data
    rng_q, rng_k = jax.random.split(self.rng, 2)
    q = jax.random.normal(rng_q, (batch_size, seq_len, n_heads, d_model))
    k = jax.random.normal(rng_k, (batch_size, seq_len, n_heads, d_model))

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    # Apply rotary embedding at original positions
    out_q1, out_k1 = rotary.apply_rotary_embedding(q, k, cos_emb, sin_emb)

    # Apply rotary embedding at shifted positions
    cos_shifted = cos_emb[shift : shift + seq_len]
    sin_shifted = sin_emb[shift : shift + seq_len]
    out_q2, out_k2 = rotary.apply_rotary_embedding(q, k, cos_shifted, sin_shifted)

    # The relative attention scores should be preserved
    # Compute attention scores for both cases
    scores1 = jnp.einsum('bqhd,bkhd->bqkh', out_q1, out_k1)
    scores2 = jnp.einsum('bqhd,bkhd->bqkh', out_q2, out_k2)

    # Scores should be equal since rotary embedding preserves relative positions
    chex.assert_trees_all_close(scores1, scores2, rtol=1e-5)

  def test_apply_rotary_embedding_parallel_vs_sequential(self) -> None:
    """Test that parallel and sequential rotary embeddings are equivalent."""
    batch_size = 2
    seq_len = 5
    n_heads = 4
    d_model = 32
    max_len = 10

    # Generate test data
    rng_q, rng_k = jax.random.split(self.rng, 2)
    query = jax.random.normal(rng_q, (batch_size, seq_len, n_heads, d_model))
    key = jax.random.normal(rng_k, (batch_size, seq_len, n_heads, d_model))

    sin_emb, cos_emb = rotary.generate_fixed_pos_embedding(d_model, max_len)

    # Parallel application
    out_query_parallel, out_key_parallel = rotary.apply_rotary_embedding(
      query, key, cos_emb, sin_emb
    )

    # Sequential application
    out_query_sequential = []
    out_key_sequential = []
    for i in range(seq_len):
      rotary_index = jnp.full((batch_size,), i, dtype=jnp.int32)
      out_query, out_key = rotary.apply_rotary_embedding(
        query[:, i : i + 1],
        key[:, i : i + 1],
        cos_emb,
        sin_emb,
        decode=True,
        rotary_index=rotary_index,
      )
      out_query_sequential.append(out_query)
      out_key_sequential.append(out_key)

    out_query_sequential = jnp.concatenate(out_query_sequential, axis=1)
    out_key_sequential = jnp.concatenate(out_key_sequential, axis=1)

    chex.assert_trees_all_close(out_query_parallel, out_query_sequential)
    chex.assert_trees_all_close(out_key_parallel, out_key_sequential)


if __name__ == '__main__':
  absltest.main()
