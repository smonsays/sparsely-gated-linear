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

from hyperlinear.evaluation import cka


class CkaTest(absltest.TestCase):
  def setUp(self) -> None:
    super().setUp()
    self.rng = jax.random.PRNGKey(42)
    self.n_samples = 100
    self.d1 = 20
    self.d2 = 30

  def test_linear_cka_identical(self) -> None:
    """Test that CKA between a matrix and itself is 1.0."""
    x = jax.random.normal(self.rng, (self.n_samples, self.d1))
    cka_val = cka.linear_cka(x, x)
    self.assertTrue(jnp.allclose(cka_val, 1.0, atol=1e-5))

  def test_linear_cka_scale_invariance(self) -> None:
    """Test that CKA is invariant to isotropic scaling."""
    k1, k2 = jax.random.split(self.rng)
    x = jax.random.normal(k1, (self.n_samples, self.d1))
    y = jax.random.normal(k2, (self.n_samples, self.d2))

    base_cka = cka.linear_cka(x, y)
    scaled_cka = cka.linear_cka(x * 5.0, y * -0.3)
    self.assertTrue(jnp.allclose(base_cka, scaled_cka, atol=1e-5))

  def test_linear_cka_matches_gram_cka(self) -> None:
    """
    Test that the efficient feature-space linear CKA matches the
    Gram-matrix based CKA mathematically.
    """
    k1, k2 = jax.random.split(self.rng)
    x = jax.random.normal(k1, (self.n_samples, self.d1))
    y = jax.random.normal(k2, (self.n_samples, self.d2))

    # Efficient feature-space linear CKA
    lin_cka = cka.linear_cka(x, y)

    # Generic Gram-matrix CKA (using linear kernels K = X X^T, L = Y Y^T)
    gr_cka = cka.gram_cka(x @ x.T, y @ y.T)

    self.assertTrue(jnp.allclose(lin_cka, gr_cka, atol=1e-5))


if __name__ == '__main__':
  absltest.main()
