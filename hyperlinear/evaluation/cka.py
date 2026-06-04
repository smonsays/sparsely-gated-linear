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
import jaxtyping as jt


@jax.jit
def center_columns(x: jt.Float[jt.Array, 'n d']) -> jt.Float[jt.Array, 'n d']:
  """Centers the columns of a matrix."""
  return x - jnp.mean(x, axis=0, keepdims=True)


@jax.jit
def linear_cka(
  x: jt.Float[jt.Array, 'n d1'],
  y: jt.Float[jt.Array, 'n d2'],
) -> jt.Float[jt.Array, '']:
  """Efficiently compute Linear Centered Kernel Alignment (CKA) in feature space.

  This avoids creating N x N Gram matrices, making it highly efficient when N > d.
  """
  xc = center_columns(x)
  yc = center_columns(y)

  # Cross-covariance norm ||X_c^T Y_c||_F^2
  cross_cov = xc.T @ yc
  numerator = jnp.sum(cross_cov**2)

  # Autocovariance norms ||X_c^T X_c||_F * ||Y_c^T Y_c||_F
  cov_x = xc.T @ xc
  cov_y = yc.T @ yc
  denominator = jnp.sqrt(jnp.sum(cov_x**2) * jnp.sum(cov_y**2))

  return numerator / jnp.maximum(denominator, 1e-12)


@jax.jit
def hsic(
  gram_x: jt.Float[jt.Array, 'n n'],
  gram_y: jt.Float[jt.Array, 'n n'],
) -> jt.Float[jt.Array, '']:
  """Hilbert-Schmidt Independence Criterion for generic Gram matrices."""
  n = gram_x.shape[0]
  h = jnp.eye(n) - 1.0 / n * jnp.ones((n, n))

  # Center the Gram matrices
  kc = h @ gram_x @ h
  lc = h @ gram_y @ h

  # trace(Kc Lc) for symmetric matrices is sum(Kc * Lc)
  numerator = jnp.sum(kc * lc)
  return numerator / (n - 1) ** 2


@jax.jit
def gram_cka(
  gram_x: jt.Float[jt.Array, 'n n'],
  gram_y: jt.Float[jt.Array, 'n n'],
) -> jt.Float[jt.Array, '']:
  """Centered Kernel Alignment for given Gram matrices (e.g., for non-linear kernels)."""
  hsic_xy = hsic(gram_x, gram_y)
  hsic_xx = hsic(gram_x, gram_x)
  hsic_yy = hsic(gram_y, gram_y)
  return hsic_xy / jnp.sqrt(jnp.maximum(hsic_xx * hsic_yy, 1e-12))
