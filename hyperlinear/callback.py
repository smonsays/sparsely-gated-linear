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

import jax
import jax.numpy as jnp

from hyperlinear.data_types import Batch
from hyperlinear.data_types import Metrics
from hyperlinear.training.experiment import Callback
from hyperlinear.training.experiment import CallbackEvent
from hyperlinear.training.experiment import Experiment
from hyperlinear.training.experiment import ExperimentState
from hyperlinear.utils import dict_filter


class ExpertsUsage(Callback):
  def __init__(
    self,
    log_level: int,
    onevent: CallbackEvent,
  ) -> None:
    super().__init__(log_level, onevent)

  def __call__(self, ctx: Experiment, exp_state: ExperimentState) -> Metrics:
    @jax.jit
    def predict_experts_used_per_layer(batch: Batch) -> dict[str, jax.Array]:
      _, variables = ctx.model.apply(
        exp_state.params,
        batch.x,
        rngs={'dropout': exp_state.rng},
        mutable='metrics',
      )
      experts_used = dict_filter(variables, 'experts_used')
      return {f'layer_{i}': x for i, x in enumerate(experts_used)}

    experts_used_list = []
    for batch in ctx.eval_loaders['test']:
      experts_used_list.append(predict_experts_used_per_layer(batch))

    return self.compute_metrics(experts_used_list)

  @functools.partial(jax.jit, static_argnames='self')
  def compute_metrics(self, experts_used_list: list[jax.Array]) -> dict[str, jax.Array]:
    """Compute expert usage metrics across all batches and per sequence."""

    @functools.partial(jnp.vectorize, signature='(n)->()')
    def min_max_ratio(bincount: jax.Array) -> jax.Array:
      """Min-max ratio

      In range [0,1], where 0 indicates low uniformity, 1 indicates high uniformity.
      """
      return jnp.min(bincount) / jnp.max(bincount)

    @functools.partial(jnp.vectorize, signature='(n)->()')
    def coeff_of_variation(bincount: jax.Array) -> jax.Array:
      """Coefficient of variation"""
      return jnp.std(bincount) / jnp.mean(bincount)

    @functools.partial(jnp.vectorize, signature='(n)->()')
    def fraction_nonzero(bincount: jax.Array) -> jax.Array:
      """Fraction of non-zero bins.

      In range [0,1], where 1 indicates all bins have at least one count.
      """
      return jnp.sum(bincount > 0) / len(bincount)

    @functools.partial(jnp.vectorize, signature='(n)->()')
    def normalized_entropy(bincount: jax.Array) -> jax.Array:
      """Normalized entropy

      In range [0,1], where 0 indicates low uniformity, 1 indicates high uniformity.
      """
      probs = bincount / jnp.sum(bincount)
      p_log_p = jnp.where(probs > 0, probs * jnp.log2(probs), jnp.zeros_like(probs))
      return -jnp.sum(p_log_p) / jnp.log2(len(bincount))

    # Concatenate all batches to compute statistics across whole eval dataset
    experts_used = jax.tree.map(lambda *args: jnp.concatenate((args)), *experts_used_list)

    def compute_metrics_per_layer(experts_used: jax.Array) -> dict[str, jax.Array]:
      return dict(
        min_max_ratio_per_sequence=jnp.mean(min_max_ratio(experts_used)),
        min_max_ratio=min_max_ratio(experts_used.flatten()),
        coeff_of_variation_per_sequence=jnp.mean(coeff_of_variation(experts_used)),
        coeff_of_variation=coeff_of_variation(experts_used.flatten()),
        fraction_nonzero_per_sequence=jnp.mean(fraction_nonzero(experts_used)),
        fraction_nonzero=fraction_nonzero(experts_used.flatten()),
        normalized_entropy_per_sequence=jnp.mean(normalized_entropy(experts_used)),
        normalized_entropy=normalized_entropy(experts_used.flatten()),
      )

    metrics_nested = jax.tree.map(compute_metrics_per_layer, experts_used)
    return {
      '_'.join(('experts', key_metric, key_layer)): val
      for (key_layer, metrics_layer) in metrics_nested.items()
      for (key_metric, val) in metrics_layer.items()
    }
