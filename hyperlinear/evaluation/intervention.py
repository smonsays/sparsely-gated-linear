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

from typing import Any
from typing import Callable

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax import struct

from hyperlinear.model.transformer import Transformer
from hyperlinear.training.loss import shift_right
from hyperlinear.utils import dict_filter


@struct.dataclass
class InterventionBatch:
  prefix_clean: jt.Int[jt.ArrayLike, 'B T']
  prefix_counterfactual: jt.Int[jt.ArrayLike, 'B T']
  answers_clean: jt.Int[jt.ArrayLike, 'B A']
  answers_counterfactual: jt.Int[jt.ArrayLike, 'B A']


@struct.dataclass
class InterventionMetrics:
  logitdiff_clean: jt.Float[jt.Array, ' B']
  logitdiff_counterfactual: jt.Float[jt.Array, ' B']
  logitdiff_intervened: jt.Float[jt.Array, ' B']

  recovered_effect: jt.Float[jt.Array, ' B']
  rank_flip: jt.Float[jt.Array, ' B']

  correct_clean: jt.Float[jt.Array, ' B']
  correct_counterfactual: jt.Float[jt.Array, ' B']


@struct.dataclass
class RouterRecord:
  prefix: jt.Int[jt.Array, 'B T']
  answer: jt.Int[jt.Array, 'B A']
  score: jt.Float[jt.Array, 'B T L C K']
  neuron: jt.Int[jt.Array, 'B T L C K']


@struct.dataclass
class InterventionRecord:
  router_clean: RouterRecord
  router_counterfactual: RouterRecord
  metrics: InterventionMetrics


def calculate_indirect_effect(
  logits_clean: jt.Float[jt.Array, 'B V'],
  logits_counterfactual: jt.Float[jt.Array, 'B V'],
  logits_intervened: jt.Float[jt.Array, 'B V'],
  batch: InterventionBatch,
) -> InterventionMetrics:
  def logit_diff(logits: jt.Float[jt.Array, 'B V']) -> jt.Float[jt.Array, ' B']:
    ans_clean_logits = jax.nn.logsumexp(
      jnp.take_along_axis(logits, batch.answers_clean, axis=-1), axis=-1
    )
    ans_cf_logits = jax.nn.logsumexp(
      jnp.take_along_axis(logits, batch.answers_counterfactual, axis=-1), axis=-1
    )
    return ans_clean_logits - ans_cf_logits

  diff_clean = logit_diff(logits_clean)
  diff_counterfactual = logit_diff(logits_counterfactual)
  diff_intervened = logit_diff(logits_intervened)

  recovered_effect = (diff_clean - diff_intervened) / (
    diff_clean - diff_counterfactual + 1e-8
  )

  ans_clean_logits_intervened = jax.nn.logsumexp(
    jnp.take_along_axis(logits_intervened, batch.answers_clean, axis=-1), axis=-1
  )
  ans_cf_logits_intervened = jax.nn.logsumexp(
    jnp.take_along_axis(logits_intervened, batch.answers_counterfactual, axis=-1),
    axis=-1,
  )
  rank_flip = (ans_cf_logits_intervened > ans_clean_logits_intervened).astype(jnp.float32)

  correct_clean = (diff_clean > 0).astype(jnp.float32)
  correct_counterfactual = (diff_counterfactual < 0).astype(jnp.float32)

  return InterventionMetrics(
    logitdiff_clean=diff_clean,
    logitdiff_counterfactual=diff_counterfactual,
    logitdiff_intervened=diff_intervened,
    recovered_effect=recovered_effect,
    rank_flip=rank_flip,
    correct_clean=correct_clean,
    correct_counterfactual=correct_counterfactual,
  )


@struct.dataclass
class InterventionRecorder:
  model: Transformer = struct.field(pytree_node=False)
  params: dict
  layers_to_patch: tuple[int, ...] = struct.field(pytree_node=False)
  position_to_patch: int
  position_to_measure: int

  @jax.jit
  def __call__(self, batch: InterventionBatch) -> InterventionRecord:
    out_clean, mutables_clean = self.model.apply(
      self.params, batch.prefix_clean, mutable=['intermediates']
    )
    out_counterfactual, mutables_counterfactual = self.model.apply(
      self.params, batch.prefix_counterfactual, mutable=['intermediates']
    )

    hidden_cf = mutables_counterfactual['intermediates']
    hidden_clean = mutables_clean['intermediates']
    gatings_patched = {
      f'blocks_{layer}': (
        hidden_cf[f'blocks_{layer}']['ffw']['router']['scores'][0],
        hidden_cf[f'blocks_{layer}']['ffw']['router']['indices'][0],
      )
      for layer in self.layers_to_patch
    }

    router_clean = RouterRecord(
      prefix=batch.prefix_clean,
      answer=batch.answers_clean,
      score=jnp.stack(dict_filter(hidden_clean, ['router', 'scores']), axis=2),
      neuron=jnp.stack(dict_filter(hidden_clean, ['router', 'indices']), axis=2),
    )
    router_counterfactual = RouterRecord(
      prefix=batch.prefix_counterfactual,
      answer=batch.answers_counterfactual,
      score=jnp.stack(dict_filter(hidden_cf, ['router', 'scores']), axis=2),
      neuron=jnp.stack(dict_filter(hidden_cf, ['router', 'indices']), axis=2),
    )

    def patch_interceptor(
      next_fun: Callable[..., Any],
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      context: Any,
    ) -> Any:
      """Patch router scores and indices for selected layers and postion."""
      if context.method_name == '__call__' and context.module.name == 'router':
        block_name = context.module.parent.parent.name
        if block_name in gatings_patched:
          scores_clean, indices_clean = next_fun(*args, **kwargs)
          scores_patched, indices_patched = gatings_patched[block_name]

          pos = self.position_to_patch
          scores_new = scores_clean.at[:, pos].set(scores_patched[:, pos])
          indices_new = indices_clean.at[:, pos].set(indices_patched[:, pos])

          return scores_new, indices_new

      return next_fun(*args, **kwargs)

    with nn.intercept_methods(patch_interceptor):
      out_intervened = self.model.apply(self.params, batch.prefix_clean)

    logits_clean = out_clean[:, self.position_to_measure, :]
    logits_counterfactual = out_counterfactual[:, self.position_to_measure, :]
    logits_intervened = out_intervened[:, self.position_to_measure, :]

    metrics = calculate_indirect_effect(
      logits_clean, logits_counterfactual, logits_intervened, batch
    )

    return InterventionRecord(
      router_clean=router_clean,
      router_counterfactual=router_counterfactual,
      metrics=metrics,
    )
