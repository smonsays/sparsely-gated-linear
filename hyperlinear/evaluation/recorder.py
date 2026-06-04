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

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from flax import struct

from hyperlinear.data_types import Dataloader
from hyperlinear.training.loss import shift_right
from hyperlinear.utils import dict_filter


@struct.dataclass
class NeuronFeatureRecord:
  sequence: jt.Int[jt.Array, 'B T']
  score: jt.Float[jt.Array, 'B T L C K']
  neuron: jt.Int[jt.Array, 'B T L C K']
  top_k_output_tokens: jt.Int[jt.Array, 'B T K']


class NeuronFeatureRecorder:
  """Record which neurons of a SparselyGatedLinear layer are activated by input examples.

  Given a pretrained transformer model with parameters `params`, sample `n_batches` from
  `dataloader`and record which neurons indices are activated along with their scores.
  """

  def __init__(
    self,
    model: nn.Module,
    params: dict,
    bos_token_id: int,
    n_top_output_tokens: int,
    seed: int,
  ) -> None:
    self.model = model
    self.params = params
    self.bos_token_id = bos_token_id
    self.seed = seed
    self.n_top_output_tokens = n_top_output_tokens

  @partial(jax.jit, static_argnames='self')
  def step(
    self, rng: jt.PRNGKeyArray, input_ids: jt.Int[jt.Array, 'B T']
  ) -> NeuronFeatureRecord:
    input_ids = shift_right(input_ids, bos_token_id=self.bos_token_id)
    logits, variables = self.model.apply(
      self.params,
      input_ids,
      rngs={'dropout': rng},
      mutable='intermediates',
    )

    _, top_k_output_tokens = jax.lax.top_k(logits, k=self.n_top_output_tokens)

    return NeuronFeatureRecord(
      sequence=input_ids,
      score=jnp.stack(dict_filter(variables, 'scores'), axis=2),
      neuron=jnp.stack(dict_filter(variables, 'indices'), axis=2),
      top_k_output_tokens=top_k_output_tokens,
    )

  def run(
    self, dataloader: Dataloader, n_batches: int | None = None
  ) -> tuple[NeuronFeatureRecord, dict[str, jt.ArrayLike]]:
    rng = jax.random.key(self.seed)
    records_list = []
    infos_list = []
    for i, batch in enumerate(dataloader):
      rng = jax.random.fold_in(rng, i)
      records_list.append(jax.tree.map(np.array, self.step(rng, batch.x)))
      infos_list.append(batch.info)
      if n_batches is not None and i == (n_batches - 1):
        break

    records = jax.tree.map(lambda *args: np.concatenate((args)), *records_list)

    if infos_list:
      infos = jax.tree.map(lambda *args: np.concatenate(args), *infos_list)
    else:
      infos = dict()

    return records, infos
