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

import abc
import enum
import logging
import os
import pickle
import time
from functools import partial
from typing import Callable
from typing import Type

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.sharding as shd
import optax
from flax import struct
from transformers import PreTrainedTokenizerFast

from hyperlinear import config_classes
from hyperlinear.data_types import Batch
from hyperlinear.data_types import Dataloader
from hyperlinear.data_types import Metrics
from hyperlinear.training import fsdp
from hyperlinear.training.logger import Logger


class CallbackEvent(enum.Enum):
  START = enum.auto()
  STEP = enum.auto()
  END = enum.auto()


class Callback(abc.ABC):
  """
  Callbacks are expected to take care of jit-compiling themselves if possible.
  """

  def __init__(self, log_level: int, onevent: CallbackEvent) -> None:
    self.log_level = log_level
    self.onevent = onevent

  @abc.abstractmethod
  def __call__(self, ctx: 'Experiment', exp_state: 'ExperimentState') -> Metrics:
    pass


@struct.dataclass
class ExperimentLoss(abc.ABC):
  apply_fn: Callable

  @abc.abstractmethod
  def __call__(
    self, params: dict, rng: chex.PRNGKey, batch: Batch
  ) -> tuple[jax.Array, Metrics]:
    pass


@struct.dataclass
class ExperimentState:
  optim: optax.OptState
  params: dict
  rng: chex.PRNGKey
  step: int


class Experiment:
  def __init__(
    self,
    config: config_classes.ExperimentConfig,
    model: nn.Module,
    loss: Type[ExperimentLoss],
    optimizer: optax.GradientTransformation,
    train_steps: int,
    train_loader: Dataloader,
    valid_loaders: dict[str, Dataloader],
    eval_loaders: dict[str, Dataloader],
    tokenizer: PreTrainedTokenizerFast,
    logger_list: tuple[Logger, ...] = (),
    callbacks: tuple[Callback, ...] = (),
    log_every: int = 1,
    log_level: int = 0,
    checkpoint: bool = False,
  ) -> None:
    self.config = config
    self.model = model
    self.loss_fn = loss(apply_fn=self.model.apply)
    self.optimizer = optimizer
    self.train_steps = train_steps
    self.train_loader = train_loader
    self.valid_loaders = valid_loaders
    self.eval_loaders = eval_loaders
    self.tokenizer = tokenizer
    self.logger_list = logger_list
    self.callbacks = callbacks
    self.log_every = log_every
    self.log_level = log_level
    self.checkpoint = checkpoint

    if self.config.sharding_strategy is not None:
      self.mesh = jax.make_mesh(
        (jax.device_count(),), ('data',), axis_types=(jax.sharding.AxisType.Auto,)
      )

  def trigger_callback(
    self, exp_state: ExperimentState, onevent: CallbackEvent
  ) -> Metrics:
    metrics = dict()
    for c in self.callbacks:
      if c.onevent == onevent and c.log_level <= self.log_level:
        metrics.update(c(exp_state=exp_state, ctx=self))

    return metrics

  def log(self, step: int, log_dict: dict, prefix: str = '') -> None:
    if prefix:
      prefix = prefix + '_'

    for logger in self.logger_list:
      logger.log(step, {prefix + k: log_dict[k] for k in log_dict})

  def reset(self, rng: chex.PRNGKey) -> ExperimentState:
    rng_exp, rng_params, rng_dropout = jax.random.split(rng, 3)
    sample_batch = Batch(x=jnp.zeros((1, self.config.dataset.seq_len), dtype=jnp.int32))
    init_rngs = {'params': rng_params, 'dropout': rng_dropout, 'target': rng_params}

    def init_params_optim(
      rng: chex.PRNGKey, inputs: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      params = self.model.init(init_rngs, inputs, mutable='params')
      optim = self.optimizer.init(params)
      return (params, optim)

    match self.config.sharding_strategy:
      case 'fsdp':
        inputs = jax.ShapeDtypeStruct(sample_batch.x.shape, sample_batch.x.dtype)
        state = jax.eval_shape(init_params_optim, rng, inputs)
        shardings = fsdp.infer_fsdp_sharding(state, self.mesh)
        (params, optim) = jax.jit(init_params_optim, out_shardings=shardings)(rng, inputs)

      case 'data':

        def replication_sharding(x: jax.Array) -> shd.Sharding | None:
          if hasattr(x, 'shape'):
            return shd.NamedSharding(self.mesh, shd.PartitionSpec())
          return None

        inputs = jax.ShapeDtypeStruct(sample_batch.x.shape, sample_batch.x.dtype)
        state = jax.eval_shape(init_params_optim, rng, inputs)
        shardings = jax.tree.map(replication_sharding, state)
        (params, optim) = jax.jit(init_params_optim, out_shardings=shardings)(rng, inputs)

      case None:
        params, optim = init_params_optim(init_rngs, sample_batch.x)

    return ExperimentState(optim=optim, params=params, rng=rng_exp, step=0)

  @staticmethod
  def load(directory: str) -> tuple[config_classes.ExperimentConfig, ExperimentState]:
    config = pickle.load(open(os.path.join(directory, 'config.pkl'), 'rb'))
    exp_state = pickle.load(open(os.path.join(directory, 'state.pkl'), 'rb'))
    return config, exp_state

  @staticmethod
  def save(config: config_classes.ExperimentConfig, exp_state: ExperimentState) -> None:
    pickle.dump(config, open(os.path.join(config.workdir, 'config.pkl'), 'wb'))
    pickle.dump(exp_state, open(os.path.join(config.workdir, 'state.pkl'), 'wb'))

  def run(self, exp_state: ExperimentState) -> ExperimentState:
    # Trigger callbacks on CallbackEvent.START
    self.log(exp_state.step, self.trigger_callback(exp_state, CallbackEvent.START))

    prev_time = time.time()
    for step, batch in enumerate(iter(self.train_loader)):
      if step >= self.train_steps:
        logging.info(f'Reached maximum number of {self.train_steps} training steps.')
        break

      exp_state, metrics = self.train_step(exp_state, batch)

      if step % self.log_every == 0:
        elapsed_time = time.time() - prev_time
        prev_time = time.time()

        batch_size, seq_len = self.config.dataset.batch_size, self.config.dataset.seq_len
        metrics = {
          **metrics,
          'steps_per_sec': self.log_every / elapsed_time,
          'tokens_per_sec': (batch_size * seq_len * self.log_every) / elapsed_time,
          'tokens_per_sec_per_device': (batch_size * seq_len * self.log_every)
          / (elapsed_time * jax.device_count()),
        }
        self.log(step, metrics, prefix='train')

        if jnp.isnan(metrics['loss']):
          raise RuntimeError('Loss is nan')

        # Validation
        for name, valid_loader in self.valid_loaders.items():
          self.log(step, self.eval(exp_state, valid_loader), prefix=name)

        # Trigger callbacks on CallbackEvent.STEP
        self.log(step, self.trigger_callback(exp_state, CallbackEvent.STEP))

    # Evaluation
    for name, eval_loader in self.eval_loaders.items():
      self.log(step, self.eval(exp_state, eval_loader), prefix=name)

    # Trigger callbacks on CallbackEvent.END
    self.log(step, self.trigger_callback(exp_state, CallbackEvent.END), prefix='callback')

    # Checkpoint
    if self.checkpoint:
      self.save(self.config, exp_state)

    return exp_state

  def eval(self, exp_state: ExperimentState, eval_loader: Dataloader) -> Metrics:
    rng = exp_state.rng
    metrics_list = list()

    for batch in iter(eval_loader):
      rng, rng_test = jax.random.split(rng)
      metrics_list.append(self.eval_step(exp_state, rng_test, batch))

    metrics = jax.tree.map(lambda *args: jnp.stack((args)), *metrics_list)
    metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics)

    return metrics

  @partial(jax.jit, static_argnames='self')
  def eval_step(
    self, exp_state: ExperimentState, rng: chex.PRNGKey, batch: Batch
  ) -> dict:
    if self.config.sharding_strategy:
      batch = jax.tree.map(partial(fsdp.shard_data, mesh=self.mesh), batch)

    return self.loss_fn(exp_state.params, rng, batch)[1]

  @partial(jax.jit, static_argnames='self', donate_argnames='exp_state')
  def train_step(
    self, exp_state: ExperimentState, batch: Batch
  ) -> tuple[ExperimentState, dict]:
    rng_grad, rng_new = jax.random.split(exp_state.rng)

    if self.config.sharding_strategy:
      batch = jax.tree.map(partial(fsdp.shard_data, mesh=self.mesh), batch)

    (loss, metrics), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
      exp_state.params, rng_grad, batch
    )

    params_update, optim = self.optimizer.update(grads, exp_state.optim, exp_state.params)
    params = optax.apply_updates(exp_state.params, params_update)

    exp_state = ExperimentState(
      params=params,
      optim=optim,
      rng=rng_new,
      step=exp_state.step + 1,
    )

    metrics.update(
      {
        'loss': loss,
        'grad_norm': optax.global_norm(grads),
        'param_norm': optax.global_norm(params),
      }
    )

    return exp_state, metrics
