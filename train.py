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

import dataclasses
import logging
import os
import pprint
import socket
import time
import uuid
from functools import partial

import jax
import jax.numpy as jnp
import optax
from absl import app
from absl import flags
from flax import traverse_util as flax_traverse
from ml_collections import config_flags

from hyperlinear import config_classes
from hyperlinear.callback import ExpertsUsage
from hyperlinear.data import slimpajama
from hyperlinear.data import tinystories
from hyperlinear.model import transformer
from hyperlinear.training import experiment
from hyperlinear.training import logger
from hyperlinear.training import loss
from hyperlinear.training import schedules
from hyperlinear.training.flops import count_forward_flops
from hyperlinear.training.flops import count_parameters

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
  name='training_config',
  default='configs/tinystories_config.py:sparsely_gated_linear',
  # default='configs/slimpajama_config.py:mlp;1',
  help_string='Training configuration.',
)


@dataclasses.dataclass
class ExperimentStatistics:
  n_flops: int
  n_tokens: int
  n_params: int
  n_steps: int


def calculate_experiment_statistics(
  flops_budget: int | None,
  train_size: int,
  model_config: config_classes.ModelConfig,
  vocab_size: int,
  batch_size: int,
  seq_len: int,
) -> ExperimentStatistics:
  """Compute ExperimentStatistics (flops, tokens, params, training steps)."""
  # Heuristic estimation of backward pass flops as 2x forward flops
  n_flops_per_sequence = 3 * count_forward_flops(model_config, seq_len)

  if flops_budget is not None:
    n_flops_per_step = n_flops_per_sequence * batch_size
    n_steps = flops_budget // n_flops_per_step
  else:
    n_steps = train_size // batch_size

  return ExperimentStatistics(
    n_flops=n_steps * batch_size * n_flops_per_sequence,
    n_tokens=n_steps * batch_size * seq_len,
    n_params=count_parameters(model_config, vocab_size),
    n_steps=n_steps,
  )


def setup_experiment(
  config: config_classes.ExperimentConfig,
  logger_list: tuple = (),
  callback_list: tuple = (),
) -> tuple[experiment.Experiment, ExperimentStatistics]:
  logging.info('Loading data...')
  match config.dataset:
    case config_classes.TinystoriesConfig():
      train_loader, valid_loaders, eval_loaders, tokenizer, data_info = (
        tinystories.create_tinystories_dataloader(
          batch_size=config.dataset.batch_size,
          seq_len=config.dataset.seq_len,
          vocab_size=config.dataset.vocab_size,
          seed=config.seed,
          tokenizer_path=config.dataset.tokenizer_path,
          n_workers=config.dataset.n_workers,
        )
      )
      loss_fn = partial(
        loss.AutoregressiveCrossEntropy,
        bos_token_id=data_info.eos_token_id,
        pad_token_id=data_info.pad_token_id,
      )
    case config_classes.SlimpajamaConfig():
      train_loader, valid_loaders, eval_loaders, tokenizer, data_info = (
        slimpajama.create_slimpajama_dataloader(
          batch_size=config.dataset.batch_size,
          seq_len=config.dataset.seq_len,
          seed=config.seed,
          variant=config.dataset.variant,
          local_files_only=config.dataset.local_files_only,
          n_workers=config.dataset.n_workers,
        )
      )
      loss_fn = partial(
        loss.AutoregressiveCrossEntropy,
        bos_token_id=data_info.eos_token_id,
        pad_token_id=data_info.pad_token_id,
      )
    case _:
      raise ValueError(f'Unknown dataset {config.dataset.name}')

  logging.info(
    f'Done loading {config.dataset.name} dataset with {data_info.train_size} sequences'
    f' of length {config.dataset.seq_len} and a vocabulary size of {data_info.vocab_size}'
  )

  exp_stats = calculate_experiment_statistics(
    flops_budget=config.flops_budget,
    train_size=data_info.train_size,
    model_config=config.model,
    vocab_size=data_info.vocab_size,
    batch_size=config.dataset.batch_size,
    seq_len=config.dataset.seq_len,
  )

  if exp_stats.n_steps > data_info.train_size // config.dataset.batch_size:
    # NOTE: This is only an approximation due to packing/splitting of sequences.
    raise ValueError(
      f'Too little training data to satisfy flop budget of {config.flops_budget} flops'
    )

  # Instantiate model
  sequence_model = transformer.Transformer(
    n_vocab=data_info.vocab_size,
    d_model=config.model.d_model,
    d_heads=config.model.d_heads,
    n_layers=config.model.n_layers,
    feedforward_config=config.model.feedforward_config,
    dtype=config.model.dtype,
    remat=config.model.remat,
  )

  # Instantiate optimizer and learning rate scheduler
  optimizer_ops = []
  n_optim_steps = exp_stats.n_steps // config.optimizer.n_gradient_accumulation

  if config.optimizer.clip_by_global_norm is not None:
    optimizer_ops.append(optax.clip_by_global_norm(config.optimizer.clip_by_global_norm))

  match config.optimizer.schedule:
    case 'cosine':
      schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optimizer.learning_rate,
        warmup_steps=config.optimizer.warmup_steps,
        decay_steps=n_optim_steps - config.optimizer.warmup_steps,
        end_value=config.optimizer.learning_rate * 0.1,
      )
    case 'constant':
      schedule = optax.constant_schedule(config.optimizer.learning_rate)
    case 'warmup_stable_linear_decay':
      decay_steps = int(config.optimizer.decay_fraction * n_optim_steps)
      schedule = schedules.warmup_stable_linear_decay_schedule(
        learning_rate=config.optimizer.learning_rate,
        warmup_steps=config.optimizer.warmup_steps,
        const_steps=n_optim_steps - config.optimizer.warmup_steps - decay_steps,
        decay_steps=decay_steps,
      )
    case 'warmup_stable_sqrt_decay':
      decay_steps = int(config.optimizer.decay_fraction * n_optim_steps)
      schedule = schedules.warmup_stable_sqrt_decay_schedule(
        learning_rate=config.optimizer.learning_rate,
        warmup_steps=config.optimizer.warmup_steps,
        const_steps=n_optim_steps - config.optimizer.warmup_steps - decay_steps,
        decay_steps=decay_steps,
      )
    case _:
      raise ValueError(f'Unknown learning rate schedule: {config.optimizer.schedule}')

  optimizer_ops.append(
    getattr(optax, config.optimizer.optimizer)(
      learning_rate=schedule,
      # Decoupled weight_decay, see https://fabian-sp.github.io/posts/2024/02/decoupling/
      weight_decay=config.optimizer.weight_decay / config.optimizer.learning_rate,
      mask=(
        lambda p: jax.tree_util.tree_map(  # mask weight decay for biases and layernorms
          lambda x: x.ndim != 1, p
        )
      )
      if config.optimizer.mask_weight_decay
      else None,
    )
  )
  optimizer = optax.MultiSteps(
    opt=optax.chain(*optimizer_ops),
    every_k_schedule=config.optimizer.n_gradient_accumulation,
    use_grad_mean=True,
  )

  # Instantiate experiment runner
  exp = experiment.Experiment(
    config=config,
    model=sequence_model,
    loss=loss_fn,
    optimizer=optimizer,
    train_steps=exp_stats.n_steps,
    train_loader=train_loader,
    valid_loaders=valid_loaders,
    eval_loaders=eval_loaders,
    tokenizer=tokenizer,
    logger_list=logger_list,
    callbacks=callback_list,
    log_every=config.log_every,
    log_level=config.log_level,
    checkpoint=config.checkpoint,
  )

  return (exp, exp_stats)


def main(argv: list[str]) -> None:
  del argv

  config: config_classes.ExperimentConfig = flags.FLAGS.training_config

  # Setup workdir and overwrite the workdir path in the config
  def setup_unique_workdir(base_path: str) -> str:
    unique_id = str(uuid.uuid4())[-4:]
    hostname = socket.gethostname()
    datetime = time.strftime('%Y%m%d_%H%M%S_')
    id = datetime + hostname + '_' + unique_id + '_{}'.format(config.name)
    workdir = os.path.join(os.getcwd(), base_path, id)
    os.makedirs(workdir, exist_ok=True)

    return workdir

  workdir = setup_unique_workdir(config.workdir)
  config = dataclasses.replace(config, workdir=workdir)
  logging.info('Workdir at {}'.format(workdir))

  # Setup loggers
  logger_list = []
  for logger_name in config.logger_names:
    match logger_name:
      case 'stdout':
        logger_list.append(logger.StandardLogger(log_dir=None))
      case 'stdout_to_disk':
        logger_list.append(logger.StandardLogger(log_dir=workdir))
      case 'wandb':
        logger_list.append(logger.WandbLogger(config, log_dir=None))
      case 'wandb_to_disk':
        logger_list.append(logger.WandbLogger(config, log_dir=workdir))
      case _:
        raise ValueError(f'Unknown logger: {logger_name}')

  # Setup callbacks
  callback_list = []
  for callback_name in config.callback_names:
    match callback_name:
      case 'experts_usage':
        callback_list.append(
          ExpertsUsage(log_level=1, onevent=experiment.CallbackEvent.STEP)
        )
      case _:
        raise ValueError(f'Unknown callback: {callback_name}')

  logging.info('Setup experiment')
  exp, exp_stats = setup_experiment(
    config, tuple(logger_list), callback_list=tuple(callback_list)
  )
  exp_state = exp.reset(jax.random.key(config.seed))

  # Log experiment statistics
  logging.info('Experiment statistics')
  exp.log(step=0, log_dict=dataclasses.asdict(exp_stats))

  logging.info(f'Running on {jax.device_count()} {jax.default_backend()}(s)')
  if config.flops_budget is not None:
    logging.info(
      f'Given {config.flops_budget:e} flops, train for {exp_stats.n_steps} steps.'
    )
  else:
    logging.info(f'Given no flop budget, train for full {exp_stats.n_steps} steps.')

  logging.info(f'Training will consume {exp_stats.n_flops:e} flops.')
  logging.info(f'Training will consume {exp_stats.n_tokens:e} tokens.')
  logging.info(f'Model has {exp_stats.n_params:e} parameters.')

  logging.info('Start training with parametrization')
  config_str = pprint.pformat(flax_traverse.flatten_dict(dataclasses.asdict(config)))
  logging.info(f'\n{config_str}')

  logging.info(jax.tree.map(jnp.shape, exp_state.params['params']))
  exp_state = exp.run(exp_state)


if __name__ == '__main__':
  # with jax.disable_jit():
  app.run(main)
