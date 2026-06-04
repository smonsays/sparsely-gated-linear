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
import logging
import os
from importlib import reload

import jax
import jax.numpy as jnp
import numpy as np

import wandb
from hyperlinear.config_classes import ExperimentConfig


class Logger(abc.ABC):
  @abc.abstractmethod
  def log(self, step: int, metrics: dict) -> None:
    pass


class StandardLogger(Logger):
  """
  Standard library logging to stdout and file.
  """

  def __init__(self, log_dir: str | None = None) -> None:
    # Need to reload logging as otherwise the logger might be captured by another library
    reload(logging)
    handlers = [logging.StreamHandler()]

    if log_dir is not None:
      # Only log outputs to disk if log_dir specified
      if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        handlers.append(logging.FileHandler(os.path.join(log_dir, 'event.log')))

    logging.basicConfig(
      level=logging.INFO,
      format='[%(levelname)-5.5s %(asctime)s] %(message)s',
      datefmt='%H:%M:%S',
      handlers=handlers,
    )

  def log(self, step: int, metrics: dict) -> None:
    metrics = {
      k: v
      for k, v in metrics.items()
      if isinstance(v, (int, float, str, np.ndarray, jnp.ndarray))
    }
    metrics_str = ' \t '.join('{}: {:.4f}'.format(str(k), v) for k, v in metrics.items())
    logging.info('step: {} \t'.format(step) + metrics_str)


class WandbLogger(Logger):
  def __init__(
    self, config: ExperimentConfig, log_dir: str | None, synchronize: bool = True
  ) -> None:
    if not synchronize:
      os.environ['WANDB_DISABLED'] = 'true'

    wandb.init(
      config=config,
      save_code=False,
      dir=log_dir,
    )

  def log(self, step: int, metrics: dict) -> None:
    # HACK(@simon): Workaround for wandb not converting jax.Array automatically
    metrics = {
      key: val if not isinstance(val, jax.Array) else np.array(val)
      for key, val in metrics.items()
    }
    wandb.log({**metrics, 'step': step}, step=step)
