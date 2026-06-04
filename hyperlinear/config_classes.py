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
from typing import Literal


@dataclasses.dataclass
class SlimpajamaConfig:
  batch_size: int
  seq_len: int
  local_files_only: bool
  variant: str
  n_workers: int
  name: str = 'slimpajama'


@dataclasses.dataclass
class TinystoriesConfig:
  batch_size: int
  seq_len: int
  vocab_size: int
  tokenizer_path: str
  n_workers: int
  name: str = 'tinystories'


@dataclasses.dataclass
class FeedforwardConfig:
  type: str
  activation: str
  d_ffw: int
  n_channels: int = 0
  knn: int = 0
  d_key: int = 0
  router_type: str = ''


@dataclasses.dataclass
class ModelConfig:
  d_model: int
  d_heads: int
  n_layers: int
  feedforward_config: FeedforwardConfig
  dtype: str = 'bfloat16'
  remat: bool = False


@dataclasses.dataclass
class OptimizerConfig:
  learning_rate: float
  weight_decay: float
  optimizer: str
  schedule: str
  warmup_steps: int
  decay_fraction: float
  n_gradient_accumulation: int
  mask_weight_decay: bool = True
  clip_by_global_norm: None | float = None


@dataclasses.dataclass
class ExperimentConfig:
  name: str
  seed: int
  model: ModelConfig
  dataset: SlimpajamaConfig | TinystoriesConfig
  optimizer: OptimizerConfig
  flops_budget: int | None
  sharding_strategy: Literal['fsdp'] | Literal['data'] | None
  workdir: str
  log_every: int
  log_level: int
  checkpoint: bool
  logger_names: tuple[str, ...]
  callback_names: tuple[str, ...]
