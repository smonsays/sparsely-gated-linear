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

from hyperlinear import config_classes


def get_feedforward_config(feedforward_type: str) -> config_classes.FeedforwardConfig:
  match feedforward_type:
    case 'mlp':
      feedforward_config = config_classes.FeedforwardConfig(
        type='mlp',
        activation='gelu',
        d_ffw=1024,
      )
    case 'sparsely_gated_linear':
      feedforward_config = config_classes.FeedforwardConfig(
        type='sparsely_gated_linear',
        d_ffw=1024,
        n_channels=8,
        knn=8,
        d_key=128,
        router_type='product_dense',
        activation='identity',
      )
    case 'peer':
      feedforward_config = config_classes.FeedforwardConfig(
        type='peer',
        activation='gelu',
        d_ffw=1024,
        n_channels=8,
        knn=8,
        d_key=128,
      )
    case 'swiglu':
      feedforward_config = config_classes.FeedforwardConfig(
        type='glu',
        activation='sigmoid',
        d_ffw=1024,
      )
    case 'moe':
      feedforward_config = config_classes.FeedforwardConfig(
        type='moe',
        activation='sigmoid',
        d_ffw=1024,
        n_channels=8,
        knn=2,
      )
    case _:
      raise ValueError('Unknown feedforward_type {feedforward_type}')

  return feedforward_config


def get_config(feedforward_type: str) -> config_classes.ExperimentConfig:
  dataset = config_classes.TinystoriesConfig(
    batch_size=256,
    vocab_size=8192,
    seq_len=1024,
    tokenizer_path='./configs/tokenizer',
    n_workers=8,
  )
  model = config_classes.ModelConfig(
    d_model=256,
    d_heads=64,
    n_layers=4,
    feedforward_config=get_feedforward_config(feedforward_type),
    dtype='bfloat16',
    remat=True,
  )

  optimizer = config_classes.OptimizerConfig(
    learning_rate=1e-3,
    weight_decay=1e-4,
    optimizer='adamw',
    schedule='warmup_stable_sqrt_decay',
    warmup_steps=100,
    decay_fraction=0.2,
    mask_weight_decay=True,
    clip_by_global_norm=2.0,
    n_gradient_accumulation=1,
  )

  config = config_classes.ExperimentConfig(
    name='tinystories',
    seed=0,
    optimizer=optimizer,
    dataset=dataset,
    model=model,
    flops_budget=None,  # int(7.254849e+15),
    sharding_strategy=None,
    workdir='logs',
    log_every=1000,
    log_level=1,
    checkpoint=True,
    logger_names=('stdout', 'wandb'),
    callback_names=(),
  )

  return config
