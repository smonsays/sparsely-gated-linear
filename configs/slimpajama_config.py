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
from hyperlinear.training.flops import count_parameters


def scaling_ladder(scale: int, feedforward_type: str) -> tuple[int, ...]:
  d_model = 128 * scale
  n_layers = 2 * scale

  match feedforward_type:
    case 'swiglu' | 'mlp':
      d_ffw = int(8 / 3 * d_model / 256) * 256  # multiple of 256 close to 8/3
    case 'moe':
      d_ffw = d_model  # following GPT-OSS
    case 'sparsely_gated_linear':
      d_ffw = (16 + 12 * scale) ** 2
    case 'peer':
      d_ffw = (32 + 24 * scale) ** 2
    case _:
      raise ValueError('Unknown feedforward_type {feedforward_type}')

  return (d_model, n_layers, d_ffw)


def get_feedforward_config(
  feedforward_type: str, d_ffw: int
) -> config_classes.FeedforwardConfig:
  match feedforward_type:
    case 'mlp':
      feedforward_config = config_classes.FeedforwardConfig(
        type='mlp',
        activation='gelu',
        d_ffw=d_ffw,
      )
    case 'sparsely_gated_linear':
      feedforward_config = config_classes.FeedforwardConfig(
        type='sparsely_gated_linear',
        d_ffw=d_ffw,
        n_channels=16,
        knn=8,
        d_key=128,
        router_type='product_dense',
        activation='identity',
      )
    case 'peer':
      feedforward_config = config_classes.FeedforwardConfig(
        type='peer',
        activation='gelu',
        d_ffw=d_ffw,
        n_channels=16,
        knn=8,
        d_key=128,
      )
    case 'swiglu':
      feedforward_config = config_classes.FeedforwardConfig(
        type='glu',
        activation='sigmoid',
        d_ffw=d_ffw,
      )
    case 'moe':
      feedforward_config = config_classes.FeedforwardConfig(
        type='moe',
        activation='sigmoid',
        d_ffw=d_ffw,
        n_channels=16,
        knn=2,
      )
    case _:
      raise ValueError('Unknown feedforward_type {feedforward_type}')

  return feedforward_config


def get_config(feedforward_type_scale: str) -> config_classes.ExperimentConfig:
  feedforward_type, scale = feedforward_type_scale.split(';')
  d_model, n_layers, d_ffw = scaling_ladder(int(scale), feedforward_type)

  dataset = config_classes.SlimpajamaConfig(
    batch_size=128,
    seq_len=2048,
    variant='full',
    local_files_only=True,
    n_workers=8,
  )
  model = config_classes.ModelConfig(
    d_model=d_model,
    d_heads=64,
    n_layers=n_layers,
    feedforward_config=get_feedforward_config(feedforward_type, d_ffw),
    dtype='bfloat16',
    remat=True,
  )

  optimizer = config_classes.OptimizerConfig(
    learning_rate=1e-3,
    weight_decay=1e-4,
    optimizer='adamw',
    schedule='warmup_stable_sqrt_decay',
    warmup_steps=1000,
    decay_fraction=0.2,
    mask_weight_decay=True,
    clip_by_global_norm=1.0,
    n_gradient_accumulation=1,
  )

  config = config_classes.ExperimentConfig(
    name='slimpajama',
    seed=0,
    optimizer=optimizer,
    dataset=dataset,
    model=model,
    flops_budget=int(1e17),
    sharding_strategy='data',
    workdir='logs',
    log_every=1000,
    log_level=1,
    checkpoint=False,
    logger_names=('stdout', 'wandb'),
    callback_names=(),
  )

  return config


if __name__ == '__main__':
  # Scaling ladder
  for scale in range(1, 10):
    for ffw_type in ['mlp', 'swiglu', 'peer', 'sparsely_gated_linear', 'moe']:
      print(f'\nscale {scale}')
      print(f'n_layers={get_config(f"{ffw_type};{scale}").model.n_layers}')
      print(f'd_model={get_config(f"{ffw_type};{scale}").model.d_model}')
      print(get_config(f'{ffw_type};{scale}').model.feedforward_config)
      print(
        f'n_params={count_parameters(get_config(f"{ffw_type};{scale}").model, 50258):.2e}'
      )
