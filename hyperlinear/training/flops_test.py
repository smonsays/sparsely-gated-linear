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

import jax
import optax
import plotly.express as px
import polars as pl
from absl.testing import absltest
from absl.testing import parameterized
from jax import flatten_util

from configs import slimpajama_config
from configs import tinystories_config
from hyperlinear.model import transformer
from hyperlinear.training import experiment
from hyperlinear.training import flops
from hyperlinear.training import loss


class FlopsTest(parameterized.TestCase):
  @parameterized.parameters(
    dict(model_name='mlp'),
    dict(model_name='swiglu'),
    dict(model_name='moe'),
    dict(model_name='peer'),
    dict(model_name='sparsely_gated_linear'),
  )
  def test_count_parameters(self, model_name: str, vocab_size: int = 16) -> None:
    """Test that count_parameters against Experiment.reset."""
    config = tinystories_config.get_config(model_name)
    sequence_model = transformer.Transformer(
      n_vocab=vocab_size,
      d_model=config.model.d_model,
      d_heads=config.model.d_heads,
      n_layers=config.model.n_layers,
      feedforward_config=config.model.feedforward_config,
      dtype=config.model.dtype,
      remat=config.model.remat,
    )
    loss_fn = partial(
      loss.AutoregressiveCrossEntropy,
      bos_token_id=0,
      pad_token_id=0,
    )
    exp = experiment.Experiment(
      config=config,
      model=sequence_model,
      loss=loss_fn,
      optimizer=optax.adam(learning_rate=1e-3),
      train_steps=1,
      train_loader=iter([]),  # Empty loader for test
      valid_loaders={},
      eval_loaders={},
      tokenizer=lambda x: x,
    )
    exp_state = exp.reset(jax.random.key(0))

    expected_n_params = flops.count_parameters(config.model, vocab_size)
    actual_n_params = flatten_util.ravel_pytree(exp_state.params['params'])[0].shape[0]
    self.assertEqual(actual_n_params, expected_n_params)

  @absltest.skip('Skipping test since it invokes a plot.')
  def test_plot_dffw_flops_comparison(
    self, vocab_size: int = 50257, scale: int = 6
  ) -> None:
    data_list = list()
    for ffw_type in [
      'mlp',
      'swiglu',
      'peer',
      'sparsely_gated_linear',
      'moe',
    ]:
      config = slimpajama_config.get_config(f'{ffw_type};{scale}')
      print(
        ffw_type[:6],
        '\t',
        f'{3 * flops.count_forward_flops(config.model, config.dataset.seq_len):e}',
      )

      for d_ffw in [32**2, 48**2, 64**2, 128**2, 256**2]:
        if ffw_type == 'moe':
          # For a modern sparse MoE, increase number of experts rather than d_ffw
          config.model.feedforward_config.d_ffw = config.model.d_model
          config.model.feedforward_config.n_channels = max(
            2, d_ffw // config.model.d_model
          )
        else:
          config.model.feedforward_config.d_ffw = d_ffw
        n_flops = 3 * flops.count_forward_flops(config.model, config.dataset.seq_len)
        n_params = flops.count_parameters(config.model, vocab_size=vocab_size)
        data_list.append(
          dict(ffw_type=ffw_type, d_ffw=d_ffw, flops=n_flops, params=n_params)
        )

    df = pl.DataFrame(data_list)
    # df.write_parquet('model-complexity_ffw.parquet')

    df = df.with_columns((pl.col('params') / pl.col('flops')).alias('params_per_flop'))
    fig = px.line(df, x='params', y='flops', color='ffw_type', log_x=True, log_y=True)
    # fig.update_xaxes(range=[6, 8])
    fig.show()

    fig = px.line(df, x='d_ffw', y='flops', color='ffw_type', log_x=True, log_y=True)
    fig.show()

    fig = px.line(df, x='d_ffw', y='params', color='ffw_type', log_x=True, log_y=True)
    fig.show()

    fig = px.line(
      df, x='d_ffw', y='params_per_flop', color='ffw_type', log_x=True, log_y=True
    )
    fig.show()

  @absltest.skip('Skipping test since it invokes a plot.')
  def test_plot_scale_flops_comparison(self, vocab_size: int = 50257) -> None:
    data_list = list()
    for ffw_type in [
      'mlp',
      'swiglu',
      'peer',
      'sparsely_gated_linear',
      'moe',
    ]:
      for scale in [3, 4, 5, 6, 7, 8, 9, 10]:
        config = slimpajama_config.get_config(f'{ffw_type};{scale}')
        n_flops = 3 * flops.count_forward_flops(config.model, config.dataset.seq_len)
        n_params = flops.count_parameters(config.model, vocab_size=vocab_size)

        data_dict = dict(ffw_type=ffw_type, scale=scale, flops=n_flops, params=n_params)
        print(data_dict)
        data_list.append(data_dict)

    # df = pl.DataFrame(data_list)
    # df.write_parquet('model-complexity_scale.parquet')


if __name__ == '__main__':
  absltest.main()
