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

import logging
import os

import einops
import jaxtyping as jt
import numpy as np
import polars as pl
import typer
from scipy import sparse
from transformers import PreTrainedTokenizerFast

from hyperlinear import config_classes
from hyperlinear.data import tinystories
from hyperlinear.evaluation.recorder import NeuronFeatureRecord
from hyperlinear.evaluation.recorder import NeuronFeatureRecorder
from hyperlinear.training.experiment import Experiment
from train import setup_experiment


def create_sparse_gating_matrix(
  lf_neurons: pl.LazyFrame,
  n_sequences: int,
  n_neurons: int,
  layer: int,
  channel: int,
  binary_gates: bool,
) -> tuple[sparse.spmatrix, pl.LazyFrame]:
  """Collect gating patterns into a sparse matrix."""
  df_pattern = (
    lf_neurons.filter(
      pl.col('neuron').is_in(range(n_neurons)),
      pl.col('layer').eq(layer),
      pl.col('channel').eq(channel),
      pl.col('sequence_id').lt(n_sequences),
      pl.col('token_id').ne(0),  # filter pad token
    )
    .group_by('sequence_id', 'position', 'channel', 'layer')
    .agg(pl.col('neuron').alias('active_neurons'), pl.col('score').alias('active_scores'))
    .sort(by=('sequence_id', 'position', 'channel', 'layer'))  # ensure stable pattern_id
    .with_row_index('pattern_id')
  )

  # Extract gating patterns as sparse matrix
  exploded = df_pattern.explode('active_neurons', 'active_scores').collect()
  row_indices = exploded['pattern_id'].to_numpy()
  col_indices = exploded['active_neurons'].to_numpy()
  values = exploded['active_scores'].to_numpy()
  if binary_gates:
    values = np.ones_like(values)

  neuron_gatings = sparse.csr_matrix(
    (values, (row_indices, col_indices)), shape=(row_indices.max() + 1, n_neurons)
  )

  neuron_gatings_metadata = df_pattern.select(
    ['pattern_id', 'sequence_id', 'position', 'layer', 'channel']
  )

  return neuron_gatings, neuron_gatings_metadata


def records_to_dataframe(
  records: NeuronFeatureRecord,
  layers_to_keep: tuple[int, ...],
  channels_to_keep: tuple[int, ...],
) -> pl.DataFrame:
  """Filter and store NeuronFeatureRecord in a long-format pl.DataFrame."""
  # Filter records based on specified layers and channels
  score = records.score[:, :, layers_to_keep][:, :, :, channels_to_keep]
  neuron = records.neuron[:, :, layers_to_keep][:, :, :, channels_to_keep]

  # Build indices for flattened representation
  B, T, L, C, K = score.shape
  channel_idx = einops.repeat(
    np.array(channels_to_keep), 'c -> (b t l c k)', b=B, t=T, l=L, c=C, k=K
  )
  layer_idx = einops.repeat(
    np.array(layers_to_keep), 'l -> (b t l c k)', b=B, t=T, l=L, c=C, k=K
  )
  position_idx = einops.repeat(np.arange(T), 't -> (b t l c k)', b=B, t=T, l=L, c=C, k=K)
  sequence_idx = einops.repeat(np.arange(B), 'b -> (b t l c k)', b=B, t=T, l=L, c=C, k=K)
  token_id = einops.repeat(
    records.sequence, 'b t -> (b t l c k)', b=B, t=T, l=L, c=C, k=K
  )

  return pl.DataFrame(
    dict(
      score=score.flatten().tolist(),
      neuron=neuron.flatten().tolist(),
      channel=channel_idx.tolist(),
      layer=layer_idx.tolist(),
      position=position_idx.tolist(),
      sequence_id=sequence_idx,
      token_id=token_id.tolist(),
    ),
    schema=dict(
      score=pl.Float32,
      neuron=pl.UInt16,
      channel=pl.UInt8,
      layer=pl.UInt8,
      position=pl.UInt16,
      sequence_id=pl.UInt16,
      token_id=pl.UInt16,
    ),
  )


def sequences_to_dataframe(
  sequences_detokenized: list[list[str]],
  sequences_tokenized: list[list[int]],
  top_k_output_tokens: list[list[list[str]]],
  infos: dict[str, jt.ArrayLike],
) -> pl.DataFrame:
  assert len(sequences_tokenized) == len(sequences_detokenized)

  data = dict(
    sequence_id=range(len(sequences_tokenized)),
    sequence=sequences_tokenized,
    detokenized=sequences_detokenized,
    top_k_output_tokens=top_k_output_tokens,
    **infos,
  )

  # Create and join corresponding example sequences
  df_sequence = pl.DataFrame(data)
  df_sequence = df_sequence.with_columns(
    pl.col('sequence_id').cast(pl.UInt16),
    pl.col('sequence').cast(pl.List(pl.UInt16)),
    pl.col('detokenized').cast(pl.List(pl.String)),
    pl.col('top_k_output_tokens').cast(pl.List(pl.List(pl.String))),
  )

  return df_sequence


def collect_gating_records(
  ckpt_path: str,
  dataset: str,
  batch_size: int,
  n_batches: int | None,
  layers: tuple[int, ...],
  channels: tuple[int, ...],
  n_top_output_tokens: int,
  seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
  # Load experiment
  exp_config, exp_state = Experiment.load(ckpt_path)
  exp_config.dataset.batch_size = batch_size
  exp_ctx, _ = setup_experiment(exp_config)

  if not isinstance(exp_config.dataset, config_classes.TinystoriesConfig):
    raise NotImplementedError

  logging.info('Record neuron features.')
  neuron_feature_recorder = NeuronFeatureRecorder(
    model=exp_ctx.model,
    params=exp_state.params,
    bos_token_id=exp_ctx.tokenizer.eos_token_id,
    seed=seed,
    n_top_output_tokens=n_top_output_tokens,
  )

  match dataset:
    case 'tinystories':
      _, valid_loaders, _, _, _ = tinystories.create_tinystories_dataloader(
        batch_size=exp_config.dataset.batch_size,
        seq_len=exp_config.dataset.seq_len,
        vocab_size=exp_config.dataset.vocab_size,
        seed=exp_config.seed,
        tokenizer_path=exp_config.dataset.tokenizer_path,
        n_workers=exp_config.dataset.n_workers,
      )
      dataloader = valid_loaders['validation']
    case _:
      raise ValueError(f'Unknown dataset: {dataset}')

  records, infos = neuron_feature_recorder.run(dataloader, n_batches=n_batches)

  logging.info('Convert records to dataframe and filter layers, channels and neurons.')
  df_neurons = records_to_dataframe(
    records, layers_to_keep=layers, channels_to_keep=channels
  )

  logging.info('Detokenize examples.')
  tokenizer_file = f'{exp_config.dataset.tokenizer_path}/tokenizer.json'
  tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
  tokenizer.add_special_tokens(tinystories.SPECIAL_TOKENS)
  sequences_detokenized = [
    tokenizer.convert_ids_to_tokens(seq) for seq in records.sequence
  ]
  top_k_output_tokens_detokenized = [
    [tokenizer.convert_ids_to_tokens(pos) for pos in seq]
    for seq in records.top_k_output_tokens
  ]
  df_sequences = sequences_to_dataframe(
    sequences_detokenized,
    records.sequence.tolist(),
    top_k_output_tokens_detokenized,
    infos,
  )

  return df_neurons, df_sequences


def main(
  ckpt_path: str = 'logs/20260313_141205_della-i20g2_5cbc_tinystories',
  dataset: str = 'tinystories',
  batch_size: int = 64,
  n_batches: int | None = 16,
  layers: list[int] = [0, 3],  # noqa: B006
  channels: list[int] = [0],  # noqa: B006
  n_top_output_tokens: int = 5,
  seed: int = 0,
) -> None:
  df_neurons, df_sequences = collect_gating_records(
    ckpt_path=ckpt_path,
    dataset=dataset,
    batch_size=batch_size,
    n_batches=n_batches,
    layers=tuple(layers),
    channels=tuple(channels),
    seed=seed,
    n_top_output_tokens=n_top_output_tokens,
  )

  filename = 'records_neuron.parquet'
  filepath = os.path.join(ckpt_path, filename)
  logging.info(
    f'Saving neurons dataframe of size {df_neurons.shape}'
    f'({df_neurons.estimated_size("gb"):.2f}GB) to {filepath}.'
  )
  df_neurons.write_parquet(filepath)

  filename = 'records_sequence.parquet'
  filepath = os.path.join(ckpt_path, filename)
  logging.info(
    f'Saving sequences dataframe of size {df_sequences.shape}'
    f'({df_sequences.estimated_size("gb"):.2f}GB) to {filepath}.'
  )
  df_sequences.write_parquet(filepath)

  # df_full = pl.DataFrame(df_neurons).join(df_sequences, on='sequence_id', how='left')


if __name__ == '__main__':
  typer.run(main)
