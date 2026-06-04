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

import os

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import typer
import umap
from plotly.graph_objs import Figure

from hyperlinear import config_classes
from hyperlinear.data import tinyknowledge
from hyperlinear.evaluation.intervention import InterventionMetrics
from hyperlinear.evaluation.intervention import InterventionRecorder
from hyperlinear.evaluation.intervention import RouterRecord
from hyperlinear.training.experiment import Experiment
from scope import create_sparse_gating_matrix
from train import setup_experiment


def records_to_dataframe(
  records: RouterRecord,
  layers_to_keep: tuple[int, ...],
  channels_to_keep: tuple[int, ...],
) -> pl.DataFrame:
  """Filter and store RouterRecord in a long-format pl.DataFrame."""
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
  token_id = einops.repeat(records.prefix, 'b t -> (b t l c k)', b=B, t=T, l=L, c=C, k=K)

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


def metrics_to_dataframe(
  metrics: InterventionMetrics, sequence_ids: np.ndarray, answers: np.ndarray
) -> pl.DataFrame:
  """Store InterventionMetrics in a pl.DataFrame."""

  data = dict(
    sequence_id=sequence_ids,
    answer_clean=answers[:, 0].tolist(),  # Top answer
    logitdiff_clean=metrics.logitdiff_clean.tolist(),
    logitdiff_counterfactual=metrics.logitdiff_counterfactual.tolist(),
    logitdiff_intervened=metrics.logitdiff_intervened.tolist(),
    recovered_effect=metrics.recovered_effect.tolist(),
    rank_flip=metrics.rank_flip.tolist(),
    correct_clean=metrics.correct_clean.tolist(),
    correct_counterfactual=metrics.correct_counterfactual.tolist(),
  )

  schema = dict(
    sequence_id=pl.UInt16,
    answer_clean=pl.UInt16,
    logitdiff_clean=pl.Float32,
    logitdiff_counterfactual=pl.Float32,
    logitdiff_intervened=pl.Float32,
    recovered_effect=pl.Float32,
    rank_flip=pl.Float32,
    correct_clean=pl.Float32,
    correct_counterfactual=pl.Float32,
  )

  return pl.DataFrame(data, schema=schema)


def sequences_to_dataframe(
  sequences_detokenized: list[list[str]],
  sequences_tokenized: list[list[int]],
) -> pl.DataFrame:
  """Store tokenized and detokenized prefixes in a pl.DataFrame."""
  assert len(sequences_tokenized) == len(sequences_detokenized)

  data = dict(
    sequence_id=range(len(sequences_tokenized)),
    sequence=sequences_tokenized,
    detokenized=sequences_detokenized,
  )

  df_sequence = pl.DataFrame(data)
  df_sequence = df_sequence.with_columns(
    pl.col('sequence_id').cast(pl.UInt16),
    pl.col('sequence').cast(pl.List(pl.UInt16)),
    pl.col('detokenized').cast(pl.List(pl.String)),
  )

  return df_sequence


def embed_gating_patterns_patched_token(
  lf_neurons: pl.LazyFrame,
  lf_sequences: pl.LazyFrame,
  n_sequences: int,
  n_neurons: int,
  layers: tuple[int, ...],
  channel: int,
  binary_gates: bool,
  position_to_patch: int,
) -> pl.DataFrame:
  """Embed gating patterns for the patched prefix token using UMAP."""

  # Filter to only include the position where the intervention was applied
  lf_neurons_final = lf_neurons.filter(pl.col('position') == position_to_patch)

  all_embeddings = []

  for layer in layers:
    neuron_gatings, neuron_gatings_metadata = create_sparse_gating_matrix(
      lf_neurons_final,
      n_sequences=n_sequences,
      n_neurons=n_neurons,
      layer=layer,
      channel=channel,
      binary_gates=binary_gates,
    )

    sparsity = 1 - neuron_gatings.nnz / (
      neuron_gatings.shape[0] * neuron_gatings.shape[1]
    )
    typer.echo(f'Layer {layer} - Sparse matrix shape: {neuron_gatings.shape}')
    typer.echo(f'Layer {layer} - Sparsity: {sparsity:.4f}')

    typer.echo(f'Running UMAP for layer {layer}...')
    reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15, verbose=True)
    embedding = reducer.fit_transform(neuron_gatings)

    # Join metadata with df_sequences to retrieve color_modifier, answer_id, etc.
    layer_df = (
      neuron_gatings_metadata.join(lf_sequences, on='sequence_id', how='left')
      .collect()
      .with_columns(  # ty:ignore[unresolved-attribute]
        x=embedding[:, 0],
        y=embedding[:, 1],
        layer=pl.lit(layer),
      )
    )

    all_embeddings.append(layer_df)

  return pl.concat(all_embeddings) if all_embeddings else pl.DataFrame()


def plot_gating_embedding(df: pd.DataFrame, color_key: str) -> Figure:
  fig = px.scatter(
    df,
    x='x',
    y='y',
    color=color_key,
    facet_col='layer',
    hover_data={color_key: True, 'x': False, 'y': False},
  )

  return fig


def main(
  ckpt_path: str,
  batch_size: int = 64,
  layers_to_patch: list[int] = [0, 1, 2, 3],  # noqa: B006
  top_answers: int = 1,
  channel: int = 0,
  position_to_patch: int = 2,
  position_to_measure: int = 3,
) -> None:
  # Load experiment
  exp_config, exp_state = Experiment.load(ckpt_path)
  exp_config.dataset.batch_size = batch_size
  exp_ctx, _ = setup_experiment(exp_config)

  if not isinstance(exp_config.dataset, config_classes.TinystoriesConfig):
    raise NotImplementedError(
      'Only TinyStories is currently supported for tinyknowledge.'
    )

  # Setup dataloader
  typer.echo('Initializing Tinyknowledge dataloader...')
  dataloader = tinyknowledge.create_tinyknowledge_interventionloader(
    tokenizer=exp_ctx.tokenizer,
    batch_size=batch_size,
    seq_len=24,
    top_answers=top_answers,
    n_workers=exp_config.dataset.n_workers,
  )

  recorder = InterventionRecorder(
    model=exp_ctx.model,
    params=exp_state.params,
    layers_to_patch=tuple(layers_to_patch),
    position_to_patch=position_to_patch,
    position_to_measure=position_to_measure,
  )

  typer.echo('Running causal interventions...')
  records_list = []
  for batch in dataloader:
    records_list.append(recorder(batch))

  records = jax.tree.map(lambda *arrays: np.concatenate(arrays, axis=0), *records_list)

  typer.echo(
    f'Causal intervention metrics \n {
      jax.tree.map(lambda x: jnp.mean(x).item(), records.metrics)
    }'
  )
  typer.echo(f'Mean Recovered Effect: {records.metrics.recovered_effect.mean():.4f}')
  typer.echo(f'Mean Rank Flip: {records.metrics.rank_flip.mean():.4f}')
  typer.echo(f'Mean Correct Clean: {records.metrics.correct_clean.mean():.4f}')
  typer.echo(
    f'Mean Correct Counterfactual: {records.metrics.correct_counterfactual.mean():.4f}'
  )

  typer.echo('Converting records to DataFrames...')

  df_neurons_clean = records_to_dataframe(
    records.router_clean,
    layers_to_keep=tuple(layers_to_patch),
    channels_to_keep=(channel,),
  )

  n_sequences = records.router_clean.prefix.shape[0]
  df_metrics = metrics_to_dataframe(
    records.metrics,
    sequence_ids=np.arange(n_sequences),
    answers=records.router_clean.answer,
  )
  df_metrics = df_metrics.with_columns(
    pl.lit(layers_to_patch).alias('layers_to_patch'),
    pl.lit(position_to_patch).alias('position_to_patch'),
  )

  layers_to_patch_str = '-'.join(map(str, layers_to_patch))
  file_suffix = f'layers_{layers_to_patch_str}_position_{position_to_patch}'
  df_metrics.write_parquet(
    os.path.join(ckpt_path, f'tinyknowledge_metrics_{file_suffix}.parquet')
  )

  sequences_detokenized = [
    exp_ctx.tokenizer.convert_ids_to_tokens(seq) for seq in records.router_clean.prefix
  ]

  df_sequences = sequences_to_dataframe(
    sequences_detokenized, records.router_clean.prefix.tolist()
  )

  # Join metrics into sequences for easier visualization
  df_sequences = df_sequences.join(df_metrics, on='sequence_id', how='left')

  # UMAP Visualization
  typer.echo('Extracting and embedding gating patterns for patched token...')
  df_embedding = embed_gating_patterns_patched_token(
    df_neurons_clean.lazy(),
    df_sequences.lazy(),
    n_sequences=n_sequences,
    n_neurons=exp_config.model.feedforward_config.d_ffw,
    layers=tuple(layers_to_patch),
    channel=channel,
    binary_gates=False,
    position_to_patch=position_to_patch,
  )

  typer.echo('Plotting embedding gating patterns...')
  df_embedding_pd = df_embedding.to_pandas()
  df_embedding_pd['answer_detokenized'] = exp_ctx.tokenizer.convert_ids_to_tokens(
    df_embedding_pd['answer_clean'].tolist()
  )
  df_embedding_pd['answer_detokenized'] = (
    df_embedding_pd['answer_detokenized']
    .str.replace('Ġ', '')
    .str.replace(' ', '')
    .str.strip()
  )
  fig = plot_gating_embedding(df_embedding_pd, 'answer_detokenized')
  fig.write_html(
    os.path.join(ckpt_path, f'tinyknowledge_gating_embedding_answer_{file_suffix}.html')
  )

  typer.echo('Done.')


if __name__ == '__main__':
  typer.run(main)
