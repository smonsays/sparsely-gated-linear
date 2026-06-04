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

import jaxtyping as jt
import numpy as np
import plotly.express as px
import polars as pl
import typer
import umap
from plotly.graph_objs import Figure
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from transformers import PreTrainedTokenizerFast

from hyperlinear.config_classes import TinystoriesConfig
from hyperlinear.evaluation import cka
from hyperlinear.training.experiment import Experiment
from scope import collect_gating_records
from scope import create_sparse_gating_matrix


def add_input_token(lf: pl.LazyFrame) -> pl.LazyFrame:
  return lf.with_columns(
    pl.col('detokenized').list.get(pl.col('position')).alias('input_token')
  )


def add_target_token(lf: pl.LazyFrame) -> pl.LazyFrame:
  return lf.with_columns(
    pl.col('detokenized')
    .list.get(pl.col('position') + 1, null_on_oob=True)
    .alias('target_token'),
  )


def create_neuron_coactivation_matrix(
  lf_neurons: pl.LazyFrame,
  n_neurons: int,
  layer: int,
  channel: int,
) -> jt.Int[jt.ArrayLike, 'n_neurons n_neurons']:
  """Collect neuron co-activation matrix showing which neurons fire together."""
  coactivation_pairs = (
    lf_neurons.filter(
      pl.col('neuron').is_in(range(n_neurons)),
      pl.col('layer').eq(layer),
      pl.col('channel').eq(channel),
    )
    .group_by(['sequence_id', 'channel', 'layer', 'position'])
    .agg(pl.col('neuron').sort().alias('active_neurons'))
    # Create all pairwise combinations by cross joining each list with itself
    .with_columns(
      pl.col('active_neurons').alias('neuron_i'),
      pl.col('active_neurons').alias('neuron_j'),
    )
    .explode(pl.col('neuron_i'))
    .explode(pl.col('neuron_j'))
    .filter(pl.col('neuron_i') != pl.col('neuron_j'))  # Filter out self-activations
    # Count occurrences of each pair
    .group_by(pl.col('neuron_i'), pl.col('neuron_j'))
    .len()
    .collect(engine='streaming')
  )

  # Build co-occurrence matrix from the pairs
  coactivation_matrix = np.zeros((n_neurons, n_neurons), dtype=np.int32)

  for row in coactivation_pairs.iter_rows():
    i, j, count = row
    coactivation_matrix[i, j] = count

  # Normalize the coactivation matrix
  row_sums = coactivation_matrix.sum(axis=1, keepdims=True)
  row_sums = np.where(row_sums == 0, 1, row_sums)
  coactivation_matrix = coactivation_matrix / row_sums

  return coactivation_matrix


def get_top_n_tokens(lf_sequences: pl.LazyFrame, n: int) -> list[int]:
  return (
    lf_sequences.select(pl.col('sequence').explode())
    .filter(pl.col('sequence').ne(0))
    .group_by('sequence')
    .len()
    .sort('len', descending=True)
    .limit(n)
    .collect()
    .get_column('sequence')
    .to_list()
  )


def get_first_occurrence_gating_weights(
  lf_neurons: pl.LazyFrame,
  token_ids: list[int],
  layer: int,
  channel: int,
  n_neurons: int,
) -> jt.Float[np.ndarray, 'n_tokens n_neurons']:
  df_first = (
    lf_neurons.filter(
      pl.col('layer').eq(layer),
      pl.col('channel').eq(channel),
      pl.col('token_id').is_in(token_ids),
    )
    .group_by('token_id')
    .agg(
      pl.col('sequence_id').first(),
      pl.col('position').first(),
    )
    .collect()
  )

  df_gating = (
    lf_neurons.filter(pl.col('layer').eq(layer), pl.col('channel').eq(channel))
    .join(df_first.lazy(), on=['token_id', 'sequence_id', 'position'], how='inner')
    .select('token_id', 'neuron', 'score')
    .collect()
  )

  y = np.zeros((len(token_ids), n_neurons), dtype=np.float32)
  token_to_idx = {token: i for i, token in enumerate(token_ids)}

  for row in df_gating.iter_rows():
    token_id, neuron, score = row
    if neuron < n_neurons:
      y[token_to_idx[token_id], neuron] = score

  return y


def get_input_embeddings(
  ckpt_path: str, token_ids: list[int]
) -> jt.Float[np.ndarray, 'n_tokens d_model']:
  _, exp_state = Experiment.load(ckpt_path)
  embedding = exp_state.params['params']['embedder']['embedding']
  return np.array(embedding[np.array(token_ids)])


def embed_gating_patterns(
  lf_neurons: pl.LazyFrame,
  lf_sequences: pl.LazyFrame,
  n_sequences: int,
  n_neurons: int,
  filter_n_most_frequent_tokens: int | None,
  layers: tuple[int, ...],
  channel: int,
  binary_gates: bool,
) -> pl.DataFrame:
  """Embed gating patterns using UMAP."""

  if filter_n_most_frequent_tokens is not None:
    top_tokens = get_top_n_tokens(lf_sequences, filter_n_most_frequent_tokens)
    lf_neurons = lf_neurons.filter(pl.col('token_id').is_in(top_tokens))

  all_embeddings = []

  for layer in layers:
    neuron_gatings, neuron_gatings_metadata = create_sparse_gating_matrix(
      lf_neurons,
      n_sequences=n_sequences,
      n_neurons=n_neurons,
      layer=layer,
      channel=channel,
      binary_gates=binary_gates,
    )
    sparsity = 1 - neuron_gatings.nnz / (
      neuron_gatings.shape[0] * neuron_gatings.shape[1]
    )
    print(f'Layer {layer} - Sparse matrix shape: {neuron_gatings.shape}')
    print(f'Layer {layer} - Sparsity: {sparsity:.4f}')

    print(f'Running UMAP for layer {layer}...')
    reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15, verbose=True)
    embedding = reducer.fit_transform(neuron_gatings)

    layer_df = (
      neuron_gatings_metadata.join(
        lf_sequences,
        on='sequence_id',
        how='left',
      )
      .pipe(add_input_token)
      .pipe(add_target_token)
      .select('input_token', 'target_token')
      .collect()
      .with_columns(  # ty:ignore[unresolved-attribute]
        x=embedding[:, 0],
        y=embedding[:, 1],
        layer=pl.lit(layer),
      )
      .drop_nulls()  # drop rows without target token
    )

    all_embeddings.append(layer_df)

  return pl.concat(all_embeddings)


def find_nearest_neighbor_gatings(
  neuron_gatings: sparse.spmatrix,
  target_gatings: sparse.spmatrix,
  n_nearest_neighbors: int,
) -> tuple[
  jt.Float[np.ndarray, 'patterns neighbors'],
  jt.Int[np.ndarray, 'patterns neighbors'],
]:
  """Find nearest neighbors for each gating pattern."""
  nearest_neighbor_estimator = NearestNeighbors(
    n_neighbors=n_nearest_neighbors,
    algorithm='auto',
    metric='cosine',
  ).fit(neuron_gatings)

  return nearest_neighbor_estimator.kneighbors(target_gatings)


def join_gating_patterns_with_sequences(
  lf_sequences: pl.LazyFrame,
  target_pattern_ids: list[int],
  neighbors: jt.Int[np.ndarray, 'patterns neighbors'],
  distances: jt.Float[np.ndarray, 'patterns neighbors'],
  lf_neuron_gatings_metadata: pl.LazyFrame,
) -> pl.LazyFrame:
  df_neighbors = pl.DataFrame(
    dict(
      pattern_id=target_pattern_ids,
      neighbor_pattern_id=neighbors.tolist(),
      distance=distances.tolist(),
    )
  )

  # Target metadata
  lf_target_meta = lf_neuron_gatings_metadata.select(
    pl.col('pattern_id'),
    pl.col('sequence_id').alias('target_sequence_id'),
    pl.col('position').alias('position'),
    pl.col('layer').alias('layer'),
    pl.col('channel').alias('channel'),
  )

  # Target sequence tokens
  lf_target_seq = lf_sequences.select(
    pl.col('sequence_id').alias('target_sequence_id'),
    pl.col('detokenized').alias('target_detokenized'),
  )

  # Neighbor metadata
  lf_neighbor_meta = lf_neuron_gatings_metadata.select(
    pl.col('pattern_id').alias('neighbor_pattern_id'),
    pl.col('sequence_id').alias('neighbor_sequence_id'),
    pl.col('position').alias('neighbor_position'),
  )

  # Neighbor sequence tokens
  lf_neighbor_seq = lf_sequences.select(
    pl.col('sequence_id').alias('neighbor_sequence_id'),
    pl.col('detokenized').alias('neighbor_detokenized'),
    pl.col('top_k_output_tokens').alias('neighbor_top_k_output_tokens_all'),
  )

  lf = (
    df_neighbors.lazy()
    .explode('neighbor_pattern_id', 'distance')
    .join(lf_target_meta, on='pattern_id', how='left')
    .join(lf_target_seq, on='target_sequence_id', how='left')
    .join(lf_neighbor_meta, on='neighbor_pattern_id', how='left')
    .join(lf_neighbor_seq, on='neighbor_sequence_id', how='left')
  )

  # Extract token_at_position
  lf = lf.with_columns(
    pl.col('target_detokenized').list.get(pl.col('position')).alias('token_at_position')
  )

  return lf


def add_neighbor_input_excerpt(lf: pl.LazyFrame, window_size: int = 5) -> pl.LazyFrame:
  """Extract an excerpt around the neighbor_position and highlight the active token."""
  return (
    lf.with_columns(pl.col('neighbor_position').cast(pl.Int32))
    .with_columns(
      pl.col('neighbor_detokenized')
      .list.get(pl.col('neighbor_position'))
      .alias('neighbor_active_token'),
      pl.max_horizontal(pl.lit(0), pl.col('neighbor_position') - window_size).alias(
        'window_start'
      ),
      pl.min_horizontal(
        pl.col('neighbor_detokenized').list.len(),
        pl.col('neighbor_position') + window_size + 1,
      ).alias('window_end'),
    )
    .with_columns(
      (pl.col('neighbor_position') - pl.col('window_start')).alias('relative_pos'),
      pl.col('neighbor_detokenized')
      .list.slice(pl.col('window_start'), pl.col('window_end') - pl.col('window_start'))
      .alias('window_tokens'),
    )
    .with_columns(
      pl.concat_str(
        [
          pl.col('window_tokens').list.slice(0, pl.col('relative_pos')).list.join(''),
          pl.lit('**'),
          pl.col('neighbor_active_token'),
          pl.lit('**'),
          pl.col('window_tokens').list.slice(pl.col('relative_pos') + 1).list.join(''),
        ]
      ).alias('input_excerpt')
    )
    .with_columns(pl.col('input_excerpt').str.replace_all('<|pad|>', '', literal=True))
  )


def add_neighbor_top_k_predictions(lf: pl.LazyFrame) -> pl.LazyFrame:
  return lf.with_columns(
    pl.col('neighbor_top_k_output_tokens_all')
    .list.get(pl.col('neighbor_position'))
    .alias('top_k_predictions')
  )


def nearest_neighbor_gating_patterns(
  lf_neurons: pl.LazyFrame,
  lf_sequences: pl.LazyFrame,
  n_sequences: int,
  n_neurons: int,
  n_nearest_neighbors: int,
  layer: int,
  channel: int,
  binary_gates: bool,
  target_sequence_id: int,
) -> pl.DataFrame:
  """
  Find nearest neighbors for neuron gating patterns and add corresponding text excerpts.
  """
  neuron_gatings, neuron_gatings_metadata = create_sparse_gating_matrix(
    lf_neurons,
    n_sequences=n_sequences,
    n_neurons=n_neurons,
    layer=layer,
    channel=channel,
    binary_gates=binary_gates,
  )

  # Filter metadata for target sequence
  target_pattern_ids = (
    neuron_gatings_metadata.filter(pl.col('sequence_id') == target_sequence_id)
    .select('pattern_id')
    .collect()['pattern_id']
    .to_list()
  )
  target_gatings = neuron_gatings[target_pattern_ids]

  distances, neighbors = find_nearest_neighbor_gatings(
    neuron_gatings=neuron_gatings,
    target_gatings=target_gatings,
    n_nearest_neighbors=n_nearest_neighbors + 1,  # +1 to compensate for self-matches
  )

  return (
    join_gating_patterns_with_sequences(
      lf_sequences, target_pattern_ids, neighbors, distances, neuron_gatings_metadata
    )
    .filter(pl.col('pattern_id') != pl.col('neighbor_pattern_id'))  # filter self-matches
    .pipe(add_neighbor_input_excerpt)
    .pipe(add_neighbor_top_k_predictions)
    .select(
      pl.col('position'),
      pl.col('token_at_position'),
      pl.col('channel'),
      pl.col('layer'),
      pl.col('neighbor_pattern_id'),
      pl.col('distance'),
      pl.col('input_excerpt'),
      pl.col('top_k_predictions'),
    )
    .sort(['layer', 'channel', 'position', 'distance'])
    .collect()
  )


def embed_input_embedding(ckpt_path: str) -> pl.DataFrame:
  exp_config, exp_state = Experiment.load(ckpt_path)

  # Get the tokenizer mapping from IDs to tokens
  tokenizer_path = exp_config.dataset.tokenizer_path
  if tokenizer_path is None:
    tokenizer_path = './configs/tokenizer'
  tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
  id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
  id_to_token = {k: id_to_token[k] for k in sorted(id_to_token)}

  # Umap embed the input embedding
  exp_config, exp_state = Experiment.load(ckpt_path)
  assert isinstance(exp_config.dataset, TinystoriesConfig)
  params_embedding = exp_state.params['params']['embedder']['embedding']
  reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15)
  embedding = reducer.fit_transform(params_embedding)

  return pl.DataFrame(
    {
      'x': embedding[:, 0],
      'y': embedding[:, 1],
      'input_token': id_to_token.values(),
      'layer': 'embedding',
    }
  )


def plot_embedding(df: pl.DataFrame, color_key: str) -> Figure:
  fig = px.scatter(
    df.to_pandas(),
    x='x',
    y='y',
    color=color_key,
    facet_col='layer',
    hover_data={color_key: True, 'x': False, 'y': False},
  )

  return fig


def main(
  ckpt_path: str,
  target_sequence_id: int = 1,
  n_nearest_neighbors: int = 20,
  batch_size: int = 64,
  n_batches: int = 2,
  n_neurons: int | None = None,
  filter_n_most_frequent_tokens: int | None = 100,
  layers: list[int] = [0, 1, 2, 3],  # noqa: B006
  channel: int = 0,
  seed: int = 0,
  n_top_output_tokens: int = 5,
) -> None:
  # os.environ['POLARS_VERBOSE'] = '1'
  n_sequences = batch_size * n_batches

  typer.echo('Embed input embedding...')
  df = embed_input_embedding(ckpt_path)
  fig = plot_embedding(df, color_key='input_token')
  fig.write_html(os.path.join(ckpt_path, 'input_embedding.html'))

  typer.echo('Collecting gating records...')
  df_neurons, df_sequences = collect_gating_records(
    ckpt_path=ckpt_path,
    dataset='tinystories',
    batch_size=batch_size,
    n_batches=n_batches,
    layers=tuple(layers),
    channels=(channel,),
    seed=seed,
    n_top_output_tokens=n_top_output_tokens,
  )
  if n_neurons is not None:
    df_neurons = df_neurons.filter(pl.col('neuron') < n_neurons)
  else:
    n_neurons = df_neurons.select(pl.col('neuron').max() + 1).item()

  df_neurons.write_parquet(os.path.join(ckpt_path, 'gating_weights.parquet'))

  lf_neurons = df_neurons.lazy()
  lf_sequences = df_sequences.lazy()

  if filter_n_most_frequent_tokens is not None:
    typer.echo('Computing CKA between input embeddings and first layer gating weights...')
    top_tokens = get_top_n_tokens(lf_sequences, filter_n_most_frequent_tokens)

    x = get_input_embeddings(ckpt_path, top_tokens)
    y = get_first_occurrence_gating_weights(
      lf_neurons, top_tokens, layer=layers[0], channel=channel, n_neurons=n_neurons
    )

    cka_score = float(cka.linear_cka(x, y))
    typer.echo(
      f'Linear CKA (Input Embeddings vs Layer {layers[0]} Gating): {cka_score:.4f}'
    )

  typer.echo('Extracting coactivation matrices...')
  for layer in layers:
    coactivation_matrix = create_neuron_coactivation_matrix(
      lf_neurons, n_neurons=n_neurons, layer=layer, channel=channel
    )
    typer.echo(
      f'Created neuron coactivation matrix for layer {layer} '
      f'with shape {coactivation_matrix.shape}'
    )
    np.save(
      os.path.join(ckpt_path, f'neuron_coactivation_matrix_layer_{layer}.npy'),
      coactivation_matrix,
    )

  typer.echo('Extracting gating embeddings...')
  df_gating_embedding = embed_gating_patterns(
    lf_neurons,
    lf_sequences,
    n_sequences=n_sequences,
    n_neurons=n_neurons,
    filter_n_most_frequent_tokens=filter_n_most_frequent_tokens,
    layers=tuple(layers),
    channel=channel,
    binary_gates=False,
  )
  df_gating_embedding.write_parquet(os.path.join(ckpt_path, 'gating_embedding.parquet'))

  fig = plot_embedding(df_gating_embedding, color_key='input_token')
  fig.write_html(os.path.join(ckpt_path, 'gating_embedding_input.html'))
  fig = plot_embedding(df_gating_embedding, color_key='target_token')
  fig.write_html(os.path.join(ckpt_path, 'gating_embedding_target.html'))

  typer.echo('Finding nearest neighbor gating patterns...')
  res = pl.concat(
    [
      nearest_neighbor_gating_patterns(
        lf_neurons,
        lf_sequences,
        n_sequences=n_sequences,
        n_neurons=n_neurons,
        n_nearest_neighbors=n_nearest_neighbors,
        layer=layer,
        channel=channel,
        binary_gates=False,
        target_sequence_id=target_sequence_id,
      )
      for layer in layers
    ]
  )
  typer.echo(res)
  res.write_parquet(os.path.join(ckpt_path, 'gating_neighbors.parquet'))
  typer.echo('Done')


if __name__ == '__main__':
  typer.run(main)
