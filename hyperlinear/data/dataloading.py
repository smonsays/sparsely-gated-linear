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
import os
from pathlib import Path

import grain.python as grain
import jaxtyping as jt
import numpy as np
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from hyperlinear.data_types import Batch


@dataclasses.dataclass
class DecodeArrayRecords(grain.MapTransform):
  @staticmethod
  def map(data: bytes) -> str:
    return data.decode(encoding='utf-8')


@dataclasses.dataclass
class TokenizeTruncatePad(grain.MapTransform):
  """Tokenize, truncate and pad a sequence to the specified length."""

  def __init__(self, tokenizer: PreTrainedTokenizerFast, seq_len: int) -> None:
    self.seq_len = seq_len
    self.tokenizer = tokenizer

  def map(self, text: str) -> jt.ArrayLike:
    tokenized = self.tokenizer(
      text,
      padding='max_length',
      max_length=self.seq_len,
      truncation=True,
      return_tensors='np',
      padding_side='right',
    )
    return tokenized['input_ids'].squeeze(0)


@dataclasses.dataclass
class Tokenize(grain.MapTransform):
  def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
    self.tokenizer = tokenizer

  def map(self, text: str) -> jt.ArrayLike:
    tokenized = self.tokenizer.encode(text, return_tensors='np')
    return tokenized.squeeze(0)


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  @staticmethod
  def map(sample: jt.ArrayLike) -> Batch:
    return Batch(x=sample)


@dataclasses.dataclass
class SplitLongSequencesPad(grain.experimental.FlatMapTransform):
  """Split sequences longer than max_length into chunks of max_length."""

  def __init__(self, max_length: int, pad_token_id: int, drop_remainder: bool) -> None:
    self.max_length = max_length
    self.pad_token_id = pad_token_id
    self.drop_remainder = drop_remainder
    self.max_fan_out = 5  # Slows down noticeably for large values

  def flat_map(self, seq: jt.ArrayLike) -> list[jt.ArrayLike]:
    def pad(x: jt.ArrayLike) -> jt.ArrayLike:
      padding = np.full(self.max_length - len(x), self.pad_token_id, dtype=x.dtype)
      return np.concatenate([x, padding])

    # Pad short sequences to max_length
    if len(seq) <= self.max_length:
      return [pad(seq)]

    # Chunk longer sequences into max_length pieces
    chunks = []
    for i in range(0, len(seq), self.max_length):
      chunk = seq[i : i + self.max_length]
      if len(chunk) == self.max_length:
        chunks.append(chunk)
      elif not self.drop_remainder:
        chunks.append(pad(chunk))
    return chunks


def load_array_record_data_source(
  dataset_path: str, file_extension: str = '.arecord'
) -> grain.ArrayRecordDataSource:
  """Load all files located at dataset_path as grain.ArrayRecordDataSource."""
  assert os.path.exists(dataset_path), f'dataset path {dataset_path} does not exist.'

  # Get files ending in `file_extension` in sorted order
  path_pattern = Path(f'{dataset_path}/*{file_extension}')
  files = path_pattern.parent.glob(path_pattern.name)
  sorted_files = sorted(files, key=lambda x: int(x.stem.split('_')[-1]))

  data_source = grain.ArrayRecordDataSource(sorted_files)
  return data_source


def create_grain_truncated_dataset(
  dataset: grain.MapDataset,
  tokenizer: PreTrainedTokenizerFast,
  batch_size: int,
  seq_len: int,
  shuffle: bool,
  add_eos_token: bool,
  n_workers: int,
  seed: int,
) -> grain.IterDataset:
  ds = dataset.seed(seed)

  if shuffle:
    ds = ds.shuffle()

  ds = ds.map(DecodeArrayRecords())
  if add_eos_token:
    ds = ds.map(lambda x: x + tokenizer.eos_token)
  ds = ds.map(TokenizeTruncatePad(tokenizer, seq_len))
  ds = ds.to_iter_dataset()

  ds = ds.batch(batch_size=batch_size, drop_remainder=True)
  ds = ds.map(NormalizeFeatures())
  ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=n_workers))

  return ds


def create_grain_packed_dataset(
  dataset: grain.MapDataset,
  tokenizer: PreTrainedTokenizerFast,
  batch_size: int,
  seq_len: int,
  shuffle: bool,
  n_workers: int,
  seed: int,
) -> grain.IterDataset:
  assert tokenizer.pad_token_id == 0, 'ConcatThenSplitIterDataset assumes pad_token_id=0'

  ds = dataset.seed(seed)

  if shuffle:
    ds = ds.shuffle()

  ds = ds.map(DecodeArrayRecords())
  ds = ds.map(lambda x: x + tokenizer.eos_token)
  ds = ds.map(Tokenize(tokenizer))
  ds = ds.map(lambda x: dict(text=x))
  ds = ds.to_iter_dataset()

  # FirstFitPackIterDataset implements first-fit packing of sequences but requires all
  # dataset sequences to already be <= seq_len
  # iter_packed = grain.experimental.FirstFitPackIterDataset(
  #   ds,
  #   length_struct=dict(text=seq_len),
  #   padding_struct=dict(text=tokenizer.pad_token_id),
  #   num_packing_bins=128,
  #   seed=seed,
  # )

  # Concat-then-split packing with sequences being split across packed sequences
  # Implicitly pads with 0
  ds = grain.experimental.ConcatThenSplitIterDataset(ds, length_struct=dict(text=seq_len))
  ds = ds.map(lambda x: x['text'])
  ds = ds.batch(batch_size=batch_size, drop_remainder=True)
  ds = ds.map(NormalizeFeatures())
  ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=n_workers))

  return ds


def create_grain_chunked_dataset(
  dataset: grain.MapDataset,
  tokenizer: PreTrainedTokenizerFast,
  batch_size: int,
  seq_len: int,
  shuffle: bool,
  n_workers: int,
  seed: int,
) -> grain.IterDataset:
  ds = dataset.seed(seed)

  if shuffle:
    ds = ds.shuffle()

  ds = ds.map(DecodeArrayRecords())
  ds = ds.map(Tokenize(tokenizer))

  # Split sequences > seq_len into chunks of seq_len and drop incomplete last chunk
  transform = SplitLongSequencesPad(seq_len, tokenizer.pad_token_id, drop_remainder=True)
  ds = grain.experimental.FlatMapMapDataset(ds, transform)

  ds = ds.to_iter_dataset()
  ds = ds.batch(batch_size=batch_size, drop_remainder=True)
  ds = ds.map(NormalizeFeatures())
  ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=n_workers))

  return ds


def create_grain_dataloader(
  data_source: grain.RandomAccessDataSource,
  tokenizer: PreTrainedTokenizerFast,
  batch_size: int,
  seq_len: int,
  shuffle: bool,
  n_workers: int,
  seed: int,
) -> grain.DataLoader:
  index_sampler = grain.IndexSampler(
    num_records=len(data_source), num_epochs=1, shuffle=shuffle, seed=seed
  )
  operations = [
    DecodeArrayRecords(),
    TokenizeTruncatePad(tokenizer, seq_len),
    grain.Batch(batch_size, drop_remainder=True),
    NormalizeFeatures(),
  ]
  return grain.DataLoader(
    data_source=data_source,
    operations=operations,
    sampler=index_sampler,
    worker_count=n_workers,
  )
