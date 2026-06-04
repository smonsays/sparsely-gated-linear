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
from typing import Iterable

import grain.python as grain
from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers import models
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from hyperlinear.data.dataloading import create_grain_truncated_dataset
from hyperlinear.data.dataloading import load_array_record_data_source
from hyperlinear.data_types import DatasetInfo

# NOTE: Ordering matters for the BPE trainer
SPECIAL_TOKENS = dict(
  pad_token='<|pad|>',
  eos_token='<|endoftext|>',
)


def train_bpe_tokenizer(
  data_iter: Iterable, vocab_size: int, save_path: str
) -> PreTrainedTokenizerFast:
  """
  Train a BPE tokenizer from scratch with a small vocabulary size.

  Args:
    data_iter: An Iterable containing the data to train the tokenizer on
    vocab_size: Size of the vocabulary (number of tokens)
    save_path: Path to save the trained tokenizer

  Returns:
    A PreTrainedTokenizerFast object
  """
  tokenizer = Tokenizer(models.BPE())
  tokenizer.normalizer = normalizers.NFKC()
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

  # Train tokenizer
  logging.info(f'Training BPE tokenizer with vocab size {vocab_size}...')
  trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=list(SPECIAL_TOKENS.values()),
  )
  tokenizer.train_from_iterator(iterator=data_iter, trainer=trainer)
  tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
  tokenizer.decoder = decoders.ByteLevel()

  # Save tokenizer
  os.makedirs(save_path, exist_ok=True)
  tokenizer.save(f'{save_path}/tokenizer.json')
  logging.info(f'Tokenizer saved to {save_path}/tokenizer.json')

  # Create a PreTrainedTokenizerFast from the trained tokenizer
  pretrained_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=f'{save_path}/tokenizer.json',
    **SPECIAL_TOKENS,
  )
  pretrained_tokenizer.save_pretrained(save_path)
  logging.info(f'PreTrainedTokenizerFast saved to {save_path}')

  return pretrained_tokenizer


def create_tinystories_dataloader(
  batch_size: int,
  seq_len: int,
  vocab_size: int,
  seed: int,
  tokenizer_path: str,
  n_workers: int,
) -> tuple[
  grain.IterDataset,
  dict[str, grain.IterDataset],
  dict[str, grain.IterDataset],
  PreTrainedTokenizerFast,
  DatasetInfo,
]:
  # Load data from disk assuming it resides in `$SCRATCH/array_records`
  dataset_name = 'roneneldan/TinyStories'.replace('/', '_')
  path = os.path.join(os.path.expandvars('$SCRATCH/array_records'), dataset_name)
  dataset = dict(
    train=grain.MapDataset.source(
      load_array_record_data_source(os.path.join(path, 'train'))
    ),
    validation=grain.MapDataset.source(
      load_array_record_data_source(os.path.join(path, 'validation'))
    ),
  )
  assert len(dataset['train']) == 2_119_719

  # Use/Train a custom BPE tokenizer with a small vocabulary
  if os.path.exists(f'{tokenizer_path}/tokenizer.json'):
    logging.info(f'Loading existing tokenizer from {tokenizer_path}')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'{tokenizer_path}/tokenizer.json')
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
  else:
    data_iter = dataset['train'].map(lambda x: x.decode()).to_iter_dataset().batch(1000)
    tokenizer = train_bpe_tokenizer(
      data_iter, vocab_size=vocab_size, save_path=tokenizer_path
    )

  for key in SPECIAL_TOKENS:
    assert getattr(tokenizer, key) == SPECIAL_TOKENS[key], 'Mismatched special tokens'

  assert len(tokenizer) == vocab_size, (
    'Mismatch between specified vocab_size and actual tokenizer vocab size'
  )

  # Create dataloaders
  train_loader = create_grain_truncated_dataset(
    dataset=dataset['train'],
    tokenizer=tokenizer,
    batch_size=batch_size,
    seq_len=seq_len,
    shuffle=True,
    add_eos_token=True,
    n_workers=n_workers,
    seed=seed,
  )
  valid_loaders = dict(
    # validation=create_grain_truncated_dataset(
    #   dataset=dataset['validation'],
    #   tokenizer=tokenizer,
    #   batch_size=batch_size,
    #   seq_len=seq_len,
    #   shuffle=False,
    #   add_eos_token=True,
    #   n_workers=n_workers,
    #   seed=seed,
    # )
  )
  eval_loaders = dict()
  dataset_info = DatasetInfo(
    vocab_size=len(tokenizer),
    train_size=len(dataset['train']),
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
  )

  return train_loader, valid_loaders, eval_loaders, tokenizer, dataset_info
