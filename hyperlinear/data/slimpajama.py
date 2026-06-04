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

import json
import os
import tempfile
from typing import Literal

import grain.python as grain
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
from transformers import PreTrainedTokenizerFast

from hyperlinear.data.dataloading import create_grain_packed_dataset
from hyperlinear.data.dataloading import create_grain_truncated_dataset
from hyperlinear.data.dataloading import load_array_record_data_source
from hyperlinear.data_types import DatasetInfo

# Fix for race condition on tokenizer, see https://github.com/huggingface/tokenizers/issues/537
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


SPECIAL_TOKENS = dict(pad_token='<|pad|>')


def create_gpt2_tokenizer(local_files_only: bool) -> GPT2TokenizerFast:
  tokenizer = AutoTokenizer.from_pretrained(
    'openai-community/gpt2', local_files_only=local_files_only
  )
  tokenizer.add_special_tokens(SPECIAL_TOKENS)
  tokenizer = set_gpt2_tokenizer_pad_id_to_zero(tokenizer)

  for key in SPECIAL_TOKENS:
    assert getattr(tokenizer, key) == SPECIAL_TOKENS[key], 'Mismatched special tokens'

  return tokenizer


def set_gpt2_tokenizer_pad_id_to_zero(tokenizer: GPT2TokenizerFast) -> GPT2TokenizerFast:
  """Set the pad_token_id to 0."""
  # HACK: Save tokenizer to disk, perform surgery on tokenizer.json and reload
  tokenizer_path = tempfile.mkdtemp()
  tokenizer.save_pretrained(tokenizer_path)
  with open(os.path.join(tokenizer_path, 'tokenizer.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

  # Find current <|pad|> ID and then set it to 0
  for item in data['added_tokens']:
    if item['content'] == '<|pad|>':
      id_pad = item['id']
      item['id'] = 0
      break

  # Ensure the ID of '!' is 0
  id_exclamation = data['model']['vocab']['!']
  assert id_exclamation == 0, 'Assuming "!" has pad_id=0 but it is not the case.'

  # Swap IDs of '!' and '<|pad|>'
  data['model']['vocab']['!'] = id_pad
  data['model']['vocab']['<|pad|>'] = id_exclamation

  with open(os.path.join(tokenizer_path, 'tokenizer.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

  return GPT2TokenizerFast.from_pretrained(tokenizer_path)


def create_slimpajama_dataloader(
  batch_size: int,
  seq_len: int,
  seed: int,
  variant: Literal['full', '6B'],
  local_files_only: bool,
  n_workers: int,
) -> tuple[
  grain.IterDataset,
  dict[str, grain.IterDataset],
  dict[str, grain.IterDataset],
  PreTrainedTokenizerFast,
  DatasetInfo,
]:
  match variant:
    case 'full':
      dataset_name = 'cerebras/SlimPajama-627B'
      train_size = 590_394_625
    case '6B':
      dataset_name = 'DKYoon/SlimPajama-6B'
      train_size = 5_489_000
    case _:
      raise ValueError(f'Unknown variant {variant} of slimpajama dataset.')

  path = os.path.join(
    os.path.expandvars('$SCRATCH/array_records'), dataset_name.replace('/', '_')
  )
  dataset = dict(
    train=grain.MapDataset.source(
      load_array_record_data_source(os.path.join(path, 'train'))
    ),
    validation=grain.MapDataset.source(
      load_array_record_data_source(os.path.join(path, 'validation'))
    ),
    test=grain.MapDataset.source(
      load_array_record_data_source(os.path.join(path, 'test'))
    ),
  )
  assert len(dataset['train']) == train_size, f'Dataset length: {len(dataset["train"])}'

  tokenizer = create_gpt2_tokenizer(local_files_only=local_files_only)

  train_loader = create_grain_packed_dataset(
    dataset=dataset['train'],
    tokenizer=tokenizer,
    batch_size=batch_size,
    seq_len=seq_len,
    shuffle=True,
    n_workers=n_workers,
    seed=seed,
  )
  validation_loaders = dict(
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

  dataset_info = DatasetInfo(
    vocab_size=len(tokenizer),
    train_size=train_size,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
  )

  # Add perplexity-based evaluation loaders
  eval_loaders = dict()
  eval_loaders['test'] = create_grain_truncated_dataset(
    dataset=dataset['test'],
    tokenizer=tokenizer,
    batch_size=batch_size,
    seq_len=seq_len,
    shuffle=False,
    add_eos_token=True,
    n_workers=n_workers,
    seed=seed,
  )

  return train_loader, validation_loaders, eval_loaders, tokenizer, dataset_info
