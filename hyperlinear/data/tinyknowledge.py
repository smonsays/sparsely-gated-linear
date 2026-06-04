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
import itertools
import logging
from typing import Iterator

import grain.python as grain
import numpy as np
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from hyperlinear.evaluation.intervention import InterventionBatch

CANONICAL_KNOWLEDGE_TINYSTORIES = {
  'ground': ['icy'],
  'turtle': ['slow'],
  'door': ['open'],
  'doctor': ['nice'],
  'wind': ['strong'],
  'food': ['ready'],
  'lake': ['big'],
  'mum': ['proud'],
}


@dataclasses.dataclass
class TinyknowledgeInterventionRecord:
  prefix_clean: list[int]
  prefix_counterfactual: list[int]
  answers_clean: list[int]
  answers_counterfactual: list[int]


@dataclasses.dataclass
class PadAndFormatAsBatch(grain.MapTransform):
  """Pad right to seq_len and format as InterventionBatch."""

  pad_token_id: int
  bos_token_id: int
  seq_len: int

  def map(self, item: TinyknowledgeInterventionRecord) -> InterventionBatch:
    prefix_clean = [self.bos_token_id, *item.prefix_clean]
    prefix_counterfactual = [self.bos_token_id, *item.prefix_counterfactual]

    seq_len_clean = len(prefix_clean)
    seq_len_cf = len(prefix_counterfactual)

    if seq_len_clean > self.seq_len:
      raise ValueError(
        f'Clean sequence length exceeded. max seq_len={self.seq_len}, got {seq_len_clean}'
      )

    if seq_len_cf > self.seq_len:
      raise ValueError(
        f'Counterfactual sequence length exceeded. max seq_len={self.seq_len}, got {seq_len_cf}'
      )

    if seq_len_clean != seq_len_cf:
      raise ValueError(
        f'Prefix lengths must match for intervention alignment. clean={seq_len_clean}, cf={seq_len_cf}'
      )

    # Pad prefix_clean
    padded_clean = np.full(self.seq_len, self.pad_token_id, dtype=np.int_)
    padded_clean[:seq_len_clean] = prefix_clean

    # Pad prefix_counterfactual
    padded_cf = np.full(self.seq_len, self.pad_token_id, dtype=np.int_)
    padded_cf[:seq_len_cf] = prefix_counterfactual

    return InterventionBatch(
      prefix_clean=padded_clean,
      prefix_counterfactual=padded_cf,
      answers_clean=np.array(item.answers_clean, dtype=np.int_),
      answers_counterfactual=np.array(item.answers_counterfactual, dtype=np.int_),
    )


def filter_and_tokenize_data(
  tokenizer: PreTrainedTokenizerFast,
  top_answers: int,
  data: dict[str, list[str]],
) -> Iterator[dict]:
  """Parse data dict, tokenize, filter and pair examples for intervention."""

  # Build tokenized prefix / answer pairs enforcing some constraints on the tokenization
  noun_to_valid_adjs = {}
  for noun, adjs in data.items():
    noun_tokens = tokenizer.encode(f' {noun}', add_special_tokens=False)
    if len(noun_tokens) != 1:
      continue

    prefix = f'The {noun} was'
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

    valid_adjs = []
    adjs_list = list(adjs.keys()) if isinstance(adjs, dict) else adjs
    for adj in adjs_list:
      # We tokenize the full sequence to allow the tokenizer to naturally handle the
      # whitespace between prefix and answer (e.g. BPE fuses it into the answer token).
      full_tokens = tokenizer.encode(f'{prefix} {adj}', add_special_tokens=False)

      # Ensure the whitespace is consistently tokenized into the answer
      if full_tokens[: len(prefix_tokens)] != prefix_tokens:
        continue

      answer_tokens = full_tokens[len(prefix_tokens) :]

      # Retain only examples where answers/adjs are single tokens
      if len(answer_tokens) != 1:
        continue

      valid_adjs.append(answer_tokens[0])

    if len(valid_adjs) < top_answers:
      continue

    top_adj_ids = valid_adjs[:top_answers]
    noun_to_valid_adjs[noun] = {
      'prefix_tokens': prefix_tokens,
      'answers': top_adj_ids,
    }

  # Generate pairs of clean and counterfactual combinations
  for noun_a, noun_b in itertools.combinations(noun_to_valid_adjs.keys(), 2):
    data_a = noun_to_valid_adjs[noun_a]
    data_b = noun_to_valid_adjs[noun_b]

    # Enforce that both prefixes have the exact same token length
    if len(data_a['prefix_tokens']) != len(data_b['prefix_tokens']):
      continue

    # Yield clean A, counterfactual B
    yield {
      'prefix_clean': data_a['prefix_tokens'],
      'prefix_counterfactual': data_b['prefix_tokens'],
      'answers_clean': data_a['answers'],
      'answers_counterfactual': data_b['answers'],
    }

    # Yield clean B, counterfactual A
    yield {
      'prefix_clean': data_b['prefix_tokens'],
      'prefix_counterfactual': data_a['prefix_tokens'],
      'answers_clean': data_b['answers'],
      'answers_counterfactual': data_a['answers'],
    }


def create_tinyknowledge_interventionloader(
  tokenizer: PreTrainedTokenizerFast,
  batch_size: int,
  seq_len: int,
  top_answers: int = 1,
  n_workers: int = 0,
) -> grain.IterDataset:
  """Create a DataLoader for the Factual Knowledge intervention task."""

  valid_records = [
    TinyknowledgeInterventionRecord(
      prefix_clean=pair['prefix_clean'],
      prefix_counterfactual=pair['prefix_counterfactual'],
      answers_clean=pair['answers_clean'],
      answers_counterfactual=pair['answers_counterfactual'],
    )
    for pair in filter_and_tokenize_data(
      tokenizer, top_answers, CANONICAL_KNOWLEDGE_TINYSTORIES
    )
  ]

  logging.info(f'Loaded {len(valid_records)} valid intervention pairs')

  # Build and return the pipeline
  ds = grain.MapDataset.source(valid_records)
  ds = ds.map(
    PadAndFormatAsBatch(
      pad_token_id=tokenizer.pad_token_id,
      bos_token_id=tokenizer.eos_token_id,
      seq_len=seq_len,
    )
  )
  ds = ds.to_iter_dataset()
  ds = ds.batch(batch_size=batch_size, drop_remainder=False)

  if n_workers > 0:
    ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers=n_workers))

  return ds
