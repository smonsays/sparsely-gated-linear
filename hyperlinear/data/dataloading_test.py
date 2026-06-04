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

import math

import grain.python as grain
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from hyperlinear.data import dataloading
from hyperlinear.data.slimpajama import create_gpt2_tokenizer


class DataloadingTest(parameterized.TestCase):
  def test_tokenize_truncate_pad(
    self, seq_len: int = 20, short_text: str = 'Hello'
  ) -> None:
    """Test TokenizeHuggingface"""
    tokenizer = create_gpt2_tokenizer(local_files_only=False)
    tokenize_transform = dataloading.TokenizeTruncatePad(tokenizer, seq_len)
    tokenized = tokenize_transform.map(short_text)
    self.assertEqual(len(tokenized), seq_len)

    n_text_tokens = len(tokenizer.encode(short_text))
    tokens_text = tokenized[:n_text_tokens]
    tokens_padding = tokenized[n_text_tokens:]

    self.assertGreater(len(tokens_padding), 0, 'Expected at least one padding token')
    self.assertTrue(
      (tokens_padding == 0).all(),
      f'Expected all padding tokens to be 0, but got: {tokens_padding}',
    )
    self.assertTrue(
      (tokens_text != 0).any(),
      'Expected at least some non-padding tokens to be non-{pad_token_id}',
    )

  @parameterized.parameters(
    dict(seq_len=16),
    dict(seq_len=7),
    dict(seq_len=3),
  )
  def test_create_grain_chunked_dataset(
    self, seq_len: int, batch_size: int = 4, n_workers: int = 0
  ) -> None:
    sample_texts = [
      b'This is a sample text for testing the evaluation dataset.',
      b'Another sample text to ensure we have enough data.',
      b'A short one!',
      b'One?',
    ]
    data_source = grain.InMemoryDataSource(sample_texts)
    dataset = grain.MapDataset.source(data_source)
    tokenizer = create_gpt2_tokenizer(local_files_only=False)
    eval_dataset = dataloading.create_grain_chunked_dataset(
      dataset=dataset,
      tokenizer=tokenizer,
      batch_size=batch_size,
      seq_len=seq_len,
      shuffle=False,
      n_workers=n_workers,
      seed=0,
    )

    batches = []
    for batch in eval_dataset:
      self.assertIsNotNone(batch)
      self.assertIsNotNone(batch.x)
      self.assertEqual(batch.x.shape[0], batch_size, 'Batch size mismatch')
      self.assertEqual(batch.x.shape[1], seq_len, 'Sequence length mismatch')
      self.assertTrue(np.issubdtype(batch.x.dtype, np.integer), 'Tokens not integers')
      batches.append(batch.x)

    data_tokens = np.concatenate(batches, axis=0).flatten()
    data_decoded = tokenizer.decode(data_tokens, skip_special_tokens=False)

    # Verify that when seq_len is large enough to fit individual sample_texts,
    # we retrieve the whole sequence (+ padding)
    sample_text_tokens = [tokenizer.encode(text.decode('utf-8')) for text in sample_texts]
    texts_that_fit = [
      text
      for text, tokens in zip(sample_texts, sample_text_tokens, strict=True)
      if len(tokens) <= seq_len
    ]

    for original_text in texts_that_fit:
      decoded_text = original_text.decode('utf-8')
      self.assertIn(
        decoded_text,
        data_decoded,
        f'Expected to find "{decoded_text}" in decoded output when seq_len={seq_len}',
      )

  def test_create_grain_packed_dataset_short_sequences(
    self, seq_len: int = 512, batch_size: int = 1, n_workers: int = 0
  ) -> None:
    data = [
      b'This is a sample text for testing the evaluation dataset.',
      b'Another sample text to ensure we have enough data.',
      b'A short one!',
      b'One?',
    ]
    data_source = grain.InMemoryDataSource(data)
    dataset = grain.MapDataset.source(data_source)
    tokenizer = create_gpt2_tokenizer(local_files_only=False)
    eval_dataset = dataloading.create_grain_packed_dataset(
      dataset=dataset,
      tokenizer=tokenizer,
      batch_size=batch_size,
      seq_len=seq_len,
      shuffle=False,
      n_workers=n_workers,
      seed=0,
    )

    data_cat = '<|endoftext|>'.join([t.decode('utf-8') for t in data]) + '<|endoftext|>'
    data_tokenized = tokenizer.encode(data_cat, return_tensors='np').squeeze()
    data_tokenized_and_padded = np.zeros((seq_len,), dtype=data_tokenized.dtype)
    data_tokenized_and_padded[: len(data_tokenized)] = data_tokenized

    for batch in eval_dataset:
      np.testing.assert_equal(batch.x.squeeze(), data_tokenized_and_padded)
      break

  def test_create_grain_packed_dataset_long_sequences(
    self, seq_len: int = 5, batch_size: int = 1, n_workers: int = 0
  ) -> None:
    data = [
      b'This is a sample text for testing the evaluation dataset.',
      b'Another sample text to ensure we have enough data.',
      b'A short one!',
      b'One?',
    ]
    data_source = grain.InMemoryDataSource(data)
    dataset = grain.MapDataset.source(data_source)
    tokenizer = create_gpt2_tokenizer(local_files_only=False)
    eval_dataset = dataloading.create_grain_packed_dataset(
      dataset=dataset,
      tokenizer=tokenizer,
      batch_size=batch_size,
      seq_len=seq_len,
      shuffle=False,
      n_workers=n_workers,
      seed=0,
    )

    # Create expected output by manually, concatenating, padding and splitting
    data_cat = '<|endoftext|>'.join([t.decode('utf-8') for t in data]) + '<|endoftext|>'
    data_tokenized = tokenizer.encode(data_cat, return_tensors='np').squeeze()
    num_chunks = math.ceil(len(data_tokenized) / seq_len)
    padded_size = num_chunks * seq_len
    pad_len = padded_size - len(data_tokenized)
    data_padded = np.pad(data_tokenized, (0, pad_len), mode='constant', constant_values=0)
    data_padded = data_padded.reshape(num_chunks, seq_len)

    all_batches = np.concatenate([batch.x for batch in eval_dataset], axis=0)
    np.testing.assert_equal(all_batches, data_padded)


if __name__ == '__main__':
  absltest.main()
