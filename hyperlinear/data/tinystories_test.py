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
import tempfile

import numpy as np
import tensorflow_datasets as tfds
from absl.testing import absltest
from absl.testing import parameterized

from hyperlinear.data import tinystories


class TinyStoriesTest(parameterized.TestCase):
  def setUp(self) -> None:
    super().setUp()
    # Create a temporary directory for tokenizer files
    self.test_dir = tempfile.TemporaryDirectory()
    self.tokenizer_path = self.test_dir.name

  def tearDown(self) -> None:
    # Clean up the temporary directory
    self.test_dir.cleanup()
    super().tearDown()

  def test_dataloader_batch_shape(self) -> None:
    """Test that the dataloader produces batches with the correct shape."""
    batch_size = 32
    seq_len = 64
    vocab_size = 8192
    n_workers = 2
    seed = 0

    train_loader = tinystories.create_tinystories_dataloader(
      batch_size=batch_size,
      seq_len=seq_len,
      vocab_size=vocab_size,
      seed=seed,
      tokenizer_path='./configs/tokenizer',
      n_workers=n_workers,
    )[0]

    first_batch = next(iter(train_loader)).x
    self.assertEqual(first_batch.shape, (batch_size, seq_len))
    self.assertEqual(first_batch.dtype, np.int64)
    self.assertTrue((first_batch >= 0).all())
    self.assertTrue((first_batch < vocab_size).all())

  def test_tinystories_tfds_benchmark(self) -> None:
    """Benchmark tinystories dataloading speed."""
    batch_size = 128
    seq_len = 512
    vocab_size = 8192
    seed = 0
    n_workers = 2
    num_iter = 100

    train_loader = tinystories.create_tinystories_dataloader(
      batch_size=batch_size,
      seq_len=seq_len,
      vocab_size=vocab_size,
      seed=seed,
      tokenizer_path='./configs/tokenizer',
      n_workers=n_workers,
    )[0]
    tfds.benchmark(train_loader, num_iter=num_iter)

  def test_train_bpe_tokenizer(self) -> None:
    """Test BPE tokenizer training and verify that it acts as expected."""
    tiny_dataset = [
      'Once upon a time, there was a little girl named Emma.',
      'Emma loved to play with her toys and read books.',
      'One day, Emma found a magic book in her room.',
      'The book had colorful pictures and funny stories.',
      'Emma read the book every day and learned new words.',
      'She shared the stories with her friends at school.',
      "Everyone loved Emma's wonderful stories from the magic book.",
      'The end of each story always made Emma smile.',
    ]
    vocab_size = 100

    tokenizer = tinystories.train_bpe_tokenizer(
      data_iter=tiny_dataset, vocab_size=vocab_size, save_path=self.tokenizer_path
    )

    # Verify tokenizer files were saved
    self.assertTrue(os.path.exists(f'{self.tokenizer_path}/tokenizer.json'))
    self.assertTrue(os.path.exists(f'{self.tokenizer_path}/tokenizer_config.json'))

    # Verify vocab size
    self.assertEqual(len(tokenizer), vocab_size)

    # Verify special tokens are configured
    self.assertEqual(tokenizer.pad_token, '<|pad|>')

    # Test tokenization
    test_text = 'Emma loved to play with her toys.'
    tokens = tokenizer(test_text, return_tensors='np', add_special_tokens=True)
    self.assertIsNotNone(tokens['input_ids'])
    self.assertGreater(len(tokens['input_ids'][0]), 0)

    # Test that we can decode the tokens back to exact original text
    decoded_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    self.assertEqual(decoded_text.strip(), test_text.strip())


if __name__ == '__main__':
  absltest.main()
