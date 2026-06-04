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

import numpy as np
import tensorflow_datasets as tfds
from absl.testing import absltest
from absl.testing import parameterized
from transformers import AutoTokenizer

from hyperlinear.data import slimpajama


class SlimpajamaTest(parameterized.TestCase):
  def test_gpt2_tokenizer(self) -> None:
    seq_len = 10
    tokenizer = slimpajama.create_gpt2_tokenizer(local_files_only=True)

    def tokenize(sample: dict) -> dict:
      tokenized = tokenizer(
        sample['text'],
        padding='max_length',
        max_length=seq_len,
        truncation=True,
        return_tensors='np',
        padding_side='right',
      )
      sample['text'] = tokenized['input_ids'].squeeze(0)
      return sample

    text = 'Emma loved to play with her toys.'
    tokens = tokenize(dict(text=text))['text']

    assert tokenizer.decode(tokens, skip_special_tokens=True) == text
    assert slimpajama.SPECIAL_TOKENS['pad_token'] in tokenizer.decode(tokens)

  def test_set_gpt2_tokenizer_pad_id_to_zero(self) -> None:
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.add_special_tokens(slimpajama.SPECIAL_TOKENS)

    assert tokenizer.pad_token_id != 0
    assert tokenizer.pad_token == '<|pad|>'
    assert tokenizer.encode('<|pad|>')[0] != 0
    assert tokenizer.encode('!')[0] == 0

    tokenizer = slimpajama.set_gpt2_tokenizer_pad_id_to_zero(tokenizer)

    assert tokenizer.pad_token_id == 0
    assert tokenizer.pad_token == '<|pad|>'
    assert tokenizer.encode('<|pad|>')[0] == 0
    assert tokenizer.encode('!')[0] != 0

  def test_dataloader_batch_shape(self) -> None:
    """Test that the dataloader produces batches with the correct shape."""
    batch_size = 32
    seq_len = 64
    seed = 0

    train_loader, _, _, _, data_info = slimpajama.create_slimpajama_dataloader(
      batch_size=batch_size,
      seq_len=seq_len,
      seed=seed,
      variant='6B',
      n_workers=0,
      local_files_only=True,
    )

    first_batch = next(iter(train_loader)).x
    self.assertEqual(first_batch.shape, (batch_size, seq_len))
    self.assertEqual(first_batch.dtype, np.int64)
    self.assertTrue((first_batch >= 0).all())
    self.assertTrue((first_batch < data_info.vocab_size).all())

  def test_slimpajama_tfds_benchmark(self) -> None:
    """Benchmark slimpajama dataloading speed."""
    batch_size = 128
    seq_len = 512
    seed = 0
    num_iter = 100

    train_loader = slimpajama.create_slimpajama_dataloader(
      batch_size=batch_size,
      seq_len=seq_len,
      seed=seed,
      variant='6B',
      n_workers=0,
      local_files_only=True,
    )[0]
    tfds.benchmark(train_loader, num_iter=num_iter)


if __name__ == '__main__':
  absltest.main()
