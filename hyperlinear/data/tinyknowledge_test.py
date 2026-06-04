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
from absl.testing import absltest
from transformers import PreTrainedTokenizerFast

from hyperlinear.data import tinyknowledge
from hyperlinear.data.tinystories import SPECIAL_TOKENS


class TinyknowledgeTest(absltest.TestCase):
  def setUp(self) -> None:
    super().setUp()

    # Load existing tokenizer
    self.tokenizer_path = './configs/tokenizer'
    self.tokenizer = PreTrainedTokenizerFast(
      tokenizer_file=f'{self.tokenizer_path}/tokenizer.json'
    )
    self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

    # Need at least two valid nouns to form intervention pairs
    self.mock_data = {
      'sky': [
        'blue',
        # Multi-token adj should get filtered
        'asdfölkj',
      ],
      'grass': [
        'green',
      ],
      # Multi-token noun should get filtered out
      'asdfölkj': [
        'wet',
      ],
    }

  def test_dataloader_batch_shape(self) -> None:
    """Test that the batch has the correct InterventionBatch types and shapes."""
    batch_size = 1
    seq_len = 16
    top_answers = 1

    dataloader = tinyknowledge.create_tinyknowledge_interventionloader(
      tokenizer=self.tokenizer,
      batch_size=batch_size,
      seq_len=seq_len,
      top_answers=top_answers,
      n_workers=0,
    )
    batch = next(iter(dataloader))

    self.assertEqual(batch.prefix_clean.shape, (batch_size, seq_len))
    self.assertEqual(batch.prefix_counterfactual.shape, (batch_size, seq_len))
    self.assertEqual(batch.answers_clean.shape, (batch_size, top_answers))
    self.assertEqual(batch.answers_counterfactual.shape, (batch_size, top_answers))
    self.assertEqual(batch.prefix_clean.dtype, np.int_)
    self.assertEqual(batch.prefix_counterfactual.dtype, np.int_)
    self.assertTrue((batch.prefix_clean >= 0).all())
    self.assertTrue((batch.prefix_counterfactual >= 0).all())

  def test_filtering(self) -> None:
    """Test that multi-token adjs and nouns are filtered out and valid pairs are generated."""
    top_answers = 1

    pairs = list(
      tinyknowledge.filter_and_tokenize_data(
        tokenizer=self.tokenizer,
        top_answers=top_answers,
        data=self.mock_data,
      )
    )

    # 2 valid nouns (sky, grass) -> 2 pairs: (sky, grass) and (grass, sky)
    self.assertEqual(len(pairs), 2)

  def test_sequence_length_error(self) -> None:
    """Test that a ValueError is raised if any prefix exceeds the specified seq_len."""
    batch_size = 1
    seq_len = 1
    top_answers = 1

    with self.assertRaises(ValueError):
      dataloader = tinyknowledge.create_tinyknowledge_interventionloader(
        tokenizer=self.tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        top_answers=top_answers,
        n_workers=0,
      )

      # Consume the generator to trigger the MapTransform
      list(dataloader)

  def test_actual_data(self) -> None:
    """Test that the actual dataset loads and contains examples."""
    dataloader = tinyknowledge.create_tinyknowledge_interventionloader(
      tokenizer=self.tokenizer,
      batch_size=1,
      seq_len=16,
      top_answers=1,
      n_workers=0,
    )
    num_examples = sum(1 for _ in dataloader)
    self.assertGreater(num_examples, 0)

  @absltest.skip('Skipping test since it prints the data.')
  def test_print_actual_data(self) -> None:
    """Call filter_and_tokenize_data on the hardcoded data and print all entries."""
    for entry in tinyknowledge.filter_and_tokenize_data(
      self.tokenizer, top_answers=1, data=tinyknowledge.CANONICAL_KNOWLEDGE_TINYSTORIES
    ):
      print(entry)


if __name__ == '__main__':
  absltest.main()
