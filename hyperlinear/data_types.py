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
from typing import Iterator
from typing import Protocol

import jaxtyping as jt
from flax import struct


@struct.dataclass
class Batch:
  x: jt.Int[jt.Array, 'B T']
  info: dict[str, jt.Shaped[jt.ArrayLike, ' B']] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DatasetInfo:
  vocab_size: int
  train_size: int
  eos_token_id: int
  pad_token_id: int


class Dataloader(Protocol):
  def __iter__(self) -> Iterator[Batch]: ...


Metrics = dict[str, jt.ArrayLike]
