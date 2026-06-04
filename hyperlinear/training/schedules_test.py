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

import plotly.express as px
from absl.testing import absltest
from absl.testing import parameterized

from hyperlinear.training import schedules


class ScheduleTest(parameterized.TestCase):
  def test_warmup_stable_linear_decay_schedule(self) -> None:
    sched = schedules.warmup_stable_linear_decay_schedule(
      learning_rate=0.01, warmup_steps=100, const_steps=200, decay_steps=300
    )
    self.assertEqual(sched(0), 0.0)
    self.assertAlmostEqual(sched(100), 0.01, places=6)
    self.assertAlmostEqual(sched(300), 0.01, places=6)
    self.assertAlmostEqual(sched(600), 0.0, places=6)

  def test_warmup_stable_sqrt_decay_schedule(self) -> None:
    sched = schedules.warmup_stable_sqrt_decay_schedule(
      learning_rate=0.01, warmup_steps=100, const_steps=200, decay_steps=300
    )
    self.assertEqual(sched(0), 0.0)
    self.assertAlmostEqual(sched(100), 0.01, places=6)
    self.assertAlmostEqual(sched(300), 0.01, places=6)
    self.assertAlmostEqual(sched(600), 0.0, places=6)

  @absltest.skip('Skipping test since it invokes a plot.')
  def test_plot_warmup_stable_linear_decay_schedule(self) -> None:
    sched = schedules.warmup_stable_linear_decay_schedule(
      learning_rate=0.01, warmup_steps=100, const_steps=200, decay_steps=300
    )
    steps = list(range(600))
    values = [sched(step) for step in steps]
    fig = px.line(x=steps, y=values, title='Warmup Stable Linear Decay Schedule')
    fig.show()

  @absltest.skip('Skipping test since it invokes a plot.')
  def test_plot_warmup_stable_sqrt_decay_schedule(self) -> None:
    total_steps = 20_000
    warmup_steps = 2000
    decay_steps = int(0.2 * total_steps)
    const_steps = total_steps - warmup_steps - decay_steps

    sched = schedules.warmup_stable_sqrt_decay_schedule(
      0.001, warmup_steps, const_steps, decay_steps
    )
    steps = list(range(total_steps))
    values = [sched(step) for step in steps]
    fig = px.line(x=steps, y=values, title='Warmup Stable Sqrt Decay Schedule')
    fig.show()


if __name__ == '__main__':
  absltest.main()
