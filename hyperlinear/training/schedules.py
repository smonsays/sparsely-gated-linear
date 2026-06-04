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

import jax
import jax.numpy as jnp
import optax


def sqrt_schedule(init_value: float, transition_steps: int) -> optax.Schedule:
  def schedule(count: int) -> jax.Array:
    count_clipped = jnp.clip(count, 0, transition_steps)
    return init_value * (1 - jnp.sqrt(count_clipped / transition_steps))

  return schedule


def warmup_stable_linear_decay_schedule(
  learning_rate: float,
  warmup_steps: int,
  const_steps: int,
  decay_steps: int,
) -> optax.Schedule:
  r"""Linear warmup -> const -> linear decay: /⎺\."""
  schedules = [
    optax.schedules.linear_schedule(
      init_value=0.0,
      end_value=learning_rate,
      transition_steps=warmup_steps,
    ),
    optax.schedules.constant_schedule(value=learning_rate),
    optax.schedules.linear_schedule(
      init_value=learning_rate,
      end_value=0.0,
      transition_steps=decay_steps,
    ),
  ]
  return optax.join_schedules(schedules, [warmup_steps, warmup_steps + const_steps])


def warmup_stable_sqrt_decay_schedule(
  learning_rate: float,
  warmup_steps: int,
  const_steps: int,
  decay_steps: int,
) -> optax.Schedule:
  r"""Linear warmup -> const -> sqrt decay: /⎺(."""
  schedules = [
    optax.schedules.linear_schedule(
      init_value=0.0,
      end_value=learning_rate,
      transition_steps=warmup_steps,
    ),
    optax.schedules.constant_schedule(value=learning_rate),
    sqrt_schedule(init_value=learning_rate, transition_steps=decay_steps),
  ]
  return optax.join_schedules(schedules, [warmup_steps, warmup_steps + const_steps])
