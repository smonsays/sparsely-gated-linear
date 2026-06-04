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
import os
import tempfile
from typing import Iterator

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from absl.testing import parameterized
from jax.typing import ArrayLike

from hyperlinear import config_classes
from hyperlinear.data_types import Batch
from hyperlinear.training import experiment
from hyperlinear.training.logger import StandardLogger
from hyperlinear.training.schedules import warmup_stable_sqrt_decay_schedule


class MockModel(nn.Module):
  """Mock model for testing purposes."""

  vocab_size: int
  d_model: int

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
    x = nn.Dense(features=self.vocab_size)(x)
    return x


class MockLoss(experiment.ExperimentLoss):
  """Mock loss function for testing."""

  def __call__(
    self, params: dict, rng: chex.PRNGKey, batch: Batch
  ) -> tuple[float, dict[str, ArrayLike]]:
    logits = self.apply_fn(params, batch.x, rngs={'dropout': rng})
    targets = jnp.roll(batch.x, -1, axis=1)  # Mock next-token prediction

    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=targets)
    loss = jnp.mean(loss)

    acc = jnp.mean(jnp.equal(jnp.argmax(logits, axis=-1), targets))
    metrics = {'loss': loss, 'acc': acc}

    return loss, metrics


@dataclasses.dataclass
class MockDataLoader:
  batch_size: int = 4
  seq_len: int = 8
  vocab_size: int = 10
  num_batches: int = 5

  def __iter__(self) -> Iterator[Batch]:
    np.random.seed(0)
    for _ in range(self.num_batches):
      x = np.random.randint(
        1, self.vocab_size, size=(self.batch_size, self.seq_len), dtype=np.int64
      )
      yield Batch(x=x)


class MockCallback(experiment.Callback):
  """Test callback that tracks when it's called."""

  def __init__(
    self,
    log_level: int = 0,
    onevent: experiment.CallbackEvent = experiment.CallbackEvent.STEP,
  ) -> None:
    super().__init__(log_level, onevent)
    self.call_count: int = 0
    self.last_step: int | None = None

  def __call__(
    self, ctx: experiment.Experiment, exp_state: experiment.ExperimentState
  ) -> dict[str, ArrayLike]:
    self.call_count += 1
    self.last_step = exp_state.step
    return dict(callback_metric=float(self.call_count))


class ExperimentTest(parameterized.TestCase):
  rng: chex.PRNGKey = jax.random.key(0)

  def setUp(self) -> None:
    super().setUp()
    # Create temporary directory for testing save/load
    self.test_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
    self.workdir: str = self.test_dir.name

    # Create test config
    self.config: config_classes.ExperimentConfig = config_classes.ExperimentConfig(
      name='test_experiment',
      seed=0,
      model=config_classes.ModelConfig(
        d_model=8,
        d_heads=4,
        n_layers=1,
        feedforward_config=config_classes.FeedforwardConfig(
          activation='gelu',
          type='mlp',
          d_ffw=16,
          d_key=-1,
          knn=-1,
          n_channels=-1,
          router_type='',
        ),
      ),
      dataset=config_classes.TinystoriesConfig(
        name='test',
        batch_size=4,
        seq_len=8,
        vocab_size=10,
        tokenizer_path='',
        n_workers=0,
      ),
      optimizer=config_classes.OptimizerConfig(
        learning_rate=0.001,
        weight_decay=0.01,
        optimizer='adamw',
        schedule='warmup_stable_sqrt_decay',
        warmup_steps=1,
        n_gradient_accumulation=1,
        decay_fraction=0.2,
      ),
      sharding_strategy=None,
      workdir=self.workdir,
      checkpoint=False,
      log_every=1,
      log_level=0,
      flops_budget=None,
      logger_names=('standard',),
      callback_names=(),
    )

  def tearDown(self) -> None:
    self.test_dir.cleanup()
    super().tearDown()

  def create_test_experiment(
    self, train_steps: int = 7, callbacks: tuple[experiment.Callback, ...] = ()
  ) -> experiment.Experiment:
    """Helper to create a test experiment."""
    model = MockModel(
      vocab_size=self.config.dataset.vocab_size,
      d_model=self.config.model.d_model,
    )
    train_loader = MockDataLoader(
      self.config.dataset.batch_size,
      self.config.dataset.seq_len,
      self.config.dataset.vocab_size,
      num_batches=train_steps,
    )
    eval_loaders = dict(
      test=MockDataLoader(
        self.config.dataset.batch_size,
        self.config.dataset.seq_len,
        self.config.dataset.vocab_size,
        num_batches=2,
      ),
    )

    n_optim_steps = train_steps // self.config.optimizer.n_gradient_accumulation
    decay_steps = int(self.config.optimizer.decay_fraction * n_optim_steps)
    schedule = warmup_stable_sqrt_decay_schedule(
      learning_rate=self.config.optimizer.learning_rate,
      warmup_steps=self.config.optimizer.warmup_steps,
      const_steps=n_optim_steps - self.config.optimizer.warmup_steps - decay_steps,
      decay_steps=decay_steps,
    )
    optimizer = optax.MultiSteps(
      opt=optax.adamw(learning_rate=schedule),
      every_k_schedule=self.config.optimizer.n_gradient_accumulation,
      use_grad_mean=True,
    )

    return experiment.Experiment(
      config=self.config,
      model=model,
      loss=MockLoss,
      optimizer=optimizer,
      train_steps=train_steps,
      train_loader=train_loader,
      valid_loaders=dict(),
      eval_loaders=eval_loaders,
      tokenizer=lambda x: x,
      logger_list=(StandardLogger(),),
      callbacks=callbacks,
      log_every=self.config.log_every,
      log_level=self.config.log_level,
    )

  def test_experiment_reset(self) -> None:
    """Test that experiment reset creates proper initial state."""
    exp = self.create_test_experiment()
    exp_state = exp.reset(self.rng)

    self.assertIsInstance(exp_state, experiment.ExperimentState)
    self.assertEqual(exp_state.step, 0)
    self.assertIsNotNone(exp_state.params)
    self.assertIsNotNone(exp_state.optim)
    self.assertIsNotNone(exp_state.rng)

  def test_train_step(self) -> None:
    """Test a single training step."""
    exp = self.create_test_experiment()
    exp_state = exp.reset(self.rng)
    batch = Batch(x=np.random.randint(1, 10, size=(4, 8), dtype=np.int64))
    new_exp_state, metrics = exp.train_step(exp_state, batch)

    # Check that state was updated
    self.assertEqual(new_exp_state.step, exp_state.step + 1)
    self.assertIsInstance(metrics, dict)
    self.assertIn('loss', metrics)
    self.assertIn('grad_norm', metrics)
    self.assertIn('param_norm', metrics)

  def test_eval_step(self) -> None:
    """Test evaluation step."""
    exp = self.create_test_experiment()
    exp_state = exp.reset(self.rng)
    batch = Batch(x=np.random.randint(1, 10, size=(4, 8), dtype=np.int64))
    metrics = exp.eval_step(exp_state, self.rng, batch)

    self.assertIsInstance(metrics, dict)
    self.assertIn('loss', metrics)
    self.assertIn('acc', metrics)

  def test_callbacks_on_start(self) -> None:
    """Test that callbacks are triggered on start."""
    callback = MockCallback(onevent=experiment.CallbackEvent.START)
    exp = self.create_test_experiment(callbacks=(callback,))
    exp_state = exp.reset(self.rng)

    # Trigger start callbacks
    metrics = exp.trigger_callback(exp_state, experiment.CallbackEvent.START)

    self.assertEqual(callback.call_count, 1)
    self.assertEqual(metrics['callback_metric'], 1.0)

  def test_save_and_load(self) -> None:
    """Test saving and loading experiment state."""
    exp = self.create_test_experiment()
    exp_state = exp.reset(self.rng)

    # Save the experiment state
    exp.save(exp.config, exp_state)

    # Check that files were created
    self.assertTrue(os.path.exists(os.path.join(self.workdir, 'config.pkl')))
    self.assertTrue(os.path.exists(os.path.join(self.workdir, 'state.pkl')))

    # Load the experiment state
    loaded_config, loaded_state = experiment.Experiment.load(self.workdir)

    # Check that loaded data matches original
    self.assertEqual(loaded_config.name, self.config.name)
    self.assertEqual(loaded_config.seed, self.config.seed)
    self.assertEqual(loaded_state.step, exp_state.step)

  def test_run_experiment_step(self) -> None:
    """Test running a complete small experiment."""
    with jax.disable_jit():
      exp = self.create_test_experiment()
      exp_state = exp.reset(self.rng)
      initial_params = exp_state.params
      batch = next(iter(MockDataLoader(num_batches=1)))

      for _ in range(3):
        exp_state, metrics = exp.train_step(exp_state, batch)
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)

      # Check that state was updated
      def diff_fn(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(x - y))

      param_diff = jax.tree_util.tree_map(diff_fn, initial_params, exp_state.params)
      total_diff = sum(jax.tree_util.tree_leaves(param_diff))
      self.assertGreater(total_diff, 0)
      self.assertEqual(exp_state.step, 3)

  def test_run_experiment_full(self) -> None:
    """Test running a complete experiment with eval handling."""
    exp = self.create_test_experiment()
    exp_state = exp.reset(self.rng)

    initial_step = exp_state.step
    final_exp_state = exp.run(exp_state)
    self.assertGreater(final_exp_state.step, initial_step)


if __name__ == '__main__':
  absltest.main()
