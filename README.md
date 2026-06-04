# Sparsely-gated tiny linear experts

Official code to reproduce experiments in [Sparsely-gated tiny linear experts](https://arxiv.org/abs/xxxx.xxxxx).

## Setup

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/) if not already done and invoke `uv sync` to install dependencies.
Then configure the following environment variables, e.g. in `.env`

```bash
SCRATCH=""  # absolute path for storing array_record datasets

# Optionally also set
HF_TOKEN=""  # huggingface token for increased rate limits when downloading datasets
WANDB_ENTITY=""  # your wandb entity name
WANDB_PROJECT="" # wandb project name
```

## Data preparation

You can prepare the training and evaluation datasets using the scripts provided in the `scripts/` directory.
To convert a Hugging Face dataset (e.g., TinyStories) into ArrayRecord format for training:

```bash
uv run --env-file .env scripts/hf_to_arrayrecord.py --hf_dataset_name roneneldan/TinyStories
```

To prepare the TinyKnowledge evaluation dataset:

```bash
uv run --env-file .env scripts/prepare_tinyknowledge_data.py
```

## Training

Models are trained using the `train.py` script.
You can specify the model and dataset configuration from the `configs/` directory.

For example, to train a sparsely-gated linear model on TinyStories:

```bash
uv run --env-file .env train.py --training_config="configs/tinystories_config.py:sparsely_gated_linear"
```

## Running sweeps

Wandb sweeps are used for orchestrating multiple experiments.
The sweep configurations are defined in the `sweeps/` directory.

Initialize a sweep (e.g., the TinyStories sweep):

```bash
uv run --env-file .env wandb sweep sweeps/tinystories.yaml
```

Then, start a sweep agent using the generated Sweep ID:

```bash
uv run --env-file .env wandb agent <USERNAME/PROJECT/SWEEP_ID>
```

## Analysis scripts

We provide several scripts for analyzing gating patterns and evaluating the learned models. You will need to provide the path to your trained model checkpoint.

### Record neuron gating features

Collect gating and activation patterns over a validation set:

```bash
uv run --env-file .env scope.py --ckpt-path <path/to/checkpoint> --dataset tinystories
```

### Evaluate gating patterns

Analyze and visualize nearest neighbors for gating patterns:

```bash
uv run --env-file .env evaluate_tinygating.py --ckpt-path <path/to/checkpoint>
```

### Evaluate causal interventions

Perform causal interventions to evaluate tiny knowledge routing:

```bash
uv run --env-file .env evaluate_tinyknowledge.py --ckpt-path <path/to/checkpoint>
```

## Citation

If you use this code in your research, please cite the paper:

```
@article{schug_sgatlin_2026,
  title={Sparsely-gated tiny linear experts}, 
  author={Simon Schug},
  year={2026},
  url = {https://arxiv.org/abs/xxxx.xxxxx},
}
```