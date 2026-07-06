<div align="center">
<h1> Sparsely gated tiny linear experts </h1>

<img src="https://us.aws.cdn.hf.co/xet-bridge-us/6a431232f40782a6fcb8952c/21f39a8bb0afb3dc04e59314b4f907e9cbda34c8c3db47545b5cd5192f610adc?response-content-type=image%2Fpng&user_id=public&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27graphical-abstract.png%3B+filename%3D%22graphical-abstract.png%22%3B&X-Xet-Cas-Uid=public&Expires=1783354498&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly91cy5hd3MuY2RuLmhmLmNvL3hldC1icmlkZ2UtdXMvNmE0MzEyMzJmNDA3ODJhNmZjYjg5NTJjLzIxZjM5YThiYjBhZmIzZGMwNGU1OTMxNGI0ZjkwN2U5Y2JkYTM0YzhjM2RiNDc1NDViNWNkNTE5MmY2MTBhZGNcXD9yZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcmdXNlcl9pZD1wdWJsaWMmcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj1pbmxpbmUlM0IrZmlsZW5hbWUlMkElM0RVVEYtOCUyNyUyN2dyYXBoaWNhbC1hYnN0cmFjdC5wbmclM0IrZmlsZW5hbWUlM0QlMjJncmFwaGljYWwtYWJzdHJhY3QucG5nJTIyJTNCJlgtWGV0LUNhcy1VaWQ9cHVibGljIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJFcG9jaFRpbWUiOjE3ODMzNTQ0OTh9fX1dfQ__&Signature=MEYCIQDz1GQTblPoUUQKZmI53HgJ8j9fGyk0CbT5MTBbRzqZ1QIhAPn4ojGZ7CMxCEUrUzdjlWYyd42LS9t7z409oaYy9NzU&Key-Pair-Id=01KAYHXK2CBJSW0YZTMNXK9W1M" alt="Graphical Abstract" width="100%"/>


[![arXiv](https://img.shields.io/badge/arXiv-2606.07414-b31b1b.svg)](https://arxiv.org/abs/2606.07414)
[![Website](https://img.shields.io/badge/🤗_Space-Website-FFD21E.svg)](https://huggingface.co/spaces/smonsays/sgatlin)

</div>


Official code to reproduce experiments in [Sparsely gated tiny linear experts](https://arxiv.org/abs/2606.07414).

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

We use [grain](https://github.com/google/grain) for dataloading from ArrayRecord files.
You can prepare training and evaluation datasets by converting a Hugging Face dataset (e.g., TinyStories) into ArrayRecord format using:

```bash
uv run --env-file .env scripts/hf_to_arrayrecord.py --hf_dataset_name roneneldan/TinyStories
```

## Training

Models are trained using the `train.py` script.
You can specify the model and dataset configuration from the `configs/` directory.

For example, to train a sparsely gated linear model on TinyStories:

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

```bibtex
@article{schug_sgatlin_2026,
  title={Sparsely gated tiny linear experts}, 
  author={Simon Schug},
  year={2026},
  url = {https://arxiv.org/abs/2606.07414},
}
```
