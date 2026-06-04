"""
Copyright (c) NXAI GmbH.
This software may be used and distributed according to the terms of the NXAI Community License Agreement.
https://github.com/NX-AI/xlstm-jax/blob/main/scripts/data_processing/hf_to_arrayrecord.py
"""

import argparse
import logging
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import datasets
import grain.python as grain
import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter

LOGGER = logging.getLogger(__name__)


def write_array_record(
  hf_dataset_name: str,
  split: str,
  out_path: str,
  process_idx: int,
  shard_start_idx: int,
  example_start_idx: int,
  example_end_idx: int,
  shard_size: int,
  data_column_name: str,
) -> None:
  """Write the dataset split to an array record file."""
  process_split = (
    f'{split}[{example_start_idx}:{example_end_idx}]'  # chunk for this process
  )

  LOGGER.info(f'[Process {process_idx}] Loading {process_split} dataset.')
  ds = datasets.load_dataset(
    hf_dataset_name,
    split=process_split,
    streaming=False,
  )

  LOGGER.info(f'[Process {process_idx}] Finished loading {process_split} dataset.')
  data_len = len(ds)
  n_shards = math.ceil(data_len / shard_size)
  start_time = time.time()
  writer = None
  for ex_idx, example in enumerate(ds):
    parsed_example = example[data_column_name]
    shard_idx = shard_start_idx + ex_idx // shard_size
    if writer is None:
      LOGGER.info(
        f'[Process {process_idx}] Writing shard {shard_idx} ({shard_idx - shard_start_idx + 1} of {n_shards}'
        ' local shards).'
      )
      shard_path = os.path.abspath(
        os.path.join(out_path, f'{split}_{shard_idx:06d}.arecord')
      )
      assert not os.path.exists(shard_path), (
        f'Shard file already exists: {shard_path}'
      )  # Do not overwrite
      # "group_size:1", because __get_item__ of ArrayRecordDataSource runs into the following logging error:
      # https://github.com/google/array_record/blob/main/python/
      # The API is of C++ ArrayRecordWriter is unfortunately not documented (afaik). But here is an example:
      # https://github.com/google/array_record/blob/main/python/array_record_module_test.py#L135C48-L135C63
      writer = ArrayRecordWriter(shard_path, 'group_size:1')
    writer.write(str.encode(parsed_example))
    if (ex_idx + 1) % shard_size == 0:
      eta = int((time.time() - start_time) / (ex_idx + 1) * (data_len - ex_idx - 1))
      eta_str = f'{eta // 3600:2d}h {eta // 60 % 60:2d}m {eta % 60:2d}s'
      LOGGER.info(
        f'[Process {process_idx}] Finished shard {shard_idx} (ETA to finish: {eta_str}).'
      )
      writer.close()
      writer = None
  if writer is not None:
    writer.close()
  LOGGER.info(f'[Process {process_idx}] Finished writing {process_split} dataset.')


def convert_dataset(
  hf_dataset_name: str,
  base_out_path: str,
  n_processes: int,
  shard_size: int = 250_000,
  data_column_name: str = 'text',
) -> None:
  """Convert dataset from HuggingFace to ArrayRecord.

  Args:
      hf_dataset_name: Huggingface dataset name, e.g. 'cerebras/SlimPajama-627B'.
      base_out_path: Base output directory for saving the preprocessed dataset.
      n_processes: Number of workers to use for the convert/map preprocessing.
      shard_size: Number of examples in each shard.
      data_column_name: The column containing the text data, e.g. "text" or "messages". Used only for non-sft data.
  """

  splits = datasets.get_dataset_split_names(hf_dataset_name)
  LOGGER.info(f'Detected splits: {splits}')

  for split in splits:
    # Load dataset from hub/cache in order to get the dataset size.
    LOGGER.info(f'Loading {split} dataset.')

    ds = datasets.load_dataset(
      hf_dataset_name,
      split=split,
      streaming=False,
    )
    dataset_size = len(ds)
    del ds  # free memory before reloading dataset chunks in multiple processes

    out_path = os.path.join(base_out_path, hf_dataset_name.replace('/', '_'))
    out_path = os.path.join(out_path, split)

    os.makedirs(out_path, exist_ok=True)
    assert not any(Path(out_path).iterdir()), f'Output directory ({out_path}) not empty.'

    # Write the dataset to an array record file.
    LOGGER.info(f'Writing {split} dataset to array record.')
    if n_processes <= 1:
      write_array_record(
        hf_dataset_name=hf_dataset_name,
        split=split,
        out_path=out_path,
        process_idx=0,
        shard_start_idx=0,
        example_start_idx=0,
        example_end_idx=dataset_size,
        shard_size=shard_size,
        data_column_name=data_column_name,
      )
    else:
      # We split the dataset into n_processes chunks (start and end idx) and convert each
      # chunk in parallel. Since the number of shards is in general not divisible by the
      # number of processes, we distribute the residual shards among the first processes.
      n_shards = math.ceil(dataset_size / shard_size)
      n_split_processes = (
        n_processes if n_processes <= n_shards else n_shards
      )  # avoid empty processes
      min_shards_per_process = n_shards // n_split_processes
      n_residuals = n_shards % n_split_processes

      n_shards_per_process = np.full(
        n_split_processes, min_shards_per_process, dtype=np.int32
      )
      n_shards_per_process[:n_residuals] += 1
      assert n_shards_per_process.sum() == n_shards, (
        'shards per process does not sum to total shards.'
      )
      shard_start_indices = np.cumsum(n_shards_per_process) - n_shards_per_process
      example_start_indices = shard_size * shard_start_indices
      example_end_indices = example_start_indices + shard_size * n_shards_per_process
      example_end_indices[-1] = dataset_size
      with Pool(n_split_processes) as pool:
        pool.starmap(
          write_array_record,
          [
            (
              hf_dataset_name,
              split,
              out_path,
              process_idx,
              shard_start_indices[process_idx],
              example_start_indices[process_idx],
              example_end_indices[process_idx],
              shard_size,
              data_column_name,
            )
            for process_idx in range(n_split_processes)
          ],
        )

    LOGGER.info(f'Finished conversion of {split} dataset.')


def load_array_record_data_source(
  dataset_path: str, file_extension: str = '.arecord'
) -> grain.ArrayRecordDataSource:
  """Take all files located at dataset_path and load as grain.ArrayRecordDataSource."""
  assert os.path.exists(dataset_path), f'dataset path {dataset_path} does not exist.'

  # Get files ending in `file_extension` in sorted order
  path_pattern = Path(f'{dataset_path}/*{file_extension}')
  files = path_pattern.parent.glob(path_pattern.name)
  sorted_files = sorted(files, key=lambda x: int(x.stem.split('_')[-1]))

  data_source = grain.ArrayRecordDataSource(sorted_files)

  return data_source


def download_hfdataset_with_retries(
  hf_dataset_name: str, n_retries: int, delay: int
) -> None:
  """Load a HuggingFace dataset with automatic retry logic."""
  for attempt in range(n_retries):
    try:
      _ = datasets.load_dataset(hf_dataset_name)
      return
    except Exception as e:
      if attempt < n_retries - 1:
        LOGGER.info(f'Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...')
        time.sleep(delay)
      else:
        LOGGER.info(f'All {n_retries} attempts failed.')
        raise


if __name__ == '__main__':
  LOG_FORMAT = '[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s]{rank} - %(message)s'
  stdout_handler = logging.StreamHandler(sys.stdout)
  logging.basicConfig(
    handlers=[stdout_handler],
    level='INFO',
    format=LOG_FORMAT.format(rank=''),
    force=True,
  )

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--hf_dataset_name',
    type=str,
    default='roneneldan/TinyStories',
    # default='cerebras/SlimPajama-627B',
    # default='DKYoon/SlimPajamas-6B',
    help='Huggingface dataset name.',
  )

  parser.add_argument(
    '--n_processes', type=int, default=32, help='Number of workers used to convert.'
  )
  parser.add_argument(
    '--data_column_name', type=str, default='text', help='Column name containing data.'
  )
  parser.add_argument(
    '--base_out_path',
    type=str,
    default='$SCRATCH/array_records',
    help='Base path to write array records to.',
  )
  args = parser.parse_args()
  args.base_out_path = os.path.expandvars(os.path.expanduser(args.base_out_path))

  LOGGER.info('Download HF dataset with retries in case it is not cached...')
  download_hfdataset_with_retries(args.hf_dataset_name, n_retries=100, delay=300)

  LOGGER.info(f'Converting to array_records with {args.n_processes} workers...')
  convert_dataset(
    hf_dataset_name=args.hf_dataset_name,
    base_out_path=args.base_out_path,
    n_processes=args.n_processes,
    data_column_name=args.data_column_name,
  )

  LOGGER.info('Testing processed splits...')
  splits = datasets.get_dataset_split_names(args.hf_dataset_name)
  for split in splits:
    split_path = os.path.join(
      args.base_out_path, args.hf_dataset_name.replace('/', '_'), split
    )
    if os.path.exists(split_path) and any(Path(split_path).iterdir()):
      LOGGER.info(f'Test loading {split} split at {split_path}')
      array_record_data_source = load_array_record_data_source(split_path)
      dataset = grain.MapDataset.source(array_record_data_source)
      LOGGER.info(f'Split {split} has {len(dataset)} examples.')
