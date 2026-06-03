import os
import re
import logging
import datetime
import torch
import pickle
import lightning as L
import multiprocessing
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
import json


def save_json(obj, filepath):
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def save_data_pickle(dataset_dict, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_pickle(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def parse_dataset_structure(root_dir):
    """
    Parses a dataset directory structured as:
        subject_id/
            session_id/
                image_seq.nii.gz
                image_seq_1.nii.gz
                scan_1.nii.gz
                ...
    Returns a dictionary:
    {
        'subject1': {
            'session1': ['image_seq.nii.gz', 'image_seq_1.nii.gz'],
            'session2': ['scan_1.nii.gz']
        },
        ...
    }
    """
    dataset_dict = defaultdict(lambda: defaultdict(list))

    for subject_id in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        for session_id in os.listdir(subject_path):
            session_path = os.path.join(subject_path, session_id)
            if not os.path.isdir(session_path):
                continue

            for file_name in os.listdir(session_path):
                if file_name.endswith('.nii.gz'):
                    dataset_dict[subject_id][session_id].append(file_name)

    print(f"Parsed {len(dataset_dict)} subjects from {root_dir}")
    total_sessions = sum(len(sessions) for sessions in dataset_dict.values())
    print(f"Total sessions found: {total_sessions}")

    return dict(dataset_dict)


def parse_dataset_preprocessed(preprocessed_dir):
    """
    Parses a flat directory of .npy files with varying naming conventions:
      - sub_XXXX_ses_Y_t1.npy
      - sub_XXXX_ses_Y_t1_Z.npy
      - sub_XXXX_ses_Y_scan_Z.npy
      - FOMO1_sub_XXXX.npy (legacy/no session)
      
    Returns a structure compatible with the nested parser:
    {
        'subject_id': {
            'ses_Y': ['sub_XXXX_ses_Y_t1.npy', ...],
            'session0': ['FOMO1_sub_XXXX.npy'] # Default if no session found
        }
    }
    """
    dataset_dict = defaultdict(lambda: defaultdict(list))
    
    # Regex patterns to capture IDs after 'sub_' and 'ses_'
    # Looks for 'sub_' followed by any alphanumeric characters until an underscore or dot
    sub_pattern = re.compile(r"sub_([a-zA-Z0-9]+)")
    ses_pattern = re.compile(r"ses_([a-zA-Z0-9]+)")

    files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.npy')]

    for file_name in files:
        # 1. Extract Subject ID
        sub_match = sub_pattern.search(file_name)
        if sub_match:
            subject_id = sub_match.group(1)
        else:
            # Skip files that don't contain 'sub_'
            continue

        # 2. Extract Session ID
        ses_match = ses_pattern.search(file_name)
        if ses_match:
            # Found an explicit session (e.g., 'ses_1' -> session_id='ses_1')
            session_id = f"ses_{ses_match.group(1)}"
        else:
            # Fallback if no 'ses_' tag exists (e.g., FOMO1_sub_1.npy)
            session_id = "ses_0"

        # 3. Build dictionary
        dataset_dict[subject_id][session_id].append(file_name)

    print(f"Parsed {len(dataset_dict)} subjects from {preprocessed_dir}")
    total_sessions = sum(len(sessions) for sessions in dataset_dict.values())
    print(f"Total sessions found: {total_sessions}")

    # Convert to standard dict for clean output
    return dict(dataset_dict)

import json
from dataclasses import dataclass
from typing import Union


@dataclass
class SplitConfig:
    splits: Union[dict[dict[list[dict]]], None] = None
    method: str = None
    param: Union[int, float] = None

    def split(self):
        return self.splits[str(self.method)][self.param]

    def train(self, idx):
        return self.split()[idx]["train"]

    def val(self, idx):
        return self.split()[idx]["val"]

    def lm_hparams(self):
        return {"split_method": self.method, "split_param": self.param}


def load_kfold_split_from_json(json_path: str, idx: int, param: int = 5, task_prefix: str = None, strip_sub: bool = False) -> SplitConfig:
    """
    Loads k-fold split from a JSON file and returns a SplitConfig instance.

    Args:
        json_path (str): Path to the JSON file.
        idx (int): Index of the fold to use.
        param (int): Number of total folds (default = 5).
        task_prefix (str): Optional prefix (e.g., "FOMO2_") to prepend to subject IDs.

    Returns:
        SplitConfig: A configured split object.
    """
    with open(json_path, "r") as f:
        folds = json.load(f)  # This should be a list of dicts (with keys: train, val)

    assert isinstance(folds, list), "Expected JSON to be a list of folds"
    assert 0 <= idx < len(folds), f"Fold index {idx} out of range (max {len(folds)-1})"

    # Optionally prefix subjects
    if task_prefix is not None:
        def transform_id(s):
            if strip_sub and s.startswith("sub_"):
                s = s.replace("sub_", "")
            return f"{task_prefix}{s}"

        for split in folds:
            split["train"] = [transform_id(s) for s in split["train"]]
            split["val"] = [transform_id(s) for s in split["val"]]

    # Wrap in SplitConfig structure
    splits = {
        "kfold": {
            param: folds
        }
    }

    return SplitConfig(splits=splits, method="kfold", param=param)




@dataclass
class SimplePathConfig:
    """A simplified path configuration for use with Yucca split configuration."""

    train_data_dir: str

    @property
    def task_dir(self) -> str:
        """For compatibility"""
        return self.train_data_dir

    def __init__(self, train_data_dir=None):
        """Initialize with either train_data_dir or task_dir (train_data_dir has priority)."""
        self.train_data_dir = train_data_dir


def setup_seed(continue_from_most_recent=False):
    """Set up a random seed for reproducibility."""
    if not continue_from_most_recent:
        dt = datetime.datetime.now()
        seed = int(dt.strftime("%m%d%H%M%S"))
    else:
        seed = None  # Will be loaded from checkpoint if available

    L.seed_everything(seed=seed, workers=True)
    return torch.initial_seed()


def find_checkpoint(version_dir, continue_from_most_recent):
    """Find the latest checkpoint if continuing training."""
    checkpoint_path = None
    if continue_from_most_recent:
        potential_checkpoint = os.path.join(version_dir, "checkpoints", "last.ckpt")
        if os.path.isfile(potential_checkpoint):
            checkpoint_path = potential_checkpoint
            logging.info(
                "Using last checkpoint and continuing training: %s", checkpoint_path
            )
    return checkpoint_path


def load_pretrained_weights(weights_path, compile_flag):
    """Load pretrained weights with handling for compiled models and PyTorch Lightning checkpoints."""
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    # Extract the state_dict from PyTorch Lightning checkpoint if needed
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Loading from PyTorch Lightning checkpoint")
        state_dict = checkpoint["state_dict"]
    else:
        print("Loading from standard model checkpoint")
        state_dict = checkpoint

    # Handle compiled checkpoints when loading to uncompiled model
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = next(iter(state_dict))
        if "_orig_mod" in first_key and not compile_flag:
            print("Converting compiled model weights to uncompiled format")
            uncompiled_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("_orig_mod.", "")
                uncompiled_state_dict[new_key] = state_dict[key]
            state_dict = uncompiled_state_dict

    return state_dict


def parallel_process(process_func, tasks, num_workers=None, desc="Processing"):
    """
    Process tasks in parallel using multiprocessing.

    Args:
        process_func: Function that processes a single task
        tasks: List of tasks to process
        num_workers: Number of parallel workers (default: CPU count - 1)
        desc: Description for the progress bar

    Returns:
        List of results from processing each task
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Processing {len(tasks)} items using {num_workers} workers")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(pool.imap(process_func, tasks), total=len(tasks), desc=desc)
        )

    # Print results summary
    successful = sum(
        1
        for result in results
        if isinstance(result, str) and not result.startswith("Error")
    )
    print(
        f"Processing complete: {successful}/{len(tasks)} items processed successfully"
    )

    # Print any errors
    errors = [
        result
        for result in results
        if isinstance(result, str) and result.startswith("Error")
    ]
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for error in errors[
            :10
        ]:  # Show only first 10 errors to avoid cluttering output
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return results
