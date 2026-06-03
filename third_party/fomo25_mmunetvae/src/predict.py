#!/usr/bin/env python3
"""
Batch FOMO25 predict.py
- Input: --data-dir containing preprocessed/sub_*/ses_1/*.nii.gz
- Output: one file per subject in --output-dir:
    task2 (seg): sub_XXX.nii.gz
    task1/3 (txt): sub_XXX.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

# project imports (same as your single-subject script)
from inference.predict import load_modalities
from data.task_configs import task1_config, task2_config, task3_config
from models.supervised_seg import SupervisedSegModel
from models.supervised_cls import SupervisedClsModel
from models.supervised_reg import SupervisedRegModel
from yucca.functional.preprocessing import preprocess_case_for_inference, reverse_preprocessing

# --------------------------
# Hardcoded (edit as needed)
# --------------------------
PREDICT_CFG = {
    "task1": {**task1_config, "default_checkpoint": "/app/models/Task001_FOMO1/best_model.ckpt"},
    "task2": {**task2_config, "default_checkpoint": "/app/models/Task002_FOMO2/best_model.ckpt"},
    "task3": {**task3_config, "default_checkpoint": "/app/models/Task003_FOMO3/best_model.ckpt"},
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "norm_op": "volume_wise_znorm",
    "patch_size": (64, 64, 64),
    "target_spacing": [1.0, 1.0, 1.0],
    "target_orientation": "RAS",
    "transpose_forward": [0, 1, 2],
    "keep_aspect_ratio_when_using_target_size": False,
    "transpose": [0, 1, 2],
    "keep_aspect_ratio": True,
    "overlap": 0.5,
}


SESSION_DIR = "ses_1"
SUBJECT_PREFIX = "sub_"
# modality tokens searched in filenames (lowercased)
TOKENS = {
    "adc": "adc",
    "dwi_b1000": "dwi_b1000",
    "flair": "flair",
    "t2s": "t2s",
    "swi": "swi",
    "t1": "t1",
    "t2": "t2",
}


def keep_largest_connected_component_3d(binary_mask):
    """
    Keep only the largest connected component in a 3D binary mask
    
    Args:
        binary_mask: 3D numpy array with 0s and 1s
    
    Returns:
        3D numpy array with only the largest connected component
    """
    if np.sum(binary_mask) == 0:
        return binary_mask
    
    # Label connected components (26-connectivity for 3D)
    labeled_array, num_features = ndimage.label(binary_mask)
    
    if num_features <= 1:
        return binary_mask
    
    # Count voxels in each component
    component_sizes = np.bincount(labeled_array.ravel())
    # Skip background (label 0)
    component_sizes[0] = 0
    
    # Find largest component
    largest_component_label = np.argmax(component_sizes)
    
    # Keep only the largest component
    largest_component_mask = (labeled_array == largest_component_label).astype(np.uint8)
    
    return largest_component_mask

# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Batch FOMO25 inference (per subject)")
    p.add_argument("--data-dir", required=True, type=str, help="Root data dir containing preprocessed/sub_*/ses_1")
    p.add_argument("--output-dir", required=True, type=str, help="Where to write per-subject outputs")
    p.add_argument("--task", choices=["task1", "task2", "task3"], default=None,
                   help="If omitted, will try to infer per subject from available modalities")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional path to .ckpt; overrides task default")
    # Split filtering
    p.add_argument("--which-split", choices=["val", "test"], default="val",
                   help="Which split to run (default: val)")
    p.add_argument("--split-idx", type=int, default=0, help="Fold index")
    p.add_argument("--split-param", type=int, default=5, help="Total folds (k)")
    p.add_argument("--split-json", type=str, default=None,
                   help="Override path to split JSON; if omitted, uses task-specific default")
    return p.parse_args()    

# --------------------------
# Discovery
# --------------------------
def get_preprocessed_dir(data_dir: Path) -> Path:
    if (data_dir / "preprocessed").is_dir():
        return data_dir / "preprocessed"
    return data_dir  # allow passing the preprocessed dir directly


def list_subject_dirs(preprocessed_dir: Path) -> List[Path]:
    return [d for d in preprocessed_dir.iterdir() if d.is_dir() and d.name.startswith(SUBJECT_PREFIX)]


def find_session_dir(subject_dir: Path) -> Path:
    ses = subject_dir / SESSION_DIR
    if not ses.is_dir():
        raise FileNotFoundError(f"Missing {SESSION_DIR} in {subject_dir}")
    return ses


def match_modalities(scan_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for f in scan_dir.glob("*.nii.gz"):
        name = f.name.lower()
        for mod, token in TOKENS.items():
            if token in name and mod not in found:
                found[mod] = f
    return found


# =============== SPLITS
def default_split_spec_for_task(task: str):
    # This is needed due to the way we generated the splits.
    # choose JSON + prefix/strip_sub by task type
    if task == "task1":  # classification
        return ("/media/jaume/T7/data/splits_final/task1/splits_final_no_test.json", "FOMO1_", False)
    if task == "task2":  # segmentation
        return ("/media/jaume/T7/data/splits_final/task2/nnunet_experiments/splits_final_no_test.json", "FOMO2_", False)
    if task == "task3":  # regression
        return ("/media/jaume/T7/data/splits_final/task3/splits_final_no_test.json", "FOMO3_", True)
    raise ValueError(f"Unknown task: {task}")


def transform_subject_id(sub_name: str, task_prefix: str, strip_sub: bool) -> str:
    # sub_name is like "sub_001"
    s = sub_name
    if strip_sub and s.startswith("sub_"):
        s = s[len("sub_"):]  # -> "001"
    return f"{task_prefix}{s}"  # e.g. "FOMO3_001" or "FOMO2_sub_001"


def load_split_ids(json_path: Path, fold_idx: int, which_split: str,
                   task_prefix: str, strip_sub: bool) -> set[str]:
    with open(json_path, "r") as f:
        folds = json.load(f)  # list of {train: [...], val: [...], (maybe) test: [...]}
    assert isinstance(folds, list) and 0 <= fold_idx < len(folds), "Invalid folds or index"
    split_key = which_split if (which_split in folds[fold_idx]) else "val"  # fallback if no 'test'
    ids = []
    for raw_id in folds[fold_idx][split_key]:
        s = raw_id
        # If caller passed JSON without prefixes, we still normalize through transform rules
        # (This mirrors your earlier helper logic.)
        if strip_sub and s.startswith("sub_"):
            s = s[len("sub_"):]
        ids.append(f"{task_prefix}{s}" if not s.startswith(task_prefix) else s)
    return set(ids)

# --------------------------
# Task logic
# --------------------------
def infer_task_from_found(found: Dict[str, Path]) -> str:
    has = set(found.keys())
    if {"t1", "t2"}.issubset(has):
        return "task3"
    if {"dwi_b1000", "flair"}.issubset(has) and ("swi" in has or "t2s" in has):
        return "task2"
    if has & {"adc", "dwi_b1000", "flair", "t2s", "swi"}:
        return "task1"
    raise ValueError("Cannot infer task from available files.")


def modalities_for_task(task: str, found: Dict[str, Path]) -> List[Path]:
    if task == "task1":
        if "dwi_b1000" not in found or "flair" not in found or "adc" not in found  or not (("swi" in found) or ("t2s" in found)):
            raise ValueError("Task1 requires dwi_b1000 + flair + adc + (swi OR t2s)")
        return [found["dwi_b1000"], found["flair"], found["adc"], (found["swi"] if "swi" in found else found["t2s"])]

    if task == "task2":
        if "dwi_b1000" not in found or "flair" not in found or not (("swi" in found) or ("t2s" in found)):
            raise ValueError("Task2 requires dwi_b1000 + flair + (swi OR t2s)")
        return [found["dwi_b1000"], found["flair"], (found["swi"] if "swi" in found else found["t2s"])]
    
    if task == "task3":
        if "t1" not in found or "t2" not in found:
            raise ValueError("Task3 requires t1 + t2")
        return [found["t1"], found["t2"]]
    
    raise ValueError(f"Unknown task: {task}")


def load_model(task: str, ckpt: str):
    if task == "task2":
        return SupervisedSegModel.load_from_checkpoint(checkpoint_path=ckpt)
    if task == "task1":
        return SupervisedClsModel.load_from_checkpoint(checkpoint_path=ckpt)
    if task == "task3":
        return SupervisedRegModel.load_from_checkpoint(checkpoint_path=ckpt)
    raise ValueError(f"Unknown task: {task}")


# --------------------------
# Single-subject predict
# --------------------------
@torch.no_grad()
def predict_subject(task: str, modality_paths: List[Path], out_path: Path, checkpoint: Optional[str]):
    # Config
    patch_size = PREDICT_CFG["patch_size"]
    target_spacing = PREDICT_CFG["target_spacing"]
    target_orientation = PREDICT_CFG["target_orientation"]
    transpose_forward = PREDICT_CFG["transpose_forward"]
    overlap = PREDICT_CFG["overlap"]
    crop_to_nonzero = PREDICT_CFG["crop_to_nonzero"]
    norm_op = PREDICT_CFG["norm_op"]
    
    # Reference image
    reference_img = nib.load(str(modality_paths[0]))

    # Load arrays (C channels)
    images = load_modalities([str(p) for p in modality_paths])

    # Preprocess
    normalization_scheme = [norm_op] * len(modality_paths)
    case_preprocessed, case_props = preprocess_case_for_inference(
        crop_to_nonzero=crop_to_nonzero,
        images=images,
        intensities=None,
        normalization_scheme=normalization_scheme,
        patch_size=patch_size,
        target_size=None,
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        allow_missing_modalities=False,
        keep_aspect_ratio=True,
        transpose_forward=transpose_forward,
    )

    # Model/device
    model = load_model(task, checkpoint).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    case_preprocessed = case_preprocessed.to(device)

    # Get prediction using sliding window
    with torch.no_grad():
        overlap = 0.5
        preds = model.model.predict(
            data=case_preprocessed.float(),
            mode="3D",
            mirror=True,
            overlap=overlap,
            patch_size=patch_size,
            sliding_window_prediction=True,
            device=device,
        )

    # Save per task
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if task == "task2":        
        num_classes = 2        
        probabilities = torch.softmax(preds, dim=1)
        
        # Back to original space
        preds_orig, _ = reverse_preprocessing(
            crop_to_nonzero=crop_to_nonzero,
            images=probabilities,
            image_properties=case_props,
            n_classes=num_classes,
            transpose_forward=transpose_forward,
            transpose_backward=[0, 1, 2],
        )
        
        # Get class prediction
        seg = np.argmax(preds_orig, axis=1)[0].astype(np.uint8)
        seg = (seg > 0.5).astype(np.uint8)

        # Keep only largest connected comp.
        seg = keep_largest_connected_component_3d(seg)        
        nib.save(nib.Nifti1Image(seg, reference_img.affine), str(out_path))
    elif task == "task1":             
        num_classes = 2    
        # The prediction are already probs.       
        probability = preds[0][1].item()  # Assuming class 1 is positive
        out_path.write_text(f"{probability:.6f}")
    elif task == "task3":
        num_classes = 1        
        # The prediction is already the median age.
        predicted_age = preds[0, 0]        
        out_path.write_text(f"{predicted_age:.6f}")
    else:
        raise ValueError(f"Unknown task: {task}")

# --------------------------
# Batch driver
# --------------------------
def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    pre_dir = get_preprocessed_dir(data_dir)

    subjects = list_subject_dirs(pre_dir)
    if not subjects:
        raise SystemExit(f"No subject folders found in {pre_dir}")

    if args.task is not None:
        # get split file + transform rules
        split_json, task_prefix, strip_sub = default_split_spec_for_task(args.task)
        if args.split_json is not None:
            split_json = args.split_json
        allowed_ids = load_split_ids(Path(split_json), args.split_idx,
                                     args.which_split, task_prefix, strip_sub)

        # keep only subjects whose transformed id is in split
        before = len(subjects)
        subjects = [
            s for s in subjects
            if transform_subject_id(s.name, task_prefix, strip_sub) in allowed_ids
        ]
        print(f"[INFO] Filtering by {args.which_split} split: {len(subjects)}/{before} subjects kept")
    else:
        if any([args.split_json, args.which_split, args.split_idx is not None]):
            print("[WARN] --task is required to use split filtering; running all subjects.")

    print(f"[INFO] Found {len(subjects)} subjects")

    for subj_dir in subjects:
        sid = subj_dir.name  # e.g., sub_001
        try:
            scan_dir = find_session_dir(subj_dir)
            found = match_modalities(scan_dir)
            task = args.task or infer_task_from_found(found)
            paths = modalities_for_task(task, found)

            # Output file name the validator expects
            ext = ".nii.gz" if task == "task2" else ".txt"
            out_path = out_dir / f"{sid}{ext}"

            print(f"[INFO] {sid}: task={task} → {out_path.name}")
            predict_subject(task, paths, out_path, args.checkpoint)

        except Exception as e:
            print(f"[ERROR] {sid}: {e}")

    print(f"[DONE] Outputs written to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
