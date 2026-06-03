import numpy as np
import argparse
import os
import shutil
import nibabel as nib
from functools import partial
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    save_pickle,
    maybe_mkdir_p as ensure_dir_exists,
    subfiles
)
# Assuming these are available in your environment
from yucca.functional.preprocessing import preprocess_case_for_training_with_label, preprocess_case_for_training_without_label
from utils.utils import parallel_process


# "modalities": ("DWI", "T2FLAIR", "SWI_OR_T2STAR"),
# "labels": {0: "background", 1: "menigioma"},
# "task_type": "segmentation",
# "num_classes": 2,
DEFAULT_CONFIG = {
    "crop_to_nonzero": True,
    "norm_op": "volume_wise_znorm",    
    "target_orientation": "RAS",
    "target_spacing": [1.0, 1.0, 1.0],
    "keep_aspect_ratio_when_using_target_size": False,
    "transpose": [0, 1, 2],
    "keep_aspect_ratio": True,
    "label_extension": ".txt",
}


def reorient_to_ras(img):
    """
    Force reorientation of a nibabel image to RAS (Right-Anterior-Superior).
    This fixes 'unexpected NIFTI axes' errors.
    """
    return nib.as_closest_canonical(img)


def process_subject_general(task_info):
    """
    Process a single subject (Train or Test) from nnU-Net raw structure.
    
    Dynamically detects the number of modalities based on _0000, _0001 suffixes.
    """
    (
        subject_id,
        images_dir,
        labels_dir, # Can be None for inference cases
        pp_config,
        target_preprocessed
    ) = task_info

    try:
        # 1. Dynamic Modality Discovery
        # Search for all files starting with {subject_id}_ and ending with .nii.gz
        # We expect files like subject_0000.nii.gz, subject_0001.nii.gz
        all_files = subfiles(images_dir, prefix=subject_id + "_", suffix=".nii.gz", join=False)
        
        # Sort files to ensure _0000 is index 0, _0001 is index 1, etc.
        all_files.sort()
        
        if not all_files:
            return f"Error: No modality files found for {subject_id} in {images_dir}"

        # Load all modality images
        image_paths = [join(images_dir, f) for f in all_files]
        images = [reorient_to_ras(nib.load(img)) for img in image_paths]

        # 2. Check for Label (Ground Truth)
        label = None
        if labels_dir:
            label_path = join(labels_dir, f"{subject_id}.nii.gz")
            if os.path.exists(label_path):
                label = nib.load(label_path)
                label = reorient_to_ras(label)
        
        # 3. Apply Preprocessing
        # We need a list of normalization ops matching the number of found modalities
        norm_ops = [DEFAULT_CONFIG.get("norm_op", "normalization_operation")] * len(images)
        pp_config = pp_config.copy()
        pp_config.pop("normalization_operation", None)  # Remove modalities key if exists

        # Extract configuration parameters
        crop_to_nonzero = DEFAULT_CONFIG["crop_to_nonzero"]
        norm_op = DEFAULT_CONFIG["norm_op"]
        keep_aspect_ratio = DEFAULT_CONFIG.get("keep_aspect_ratio", True)

        # Define preprocessing parameters
        normalization_scheme = [norm_op] * len(images)
        target_spacing = DEFAULT_CONFIG["target_spacing"]
        target_orientation = DEFAULT_CONFIG["target_orientation"]
        transpose_forward=DEFAULT_CONFIG["transpose"]

        if label is not None:
            preprocessed_data, preprocessed_label, properties = (
                preprocess_case_for_training_with_label(
                    images=images,
                    label=label,
                    normalization_operation=norm_ops,
                    allow_missing_modalities=False,
                    crop_to_nonzero=crop_to_nonzero,
                    intensities=None,  # Use default intensity normalization
                    target_size=None,  # We use target_spacing instead
                    target_spacing=target_spacing,
                    target_orientation=target_orientation,
                    keep_aspect_ratio_when_using_target_size=keep_aspect_ratio,
                    transpose=transpose_forward
                )
            )
            # Structure: [img1, img2, ..., label]
            data_to_save = preprocessed_data + [preprocessed_label]
        else:
            # Inference case (no label)
            preprocessed_data, properties = (
                preprocess_case_for_training_without_label(
                    images=images,
                    normalization_operation=norm_ops,
                    allow_missing_modalities=False,
                    crop_to_nonzero=crop_to_nonzero,
                    intensities=None,  # Use default intensity normalization
                    target_size=None,  # We use target_spacing instead
                    target_spacing=target_spacing,
                    target_orientation=target_orientation,
                    keep_aspect_ratio_when_using_target_size=keep_aspect_ratio,
                    transpose=transpose_forward
                )
            )
            # Structure: [img1, img2, ...] (Label is implicitly None or omitted)
            data_to_save = preprocessed_data

        # 4. Save Output
        # Result: PREFIX_subject_id.npy
        save_path = join(target_preprocessed, f"{subject_id}")
        
        # Save as object array
        np.save(save_path + ".npy", np.array(data_to_save, dtype=object))
        save_pickle(properties, save_path + ".pkl")

        return f"Processed {subject_id} ({'Train' if label else 'Test'})"

    except Exception as e:
        return f"Error processing {subject_id}: {str(e)}"


def convert_general_nnunet(
    source_path: str,
    out_path: str,
    num_workers=None,
):
    """
    Parses a standard nnU-Net folder, processes both Tr and Ts, and saves to one folder.
    """
    pp_config = DEFAULT_CONFIG
    
    # 1. Define Paths
    images_tr = join(source_path, "imagesTr")
    labels_tr = join(source_path, "labelsTr")
    images_ts = join(source_path, "imagesTs")
    labels_ts = join(source_path, "labelsTs") # Sometimes test sets have labels for validation

    # Output directory
    target_preprocessed = join(out_path, 'data')
    ensure_dir_exists(target_preprocessed)

    tasks = []

    # 2. Gather Training Subjects
    if os.path.exists(images_tr) and os.path.exists(labels_tr):
        # We rely on labelsTr to get the subject IDs (cleanest way)
        tr_files = subfiles(labels_tr, suffix=".nii.gz", join=False)
        subjects_tr = [x[:-7] for x in tr_files] # Strip .nii.gz
        
        print(f"Found {len(subjects_tr)} training subjects.")
        
        for sub in subjects_tr:
            tasks.append((sub, images_tr, labels_tr, pp_config, target_preprocessed))
    
    # 3. Gather Test Subjects
    if os.path.exists(images_ts):
        # For imagesTs, we must be careful not to count _0000, _0001 as different subjects.
        # Strategy: Get all files, split by _, remove last part (channel), unique them.
        ts_files = subfiles(images_ts, suffix=".nii.gz", join=False)
        
        # Logic: "subject123_0000.nii.gz" -> "subject123"
        # We assume the last segment after the last '_' is the channel identifier.
        subjects_ts = set()
        for f in ts_files:
            # Split by underscore and rejoin all but the last part
            parts = f.rsplit("_", 1)
            if len(parts) > 1:
                subjects_ts.add(parts[0])
            else:
                print(f"Warning: Unexpected filename format in imagesTs: {f}")
        
        subjects_ts = sorted(list(subjects_ts))
        print(f"Found {len(subjects_ts)} test subjects.")

        # Determine if labelsTs exists for these subjects
        use_labels_ts = labels_ts if os.path.exists(labels_ts) else None

        for sub in subjects_ts:
            tasks.append((sub, images_ts, use_labels_ts, pp_config, target_preprocessed))

    if not tasks:
        print("No subjects found in imagesTr or imagesTs.")
        return

    # 4. Run Parallel Processing
    parallel_process(
        process_subject_general, tasks, num_workers, desc=f"Converting nnU-Net"
    )

    # Copy the dataset JSON if exists
    dataset_json_src = join(source_path, "dataset.json")
    if os.path.exists(dataset_json_src):
        dataset_json_dst = join(out_path, "dataset.json")
        shutil.copyfile(dataset_json_src, dataset_json_dst)

    print(f"Preprocessing completed. Data saved to {target_preprocessed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", type=str, required=True, help="Path to nnU-Net raw data (parent of imagesTr/Ts)"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path where the output folder (Task002_FOMO2) will be created",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    args = parser.parse_args()
    
    convert_general_nnunet(
        source_path=args.in_path, out_path=args.out_path, num_workers=args.num_workers
    )