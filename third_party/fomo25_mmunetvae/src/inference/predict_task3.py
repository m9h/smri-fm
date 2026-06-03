#!/usr/bin/env python

import argparse

from data.task_configs import task3_config
from inference.predict import predict_from_config

# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task3_config,
    # Add inference-specific configs
    "model_path": "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/FOMO-MRI/model_test_jaume_6/Task003_FOMO3/mmunetvae/version_0/checkpoints/best_model.ckpt",
    "patch_size": (64, 64, 64),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on FOMO Task 3 (Brain Age Regression)"
    )

    # Input and output paths using modality names from task config
    parser.add_argument(
        "--t1", type=str, required=True, help="Path to T1 image (NIfTI format)"
    )
    parser.add_argument(
        "--t2", type=str, required=True, help="Path to T2 image (NIfTI format)"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output path for prediction"
    )

    # Parse arguments
    args = parser.parse_args()

    # Map arguments to modality paths in expected order from task config
    modality_paths = [args.t1, args.t2]

    # Run prediction using the shared prediction logic
    predict_from_config(
        modality_paths=modality_paths,
        output_path=args.out,
        predict_config=predict_config,
    )


if __name__ == "__main__":
    main()
