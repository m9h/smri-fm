import numpy as np

from yucca.functional.transforms.croppad import select_foreground_voxel_to_include


def croppad_3D_case_from_3D(
    image,
    image_properties,
    label,
    patch_size,
    p_oversample_foreground,
    target_image_shape,
    target_label_shape,
    crop_start_idx=None,
    **pad_kwargs,
):
    # Determine where to crop from if not provided
    if crop_start_idx is None:
        crop_start_idx = []
        if len(image_properties["foreground_locations"]) == 0 or np.random.uniform() >= p_oversample_foreground:
            for d in range(3):
                max_crop = max(0, image.shape[d + 1] - patch_size[d])
                crop_start_idx.append(np.random.randint(0, max_crop + 1))
        else:
            location = select_foreground_voxel_to_include(image_properties)
            for d in range(3):
                min_idx = max(0, location[d] - patch_size[d])
                max_idx = min(location[d], image.shape[d + 1] - patch_size[d])
                crop_start_idx.append(np.random.randint(min_idx, max_idx + 1))

    # Crop image
    crop_end_idx = [min(crop_start_idx[d] + patch_size[d], image.shape[d + 1]) for d in range(3)]
    cropped = image[
        :,
        crop_start_idx[0]:crop_end_idx[0],
        crop_start_idx[1]:crop_end_idx[1],
        crop_start_idx[2]:crop_end_idx[2],
    ]

    # Initialize target shape
    image_out = np.zeros(target_image_shape, dtype=image.dtype)

    # Center the crop in the target tensor
    insert_idx = [
        (target_image_shape[d + 1] - cropped.shape[d + 1]) // 2
        for d in range(3)
    ]
    image_out[
        :,
        insert_idx[0]:insert_idx[0] + cropped.shape[1],
        insert_idx[1]:insert_idx[1] + cropped.shape[2],
        insert_idx[2]:insert_idx[2] + cropped.shape[3],
    ] = cropped

    # Handle label similarly
    label_out = None
    if label is not None:
        label_cropped = label[
            :,
            crop_start_idx[0]:crop_end_idx[0],
            crop_start_idx[1]:crop_end_idx[1],
            crop_start_idx[2]:crop_end_idx[2],
        ]
        label_out = np.zeros(target_label_shape, dtype=label.dtype)
        label_out[
            :,
            insert_idx[0]:insert_idx[0] + label_cropped.shape[1],
            insert_idx[1]:insert_idx[1] + label_cropped.shape[2],
            insert_idx[2]:insert_idx[2] + label_cropped.shape[3],
        ] = label_cropped

    return image_out, label_out, crop_start_idx
