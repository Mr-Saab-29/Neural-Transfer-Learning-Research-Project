from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from nibabel.processing import resample_to_output


def standardize_geometry(
    nifti_path,
    target_spacing=(2.0, 2.0, 2.0)
):
    """
    Load a 3D MRI, reorient to RAS,
    and resample to the target voxel spacing.
    """

    img = nib.load(str(nifti_path))

    if len(img.shape) != 3:
        raise ValueError(
            f"Expected a 3D MRI, got {img.shape}"
        )

    img = nib.as_closest_canonical(img)

    img = resample_to_output(
        img,
        voxel_sizes=target_spacing,
        order=1
    )

    return img


def normalize_mri(img):
    """
    Apply foreground Z-score normalization.

    Nonzero voxels are treated as an approximate
    foreground mask, not a verified brain mask.
    """

    volume = img.get_fdata(dtype=np.float32)

    mask = np.isfinite(volume) & (volume != 0)

    if not np.any(mask):
        raise ValueError(
            "MRI contains no valid foreground voxels."
        )

    mean = volume[mask].mean()
    std = volume[mask].std()

    if std < 1e-8:
        raise ValueError(
            "Foreground intensity variance is too small."
        )

    normalized = np.zeros_like(
        volume,
        dtype=np.float32
    )

    normalized[mask] = (
        volume[mask] - mean
    ) / std

    return normalized


def center_crop_or_pad(
    volume,
    target_shape=(128, 128, 128)
):
    """
    Center crop or zero-pad a 3D MRI
    to the requested dimensions.
    """

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, got {volume.ndim}D"
        )

    output = np.zeros(
        target_shape,
        dtype=np.float32
    )

    source_slices = []
    target_slices = []

    for current_size, target_size in zip(
        volume.shape,
        target_shape
    ):

        copy_size = min(
            current_size,
            target_size
        )

        source_start = (
            current_size - copy_size
        ) // 2

        target_start = (
            target_size - copy_size
        ) // 2

        source_slices.append(
            slice(
                source_start,
                source_start + copy_size
            )
        )

        target_slices.append(
            slice(
                target_start,
                target_start + copy_size
            )
        )

    output[tuple(target_slices)] = (
        volume[tuple(source_slices)]
    )

    return output


def preprocess_mri(
    nifti_path,
    target_spacing=(2.0, 2.0, 2.0),
    target_shape=(128, 128, 128),
    return_qc=False
):
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    nifti_path : str or Path
        Path to a 3D NIfTI MRI.

    target_spacing : tuple
        Target voxel spacing in millimeters.

    target_shape : tuple
        Final spatial dimensions.

    Returns
    -------
    torch.Tensor
        Float32 tensor with shape [1, D, H, W].
    """

    # 1. Geometry standardization
    img = standardize_geometry(
        nifti_path,
        target_spacing=target_spacing
    )

    # Measure the original intensity mask, before normalization can create zeros.
    values = img.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Resampled MRI contains nonfinite intensities.")
    mask = values != 0
    cropped_mask = center_crop_or_pad(mask, target_shape).astype(bool)
    before = int(mask.sum())
    after = int(cropped_mask.sum())
    qc = {
        "resampled_shape": list(img.shape),
        "spacing_mm": [float(v) for v in img.header.get_zooms()[:3]],
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "foreground_before_crop": before,
        "foreground_after_crop": after,
        "foreground_retention": after / before if before else 0.0,
        "foreground_fraction": float(cropped_mask.mean()),
    }

    # 2. Intensity normalization
    volume = normalize_mri(img)

    # 3. Fixed spatial dimensions
    volume = center_crop_or_pad(
        volume,
        target_shape=target_shape
    )

    # 4. Convert to PyTorch tensor
    volume = np.ascontiguousarray(volume)

    tensor = torch.from_numpy(
        volume
    ).unsqueeze(0)

    qc["foreground_mean"] = float(volume[cropped_mask].mean()) if after else None
    qc["foreground_std"] = float(volume[cropped_mask].std()) if after else None
    qc["tensor_shape"] = list(tensor.shape)
    if not after or not torch.isfinite(tensor).all():
        raise ValueError("Output is empty or contains nonfinite values.")
    return (tensor, qc) if return_qc else tensor
