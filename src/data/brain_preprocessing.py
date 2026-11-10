"""Mask-based preprocessing, separate from the preserved Assignment 03 baseline."""
import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from scipy.ndimage import label

from src.data.preprocessing import standardize_geometry


def validate_mask(image, mask_image):
    if (len(mask_image.shape) != 3 or mask_image.shape != image.shape
            or not np.allclose(mask_image.affine, image.affine, atol=1e-4, rtol=0)):
        raise ValueError('Brain mask must match the native image shape and affine')
    mask = mask_image.get_fdata(dtype=np.float32)
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('Brain mask must contain only finite binary values')
    if not mask.any() or mask.all():
        raise ValueError('Brain mask is empty or covers the entire image')
    return mask.astype(bool)


def crop_at(volume, start, shape):
    """Crop with a shared integer origin; pad out-of-bounds regions with zero."""
    output = np.zeros(shape, dtype=volume.dtype)
    source = []
    destination = []
    for origin, length, size in zip(start, shape, volume.shape):
        low, high = max(0, int(origin)), min(size, int(origin) + length)
        if high <= low:
            return output
        source.append(slice(low, high))
        destination.append(slice(low - origin, high - origin))
    output[tuple(destination)] = volume[tuple(source)]
    return output


def preprocess_brain(raw_path, mask_path, target_shape=(128, 128, 128)):
    native = nib.load(str(raw_path))
    native_mask_image = nib.load(str(mask_path))
    native_mask = validate_mask(native, native_mask_image)
    native_values = native.get_fdata(dtype=np.float32)
    if not np.isfinite(native_values).all():
        raise ValueError('Nonfinite native intensities')
    native_foreground = native_values != 0
    if not native_foreground.any():
        raise ValueError('No native foreground')
    img = standardize_geometry(raw_path)
    values = img.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite resampled intensities')
    mask = resample_from_to(native_mask_image, (img.shape, img.affine), order=0).get_fdata() > .5
    coordinates = np.array(np.nonzero(mask))
    if coordinates.shape[1] == 0:
        raise ValueError('Brain mask vanished during resampling')
    low, high = coordinates.min(axis=1), coordinates.max(axis=1)
    center = (low + high) / 2
    start = np.floor(center - (np.array(target_shape) - 1) / 2).astype(int)
    cropped = crop_at(values, start, target_shape)
    brain = crop_at(mask, start, target_shape)
    before, after = int(mask.sum()), int(brain.sum())
    if not after:
        raise ValueError('Empty cropped brain')
    mean, std = float(cropped[brain].mean()), float(cropped[brain].std())
    if not np.isfinite([mean, std]).all() or std < 1e-8:
        raise ValueError('Invalid or constant brain intensities')
    normalized = np.zeros(target_shape, dtype=np.float32)
    normalized[brain] = (cropped[brain] - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError('Nonfinite normalized tensor')
    # Preserve world coordinates after the integer crop/pad translation.
    affine = img.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(img.affine, start)
    components, count = label(mask)
    largest = int(np.bincount(components.ravel())[1:].max())
    touching = any(np.take(brain, index, axis=axis).any()
                   for axis in range(3) for index in (0, -1))
    native_touching = any(np.take(native_mask, index, axis=axis).any()
                          for axis in range(3) for index in (0, -1))
    qc = {
        'brain_retention': after / before,
        'brain_voxels_before': before, 'brain_voxels_after': after,
        'brain_volume_ml': before * 8 / 1000,
        'brain_fraction': float(brain.mean()),
        'brain_mean': float(normalized[brain].mean()),
        'brain_std': float(normalized[brain].std()),
        'native_foreground_retention': float((native_mask & native_foreground).sum() / native_foreground.sum()),
        'largest_component_fraction': largest / before,
        'mask_components': count, 'touches_output_boundary': bool(touching),
        'touches_native_boundary': bool(native_touching),
        'crop_origin_voxels': start.tolist(), 'brain_bbox_min': low.tolist(),
        'brain_bbox_max': high.tolist(), 'spacing_mm': [2., 2., 2.],
        'orientation': ''.join(nib.aff2axcodes(affine)),
        'tensor_shape': [1, *target_shape], 'output_affine': affine.tolist(),
    }
    return (torch.from_numpy(normalized).unsqueeze(0),
            nib.Nifti1Image(brain.astype(np.uint8), affine),
            nib.Nifti1Image(cropped, affine), qc)


def review_reasons(qc, dataset):
    reasons = []
    if qc['brain_retention'] < .999:
        reasons.append('brain_crop_loss')
    if qc['touches_output_boundary']:
        reasons.append('mask_touches_output_boundary')
    if qc['touches_native_boundary']:
        reasons.append('mask_touches_native_boundary')
    if qc['largest_component_fraction'] < .99:
        reasons.append('disconnected_mask')
    if not 600 <= qc['brain_volume_ml'] <= 2000:
        reasons.append('brain_volume_outside_review_range')
    if dataset == 'PT005_IXI' and qc['native_foreground_retention'] < .95:
        reasons.append('IXI_existing_foreground_removed')
    return reasons
