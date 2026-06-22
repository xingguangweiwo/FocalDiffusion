"""
Camera-related utilities for Focal Diffusion
Handles EXIF data, camera parameters, and optical calculations
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import exifread
import logging
import json

logger = logging.getLogger(__name__)


def _safe_exif_number(value) -> Optional[float]:
    text = str(value).strip()
    if text.lower() == 'infinity':
        return None
    if '/' in text:
        a,b=text.split('/',1)
        return float(a)/max(float(b),1e-12)
    return float(text)


def parse_exif_data(image_path: Union[str, Path]) -> Optional[Dict[str, float]]:
    """
    Parse camera parameters from EXIF data

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with camera parameters or None if EXIF not available
    """
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)

        camera_params = {}

        # Focal length
        if 'EXIF FocalLength' in tags:
            focal_length = tags['EXIF FocalLength']
            camera_params['focal_length'] = _safe_exif_number(focal_length)

        # Aperture (F-number)
        if 'EXIF FNumber' in tags:
            f_number = tags['EXIF FNumber']
            camera_params['aperture'] = _safe_exif_number(f_number)
        elif 'EXIF ApertureValue' in tags:
            # APEX value: F = 2^(Av/2)
            av = _safe_exif_number(tags['EXIF ApertureValue'])
            camera_params['aperture'] = 2 ** (av / 2)

        # Focus distance (if available)
        if 'EXIF SubjectDistance' in tags:
            distance = tags['EXIF SubjectDistance']
            if str(distance) != 'Infinity':
                camera_params['focus_distance'] = _safe_exif_number(distance)

        # Sensor size (estimate from camera model)
        if 'Image Model' in tags:
            model = str(tags['Image Model'])
            camera_params['sensor_size'] = estimate_sensor_size(model)

        # ISO
        if 'EXIF ISOSpeedRatings' in tags:
            camera_params['iso'] = int(str(tags['EXIF ISOSpeedRatings']))

        # Exposure time
        if 'EXIF ExposureTime' in tags:
            camera_params['exposure_time'] = _safe_exif_number(tags['EXIF ExposureTime'])

        return camera_params if camera_params else None

    except Exception as e:
        logger.warning(f"Failed to parse EXIF data from {image_path}: {e}")
        return None


def estimate_sensor_size(camera_model: str) -> float:
    """
    Estimate sensor size based on camera model

    Args:
        camera_model: Camera model string from EXIF

    Returns:
        Sensor diagonal in meters (default to full frame if unknown)
    """
    # Common sensor sizes (diagonal in mm)
    sensor_sizes = {
        # Full frame
        'full_frame': 43.3,
        'canon_5d': 43.3,
        'nikon_d850': 43.3,
        'sony_a7': 43.3,

        # APS-C
        'aps_c': 28.3,
        'canon_7d': 26.7,
        'nikon_d7500': 28.3,
        'fujifilm_x': 28.3,

        # Micro Four Thirds
        'm43': 21.6,
        'olympus': 21.6,
        'panasonic_g': 21.6,

        # 1 inch
        '1_inch': 15.9,
        'sony_rx100': 15.9,

        # Smartphone
        'iphone': 7.5,
        'pixel': 7.5,
        'smartphone': 7.5,
    }

    # Try to match camera model
    model_lower = camera_model.lower()

    for key, size in sensor_sizes.items():
        if key in model_lower:
            return size / 1000  # Convert to meters

    # Default to full frame
    logger.info(f"Unknown camera model '{camera_model}', defaulting to full frame sensor")
    return 43.3 / 1000


def estimate_focal_plane_distances(
        num_images: int,
        near_focus: float = 0.3,
        far_focus: float = 10.0,
        distribution: str = 'log',
) -> List[float]:
    """
    Estimate focus distances when EXIF data is not available

    Args:
        num_images: Number of images in focal stack
        near_focus: Nearest focus distance in meters
        far_focus: Farthest focus distance in meters
        distribution: 'linear' or 'log' spacing

    Returns:
        List of estimated focus distances
    """
    if distribution == 'linear':
        distances = np.linspace(near_focus, far_focus, num_images)
    elif distribution == 'log':
        distances = np.logspace(
            np.log10(near_focus),
            np.log10(far_focus),
            num_images
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return distances.tolist()


def save_camera_config(config: Dict, save_path: Union[str, Path]) -> None:
    """Save camera configuration to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved camera configuration to {save_path}")


def load_camera_config(config_path: Union[str, Path]) -> Dict:
    """Load camera configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config