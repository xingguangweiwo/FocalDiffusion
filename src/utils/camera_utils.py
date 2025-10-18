"""
Camera-related utilities for Focal Diffusion
Handles EXIF data, camera parameters, and optical calculations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import exifread
import logging
import json

logger = logging.getLogger(__name__)


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
            camera_params['focal_length'] = eval(str(focal_length))

        # Aperture (F-number)
        if 'EXIF FNumber' in tags:
            f_number = tags['EXIF FNumber']
            camera_params['aperture'] = eval(str(f_number))
        elif 'EXIF ApertureValue' in tags:
            # APEX value: F = 2^(Av/2)
            av = eval(str(tags['EXIF ApertureValue']))
            camera_params['aperture'] = 2 ** (av / 2)

        # Focus distance (if available)
        if 'EXIF SubjectDistance' in tags:
            distance = tags['EXIF SubjectDistance']
            if str(distance) != 'Infinity':
                camera_params['focus_distance'] = eval(str(distance))

        # Sensor size (estimate from camera model)
        if 'Image Model' in tags:
            model = str(tags['Image Model'])
            camera_params['sensor_size'] = estimate_sensor_size(model)

        # ISO
        if 'EXIF ISOSpeedRatings' in tags:
            camera_params['iso'] = int(str(tags['EXIF ISOSpeedRatings']))

        # Exposure time
        if 'EXIF ExposureTime' in tags:
            camera_params['exposure_time'] = eval(str(tags['EXIF ExposureTime']))

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


def estimate_focus_distances(
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


def compute_circle_of_confusion(
        object_distance: float,
        focus_distance: float,
        focal_length: float,
        aperture: float,
        sensor_width: float = 0.036,  # Full frame width in meters
) -> float:
    """
    Compute Circle of Confusion for a given object distance

    Args:
        object_distance: Distance to object in meters
        focus_distance: Current focus distance in meters
        focal_length: Focal length in meters
        aperture: Aperture f-number
        sensor_width: Sensor width in meters

    Returns:
        Circle of Confusion diameter in meters
    """
    # Compute image distances using thin lens equation
    try:
        v_object = 1 / (1 / focal_length - 1 / object_distance)
        v_focus = 1 / (1 / focal_length - 1 / focus_distance)
    except ZeroDivisionError:
        return 0.0

    # Magnification at focus distance
    magnification = v_focus / focus_distance

    # Entrance pupil diameter
    pupil_diameter = focal_length / aperture

    # Circle of Confusion
    coc = abs(magnification * (v_object - v_focus) * pupil_diameter / v_focus)

    return coc


def get_depth_of_field(
        focus_distance: float,
        focal_length: float,
        aperture: float,
        coc_limit: float = 0.03e-3,  # 0.03mm for full frame
) -> Tuple[float, float, float]:
    """
    Calculate depth of field parameters

    Args:
        focus_distance: Focus distance in meters
        focal_length: Focal length in meters
        aperture: Aperture f-number
        coc_limit: Circle of Confusion limit in meters

    Returns:
        (near_distance, far_distance, hyperfocal_distance)
    """
    # Hyperfocal distance
    hyperfocal = (focal_length ** 2) / (aperture * coc_limit) + focal_length

    # Near and far limits
    denominator_near = hyperfocal - focus_distance
    if denominator_near <= 0:
        near_distance = focus_distance / 2
    else:
        near_distance = (focus_distance * (hyperfocal - focal_length)) / denominator_near

    if focus_distance >= hyperfocal:
        far_distance = float('inf')
    else:
        denominator_far = hyperfocal - focus_distance
        if denominator_far <= 0:
            far_distance = float('inf')
        else:
            far_distance = (focus_distance * (hyperfocal - focal_length)) / denominator_far

    return max(0, near_distance), far_distance, hyperfocal


def focal_length_to_fov(
        focal_length: float,
        sensor_width: float = 0.036,  # Full frame width in meters
) -> float:
    """
    Convert focal length to horizontal field of view

    Args:
        focal_length: Focal length in meters
        sensor_width: Sensor width in meters

    Returns:
        Horizontal field of view in degrees
    """
    fov_rad = 2 * np.arctan(sensor_width / (2 * focal_length))
    fov_deg = np.degrees(fov_rad)
    return fov_deg


def fov_to_focal_length(
        fov: float,
        sensor_width: float = 0.036,  # Full frame width in meters
) -> float:
    """
    Convert field of view to focal length

    Args:
        fov: Horizontal field of view in degrees
        sensor_width: Sensor width in meters

    Returns:
        Focal length in meters
    """
    fov_rad = np.radians(fov)
    focal_length = sensor_width / (2 * np.tan(fov_rad / 2))
    return focal_length


def compute_blur_kernel_size(
        coc: float,
        pixel_size: float,
        scaling_factor: float = 2.0,
) -> int:
    """
    Compute Gaussian blur kernel size from Circle of Confusion

    Args:
        coc: Circle of Confusion in meters
        pixel_size: Pixel size in meters
        scaling_factor: Factor to convert CoC to kernel size

    Returns:
        Kernel size (odd integer)
    """
    # Convert CoC to pixels
    coc_pixels = coc / pixel_size

    # Kernel size (ensure odd)
    kernel_size = int(2 * scaling_factor * coc_pixels + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    return max(1, kernel_size)


def generate_camera_config(
        focal_length: float = 50.0,  # mm
        aperture: float = 2.8,
        sensor_type: str = 'full_frame',
        num_images: int = 5,
        focus_mode: str = 'auto',
        custom_focus_distances: Optional[List[float]] = None,
) -> Dict:
    """
    Generate camera configuration for focal stack capture

    Args:
        focal_length: Focal length in mm
        aperture: Aperture f-number
        sensor_type: Sensor type ('full_frame', 'aps_c', 'm43', etc.)
        num_images: Number of images in focal stack
        focus_mode: 'auto', 'manual', or 'custom'
        custom_focus_distances: Custom focus distances if mode is 'custom'

    Returns:
        Camera configuration dictionary
    """
    # Sensor sizes (width, height) in mm
    sensor_dims = {
        'full_frame': (36.0, 24.0),
        'aps_c': (23.6, 15.7),
        'm43': (17.3, 13.0),
        '1_inch': (13.2, 8.8),
        'smartphone': (6.0, 4.5),
    }

    if sensor_type not in sensor_dims:
        logger.warning(f"Unknown sensor type '{sensor_type}', using full frame")
        sensor_type = 'full_frame'

    sensor_width, sensor_height = sensor_dims[sensor_type]
    sensor_diagonal = np.sqrt(sensor_width ** 2 + sensor_height ** 2)

    # Convert to meters
    focal_length_m = focal_length / 1000
    sensor_width_m = sensor_width / 1000

    # Calculate field of view
    fov_h = focal_length_to_fov(focal_length_m, sensor_width_m)
    fov_v = focal_length_to_fov(focal_length_m, sensor_height / 1000)

    # Determine focus distances
    if focus_mode == 'custom' and custom_focus_distances:
        focus_distances = custom_focus_distances
    elif focus_mode == 'manual':
        # Manual mode with predefined distances
        focus_distances = [0.3, 0.5, 1.0, 2.0, 5.0, 10.0][:num_images]
    else:
        # Auto mode - compute based on hyperfocal distance
        hyperfocal = (focal_length_m ** 2) / (aperture * 0.03e-3) + focal_length_m

        # Distribute focus distances around hyperfocal
        near = 0.3
        far = min(hyperfocal * 2, 20.0)
        focus_distances = estimate_focus_distances(num_images, near, far, 'log')

    # Compute depth of field for each focus distance
    dof_info = []
    for fd in focus_distances:
        near, far, _ = get_depth_of_field(fd, focal_length_m, aperture)
        dof_info.append({
            'focus_distance': fd,
            'near_limit': near,
            'far_limit': far,
            'total_dof': far - near if far != float('inf') else 'inf',
        })

    config = {
        'camera': {
            'focal_length': focal_length,
            'focal_length_m': focal_length_m,
            'aperture': aperture,
            'sensor_type': sensor_type,
            'sensor_width_mm': sensor_width,
            'sensor_height_mm': sensor_height,
            'sensor_diagonal_mm': sensor_diagonal,
            'fov_horizontal': fov_h,
            'fov_vertical': fov_v,
        },
        'capture': {
            'num_images': num_images,
            'focus_mode': focus_mode,
            'focus_distances': focus_distances,
            'dof_info': dof_info,
        },
        'settings': {
            'iso': 'auto',  # Recommend auto ISO
            'exposure_mode': 'aperture_priority',  # Keep aperture constant
            'white_balance': 'fixed',  # Keep WB constant across stack
            'image_format': 'raw+jpeg',  # Capture both
        },
    }

    return config


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