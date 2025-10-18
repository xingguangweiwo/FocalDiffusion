"""
Dataset implementation for FocalDiffusion
Handles HyperSim and Virtual KITTI datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FocalStackDataset(Dataset):
    """Base dataset for focal stack data"""

    def __init__(
            self,
            data_root: Union[str, Path],
            filelist_path: Optional[Union[str, Path]] = None,
            image_size: Tuple[int, int] = (512, 512),
            focal_stack_size: int = 5,
            focal_range: Tuple[float, float] = (0.3, 10.0),
            transform=None,
            max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.focal_stack_size = focal_stack_size
        self.focal_range = focal_range
        self.transform = transform

        # Load file list
        if filelist_path:
            self.samples = self._load_filelist(filelist_path)
        else:
            self.samples = self._scan_directory()

        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info(f"Loaded {len(self.samples)} samples")

    def _load_filelist(self, filelist_path: Union[str, Path]) -> List[Dict]:
        """Load sample paths from file list"""
        samples = []
        with open(filelist_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse line format: scene_path,depth_path,num_images
                    parts = line.split(',')
                    if len(parts) >= 2:
                        samples.append({
                            'scene_path': parts[0],
                            'depth_path': parts[1],
                            'num_images': int(parts[2]) if len(parts) > 2 else self.focal_stack_size
                        })
        return samples

    def _scan_directory(self) -> List[Dict]:
        """Scan directory for samples"""
        samples = []
        # Override in dataset-specific classes
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # Load focal stack
        focal_stack = self._load_focal_stack(sample_info)

        # Load depth
        depth = self._load_depth(sample_info)

        # Generate all-in-focus (simplified - use sharpest regions)
        all_in_focus = self._generate_all_in_focus(focal_stack)

        # Generate focus distances
        focus_distances = self._generate_focus_distances(focal_stack.shape[0])

        # Apply transforms
        if self.transform:
            focal_stack, depth, all_in_focus = self.transform(
                focal_stack, depth, all_in_focus
            )

        return {
            'focal_stack': focal_stack,
            'depth': depth,
            'all_in_focus': all_in_focus,
            'focus_distances': focus_distances,
            'camera_params': self._get_camera_params(sample_info),
            'sample_path': sample_info.get('scene_path', ''),
        }

    def _load_focal_stack(self, sample_info: Dict) -> torch.Tensor:
        """Load focal stack images"""
        scene_path = self.data_root / sample_info['scene_path']

        # Find all images in the scene directory
        image_files = sorted(scene_path.glob('*.png')) + sorted(scene_path.glob('*.jpg'))

        # Load up to focal_stack_size images
        images = []
        for img_file in image_files[:self.focal_stack_size]:
            img = Image.open(img_file).convert('RGB')
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            images.append(img_tensor)

        # Pad if necessary
        while len(images) < self.focal_stack_size:
            images.append(images[-1].clone() if images else torch.zeros(3, *self.image_size))

        return torch.stack(images)

    def _load_depth(self, sample_info: Dict) -> torch.Tensor:
        """Load depth map"""
        depth_path = self.data_root / sample_info['depth_path']

        if depth_path.suffix == '.npy':
            depth = np.load(depth_path)
        elif depth_path.suffix == '.exr':
            # Use OpenEXR if available
            import cv2
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        else:
            # Load as image and convert
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img).astype(np.float32)

            # Normalize depth to meters (assuming 16-bit depth)
            if depth.max() > 255:
                depth = depth / 1000.0  # mm to meters

        # Resize
        depth_pil = Image.fromarray(depth)
        depth_pil = depth_pil.resize(self.image_size, Image.Resampling.NEAREST)
        depth = np.array(depth_pil)

        return torch.from_numpy(depth).unsqueeze(0)

    def _generate_all_in_focus(self, focal_stack: torch.Tensor) -> torch.Tensor:
        """Generate all-in-focus image from focal stack"""
        # Simple approach: take maximum gradient magnitude
        N, C, H, W = focal_stack.shape

        # Compute gradient magnitude for each image
        gradients = []
        for i in range(N):
            img = focal_stack[i]
            # Compute gradients
            grad_x = torch.abs(img[:, :, 1:] - img[:, :, :-1])
            grad_y = torch.abs(img[:, 1:, :] - img[:, :-1, :])

            # Pad to original size
            grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
            grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))

            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=0)
            gradients.append(grad_mag)

        gradients = torch.stack(gradients)

        # Find sharpest image at each pixel
        best_indices = torch.argmax(gradients, dim=0)

        # Create all-in-focus by selecting from focal stack
        all_in_focus = torch.zeros_like(focal_stack[0])
        for i in range(N):
            mask = (best_indices == i).unsqueeze(0).expand(C, -1, -1)
            all_in_focus += focal_stack[i] * mask.float()

        return all_in_focus

    def _generate_focus_distances(self, num_images: int) -> torch.Tensor:
        """Generate focus distances"""
        # Log-spaced distances
        distances = torch.logspace(
            np.log10(self.focal_range[0]),
            np.log10(self.focal_range[1]),
            num_images
        )
        return distances

    def _get_camera_params(self, sample_info: Dict) -> Dict[str, torch.Tensor]:
        """Get camera parameters"""
        # Default camera parameters
        return {
            'focal_length': torch.tensor(0.050),  # 50mm in meters
            'aperture': torch.tensor(2.8),
            'sensor_size': torch.tensor(0.036),  # Full frame
        }


class HyperSimDataset(FocalStackDataset):
    """HyperSim dataset implementation"""

    def _scan_directory(self) -> List[Dict]:
        """Scan HyperSim directory structure"""
        samples = []

        # HyperSim structure: scenes/sceneXX/images/frameXXXX.tonemap.jpg
        scenes_dir = self.data_root / 'scenes'
        if scenes_dir.exists():
            for scene_dir in sorted(scenes_dir.iterdir()):
                if scene_dir.is_dir():
                    images_dir = scene_dir / 'images'
                    depth_dir = scene_dir / 'depth'

                    if images_dir.exists() and depth_dir.exists():
                        # Group images by frame
                        frame_groups = {}
                        for img_file in images_dir.glob('*.jpg'):
                            frame_id = img_file.stem.split('.')[0]
                            if frame_id not in frame_groups:
                                frame_groups[frame_id] = []
                            frame_groups[frame_id].append(img_file)

                        for frame_id, img_files in frame_groups.items():
                            depth_file = depth_dir / f"{frame_id}.depth.exr"
                            if depth_file.exists():
                                samples.append({
                                    'scene_path': str(images_dir.relative_to(self.data_root)),
                                    'depth_path': str(depth_file.relative_to(self.data_root)),
                                    'frame_id': frame_id,
                                    'num_images': len(img_files),
                                })

        return samples


class VirtualKITTIDataset(FocalStackDataset):
    """Virtual KITTI dataset implementation"""

    def _scan_directory(self) -> List[Dict]:
        """Scan Virtual KITTI directory structure"""
        samples = []

        # Virtual KITTI structure: Scene01/frames/rgb/Camera_0/rgbXXXXX.jpg
        for scene_dir in sorted(self.data_root.iterdir()):
            if scene_dir.is_dir() and scene_dir.name.startswith('Scene'):
                rgb_dir = scene_dir / 'frames' / 'rgb' / 'Camera_0'
                depth_dir = scene_dir / 'frames' / 'depth' / 'Camera_0'

                if rgb_dir.exists() and depth_dir.exists():
                    for rgb_file in sorted(rgb_dir.glob('*.jpg')):
                        frame_num = rgb_file.stem.replace('rgb', '')
                        depth_file = depth_dir / f"depth{frame_num}.png"

                        if depth_file.exists():
                            samples.append({
                                'scene_path': str(rgb_dir.relative_to(self.data_root)),
                                'depth_path': str(depth_file.relative_to(self.data_root)),
                                'frame_id': frame_num,
                                'num_images': 1,  # Will be augmented
                            })

        return samples


def create_dataloader(
        dataset_type: str = None,
        filelist_path: Optional[Union[str, Path]] = None,
        data_root: Union[str, Path] = './data',
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
        focal_stack_size: int = 5,
        focal_range: Tuple[float, float] = (0.3, 10.0),
        augmentation: bool = False,
        shuffle: bool = True,
        max_samples: Optional[int] = None,
        **kwargs
) -> DataLoader:
    """Create dataloader for training or evaluation"""

    # Select dataset class
    if filelist_path:
        # Use base class with file list
        dataset_class = FocalStackDataset
    elif dataset_type == 'hypersim':
        dataset_class = HyperSimDataset
    elif dataset_type == 'virtual_kitti':
        dataset_class = VirtualKITTIDataset
    else:
        dataset_class = FocalStackDataset

    # Setup augmentation
    transform = None
    if augmentation:
        from .augmentation import FocalAugmentation
        transform = FocalAugmentation()

    # Create dataset
    dataset = dataset_class(
        data_root=data_root,
        filelist_path=filelist_path,
        image_size=image_size,
        focal_stack_size=focal_stack_size,
        focal_range=focal_range,
        transform=transform,
        max_samples=max_samples,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if shuffle else False,
    )

    return dataloader