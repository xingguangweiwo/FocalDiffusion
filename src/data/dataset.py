"""Dataset implementation for FocalDiffusion.

The dataset supports loading pre-rendered focal stacks or synthesising them on the
fly from all-in-focus RGB images and metric depth maps (including HyperSim HDF5
files)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .focal_simulator import FocalStackSimulator

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}


class FocalStackDataset(Dataset):
    """Dataset that yields focal stacks, depth maps and auxiliary metadata."""

    def __init__(
        self,
        data_root: Union[str, Path],
        filelist_path: Optional[Union[str, Path]] = None,
        image_size: Tuple[int, int] = (512, 512),
        focal_stack_size: int = 5,
        focal_range: Tuple[float, float] = (0.3, 10.0),
        transform=None,
        max_samples: Optional[int] = None,
        generate_focal_stack: bool = False,
        camera_defaults: Optional[Dict[str, float]] = None,
        simulator_kwargs: Optional[Dict[str, float]] = None,
        depth_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.focal_stack_size = focal_stack_size
        self.focal_range = focal_range
        self.transform = transform
        self.generate_focal_stack = generate_focal_stack
        self.simulator_kwargs = simulator_kwargs or {}
        self.depth_bounds = depth_bounds

        self.camera_defaults = {
            "f_number": 8.0,
            "focal_length": 50e-3,
            "pixel_size": 1.2e-5,
        }
        if camera_defaults:
            self.camera_defaults.update({k: float(v) for k, v in camera_defaults.items()})

        # Load file list or scan the directory
        if filelist_path:
            self.samples = self._load_filelist(filelist_path)
        else:
            self.samples = self._scan_directory()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info("Loaded %d samples", len(self.samples))
        self._simulator: Optional[FocalStackSimulator] = None

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @property
    def simulator(self) -> FocalStackSimulator:
        if self._simulator is None:
            self._simulator = FocalStackSimulator(**self.simulator_kwargs)
        return self._simulator

    def _resolve_path(self, path_like: Union[str, Path]) -> Path:
        path = Path(path_like)
        if not path.is_absolute():
            path = self.data_root / path
        return path

    # ---------------------------------------------------------------------
    # File list handling
    # ---------------------------------------------------------------------
    def _load_filelist(self, filelist_path: Union[str, Path]) -> List[Dict]:
        samples: List[Dict] = []
        with Path(filelist_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("{"):
                    entry = json.loads(line)
                    samples.append(self._normalise_sample(entry))
                else:
                    samples.append(self._parse_legacy_entry(line))
        return samples

    def _parse_legacy_entry(self, line: str) -> Dict:
        tokens = [part.strip() for part in line.replace(",", " ").split() if part.strip()]
        if len(tokens) < 2:
            raise ValueError(f"Invalid filelist entry: {line}")

        sample: Dict = {}

        first_token, second_token = tokens[0], tokens[1]
        first_ext = Path(first_token).suffix.lower()

        if first_ext in IMAGE_EXTENSIONS:
            sample["all_in_focus"] = first_token
        else:
            sample["focal_stack_dir"] = first_token

        sample["depth_path"] = second_token

        for extra in tokens[2:]:
            if "=" in extra:
                key, value = extra.split("=", 1)
                sample[key.strip()] = self._coerce_value(value.strip())
                continue

            ext = Path(extra).suffix.lower()
            if ext in IMAGE_EXTENSIONS and "all_in_focus" not in sample:
                sample["all_in_focus"] = extra
                continue

            try:
                numeric = float(extra)
            except ValueError:
                continue

            if numeric.is_integer():
                sample["num_images"] = int(numeric)
            else:
                sample["num_images"] = int(round(numeric))

        if "all_in_focus" in sample and "focal_stack_dir" not in sample:
            sample.setdefault("generate_focal_stack", True)

        return sample

    @staticmethod
    def _coerce_value(raw: str):
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        try:
            integer = int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
        else:
            return integer

    def _normalise_sample(self, entry: Dict) -> Dict:
        sample: Dict = {}

        depth_key = entry.get("depth") or entry.get("depth_path")
        if not depth_key:
            raise ValueError("Each entry must provide a depth or depth_path field")
        sample["depth_path"] = depth_key

        focal_key = entry.get("focal_stack") or entry.get("focal_stack_dir") or entry.get("scene_path")
        if focal_key:
            sample["focal_stack_dir"] = focal_key

        if entry.get("all_in_focus") or entry.get("aif") or entry.get("aif_path"):
            sample["all_in_focus"] = entry.get("all_in_focus") or entry.get("aif") or entry.get("aif_path")

        if "focus_distances" in entry:
            sample["focus_distances"] = [float(v) for v in entry["focus_distances"]]

        if "focus_range" in entry and entry["focus_range"]:
            sample["focus_range"] = [float(v) for v in entry["focus_range"]]

        if "depth_dataset" in entry:
            sample["depth_dataset"] = entry["depth_dataset"]
        if "depth_scale" in entry:
            sample["depth_scale"] = float(entry["depth_scale"])
        if "depth_shift" in entry:
            sample["depth_shift"] = float(entry["depth_shift"])
        if "depth_range" in entry:
            dr = entry["depth_range"]
            if len(dr) != 2:
                raise ValueError("depth_range must contain [min, max]")
            sample["depth_range"] = [float(dr[0]), float(dr[1])]

        if "generate_focal_stack" in entry:
            sample["generate_focal_stack"] = bool(entry["generate_focal_stack"])

        if "num_images" in entry:
            sample["num_images"] = int(entry["num_images"])

        if "camera" in entry and isinstance(entry["camera"], dict):
            sample["camera"] = {k: float(v) for k, v in entry["camera"].items()}

        if "transpose_depth" in entry:
            sample["transpose_depth"] = bool(entry["transpose_depth"])

        return sample

    def _scan_directory(self) -> List[Dict]:
        # Dataset-specific subclasses override this when file lists are unavailable.
        return []

    # ---------------------------------------------------------------------
    # PyTorch dataset interface
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        depth, depth_meta = self._load_depth(sample_info)
        use_generated = self._should_generate_stack(sample_info)

        if use_generated:
            focus_distances = self._get_focus_distances(sample_info, self.focal_stack_size)
            all_in_focus = self._load_all_in_focus(sample_info)
            focal_stack = self.simulator.generate(
                all_in_focus,
                depth.squeeze(0),
                focus_distances,
                sample_info.get("camera"),
            )
        else:
            focal_stack = self._load_focal_stack(sample_info)
            focus_distances = self._get_focus_distances(sample_info, focal_stack.shape[0])
            all_in_focus = self._load_all_in_focus(sample_info, fallback=focal_stack)

        if focal_stack.shape[0] != focus_distances.numel():
            focus_distances = self._align_focus_distances(focus_distances, focal_stack.shape[0])

        camera_params = self._compose_camera_params(sample_info, depth_meta)

        sample = {
            "focal_stack": focal_stack,
            "depth": depth,
            "all_in_focus": all_in_focus,
            "focus_distances": focus_distances,
            "camera_params": camera_params,
            "sample_path": sample_info.get("focal_stack_dir") or sample_info.get("depth_path", ""),
            "depth_range": torch.tensor(
                [depth_meta["min"], depth_meta["max"]], dtype=torch.float32
            ),
            "valid_mask": depth_meta["mask"],
        }

        if self.transform:
            sample["focal_stack"], sample["depth"], sample["all_in_focus"] = self.transform(
                sample["focal_stack"], sample["depth"], sample["all_in_focus"]
            )

        return sample

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _should_generate_stack(self, sample_info: Dict) -> bool:
        if "generate_focal_stack" in sample_info:
            return bool(sample_info["generate_focal_stack"])
        if self.generate_focal_stack:
            return True
        return "all_in_focus" in sample_info and "focal_stack_dir" not in sample_info

    def _load_all_in_focus(
        self, sample_info: Dict, fallback: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        path = sample_info.get("all_in_focus")
        if path:
            return self._load_image(self._resolve_path(path))
        if fallback is not None:
            return self._generate_all_in_focus(fallback)
        raise ValueError(
            "Sample requires all_in_focus image or pre-rendered focal stack"
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            array = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_focal_stack(self, sample_info: Dict) -> torch.Tensor:
        dir_key = sample_info.get("focal_stack_dir") or sample_info.get("scene_path")
        if not dir_key:
            raise ValueError("Sample must include a focal_stack_dir when not generating")

        scene_dir = self._resolve_path(dir_key)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Focal stack directory {scene_dir} does not exist")

        image_files = []
        patterns = sample_info.get("image_glob")
        if patterns:
            if isinstance(patterns, str):
                patterns = [patterns]
            for pattern in patterns:
                image_files.extend(sorted(scene_dir.glob(pattern)))
        else:
            image_files.extend(sorted(scene_dir.glob("*.png")))
            image_files.extend(sorted(scene_dir.glob("*.jpg")))
            image_files.extend(sorted(scene_dir.glob("*.jpeg")))

        if not image_files:
            raise FileNotFoundError(f"No focal stack images found in {scene_dir}")

        max_images = sample_info.get("num_images") or self.focal_stack_size
        if max_images <= 0:
            max_images = len(image_files)

        selected_files = image_files[:max_images]
        images = [self._load_image(path) for path in selected_files]

        while len(images) < max_images:
            images.append(images[-1].clone())

        return torch.stack(images, dim=0)

    def _get_focus_distances(self, sample_info: Dict, num_images: int) -> torch.Tensor:
        if "focus_distances" in sample_info:
            distances = torch.as_tensor(sample_info["focus_distances"], dtype=torch.float32)
        else:
            focus_range = sample_info.get("focus_range", self.focal_range)
            near, far = focus_range
            distances = torch.logspace(
                np.log10(max(near, 1e-3)),
                np.log10(max(far, near + 1e-3)),
                steps=max(num_images, 1),
                dtype=torch.float32,
            )
        return self._align_focus_distances(distances, num_images)

    def _align_focus_distances(self, distances: torch.Tensor, num_images: int) -> torch.Tensor:
        if distances.numel() == num_images:
            return distances
        if distances.numel() > num_images:
            return distances[:num_images]
        if distances.numel() == 0:
            distances = torch.full((1,), 1.0, dtype=torch.float32)
        pad_value = distances[-1]
        padding = pad_value.repeat(num_images - distances.numel())
        return torch.cat([distances, padding], dim=0)

    def _compose_camera_params(self, sample_info: Dict, depth_meta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        params: Dict[str, torch.Tensor] = {}
        dtype = torch.float32

        camera_values = dict(self.camera_defaults)
        camera_values.update(sample_info.get("camera", {}))

        for key, value in camera_values.items():
            params[key] = torch.tensor(float(value), dtype=dtype)

        if "aperture" in sample_info.get("camera", {}):
            params["aperture"] = torch.tensor(
                float(sample_info["camera"]["aperture"]), dtype=dtype
            )
        else:
            params["aperture"] = params["focal_length"] / torch.clamp(
                params["f_number"], min=1e-6
            )

        params["depth_min"] = torch.tensor(depth_meta["min"], dtype=dtype)
        params["depth_max"] = torch.tensor(depth_meta["max"], dtype=dtype)
        return params

    def _load_depth(self, sample_info: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        depth_path = self._resolve_path(sample_info["depth_path"])

        if depth_path.suffix.lower() in {".h5", ".hdf5"}:
            depth = self._load_hdf5_depth(depth_path, sample_info.get("depth_dataset"))
        elif depth_path.suffix.lower() == ".npy":
            depth = np.load(depth_path)
        elif depth_path.suffix.lower() in {".exr", ".pfm"}:
            try:
                import cv2  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    f"OpenCV is required to read depth map {depth_path}"
                ) from exc
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        else:
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img)

        depth = depth.astype(np.float32)

        scale = sample_info.get("depth_scale", 1.0)
        shift = sample_info.get("depth_shift", 0.0)
        depth = depth * float(scale) + float(shift)

        valid_mask = np.isfinite(depth) & (depth > 0)
        if not np.any(valid_mask):
            raise ValueError(f"Depth map {depth_path} does not contain valid samples")

        if sample_info.get("transpose_depth"):
            depth = depth.T
            valid_mask = valid_mask.T

        depth_bounds = self.depth_bounds or (None, None)
        range_override = sample_info.get("depth_range")

        valid_depth = depth[valid_mask]
        depth_min = float(valid_depth.min())
        depth_max = float(valid_depth.max())

        if depth_bounds[0] is not None:
            depth_min = max(depth_min, float(depth_bounds[0]))
        if depth_bounds[1] is not None:
            depth_max = min(depth_max, float(depth_bounds[1]))

        if range_override:
            depth_min = max(depth_min, float(range_override[0]))
            depth_max = min(depth_max, float(range_override[1]))

        if depth_max <= depth_min:
            depth_max = depth_min + 1e-3

        depth = np.clip(depth, depth_min, depth_max)

        depth_image = Image.fromarray(depth)
        depth_image = depth_image.resize(self.image_size, Image.Resampling.NEAREST)
        depth_resized = np.array(depth_image, dtype=np.float32)

        mask_image = Image.fromarray((valid_mask.astype(np.uint8) * 255))
        mask_image = mask_image.resize(self.image_size, Image.Resampling.NEAREST)
        mask_resized = np.array(mask_image) > 0

        masked_depth = depth_resized[mask_resized]
        if masked_depth.size > 0:
            depth_min = float(masked_depth.min())
            depth_max = float(masked_depth.max())

        depth_tensor = torch.from_numpy(depth_resized).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_resized).to(torch.bool)

        return depth_tensor, {"min": depth_min, "max": depth_max, "mask": mask_tensor}

    def _load_hdf5_depth(self, path: Path, dataset_name: Optional[str]) -> np.ndarray:
        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("h5py is required to read HyperSim depth maps") from exc

        with h5py.File(path, "r") as handle:
            if dataset_name:
                dataset_names = [dataset_name]
            else:
                dataset_names = [
                    "depth",
                    "depth_meters",
                    "dataset",
                    "depth_m",
                ]
                dataset_names.extend(handle.keys())

            for name in dataset_names:
                if name in handle:
                    data = handle[name][()]
                    if data.ndim == 3 and data.shape[0] == 1:
                        data = data[0]
                    return np.array(data)
                if name.startswith("/") and name[1:] in handle:
                    data = handle[name[1:]][()]
                    if data.ndim == 3 and data.shape[0] == 1:
                        data = data[0]
                    return np.array(data)

        raise KeyError(
            f"Unable to locate a depth dataset inside {path}; tried {dataset_names}"
        )

    def _generate_all_in_focus(self, focal_stack: torch.Tensor) -> torch.Tensor:
        num_images, channels, height, width = focal_stack.shape
        gradients = []
        for i in range(num_images):
            img = focal_stack[i]
            grad_x = torch.abs(img[:, :, 1:] - img[:, :, :-1])
            grad_y = torch.abs(img[:, 1:, :] - img[:, :-1, :])
            grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
            grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=0)
            gradients.append(grad_mag)

        gradients = torch.stack(gradients, dim=0)
        best_indices = torch.argmax(gradients, dim=0)
        all_in_focus = torch.zeros((channels, height, width), dtype=focal_stack.dtype)
        for i in range(num_images):
            mask = (best_indices == i).unsqueeze(0)
            all_in_focus += focal_stack[i] * mask
        return all_in_focus


class HyperSimDataset(FocalStackDataset):
    """HyperSim dataset implementation."""

    def _scan_directory(self) -> List[Dict]:
        samples: List[Dict] = []
        scenes_dir = self.data_root / "scenes"
        if not scenes_dir.exists():
            return samples

        for scene_dir in sorted(scenes_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            images_dir = scene_dir / "images"
            depth_dir = scene_dir / "depth"
            if not images_dir.exists() or not depth_dir.exists():
                continue

            frame_groups: Dict[str, List[Path]] = {}
            for img_file in images_dir.glob("*.jpg"):
                frame_id = img_file.stem.split(".")[0]
                frame_groups.setdefault(frame_id, []).append(img_file)

            for frame_id, img_files in frame_groups.items():
                depth_file = depth_dir / f"{frame_id}.depth.exr"
                if depth_file.exists():
                    samples.append(
                        {
                            "focal_stack_dir": str(images_dir.relative_to(self.data_root)),
                            "depth_path": str(depth_file.relative_to(self.data_root)),
                            "num_images": len(img_files),
                        }
                    )
        return samples


class VirtualKITTIDataset(FocalStackDataset):
    """Virtual KITTI dataset implementation."""

    def _scan_directory(self) -> List[Dict]:
        samples: List[Dict] = []
        for scene_dir in sorted(self.data_root.iterdir()):
            if not scene_dir.is_dir() or not scene_dir.name.startswith("Scene"):
                continue

            rgb_dir = scene_dir / "frames" / "rgb" / "Camera_0"
            depth_dir = scene_dir / "frames" / "depth" / "Camera_0"
            if not rgb_dir.exists() or not depth_dir.exists():
                continue

            for rgb_file in sorted(rgb_dir.glob("*.jpg")):
                frame_num = rgb_file.stem.replace("rgb", "")
                depth_file = depth_dir / f"depth{frame_num}.png"
                if depth_file.exists():
                    samples.append(
                        {
                            "focal_stack_dir": str(rgb_dir.relative_to(self.data_root)),
                            "depth_path": str(depth_file.relative_to(self.data_root)),
                            "num_images": 1,
                        }
                    )
        return samples


def create_dataloader(
    dataset_type: Optional[str] = None,
    filelist_path: Optional[Union[str, Path]] = None,
    data_root: Union[str, Path] = "./data",
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    focal_stack_size: int = 5,
    focal_range: Tuple[float, float] = (0.3, 10.0),
    augmentation: bool = False,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    **dataset_kwargs,
) -> DataLoader:
    """Create dataloader for training or evaluation."""

    if filelist_path or sources:
        dataset_class = FocalStackDataset
    elif dataset_type == "hypersim":
        dataset_class = HyperSimDataset
    elif dataset_type == "virtual_kitti":
        dataset_class = VirtualKITTIDataset
    else:
        dataset_class = FocalStackDataset

    transform = None
    if augmentation:
        from .augmentation import FocalAugmentation

        transform = FocalAugmentation()

    def _build_dataset(source_kwargs: Dict[str, Any]) -> Dataset:
        merged_kwargs = dict(dataset_kwargs)
        merged_kwargs.update(source_kwargs.get("dataset_kwargs", {}))

        root_override = source_kwargs.get("data_root", data_root)
        filelist_override = source_kwargs.get("filelist", filelist_path)

        if issubclass(dataset_class, FocalStackDataset) and filelist_override is None:
            raise ValueError("filelist must be provided when using filelist datasets")

        return dataset_class(
            data_root=root_override,
            filelist_path=filelist_override,
            image_size=image_size,
            focal_stack_size=focal_stack_size,
            focal_range=focal_range,
            transform=transform,
            max_samples=source_kwargs.get("max_samples", max_samples),
            **merged_kwargs,
        )

    datasets: List[Dataset]

    if sources:
        datasets = [_build_dataset(source) for source in sources]
    else:
        datasets = [
            _build_dataset({
                "data_root": data_root,
                "filelist": filelist_path,
                "max_samples": max_samples,
                "dataset_kwargs": {},
            })
        ]

    dataset: Dataset
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
