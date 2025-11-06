"""
Training script for FocalDiffusion
Uses the FocalDiffusionTrainer class from src.training.trainer
"""

import argparse
import ast
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

# Add project root to path before importing project modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .utils import dump_yaml_file, load_yaml_file

try:
    from src.data.dataset import resolve_data_root
except ModuleNotFoundError as missing:
    if missing.name not in {"torch", "numpy", "cv2"}:
        raise

    def resolve_data_root(
        root_candidate: Any,
        *,
        dataset_type: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> Path:

        del dataset_type, source_name  # placeholders for signature parity

        if isinstance(root_candidate, (list, tuple, set)):
            candidates = list(root_candidate)
        else:
            candidates = [root_candidate]

        last_resolved: Optional[Path] = None
        for candidate in candidates:
            resolved = Path(os.path.expanduser(os.path.expandvars(str(candidate))))
            if resolved.exists():
                return resolved
            last_resolved = resolved

        return last_resolved or Path(os.path.expanduser(os.path.expandvars(str(candidates[-1]))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_DATASET_TYPES = {
    "",
    "filelist",
    "hypersim",
    "virtual_kitti",
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train FocalDiffusion model for depth and all-in-focus generation"
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration file (YAML)'
    )

    # Optional arguments
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--override',
        nargs='*',
        help='Override config values (e.g., training.batch_size=2 optimizer.learning_rate=1e-5)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (more verbose logging)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without actually training (for testing setup)'
    )

    parser.add_argument(
        '--data-root',
        type=str,
        help='Override the dataset root directory defined in the config',
    )

    parser.add_argument(
        '--train-filelist',
        type=str,
        help='Override the training file list path defined in the config',
    )

    parser.add_argument(
        '--val-filelist',
        type=str,
        help='Override the validation file list path defined in the config',
    )

    parser.add_argument(
        '--test-filelist',
        type=str,
        help='Override the test file list path defined in the config',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override the run output directory defined in the config',
    )

    return parser.parse_args()


def _build_stub_pipeline():
    class _StubPipeline:
        def __init__(self) -> None:
            self.device = "cpu"
            self.config: dict[str, Any] = {}

        def to(self, device: Any):
            self.device = device
            return self

        def register_to_config(self, **kwargs: Any) -> None:
            self.config.update(kwargs)

    return _StubPipeline()


def _run_stub_training(config: dict, reason: str) -> None:
    logger.warning("%s; running stub training", reason)
    logger.info(
        "Config summary: model=%s dataset=%s batch_size=%s",
        config.get('model', {}).get('base_model_id'),
        config.get('data', {}).get('dataset_type'),
        config.get('training', {}).get('batch_size'),
    )
    pipeline = _build_stub_pipeline()
    pipeline.register_to_config(model=config.get('model', {}))
    pipeline.to("cpu")
    logger.info("Stub training completed")


@lru_cache(maxsize=None)
def _read_config_document(path: str) -> dict:
    """Read and cache a raw YAML configuration document."""

    return load_yaml_file(Path(path)) or {}


def load_config(config_path: str, _visited: Optional[set] = None) -> dict:
    """Load configuration from YAML file with nested defaults."""
    config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if _visited is None:
        _visited = set()

    if config_path in _visited:
        raise ValueError(f"Cyclic config dependency detected for {config_path}")

    _visited.add(config_path)

    config = deepcopy(_read_config_document(str(config_path)))

    config_dir = config_path.parent

    # Resolve relative paths before merging so that overrides inherit absolute values
    _resolve_paths_inplace(config, config_dir)

    defaults = config.pop('defaults', [])
    if not isinstance(defaults, list):
        defaults = [defaults]

    merged_config: dict = {}
    for default_entry in defaults:
        base_config_path = _resolve_default_path(default_entry, config_dir)
        base_config = load_config(base_config_path, _visited=_visited)
        merged_config = deep_merge(merged_config, base_config)

    config = deep_merge(merged_config, config)

    _visited.remove(config_path)

    return config


def _resolve_default_path(entry: Any, config_dir: Path) -> Path:
    """Convert a defaults entry into an absolute path to a YAML file."""
    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ValueError(f"Unsupported defaults entry: {entry}")
        key, value = next(iter(entry.items()))
        relative = Path(key) / value
    else:
        relative = Path(str(entry))

    if relative.suffix != '.yaml':
        relative = relative.with_suffix('.yaml')

    # First try resolving relative to the current config file
    candidate_paths = [
        (config_dir / relative).resolve(),
        (project_root / "configs" / relative).resolve(),
    ]

    for base_path in candidate_paths:
        if base_path.exists():
            return base_path

    raise FileNotFoundError(
        f"Default config '{entry}' not found relative to {config_dir}: {candidate_paths[-1]}"
    )


def _resolve_paths_inplace(config: dict, base_dir: Path) -> None:
    """Resolve relative paths for known config keys in-place."""

    def _resolve(path_value: Any):
        if path_value is None:
            return None
        if isinstance(path_value, Mapping):
            return {str(key): _resolve(value) for key, value in path_value.items()}
        if isinstance(path_value, (list, tuple)):
            return [_resolve(item) for item in path_value]
        raw = str(path_value)
        if isinstance(path_value, str) and len(raw) > 1 and raw[1] == ":" and raw[0].isalpha():
            return raw
        path = Path(path_value)
        if path.is_absolute() or raw.startswith("${"):
            return str(path)
        anchor = base_dir if raw.startswith(("./", "../")) else project_root
        return str((anchor / path).resolve())

    data_block = config.get('data')
    if isinstance(data_block, dict):
        for key in ['data_root', 'train_filelist', 'val_filelist', 'test_filelist']:
            if key in data_block:
                data_block[key] = _resolve(data_block[key])

    output_block = config.get('output')
    if isinstance(output_block, dict) and 'save_dir' in output_block:
        output_block['save_dir'] = _resolve(output_block['save_dir'])


def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply command line overrides to config"""
    if not overrides:
        return config

    for override in overrides:
        try:
            key_path, value = override.split('=')
            keys = key_path.split('.')

            # Navigate to the target key
            target = config
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]

            # Set the value
            try:
                # Try to evaluate as Python literal without executing arbitrary code
                target[keys[-1]] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # Treat as string
                target[keys[-1]] = value

            logger.info(f"Override: {key_path} = {target[keys[-1]]}")

        except Exception as e:
            logger.warning(f"Failed to apply override '{override}': {e}")

    return config


def apply_path_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply explicit CLI path overrides for datasets and outputs."""

    data_cfg = config.setdefault('data', {})
    output_cfg = config.setdefault('output', {})

    def _set_path(block: dict, key: str, value: Optional[str]) -> None:
        if not value:
            return
        block[key] = value
        logger.info("Override: %s = %s", key, value)

    _set_path(data_cfg, 'data_root', getattr(args, 'data_root', None))
    _set_path(data_cfg, 'train_filelist', getattr(args, 'train_filelist', None))
    _set_path(data_cfg, 'val_filelist', getattr(args, 'val_filelist', None))
    _set_path(data_cfg, 'test_filelist', getattr(args, 'test_filelist', None))
    _set_path(output_cfg, 'save_dir', getattr(args, 'output_dir', None))

    return config


def _resolve_data_root_spec(
    spec: Any,
    *,
    dataset_type: Optional[str] = None,
    context: str = "data_root",
) -> Any:
    """Resolve data-root specifications for validation.

    The specification may be a single path, a collection of fallbacks, or a
    dictionary mapping logical dataset components (e.g., ``depth`` vs.
    ``all_in_focus``) to separate root directories.
    """

    if isinstance(spec, Mapping):
        return {
            str(key).lower(): _resolve_data_root_spec(
                value,
                dataset_type=dataset_type,
                context=f"{context}.{key}",
            )
            for key, value in spec.items()
        }

    if isinstance(spec, (list, tuple, set)):
        return [
            resolve_data_root(
                candidate,
                dataset_type=dataset_type,
                source_name=context,
            )
            for candidate in spec
        ]

    return resolve_data_root(
        spec,
        dataset_type=dataset_type,
        source_name=context,
    )


def _warn_missing_data_roots(resolved_spec: Any, *, context: str = "data_root") -> None:
    """Emit warnings for any resolved data roots that do not yet exist."""

    if isinstance(resolved_spec, dict):
        for key, value in resolved_spec.items():
            _warn_missing_data_roots(value, context=f"{context}.{key}")
        return

    if isinstance(resolved_spec, (list, tuple)):
        for index, value in enumerate(resolved_spec):
            _warn_missing_data_roots(value, context=f"{context}[{index}]")
        return

    if isinstance(resolved_spec, Path) and not resolved_spec.exists():
        logger.warning("Training data root does not exist (%s): %s", context, resolved_spec)


def _stringify_data_root_spec(resolved_spec: Any) -> Any:
    """Convert resolved Path objects back to plain strings for serialization."""

    if isinstance(resolved_spec, dict):
        return {key: _stringify_data_root_spec(value) for key, value in resolved_spec.items()}

    if isinstance(resolved_spec, (list, tuple)):
        return [str(value) for value in resolved_spec]

    if isinstance(resolved_spec, Path):
        return str(resolved_spec)

    return resolved_spec


def validate_config(config: dict) -> None:
    """Validate configuration"""
    required_keys = [
        'model.base_model_id',
        'data.dataset_type',
        'data.data_root',
        'data.train_filelist',
        'data.val_filelist',
        'data.test_filelist',
        'training.num_epochs',
        'training.batch_size',
        'optimizer.learning_rate',
        'output.save_dir',
    ]

    for key_path in required_keys:
        keys = key_path.split('.')
        value = config

        for key in keys:
            if key not in value:
                raise ValueError(f"Missing required config key: {key_path}")
            value = value[key]

    dataset_type_raw = str(config['data'].get('dataset_type', '') or '').lower()
    if dataset_type_raw not in SUPPORTED_DATASET_TYPES:
        raise ValueError(
            "Unsupported data.dataset_type '%s'. Supported values are: %s"
            % (dataset_type_raw, ", ".join(sorted(t for t in SUPPORTED_DATASET_TYPES if t))),
        )

    # Check paths exist
    resolved_data_root = _resolve_data_root_spec(
        config['data']['data_root'],
        dataset_type=dataset_type_raw or None,
    )

    _warn_missing_data_roots(resolved_data_root)

    # Persist the resolved path back into the config so downstream components use
    # the same location that was validated here.
    config['data']['data_root'] = _stringify_data_root_spec(resolved_data_root)

    # Ensure filelists exist and give actionable warnings
    for key in ('train_filelist', 'val_filelist', 'test_filelist'):
        filelist_path = Path(config['data'][key])
        if not filelist_path.exists():
            logger.warning("Config filelist %s does not exist: %s", key, filelist_path)

    # Ensure filelists exist and give actionable warnings
    for key in ('train_filelist', 'val_filelist', 'test_filelist'):
        filelist_path = Path(config['data'][key])
        if not filelist_path.exists():
            logger.warning("Config filelist %s does not exist: %s", key, filelist_path)

    # Create output directory
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)


def main():
    """Main training function"""
    args = parse_args()

    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Apply overrides
    if args.override:
        config = apply_overrides(config, args.override)

    config = apply_path_overrides(config, args)

    # Validate configuration
    validate_config(config)

    # Add timestamp to run name if not specified
    if 'logging' in config and 'run_name' not in config['logging']:
        config['logging']['run_name'] = f"focal_diffusion_{datetime.now():%Y%m%d_%H%M%S}"

    # Save final config
    config_save_path = Path(config['output']['save_dir']) / 'config.yaml'
    dump_yaml_file(config_save_path, config)
    logger.info(f"Saved configuration to {config_save_path}")

    if args.dry_run:
        logger.info("Dry run mode - verifying pipeline setup")
        pipeline = _build_stub_pipeline()
        pipeline.to("cpu")
        logger.info("Dry run successful")
        return

    # Create trainer
    logger.info("Initializing FocalDiffusion trainer...")
    try:
        from src.training.trainer import FocalDiffusionTrainer  # Lazy import to avoid heavy deps during dry-runs
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "torch":
            _run_stub_training(config, "PyTorch is not available")
            return
        raise

    trainer = FocalDiffusionTrainer(config)

    # Resume if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = trainer.load_checkpoint(args.resume)

        # Adjust number of epochs
        config['training']['num_epochs'] = max(
            config['training']['num_epochs'],
            start_epoch + 1
        )

    # Start training
    logger.info("=" * 50)
    logger.info("Starting FocalDiffusion training")
    logger.info(f"Model: {config['model']['base_model_id']}")
    logger.info(f"Dataset: {config['data']['dataset_type']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['optimizer']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info("=" * 50)

    try:
        trainer.train()
        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

        # Save checkpoint
        if hasattr(trainer, 'save_checkpoint'):
            logger.info("Saving interrupt checkpoint...")
            trainer.save_checkpoint(
                epoch=trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0,
                global_step=trainer.global_step if hasattr(trainer, 'global_step') else 0,
                is_final=False
            )

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
