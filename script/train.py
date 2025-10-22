"""
Training script for FocalDiffusion
Uses the FocalDiffusionTrainer class from src.training.trainer
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional
import logging
from datetime import datetime

# Add project root to path before importing project modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .utils import dump_yaml_file, load_yaml_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    return parser.parse_args()


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

    config = load_yaml_file(config_path) or {}

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

    def _resolve(path_value: Optional[str]) -> Optional[str]:
        if path_value is None:
            return None
        path = Path(path_value)
        if path.is_absolute() or str(path_value).startswith("${"):
            return str(path)
        anchor = base_dir if str(path_value).startswith(('./', '../')) else project_root
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
                # Try to evaluate as Python literal
                target[keys[-1]] = eval(value)
            except:
                # Treat as string
                target[keys[-1]] = value

            logger.info(f"Override: {key_path} = {target[keys[-1]]}")

        except Exception as e:
            logger.warning(f"Failed to apply override '{override}': {e}")

    return config


def validate_config(config: dict) -> None:
    """Validate configuration"""
    required_keys = [
        'model.base_model_id',
        'data.dataset_type',
        'data.data_root',
        'data.train_filelist',
        'data.val_filelist',
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

    # Check paths exist
    if not Path(config['data']['data_root']).exists():
        logger.warning(
            f"Training data root does not exist: {config['data']['data_root']}"
        )

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
        logger.info("Dry run mode - exiting without training")
        return

    # Create trainer
    logger.info("Initializing FocalDiffusion trainer...")
    from src.training.trainer import FocalDiffusionTrainer  # Lazy import to avoid heavy deps during dry-runs

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
