"""
FocalDiffusion Trainer Class
Main trainer implementation for FocalDiffusion
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import HfFolder
from huggingface_hub.errors import GatedRepoError
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import wandb

from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class FocalDiffusionTrainer:
    """Main trainer class for FocalDiffusion using file lists"""

    def __init__(self, config: dict):
        self.config = config
        self.setup_logging()
        self.setup_accelerator()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_tracking()

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Cache for prompt embeddings used during training
        self._empty_prompt_cache: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _resolve_hf_token(model_cfg: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Return the Hugging Face token and a string describing its origin."""

        if model_cfg.get('auth_token'):
            return model_cfg['auth_token'], "the experiment config"

        env_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        if env_token:
            return env_token, "environment variables"

        cached_token = HfFolder.get_token()
        if cached_token:
            return cached_token, "huggingface-cli cache"

        return None, "missing"

    def setup_data(self):
        """Setup datasets and dataloaders using file lists"""
        from ..data import create_dataloader

        logger.info("Setting up data loaders from file lists...")

        dataset_cfg = self.config['data']
        dataset_kwargs_cfg = dataset_cfg.get('dataset_kwargs') or {}
        base_dataset_kwargs = {
            key: value for key, value in dataset_kwargs_cfg.items()
            if key not in ('train', 'val')
        }
        if any(key in dataset_kwargs_cfg for key in ('train', 'val')):
            train_dataset_kwargs = {**base_dataset_kwargs, **dataset_kwargs_cfg.get('train', {})}
            val_dataset_kwargs = {**base_dataset_kwargs, **dataset_kwargs_cfg.get('val', {})}
        else:
            train_dataset_kwargs = dict(base_dataset_kwargs)
            val_dataset_kwargs = dict(base_dataset_kwargs)

        dataset_type = dataset_cfg.get('dataset_type')
        train_sources = dataset_cfg.get('train_sources')
        val_sources = dataset_cfg.get('val_sources')

        # Training dataloader
        train_filelist = dataset_cfg.get('train_filelist')
        if train_sources:
            logger.info("Loading training data from %d sources", len(train_sources))
        else:
            logger.info(f"Loading training data from: {train_filelist}")

        self.train_dataloader = create_dataloader(
            dataset_type=dataset_type,
            filelist_path=train_filelist,
            data_root=dataset_cfg.get('data_root', "./data"),
            batch_size=self.config['training']['batch_size'],
            num_workers=dataset_cfg['num_workers'],
            image_size=tuple(dataset_cfg['image_size']),
            focal_stack_size=dataset_cfg['focal_stack_size'],
            focal_range=tuple(dataset_cfg['focal_range']),
            augmentation=True,
            shuffle=True,
            max_samples=dataset_cfg.get('max_train_samples'),
            sources=train_sources,
            **train_dataset_kwargs,
        )

        # Validation dataloader
        val_filelist = dataset_cfg.get('val_filelist')
        if val_sources:
            logger.info("Loading validation data from %d sources", len(val_sources))
        else:
            logger.info(f"Loading validation data from: {val_filelist}")

        self.val_dataloader = create_dataloader(
            dataset_type=dataset_type,
            filelist_path=val_filelist,
            data_root=dataset_cfg.get('data_root', "./data"),
            batch_size=self.config['training']['batch_size'],
            num_workers=dataset_cfg['num_workers'],
            image_size=tuple(dataset_cfg['image_size']),
            focal_stack_size=dataset_cfg['focal_stack_size'],
            focal_range=tuple(dataset_cfg['focal_range']),
            augmentation=False,
            shuffle=False,
            max_samples=dataset_cfg.get('max_val_samples'),
            sources=val_sources,
            **val_dataset_kwargs,
        )

        logger.info(f"Train samples: {len(self.train_dataloader.dataset)}")
        logger.info(f"Val samples: {len(self.val_dataloader.dataset)}")

        # Log sample paths for verification
        if logger.isEnabledFor(logging.DEBUG):
            sample = self.train_dataloader.dataset[0]
            logger.debug(f"Sample data path: {sample.get('sample_path', 'N/A')}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['logging'].get('level', 'INFO').upper())

        # Create log directory
        log_dir = Path(self.config['output']['save_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"),
                logging.StreamHandler()
            ]
        )

    def setup_accelerator(self):
        """Setup distributed training with Accelerator"""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            mixed_precision=self.config['training'].get('mixed_precision', 'no'),
            log_with=['wandb'] if self.config['logging'].get('use_wandb', False) else None,
            project_dir=self.config['output']['save_dir'],
        )

        if self.config['training'].get('seed'):
            set_seed(self.config['training']['seed'])

    def setup_model(self):
        """Initialize model components"""
        logger.info("Loading SD3.5 base model...")

        # Import here to avoid circular imports
        from ..pipelines import FocalDiffusionPipeline
        from ..models import FocalStackProcessor, CameraInvariantEncoder
        from ..models.dual_decoder import DualOutputDecoder

        # Load base SD3.5 pipeline
        model_cfg = self.config['model']
        requested_model_id = model_cfg['base_model_id']
        base_model_id = requested_model_id
        variant = model_cfg.get('variant')

        if variant is None:
            for suffix in ("tensorrt", "onnx", "fp16", "fp32"):
                marker = f"-{suffix}"
                if base_model_id.endswith(marker):
                    base_model_id = base_model_id[: -len(marker)]
                    variant = suffix
                    break

        if (variant or "").lower() == "tensorrt":
            logger.warning(
                "TensorRT checkpoints such as '%s' only contain serialized inference engines. "
                "Falling back to the diffusers weights 'stabilityai/stable-diffusion-3.5-large' for training.",
                requested_model_id,
            )
            base_model_id = "stabilityai/stable-diffusion-3.5-large"
            variant = None

        auth_token, token_source = self._resolve_hf_token(model_cfg)

        if auth_token:
            logger.info("Using Hugging Face token from %s", token_source)
        else:
            logger.warning(
                "No Hugging Face token found. If '%s' is a gated model, please run "
                "`huggingface-cli login` or set `model.auth_token` in your config.",
                base_model_id,
            )

        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16 if self.config['training']['mixed_precision'] == 'fp16' else torch.float32,
                token=auth_token if auth_token else None,
                variant=variant,
            )
        except GatedRepoError as exc:
            hint_lines = [
                "Access to the requested Stable Diffusion checkpoint is gated.",
                "Visit https://huggingface.co/stabilityai/stable-diffusion-3.5-large to request access.",
            ]
            if auth_token:
                hint_lines.append(
                    "The provided Hugging Face token was rejected. Double-check that the token belongs "
                    "to an account with access to the model."
                )
            else:
                hint_lines.append(
                    "No token was supplied. Run `huggingface-cli login` or set `model.auth_token` / "
                    "the HF_TOKEN environment variable, then retry."
                )
            raise RuntimeError(" ".join(hint_lines)) from exc
        except FileNotFoundError as exc:
            cache_root_env = os.environ.get('HUGGINGFACE_HUB_CACHE')
            if cache_root_env:
                cache_root = Path(cache_root_env)
            else:
                hf_home = os.environ.get('HF_HOME')
                cache_root = (Path(hf_home) if hf_home else Path.home() / '.cache' / 'huggingface') / 'hub'

            repo_cache = cache_root / f"models--{base_model_id.replace('/', '--')}"
            hint_lines = [
                "Missing checkpoint shard(s) detected in the local Hugging Face cache.",
                f"Remove the directory '{repo_cache}' and retry the download to restore the model files.",
                "After cleanup, rerun the training script to trigger a fresh download."
            ]
            raise RuntimeError(" ".join(hint_lines)) from exc

        # Initialize focal-specific components
        logger.info("Initializing focal components...")

        self.focal_processor = FocalStackProcessor(
            feature_dim=self.config['model']['feature_dim'],
            num_scales=self.config['model']['num_scales'],
            max_sequence_length=self.config['model']['max_focal_stack_size'],
        )

        self.camera_encoder = CameraInvariantEncoder(
            output_dim=self.config['model']['feature_dim'],
        )

        self.dual_decoder = DualOutputDecoder(
            in_channels=pipe.vae.config.latent_channels,
            out_channels_depth=1,
            out_channels_rgb=pipe.vae.config.latent_channels,
        )

        # Create FocalDiffusion pipeline
        self.pipeline = FocalDiffusionPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            text_encoder_3=pipe.text_encoder_3,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            tokenizer_3=pipe.tokenizer_3,
            transformer=pipe.transformer,
            scheduler=pipe.scheduler,
            focal_processor=self.focal_processor,
            camera_encoder=self.camera_encoder,
            dual_decoder=self.dual_decoder,
        )

        # Ensure all components live on the accelerator device before training begins.
        # DiffusionPipeline.to(...) moves every registered nn.Module, while keeping the
        # object graph intact, so our existing references stay valid.
        target_device = self.accelerator.device if hasattr(self, "accelerator") else torch.device("cpu")
        self.pipeline.to(target_device)

        # Refresh local handles in case the pipeline replaced any modules internally.
        self.focal_processor = self.pipeline.focal_processor
        self.camera_encoder = self.pipeline.camera_encoder
        self.dual_decoder = self.pipeline.dual_decoder

        # Configure trainable parameters
        self._configure_trainable_params()

        # Setup EMA if enabled
        if self.config['training'].get('use_ema'):
            self.ema = EMAModel(
                self.focal_processor.parameters(),
                decay=self.config['training']['ema_decay']
            )
        else:
            self.ema = None

    def _configure_trainable_params(self):
        """Configure which parameters to train"""
        train_config = self.config['training']['trainable_modules']

        # Freeze VAE and text encoders
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        if hasattr(self.pipeline, 'text_encoder_3') and self.pipeline.text_encoder_3 is not None:
            self.pipeline.text_encoder_3.requires_grad_(False)

        # Configure transformer
        if train_config['transformer'] == 'frozen':
            self.pipeline.transformer.requires_grad_(False)
        elif train_config['transformer'] == 'lora':
            self._add_lora_to_transformer()
        elif train_config['transformer'] == 'attention_only':
            for name, param in self.pipeline.transformer.named_parameters():
                if 'attn' not in name:
                    param.requires_grad_(False)
        elif train_config['transformer'] == 'full':
            self.pipeline.transformer.requires_grad_(True)

        # Always train focal components
        self.focal_processor.requires_grad_(True)
        self.camera_encoder.requires_grad_(True)
        self.dual_decoder.requires_grad_(True)

        # Log trainable parameters
        pipeline_params = list(self._iter_pipeline_parameters())
        total_params = sum(param.numel() for param in pipeline_params)
        trainable_params = sum(param.numel() for param in pipeline_params if param.requires_grad)
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def _add_lora_to_transformer(self):
        """Add LoRA layers to transformer"""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config['training']['lora_rank'],
                lora_alpha=self.config['training']['lora_alpha'],
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.1,
            )

            self.pipeline.transformer = get_peft_model(self.pipeline.transformer, lora_config)
            logger.info("Added LoRA to transformer")
        except ImportError:
            logger.warning("PEFT not installed, using full fine-tuning instead")
            self.pipeline.transformer.requires_grad_(True)

    def setup_optimization(self):
        """Setup optimizer and scheduler"""
        from ..training.optimizers import get_optimizer

        # Get trainable parameters
        trainable_params = list(self._iter_pipeline_parameters(only_trainable=True))

        # Create optimizer
        self.optimizer = get_optimizer(
            trainable_params,
            optimizer_type=self.config['optimizer'].get('type', 'adamw'),
            learning_rate=self.config['optimizer']['learning_rate'],
            weight_decay=self.config['optimizer']['weight_decay'],
            betas=tuple(self.config['optimizer']['betas']),
        )

        # Create scheduler
        num_training_steps = len(self.train_dataloader) * self.config['training']['num_epochs']
        num_training_steps = num_training_steps // self.config['training']['gradient_accumulation_steps']

        self.lr_scheduler = get_scheduler(
            self.config['scheduler']['type'],
            optimizer=self.optimizer,
            num_warmup_steps=self.config['scheduler']['warmup_steps'],
            num_training_steps=num_training_steps,
        )

        # Prepare with accelerator
        (
            self.pipeline,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.pipeline,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler
        )

        try:
            from ..pipelines import FocalInjectedSD3Transformer

            unwrapped = self.accelerator.unwrap_model(self.pipeline)
            transformer = getattr(unwrapped, "transformer", None)
            if isinstance(transformer, FocalInjectedSD3Transformer):
                _ = transformer.base_transformer
        except (ImportError, AttributeError):
            pass

        # Setup gradient scaler for mixed precision
        self.scaler = GradScaler() if self.config['training']['mixed_precision'] == 'fp16' else None

    def _iter_pipeline_parameters(self, only_trainable: bool = False):
        """Yield pipeline parameters, tolerating older snapshots without helpers."""

        if hasattr(self.pipeline, "parameters"):
            for param in self.pipeline.parameters():
                if not only_trainable or param.requires_grad:
                    yield param
            return

        if hasattr(self.pipeline, "_iter_registered_modules"):
            for _, module in self.pipeline._iter_registered_modules():
                for param in module.parameters():
                    if not only_trainable or param.requires_grad:
                        yield param
            return

        raise AttributeError(
            "FocalDiffusionPipeline is missing parameter accessors. "
            "Please update src/pipelines/focal_diffusion_pipeline.py."
        )

    def setup_tracking(self):
        """Setup experiment tracking"""
        if self.accelerator.is_main_process:
            if self.config['logging'].get('use_wandb'):
                run_name = self.config['logging'].get('run_name')
                if not run_name:
                    run_name = f"focal_diffusion_{datetime.now():%Y%m%d_%H%M%S}"

                self.accelerator.init_trackers(
                    project_name=self.config['logging'].get('project_name', 'focal-diffusion'),
                    config=self.config,
                    init_kwargs={"wandb": {
                        "name": run_name,
                        "tags": self.config['logging'].get('tags', []),
                    }}
                )

            # Setup checkpoint directory
            self.checkpoint_dir = Path(self.config['output']['save_dir']) / 'checkpoints'
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save config and file lists
            with open(self.checkpoint_dir / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2)

            # Copy file lists for reproducibility
            import shutil
            for key in ['train_filelist', 'val_filelist', 'test_filelist']:
                if key in self.config['data']:
                    src = Path(self.config['data'][key])
                    if src.exists():
                        dst = self.checkpoint_dir / src.name
                        shutil.copy2(src, dst)
                        logger.info(f"Copied {src} to {dst}")

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        best_val_loss = float('inf')

        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch

            # Training epoch
            train_loss = self.train_epoch(epoch, self.global_step)
            self.global_step += len(self.train_dataloader)

            # Validation
            if epoch % self.config['training']['val_every_n_epochs'] == 0:
                val_metrics = self.validate(epoch)

                # Log metrics
                if self.accelerator.is_main_process:
                    self.accelerator.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    }, step=self.global_step)

                # Save best checkpoint
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, self.global_step, is_best=True)

            # Regular checkpoint
            if epoch % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch, self.global_step)

        # Final checkpoint
        self.save_checkpoint(epoch, self.global_step, is_final=True)

        logger.info("Training completed!")
    def train_epoch(self, epoch: int, global_step: int) -> float:
        """Train for one epoch"""
        from ..training.losses import FocalDiffusionLoss

        self.pipeline.train()
        epoch_loss = 0.0

        # Initialize loss function
        loss_fn = FocalDiffusionLoss(
            diffusion_weight=self.config['losses']['diffusion_weight'],
            depth_weight=self.config['losses']['depth_weight'],
            rgb_weight=self.config['losses']['rgb_weight'],
            consistency_weight=self.config['losses']['consistency_weight'],
            perceptual_weight=self.config['losses'].get('perceptual_weight', 0.05),
        ).to(self.accelerator.device)

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process
        )

        prompt_embeds, pooled_prompt_embeds = self._get_empty_prompt_embeddings()

        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.pipeline):
                # Prepare batch data on device
                device = self.accelerator.device
                focal_stack = batch['focal_stack'].to(device)
                focus_distances = batch['focus_distances'].to(device=device, dtype=focal_stack.dtype)
                depth_gt = batch['depth'].to(device)
                rgb_gt = batch['all_in_focus'].to(device)
                depth_range_tensor = batch.get('depth_range')
                if depth_range_tensor is not None:
                    depth_range_tensor = depth_range_tensor.to(device=device, dtype=depth_gt.dtype)
                depth_mask = batch.get('valid_mask')
                if depth_mask is not None:
                    depth_mask = depth_mask.to(device=device)

                camera_params = batch.get('camera_params')

                if camera_params is not None:
                    batch_size = focal_stack.shape[0]
                    converted_params = {}
                    for key, value in camera_params.items():
                        if isinstance(value, torch.Tensor):
                            converted_params[key] = value.to(device=device, dtype=focal_stack.dtype)
                        else:
                            converted_params[key] = torch.full(
                                (batch_size,),
                                float(value),
                                device=device,
                                dtype=focal_stack.dtype,
                            )
                    camera_params = converted_params

                # Normalize inputs to match pipeline preprocessing
                focal_stack = (focal_stack * 2.0) - 1.0
                rgb_target = (rgb_gt * 2.0) - 1.0

                # Extract focal features
                focal_features = self.focal_processor(
                    focal_stack,
                    focus_distances,
                    camera_params
                )

                focal_features = {
                    key: value.to(self.pipeline.transformer.dtype)
                    if isinstance(value, torch.Tensor) and value is not None else value
                    for key, value in focal_features.items()
                }

                # Encode RGB target into VAE latent space
                vae_dtype = next(self.pipeline.vae.parameters()).dtype
                rgb_latent_input = rgb_target.to(dtype=vae_dtype)
                latents_dist = self.pipeline.vae.encode(rgb_latent_input).latent_dist
                latents = latents_dist.sample() * self.pipeline.vae.config.scaling_factor

                # Sample noise for diffusion training
                noise = torch.randn_like(latents)

                scheduler = self.pipeline.scheduler

                noise_levels = None

                if hasattr(scheduler, "add_noise"):
                    timesteps = torch.randint(
                        0,
                        scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                elif hasattr(scheduler, "scale_noise"):
                    timesteps_attr = getattr(scheduler, "timesteps", None)
                    needs_timesteps = timesteps_attr is None
                    if not needs_timesteps:
                        try:
                            needs_timesteps = len(timesteps_attr) == 0  # type: ignore[arg-type]
                        except TypeError:
                            needs_timesteps = False

                    if needs_timesteps:
                        if not hasattr(scheduler, "set_timesteps"):
                            raise AttributeError(
                                "The active scheduler exposes `scale_noise` but does not provide"
                                " `set_timesteps`, preventing training-time noise scaling."
                            )

                        num_train_timesteps = getattr(scheduler.config, "num_train_timesteps", None)
                        if num_train_timesteps is None:
                            raise AttributeError(
                                "Scheduler is missing `config.num_train_timesteps`, which is required"
                                " to initialize the training timestep schedule."
                            )

                        scheduler.set_timesteps(num_train_timesteps, device=device)
                        timesteps_attr = getattr(scheduler, "timesteps", None)

                    if timesteps_attr is None:
                        raise RuntimeError(
                            "Failed to initialize scheduler timesteps for scale_noise training."
                        )

                    if not torch.is_tensor(timesteps_attr):
                        schedule_timesteps = torch.as_tensor(timesteps_attr, device=device)
                    else:
                        schedule_timesteps = timesteps_attr.to(device=device)

                    if schedule_timesteps.ndim != 1 or schedule_timesteps.numel() == 0:
                        raise RuntimeError(
                            "Scheduler timesteps must be a 1D tensor with at least one entry to"
                            " perform scale_noise training."
                        )

                    step_indices = torch.randint(
                        0,
                        schedule_timesteps.shape[0],
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                    timesteps = schedule_timesteps.index_select(0, step_indices)
                    noisy_latents = scheduler.scale_noise(latents, timesteps, noise)
                    if hasattr(scheduler, "sigmas"):
                        sigma_schedule = scheduler.sigmas.to(device=device, dtype=latents.dtype)
                        sigma_indices = step_indices.clamp(max=sigma_schedule.shape[0] - 1)
                        noise_levels = sigma_schedule.index_select(0, sigma_indices)
                else:
                    raise AttributeError(
                        "The active scheduler does not implement `add_noise` or `scale_noise`,"
                        " which are required for forward diffusion during training."
                    )
                if hasattr(self.pipeline.scheduler, "scale_model_input"):
                    model_input = self.pipeline.scheduler.scale_model_input(noisy_latents, timesteps)
                else:
                    model_input = noisy_latents

                model_input = model_input.to(self.pipeline.transformer.dtype)
                noise = noise.to(self.pipeline.transformer.dtype)

                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    # Predict noise
                    noise_pred = self.pipeline.transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        focal_features=focal_features,
                        return_dict=False,
                    )[0]

                    clean_latent_pred = self._predict_clean_latents(
                        noisy_latents=noisy_latents,
                        noise_pred=noise_pred,
                        timesteps=timesteps,
                        noise_levels=noise_levels,
                        reference_latents=latents,
                    )

                    # Decode auxiliary predictions for multi-task losses
                    depth_logits, rgb_latent_pred = self.dual_decoder(clean_latent_pred)

                    depth_probs = torch.sigmoid(depth_logits)
                    depth_probs = F.interpolate(
                        depth_probs,
                        size=depth_gt.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                    if depth_range_tensor is not None:
                        depth_min = depth_range_tensor[:, 0].view(-1, 1, 1)
                        depth_max = depth_range_tensor[:, 1].view(-1, 1, 1)
                    elif camera_params is not None and 'depth_min' in camera_params and 'depth_max' in camera_params:
                        depth_min = camera_params['depth_min'].to(dtype=depth_gt.dtype).view(-1, 1, 1)
                        depth_max = camera_params['depth_max'].to(dtype=depth_gt.dtype).view(-1, 1, 1)
                    else:
                        depth_min = depth_gt.amin(dim=(-2, -1), keepdim=True)
                        depth_max = depth_gt.amax(dim=(-2, -1), keepdim=True)

                    depth_range = (depth_max - depth_min).clamp(min=1e-6)
                    depth_pred = depth_probs * depth_range + depth_min

                    rgb_recon = self.pipeline.vae.decode(
                        rgb_latent_pred / self.pipeline.vae.config.scaling_factor,
                        return_dict=False
                    )[0]
                    rgb_recon = rgb_recon.clamp(-1, 1)

                    # Cast tensors for stable loss computation
                    noise_pred = noise_pred.float()
                    noise_target = noise.float()
                    depth_pred = depth_pred.float()
                    depth_target = depth_gt.float()
                    rgb_recon = rgb_recon.float()
                    rgb_target_fp32 = rgb_target.float()

                    # Compute losses
                    loss_dict = loss_fn(
                        noise_pred=noise_pred,
                        noise_target=noise_target,
                        depth_pred=depth_pred,
                        depth_target=depth_target,
                        depth_mask=depth_mask,
                        rgb_pred=rgb_recon,
                        rgb_target=rgb_target_fp32,
                        focal_features=focal_features,
                    )
                    loss = loss_dict['total']

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.config['training'].get('max_grad_norm'):
                    clip_params = list(self._iter_pipeline_parameters(only_trainable=True))
                    if clip_params:
                        self.accelerator.clip_grad_norm_(
                            clip_params,
                            self.config['training']['max_grad_norm']
                        )

                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Update EMA
                if self.ema is not None:
                    self.ema.step(self.focal_processor.parameters())

                # Logging
                epoch_loss += loss.item()
                if step % self.config['logging']['log_every_n_steps'] == 0:
                    progress_bar.set_postfix({
                        'loss': loss.item(),
                        'lr': self.lr_scheduler.get_last_lr()[0]
                    })

                    if self.accelerator.is_main_process:
                        self.accelerator.log({
                            'train_loss_step': loss.item(),
                            'learning_rate': self.lr_scheduler.get_last_lr()[0],
                            **{f'train_{k}': v.item() for k, v in loss_dict.items() if k != 'total'}
                        }, step=global_step + step)

        return epoch_loss / len(self.train_dataloader)

    def _get_empty_prompt_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache and return the embeddings for the empty prompt used during training."""

        cache = self._empty_prompt_cache
        dtype = self.pipeline.transformer.dtype
        device = self.accelerator.device

        if cache and cache.get("dtype") == dtype and cache.get("device") == device:
            return cache["prompt_embeds"], cache["pooled_prompt_embeds"]

        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt="",
                prompt_2="",
                prompt_3="",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                device=device,
            )

        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype).detach()
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype).detach()

        self._empty_prompt_cache = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "dtype": dtype,
            "device": device,
        }

        return prompt_embeds, pooled_prompt_embeds

    def _predict_clean_latents(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        noise_levels: Optional[torch.Tensor] = None,
        reference_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct the clean latent sample from the model's noise prediction."""

        scheduler = self.pipeline.scheduler
        latent_dtype = noisy_latents.dtype
        device = noisy_latents.device

        noise_pred = noise_pred.to(dtype=latent_dtype)

        if noise_levels is not None:
            sigma = noise_levels.to(device=device, dtype=latent_dtype)
            sigma = sigma.view(-1, *[1] * (noisy_latents.ndim - 1))
            denom = 1.0 - sigma
            near_zero = denom.abs() < 1e-6
            denom = denom.clamp(min=1e-6)
            clean = (noisy_latents - sigma * noise_pred) / denom

            if reference_latents is not None and torch.any(near_zero):
                fallback = reference_latents.to(device=device, dtype=latent_dtype).detach()
                clean = torch.where(near_zero, fallback, clean)

            return clean

        if hasattr(scheduler, "alphas_cumprod"):
            alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=latent_dtype)
            timestep_indices = timesteps.to(device=device)
            if timestep_indices.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint8):
                timestep_indices = timestep_indices.long()
            timestep_indices = timestep_indices.clamp(min=0, max=alphas_cumprod.shape[0] - 1)
            alpha_prod_t = alphas_cumprod.index_select(0, timestep_indices)
            alpha = alpha_prod_t.sqrt().view(-1, *[1] * (noisy_latents.ndim - 1))
            sigma = (1 - alpha_prod_t).sqrt().view(-1, *[1] * (noisy_latents.ndim - 1))
            clean = (noisy_latents - sigma * noise_pred) / alpha.clamp(min=1e-6)
            return clean

        raise RuntimeError(
            "Unable to recover clean latents for the active scheduler."
        )

    def validate(self, epoch: int) -> dict:
        """Validation loop"""
        from ..utils.metrics import compute_metrics

        self.pipeline.eval()
        val_metrics = {'loss': 0.0, 'abs_rel': 0.0, 'rmse': 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                    self.val_dataloader,
                    desc="Validation",
                    disable=not self.accelerator.is_local_main_process
            ):
                # Generate predictions
                output = self.pipeline(
                    focal_stack=batch['focal_stack'],
                    focus_distances=batch['focus_distances'],
                    camera_params=batch.get('camera_params'),
                    num_inference_steps=self.config['validation']['num_inference_steps'],
                    guidance_scale=self.config['validation']['guidance_scale'],
                    output_type='pt',
                    return_dict=True,
                )

                # Compute metrics
                depth_gt = batch['depth'].to(output.depth_map.device).squeeze(1)
                mask = batch.get('valid_mask')
                if mask is not None:
                    mask = mask.to(output.depth_map.device)

                metrics = compute_metrics(
                    output.depth_map,
                    depth_gt,
                    mask=mask,
                )

                # Accumulate
                for k, v in metrics.items():
                    if k in val_metrics:
                        val_metrics[k] += v

                if mask is not None:
                    val_loss = F.l1_loss(output.depth_map[mask], depth_gt[mask])
                else:
                    val_loss = F.l1_loss(output.depth_map, depth_gt)
                val_metrics['loss'] += val_loss.item()

                num_batches += 1

        # Average metrics
        for k in val_metrics:
            val_metrics[k] /= num_batches

        return val_metrics

    def save_checkpoint(
            self,
            epoch: int,
            global_step: int,
            is_best: bool = False,
            is_final: bool = False
    ):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return

        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'focal_processor_state_dict': self.accelerator.unwrap_model(self.focal_processor).state_dict(),
            'camera_encoder_state_dict': self.accelerator.unwrap_model(self.camera_encoder).state_dict(),
            'dual_decoder_state_dict': self.accelerator.unwrap_model(self.dual_decoder).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config,
        }

        # Add transformer state if training it
        if self.config['training']['trainable_modules']['transformer'] != 'frozen':
            checkpoint['transformer_state_dict'] = self.accelerator.unwrap_model(
                self.pipeline.transformer
            ).state_dict()

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        # Determine filename
        if is_best:
            filename = 'best.pt'
        elif is_final:
            filename = 'final.pt'
        else:
            filename = f'checkpoint_epoch_{epoch}.pt'

        # Save
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint: {filename}")

        # Keep only last N checkpoints
        if self.config['output'].get('save_top_k'):
            self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints"""
        save_top_k = self.config['output'].get('save_top_k', 3)

        # List all checkpoint files
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))

        # Remove old checkpoints
        if len(checkpoints) > save_top_k:
            for checkpoint in checkpoints[:-save_top_k]:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint.name}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)

        # Load model states
        self.focal_processor.load_state_dict(checkpoint['focal_processor_state_dict'])
        self.camera_encoder.load_state_dict(checkpoint['camera_encoder_state_dict'])
        self.dual_decoder.load_state_dict(checkpoint['dual_decoder_state_dict'])

        if 'transformer_state_dict' in checkpoint:
            self.pipeline.transformer.load_state_dict(checkpoint['transformer_state_dict'])

        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load EMA if available
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

        logger.info(f"Resumed from epoch {epoch}, global step {global_step}")

        return epoch, global_step
