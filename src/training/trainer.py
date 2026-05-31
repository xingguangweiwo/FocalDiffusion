"""
FocalDiffusion Trainer Class
Main trainer implementation for FocalDiffusion
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import json
from contextlib import contextmanager

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
from huggingface_hub import get_token as hf_get_token
from huggingface_hub.errors import GatedRepoError
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import wandb

from typing import Any, Dict, List, Optional, Tuple

from .sd3_objective import predict_clean_latents_from_flow, sample_sd3_flow_matching_batch
from ..training.losses import FocusConsistencyCritic


logger = logging.getLogger(__name__)


@contextmanager
def temporarily_freeze(module):
    prev = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, req in zip(module.parameters(), prev):
            p.requires_grad_(req)


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

        cached_token = hf_get_token()
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
            resize_mode=dataset_cfg.get('resize_mode', 'letterbox'),
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
            resize_mode=dataset_cfg.get('resize_mode', 'letterbox'),
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
        from ..models import FocalStackProcessor
        from ..models.focal_evidence import FocalEvidenceHead
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
            num_scales=self.config['model'].get('num_scales', 1),
            max_sequence_length=self.config['model']['max_focal_stack_size'],
            focal_encoder_type=self.config['model'].get('focal_encoder_type','focal_sweep'),
            patch_size=self.config['model'].get('patch_size',8),
            focal_attention_heads=self.config['model'].get('focal_attention_heads',8),
            focal_attention_depth=self.config['model'].get('focal_attention_depth',2),
        )

        self.focal_evidence_head = FocalEvidenceHead(
            hidden=self.config["model"].get("focal_evidence_hidden", 48),
            temperature=self.config["model"].get("focal_evidence_temperature", 0.07),
        )

        self.dual_decoder = DualOutputDecoder(
            in_channels=pipe.vae.config.latent_channels,
            out_channels_depth=1,
            out_channels_rgb=pipe.vae.config.latent_channels,
        )

        self.focus_consistency_critic = FocusConsistencyCritic()

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
            focal_evidence_head=self.focal_evidence_head,
            dual_decoder=self.dual_decoder,
        )

        # Ensure all components live on the accelerator device before training begins.
        # DiffusionPipeline.to(...) moves every registered nn.Module, while keeping the
        # object graph intact, so our existing references stay valid.
        target_device = self.accelerator.device if hasattr(self, "accelerator") else torch.device("cpu")
        self.pipeline.to(target_device)

        # Refresh local handles in case the pipeline replaced any modules internally.
        self.focal_processor = self.pipeline.focal_processor
        self.focal_evidence_head = self.pipeline.focal_evidence_head
        self.dual_decoder = self.pipeline.dual_decoder
        self.focus_consistency_critic = self.focus_consistency_critic.to(target_device)

        # Configure trainable parameters
        self._configure_trainable_params()

        # Setup EMA if enabled
        if self.config['training'].get('use_ema'):
            ema_params = self._ema_parameters()
            self.ema = EMAModel(
                ema_params,
                decay=self.config['training']['ema_decay']
            ) if ema_params else None
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
            trainable_transformer_tokens = (
                'attn',
                'condition_adapter',
                'condition_scale',
                'pre_scale',
                'post_scale',
            )
            for name, param in self.pipeline.transformer.named_parameters():
                if not any(token in name for token in trainable_transformer_tokens):
                    param.requires_grad_(False)
        elif train_config['transformer'] == 'full':
            self.pipeline.transformer.requires_grad_(True)

        # Always train focal components
        self.focal_processor.requires_grad_(True)
        self.focal_evidence_head.requires_grad_(True)
        self.dual_decoder.requires_grad_(True)
        self.focus_consistency_critic.requires_grad_(self.config['training'].get('train_focus_critic', False))

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

        if getattr(self, 'focus_consistency_critic', None) is not None:
            trainable_params += [p for p in self.focus_consistency_critic.parameters() if p.requires_grad]

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
            self.focus_consistency_critic,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.pipeline,
            self.focus_consistency_critic,
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


    def _ema_parameters(self):
        """Materialize trainable parameters tracked by EMA."""

        return [param for param in self._iter_pipeline_parameters(only_trainable=True)]

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
        self.focus_consistency_critic.train()
        epoch_loss = 0.0

        supervision_mode = self.config['training'].get('supervision_mode', 'supervised')
        if supervision_mode not in {"supervised", "semi_supervised"}:
            raise ValueError(
                "training.supervision_mode must be 'supervised' or 'semi_supervised'; "
                f"got {supervision_mode!r}."
            )

        # Initialize loss function
        loss_fn = FocalDiffusionLoss(
            diffusion_weight=self.config['losses']['diffusion_weight'],
            depth_weight=self.config['losses']['depth_weight'],
            rgb_weight=self.config['losses']['rgb_weight'],
            focus_energy_weight=self.config['losses'].get('focus_energy_weight', 0.5),
            tau_contrast_weight=self.config['losses'].get('tau_contrast_weight', 0.2),
            stack_contrast_weight=self.config['losses'].get('stack_contrast_weight', 0.2),
            mismatch_contrast_weight=self.config['losses'].get('mismatch_contrast_weight', 0.2),
            shape_candidate_contrast_weight=self.config['losses'].get('shape_candidate_contrast_weight', 0.2),
            uncertainty_weight=self.config['losses'].get('uncertainty_weight', 0.05),
            aif_highpass_weight=self.config['losses'].get('aif_highpass_weight', 0.1),
            supervision_mode=supervision_mode,
        ).to(self.accelerator.device)

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process
        )

        prompt_embeds, pooled_prompt_embeds = self._get_empty_prompt_embeddings()
        effective_steps = 0

        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.pipeline):
                # Prepare batch data on device
                device = self.accelerator.device
                focal_stack = batch['focal_stack'].to(device)
                focus_distances = batch['focus_distances'].to(device=device, dtype=focal_stack.dtype)
                depth_gt = batch.get('depth')
                rgb_gt = batch.get('all_in_focus')
                if depth_gt is not None:
                    depth_gt = depth_gt.to(device)
                if rgb_gt is not None:
                    rgb_gt = rgb_gt.to(device)
                if supervision_mode == "supervised":
                    if depth_gt is None:
                        raise ValueError("supervised mode requires batch['depth'], but it is missing.")
                    if rgb_gt is None:
                        raise ValueError("supervised mode requires batch['all_in_focus'], but it is missing.")
                if supervision_mode == "semi_supervised" and rgb_gt is None:
                    logger.warning(
                        "Skipping batch in semi_supervised mode because all_in_focus is missing; "
                        "SD3 flow-matching clean latent target is unavailable."
                    )
                    continue
                depth_range_tensor = batch.get('depth_range')
                if depth_range_tensor is not None and depth_gt is not None:
                    depth_range_tensor = depth_range_tensor.to(device=device, dtype=depth_gt.dtype)
                depth_mask = batch.get('valid_mask')
                if depth_mask is not None:
                    depth_mask = depth_mask.to(device=device)

                # Normalize inputs
                if focal_stack.min() >= 0 and focal_stack.max() <= 1:
                    focal_stack = (focal_stack * 2.0) - 1.0
                rgb_target = (rgb_gt * 2.0) - 1.0 if rgb_gt is not None else None

                # Extract focal features for SD3 conditioning.
                focal_features = self.focal_processor(focal_stack, focus_distances)

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

                # Sample the SD3 flow-matching training objective.  The transformer
                # predicts the flow direction (noise - clean_latents), not DDPM epsilon.
                scheduler = self.pipeline.scheduler
                flow_batch = sample_sd3_flow_matching_batch(scheduler, latents)
                timesteps = flow_batch.timesteps
                noisy_latents = flow_batch.noisy_latents
                flow_target = flow_batch.target

                if hasattr(self.pipeline.scheduler, "scale_model_input"):
                    model_input = self.pipeline.scheduler.scale_model_input(noisy_latents, timesteps)
                else:
                    model_input = noisy_latents

                model_input = model_input.to(self.pipeline.transformer.dtype)
                flow_target = flow_target.to(self.pipeline.transformer.dtype)
                batch_prompt_embeds, batch_pooled_prompt_embeds = self._repeat_prompt_embeddings(
                    prompt_embeds,
                    pooled_prompt_embeds,
                    batch_size=model_input.shape[0],
                )

                # Forward pass
                with autocast(enabled=self.scaler is not None):
                    # Predict the SD3 flow direction.
                    diffusion_pred = self.pipeline.transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        encoder_hidden_states=batch_prompt_embeds,
                        pooled_projections=batch_pooled_prompt_embeds,
                        focal_features=focal_features,
                        return_dict=False,
                    )[0]

                    clean_latent_pred = predict_clean_latents_from_flow(
                        noisy_latents=noisy_latents,
                        model_pred=diffusion_pred,
                        sigmas=flow_batch.sigmas,
                    )

                    # Decode auxiliary predictions for multi-task losses
                    if rgb_target is None:
                        raise ValueError("all_in_focus is required to compute the SD3 flow-matching reconstruction losses.")

                    decoder_outputs = self.dual_decoder(clean_latent_pred)
                    shape_norm = decoder_outputs["shape_norm"]
                    uncertainty = decoder_outputs["uncertainty"]
                    focal_evidence = self.focal_evidence_head(focal_stack.float(), focus_distances.float())
                    depth_focus = F.interpolate(focal_evidence["depth_focus"], size=shape_norm.shape[-2:], mode="bilinear", align_corners=False)
                    focus_reliability = F.interpolate(focal_evidence["focus_reliability"], size=shape_norm.shape[-2:], mode="bilinear", align_corners=False)
                    focus_entropy = F.interpolate(focal_evidence["focus_entropy"], size=shape_norm.shape[-2:], mode="bilinear", align_corners=False)
                    depth_prior = shape_norm
                    depth_final = focus_reliability * depth_focus + (1.0 - focus_reliability) * depth_prior
                    uncertainty_decoder = uncertainty
                    uncertainty_disagree = torch.abs(depth_focus - depth_prior)
                    uncertainty_focus = focus_entropy
                    uncertainty_final = (0.5 * uncertainty_decoder + 0.25 * uncertainty_focus + 0.25 * uncertainty_disagree).clamp(0, 1)
                    rgb_latent_pred = decoder_outputs["aif_latents"]
                    rgb_recon = self.pipeline.vae.decode(
                        rgb_latent_pred / self.pipeline.vae.config.scaling_factor,
                        return_dict=False
                    )[0]
                    rgb_recon = rgb_recon.clamp(-1, 1)

                    # Cast tensors for stable loss computation
                    diffusion_pred = diffusion_pred.float()
                    diffusion_target = flow_target.float()
                    depth_target = depth_gt.float() if depth_gt is not None else None
                    rgb_recon = rgb_recon.float()
                    rgb_target_fp32 = rgb_target.float()

                    # Compute losses; the legacy focus critic is optional ablation only.
                    if self.config['training'].get('train_focus_critic', False):
                        critic_outputs = self.focus_consistency_critic(
                            focal_stack.float(), focus_distances.float(), depth_final.float().detach()
                        )
                        critic_warmup_steps = int(self.config['training'].get('critic_warmup_steps', 0))
                        if (global_step + step) < critic_warmup_steps:
                            critic_generator_outputs = None
                        else:
                            with temporarily_freeze(self.focus_consistency_critic):
                                critic_generator_outputs = self.focus_consistency_critic(
                                    focal_stack.float(), focus_distances.float(), depth_final.float()
                                )
                    else:
                        critic_outputs = None
                        critic_generator_outputs = None
                    loss_dict = loss_fn(
                        diffusion_pred=diffusion_pred,
                        diffusion_target=diffusion_target,
                        depth_target=depth_target,
                        depth_mask=depth_mask,
                        rgb_pred=rgb_recon,
                        rgb_target=rgb_target_fp32,
                        shape_norm=depth_final.float(),
                        depth_prior=depth_prior.float(),
                        depth_focus=depth_focus.float(),
                        depth_final=depth_final.float(),
                        uncertainty=uncertainty_final.float(),
                        focus_prob=focal_evidence["focus_prob"].float(),
                        focus_entropy=focus_entropy.float(),
                        focus_reliability=focus_reliability.float(),
                        focus_distances=focus_distances.float(),
                        focal_stack=focal_stack.float(),
                        critic_outputs=critic_outputs,
                        critic_generator_outputs=critic_generator_outputs,
                        depth_range=depth_range_tensor.float() if depth_range_tensor is not None else None,
                    )
                    loss = loss_dict['total']

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping
                if self.config['training'].get('max_grad_norm'):
                    clip_params = list(self._iter_pipeline_parameters(only_trainable=True))
                    if getattr(self, "focus_consistency_critic", None) is not None:
                        clip_params += [
                            p for p in self.focus_consistency_critic.parameters()
                            if p.requires_grad
                        ]
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
                    self.ema.step(self._ema_parameters())

                # Logging
                effective_steps += 1
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
                            **{f'train_{k}': v.item() for k, v in loss_dict.items() if k != 'total' and isinstance(v, torch.Tensor)},
                            'train_focus_entropy_mean': focus_entropy.mean().item(),
                            'train_focus_reliability_mean': focus_reliability.mean().item(),
                            'train_depth_prior_focus_disagreement': torch.abs(depth_focus - depth_prior).mean().item(),
                        }, step=global_step + step)

        if effective_steps == 0:
            return 0.0
        return epoch_loss / effective_steps

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

    @staticmethod
    def _repeat_prompt_embeddings(
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Repeat cached prompt embeddings so their batch matches the current data batch."""

        if prompt_embeds.shape[0] == batch_size and pooled_prompt_embeds.shape[0] == batch_size:
            return prompt_embeds, pooled_prompt_embeds

        if prompt_embeds.shape[0] != 1 or pooled_prompt_embeds.shape[0] != 1:
            raise ValueError(
                "Cached empty prompt embeddings must have batch size 1 or match the current "
                f"training batch size ({batch_size}); got prompt batch {prompt_embeds.shape[0]} "
                f"and pooled prompt batch {pooled_prompt_embeds.shape[0]}."
            )

        repeat_shape = (batch_size,) + (1,) * (prompt_embeds.ndim - 1)
        pooled_repeat_shape = (batch_size,) + (1,) * (pooled_prompt_embeds.ndim - 1)
        return (
            prompt_embeds.repeat(repeat_shape),
            pooled_prompt_embeds.repeat(pooled_repeat_shape),
        )

    def validate(self, epoch: int) -> dict:
        """Validation loop."""
        from .validation import run_validation

        return run_validation(self, epoch)

    def save_checkpoint(
            self,
            epoch: int,
            global_step: int,
            is_best: bool = False,
            is_final: bool = False
    ):
        """Save model checkpoint."""
        from .checkpointing import save_checkpoint

        save_checkpoint(self, epoch, global_step, is_best=is_best, is_final=is_final)

    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints."""
        from .checkpointing import cleanup_checkpoints

        cleanup_checkpoints(self)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training."""
        from .checkpointing import load_checkpoint

        return load_checkpoint(self, checkpoint_path)
