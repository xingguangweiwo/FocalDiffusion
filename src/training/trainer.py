"""
FocalStackGeneration Trainer Class
Main trainer implementation for FocalStackGeneration
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import get_token as hf_get_token
from huggingface_hub.errors import GatedRepoError
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from typing import Any, Dict, Optional, Tuple

from .sd3_objective import predict_clean_latents_from_flow, sample_sd3_flow_matching_batch
from ..models.focal_evidence_encoder import build_physical_evidence_features


logger = logging.getLogger(__name__)


class FocalStackGenerationTrainer:
    """Main trainer class for FocalStackGeneration using file lists"""

    def __init__(self, config: dict):
        self.config = config
        self.setup_logging()
        self.setup_accelerator()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_tracking()

        self.current_epoch = 0
        self.global_step = 0
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

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config['logging'].get('level', 'INFO').upper())
        log_dir = Path(self.config['output']['save_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"),
                logging.StreamHandler(),
            ],
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
        train_filelist = dataset_cfg.get('train_filelist')
        val_filelist = dataset_cfg.get('val_filelist')

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

    def setup_model(self):
        """Initialize model components"""
        logger.info("Loading SD3.5 base model...")
        from ..pipelines import FocalStackGenerationPipeline
        from ..models import FocalStackProcessor
        from ..models.focal_evidence_encoder import FocalEvidenceEncoder, PhysicalEvidenceEstimator
        from ..models.task_output_decoder import TaskOutputDecoder

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
            raise RuntimeError(
                "Access to the requested Stable Diffusion checkpoint is gated. "
                "Request access on Hugging Face, then login or set model.auth_token / HF_TOKEN."
            ) from exc
        except FileNotFoundError as exc:
            cache_root_env = os.environ.get('HUGGINGFACE_HUB_CACHE')
            if cache_root_env:
                cache_root = Path(cache_root_env)
            else:
                hf_home = os.environ.get('HF_HOME')
                cache_root = (Path(hf_home) if hf_home else Path.home() / '.cache' / 'huggingface') / 'hub'
            repo_cache = cache_root / f"models--{base_model_id.replace('/', '--')}"
            raise RuntimeError(
                f"Missing checkpoint shard(s) in the Hugging Face cache. Remove '{repo_cache}' and retry."
            ) from exc

        logger.info("Initializing focal components...")
        self.focal_processor = FocalStackProcessor(
            feature_dim=self.config['model']['feature_dim'],
            num_scales=self.config['model'].get('num_scales', 1),
            max_sequence_length=self.config['model']['max_focal_stack_size'],
            focal_encoder_type=self.config['model'].get('focal_encoder_type', 'focal_sweep'),
            patch_size=self.config['model'].get('patch_size', 8),
            focal_attention_heads=self.config['model'].get('focal_attention_heads', 8),
            focal_attention_depth=self.config['model'].get('focal_attention_depth', 2),
        )
        self.focal_evidence_head = FocalEvidenceEncoder(
            hidden=self.config["model"].get("focal_evidence_hidden", 48),
            temperature=self.config["model"].get("focal_evidence_temperature", 0.07),
        )
        self.physical_evidence_support_head = PhysicalEvidenceEstimator(
            in_channels=5,
            hidden=self.config["model"].get("physical_evidence_support_hidden", 16),
        )

        self.task_output_decoder = TaskOutputDecoder(
            in_channels=pipe.vae.config.latent_channels,
            out_channels_depth=1,
            out_channels_rgb=pipe.vae.config.latent_channels,
        )


        # Create FocalStackGeneration pipeline
        self.pipeline = FocalStackGenerationPipeline(
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
            task_output_decoder=self.task_output_decoder,
            physical_evidence_support_head=self.physical_evidence_support_head,
        )

        target_device = self.accelerator.device if hasattr(self, "accelerator") else torch.device("cpu")
        self.pipeline.to(target_device)
        self.focal_processor = self.pipeline.focal_processor
        self.focal_evidence_head = self.pipeline.focal_evidence_head
        self.task_output_decoder = self.pipeline.task_output_decoder
        self.physical_evidence_support_head = self.pipeline.physical_evidence_support_head

        # Configure trainable parameters
        self._configure_trainable_params()

        if self.config['training'].get('use_ema'):
            ema_params = self._ema_parameters()
            self.ema = EMAModel(ema_params, decay=self.config['training']['ema_decay']) if ema_params else None
        else:
            self.ema = None

    def _configure_trainable_params(self):
        """Configure which parameters to train"""
        train_config = self.config['training']['trainable_modules']
        self.pipeline.vae.requires_grad_(False)
        self.pipeline.text_encoder.requires_grad_(False)
        self.pipeline.text_encoder_2.requires_grad_(False)
        if hasattr(self.pipeline, 'text_encoder_3') and self.pipeline.text_encoder_3 is not None:
            self.pipeline.text_encoder_3.requires_grad_(False)

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

        self.focal_processor.requires_grad_(True)
        self.focal_evidence_head.requires_grad_(True)
        self.task_output_decoder.requires_grad_(True)
        self.physical_evidence_support_head.requires_grad_(True)

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
        trainable_params = list(self._iter_pipeline_parameters(only_trainable=True))


        # Create optimizer
        self.optimizer = get_optimizer(
            trainable_params,
            optimizer_type=self.config['optimizer'].get('type', 'adamw'),
            learning_rate=self.config['optimizer']['learning_rate'],
            weight_decay=self.config['optimizer']['weight_decay'],
            betas=tuple(self.config['optimizer']['betas']),
        )
        num_training_steps = len(self.train_dataloader) * self.config['training']['num_epochs']
        num_training_steps = num_training_steps // self.config['training']['gradient_accumulation_steps']
        self.lr_scheduler = get_scheduler(
            self.config['scheduler']['type'],
            optimizer=self.optimizer,
            num_warmup_steps=self.config['scheduler']['warmup_steps'],
            num_training_steps=num_training_steps,
        )
        (
            self.pipeline,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.pipeline,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.lr_scheduler,
        )
        self.scaler = GradScaler() if self.config['training']['mixed_precision'] == 'fp16' else None

    def _iter_pipeline_parameters(self, only_trainable: bool = False):
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
        raise AttributeError("FocalStackGenerationPipeline is missing parameter accessors.")

    def _ema_parameters(self):
        return [param for param in self._iter_pipeline_parameters(only_trainable=True)]

    def setup_tracking(self):
        """Setup experiment tracking"""
        if self.accelerator.is_main_process:
            if self.config['logging'].get('use_wandb'):
                run_name = self.config['logging'].get('run_name') or f"focal_diffusion_{datetime.now():%Y%m%d_%H%M%S}"
                self.accelerator.init_trackers(
                    project_name=self.config['logging'].get('project_name', 'focal-diffusion'),
                    config=self.config,
                    init_kwargs={"wandb": {"name": run_name, "tags": self.config['logging'].get('tags', [])}},
                )
            self.checkpoint_dir = Path(self.config['output']['save_dir']) / 'checkpoints'
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_dir / 'config.json', 'w') as f:
                json.dump(self.config, f, indent=2)

    def train(self):
        logger.info("Starting training...")
        best_val_loss = float('inf')
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            train_loss = self.train_epoch(epoch, self.global_step)
            self.global_step += len(self.train_dataloader)
            if epoch % self.config['training']['val_every_n_epochs'] == 0:
                val_metrics = self.validate(epoch)
                if self.accelerator.is_main_process:
                    self.accelerator.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        **{f'val_{k}': v for k, v in val_metrics.items()},
                    }, step=self.global_step)
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, self.global_step, is_best=True)
            if epoch % self.config['training']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch, self.global_step)
        self.save_checkpoint(epoch, self.global_step, is_final=True)
        logger.info("Training completed!")

    def train_epoch(self, epoch: int, global_step: int) -> float:
        from ..training.losses import FocalStackGenerationLoss

        self.pipeline.train()
        epoch_loss = 0.0
        effective_steps = 0
        supervision_mode = self.config['training'].get('supervision_mode', 'supervised')
        if supervision_mode not in {"supervised", "semi_supervised"}:
            raise ValueError("training.supervision_mode must be 'supervised' or 'semi_supervised'.")

        loss_fn = FocalStackGenerationLoss(
            diffusion_weight=self.config['losses']['diffusion_weight'],
            depth_weight=self.config['losses']['depth_weight'],
            rgb_weight=self.config['losses']['rgb_weight'],
            focal_posterior_kl_weight=self.config['losses'].get('focal_posterior_kl_weight', 0.2),
            focus_depth_weight=self.config['losses'].get('focus_depth_weight', 0.2),
            prior_depth_weight=self.config['losses'].get('prior_depth_weight', 0.05),
            all_in_focus_focal_evidence_weight=self.config['losses'].get('all_in_focus_focal_evidence_weight', 0.1),
            uncertainty_focus_weight=self.config['losses'].get('uncertainty_focus_weight', 0.05),
            uncertainty_error_weight=self.config['losses'].get('uncertainty_error_weight', 0.05),
            gate_calibration_weight=self.config['losses'].get('gate_calibration_weight', 0.05),
            posterior_consistency_weight=self.config['losses'].get('posterior_consistency_weight', 0.02),
            depth_affinity_smoothness_weight=self.config['losses'].get('depth_affinity_smoothness_weight', 0.01),
            gate_consistency_weight=self.config['losses'].get('gate_consistency_weight', 0.0),
            focal_axis_smoothness_weight=self.config['losses'].get('focal_axis_smoothness_weight', 0.0),
            local_affinity_sigma=self.config['losses'].get('local_affinity_sigma', 0.10),
            focus_target_temperature=self.config['losses'].get('focus_target_temperature', 0.07),
            focal_target_type=self.config["losses"].get("focal_target_type", "normalized"),
            coc_posterior_temperature=self.config["losses"].get("coc_posterior_temperature", 1.0),
            supervision_mode=supervision_mode,
        )
        loss_fn = loss_fn.to(self.accelerator.device)

        prompt_embeds, pooled_prompt_embeds = self._get_empty_prompt_embeddings()
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.pipeline):
                device = self.accelerator.device
                focal_stack = batch['focal_stack'].to(device)
                focal_plane_distances = batch['focal_plane_distances'].to(device)
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
                camera_params = batch.get('camera_params')
                if camera_params is not None:
                    camera_params = {
                        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                        for key, value in camera_params.items()
                    }

                # Normalize inputs
                if focal_stack.min() >= 0 and focal_stack.max() <= 1:
                    focal_stack = (focal_stack * 2.0) - 1.0
                rgb_target = (rgb_gt * 2.0) - 1.0 if rgb_gt is not None else None

                # Extract focal features for SD3 conditioning.
                focal_features = self.focal_processor(focal_stack, focal_plane_distances)

                focal_features = {
                    key: value.to(self.pipeline.transformer.dtype)
                    if isinstance(value, torch.Tensor) and value is not None else value
                    for key, value in focal_features.items()
                }

                vae_dtype = next(self.pipeline.vae.parameters()).dtype
                rgb_latent_input = rgb_target.to(dtype=vae_dtype)
                latents_dist = self.pipeline.vae.encode(rgb_latent_input).latent_dist
                latents = latents_dist.sample() * self.pipeline.vae.config.scaling_factor
                flow_batch = sample_sd3_flow_matching_batch(self.pipeline.scheduler, latents)
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

                with autocast(enabled=self.scaler is not None):
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
                    if rgb_target is None:
                        raise ValueError("all_in_focus is required to compute the SD3 flow-matching reconstruction losses.")

                    decoder_outputs = self.task_output_decoder(clean_latent_pred)
                    generated_depth_canonical = decoder_outputs["generated_depth_canonical"]
                    uncertainty = decoder_outputs["uncertainty"]
                    focal_evidence = self.focal_evidence_head(focal_stack.float(), focal_plane_distances.float())
                    focal_depth_canonical = F.interpolate(focal_evidence["focal_depth_canonical"], size=generated_depth_canonical.shape[-2:], mode="bilinear", align_corners=False)
                    focal_entropy = F.interpolate(focal_evidence["focal_entropy"], size=generated_depth_canonical.shape[-2:], mode="bilinear", align_corners=False)
                    focal_posterior = F.interpolate(
                        focal_evidence["focal_posterior"],
                        size=generated_depth_canonical.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    focal_posterior = focal_posterior / focal_posterior.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    generative_uncertainty = uncertainty
                    support_inputs, support_maps = build_physical_evidence_features(
                        focal_posterior=focal_posterior,
                        focal_entropy=focal_entropy,
                        focal_depth_canonical=focal_depth_canonical,
                        generated_depth_canonical=generated_depth_canonical,
                        generative_uncertainty=generative_uncertainty,
                    )
                    support_outputs = self.physical_evidence_support_head(support_inputs)
                    focal_evidence_weight = support_outputs["focal_evidence_weight"]
                    generative_prior_weight = support_outputs["generative_prior_weight"]
                    abstention_weight = support_outputs["abstention_weight"]
                    gate_sum = (focal_evidence_weight + generative_prior_weight).clamp(min=1e-6)
                    focal_evidence_weight_norm = focal_evidence_weight / gate_sum
                    generative_prior_weight_norm = generative_prior_weight / gate_sum
                    final_depth_canonical = focal_evidence_weight_norm * focal_depth_canonical + generative_prior_weight_norm * generated_depth_canonical
                    uncertainty_final = torch.maximum(
                        support_outputs["uncertainty_final"],
                        abstention_weight,
                    ).clamp(0.0, 1.0)
                    rgb_latent_pred = decoder_outputs["all_in_focus_latents"]
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

                    loss_dict = loss_fn(
                        diffusion_pred=diffusion_pred.float(),
                        diffusion_target=flow_target.float(),
                        depth_target=depth_gt.float() if depth_gt is not None else None,
                        depth_mask=depth_mask,
                        rgb_pred=rgb_recon,
                        rgb_target=rgb_target_fp32,
                        generated_depth_canonical=generated_depth_canonical.float(),
                        focal_depth_canonical=focal_depth_canonical.float(),
                        final_depth_canonical=final_depth_canonical.float(),
                        uncertainty=uncertainty_final.float(),
                        focal_posterior=focal_posterior.float(),
                        focal_entropy=focal_entropy.float(),
                        focal_plane_distances=focal_plane_distances.float(),
                        focal_stack=focal_stack.float(),
                        depth_range=depth_range_tensor.float() if depth_range_tensor is not None else None,
                        focal_evidence_weight=focal_evidence_weight_norm.float(),
                        generative_prior_weight=generative_prior_weight_norm.float(),
                        abstention_weight=abstention_weight.float(),
                        physical_evidence_support=support_outputs["physical_evidence_support"].float(),
                        camera_params=camera_params,
                    )
                    loss = loss_dict['total']

                self.accelerator.backward(loss)
                if self.config['training'].get('max_grad_norm'):
                    clip_params = list(self._iter_pipeline_parameters(only_trainable=True))
                    if clip_params:
                        self.accelerator.clip_grad_norm_(clip_params, self.config['training']['max_grad_norm'])
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if self.ema is not None:
                    self.ema.step(self._ema_parameters())

                effective_steps += 1
                epoch_loss += loss.item()
                if step % self.config['logging']['log_every_n_steps'] == 0:
                    progress_bar.set_postfix({'loss': loss.item(), 'lr': self.lr_scheduler.get_last_lr()[0]})
                    if self.accelerator.is_main_process:
                        self.accelerator.log({
                            'train_loss_step': loss.item(),
                            'learning_rate': self.lr_scheduler.get_last_lr()[0],
                            **{f'train_{k}': v.item() for k, v in loss_dict.items() if k != 'total' and isinstance(v, torch.Tensor)},
                            'train_focal_entropy_mean': focal_entropy.mean().item(),
                            'train_focal_peak_confidence_mean': support_maps['focal_peak_confidence'].mean().item(),
                            'train_focal_evidence_weight_mean': focal_evidence_weight_norm.mean().item(),
                            'train_generative_prior_weight_mean': generative_prior_weight_norm.mean().item(),
                            'train_abstention_weight_mean': abstention_weight.mean().item(),
                            'train_physical_evidence_support_mean': support_outputs['physical_evidence_support'].mean().item(),
                            'train_generated_focal_depth_disagreement': support_maps['depth_disagreement'].mean().item(),
                        }, step=global_step + step)

        if effective_steps == 0:
            return 0.0
        return epoch_loss / effective_steps

    def _get_empty_prompt_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if prompt_embeds.shape[0] == batch_size and pooled_prompt_embeds.shape[0] == batch_size:
            return prompt_embeds, pooled_prompt_embeds
        if prompt_embeds.shape[0] != 1 or pooled_prompt_embeds.shape[0] != 1:
            raise ValueError(
                "Cached empty prompt embeddings must have batch size 1 or match the current training batch size."
            )
        repeat_shape = (batch_size,) + (1,) * (prompt_embeds.ndim - 1)
        pooled_repeat_shape = (batch_size,) + (1,) * (pooled_prompt_embeds.ndim - 1)
        return (
            prompt_embeds.repeat(repeat_shape),
            pooled_prompt_embeds.repeat(pooled_repeat_shape),
        )

    def validate(self, epoch: int) -> dict:
        from .validation import run_validation
        return run_validation(self, epoch)

    def save_checkpoint(self, epoch: int, global_step: int, is_best: bool = False, is_final: bool = False):
        from .checkpointing import save_checkpoint
        save_checkpoint(self, epoch, global_step, is_best=is_best, is_final=is_final)

    def _cleanup_checkpoints(self):
        from .checkpointing import cleanup_checkpoints
        cleanup_checkpoints(self)

    def load_checkpoint(self, checkpoint_path: str):
        from .checkpointing import load_checkpoint
        return load_checkpoint(self, checkpoint_path)
