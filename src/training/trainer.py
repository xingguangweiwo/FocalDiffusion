"""
FocalStackGeneration Trainer Class
Main trainer implementation for FocalStackGeneration
"""

import os
import logging
import math
from pathlib import Path
from datetime import datetime
import json
import hashlib

import torch
import torch.nn.functional as F
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import get_token as hf_get_token
from huggingface_hub.errors import GatedRepoError
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from typing import Any, Dict, Iterable, Optional, Tuple

from .sd3_objective import predict_clean_latents_from_flow, sample_sd3_flow_matching_batch
from ..models.focal_evidence_encoder import build_physical_evidence_features
from ..models.physics_modules import FocalPhysicalVerifier
from ..utils.image_utils import resize_probability_volume, to_model_range, to_unit_range


logger = logging.getLogger(__name__)


class FocalStackGenerationTrainer:
    """Main trainer class for FocalStackGeneration using file lists"""

    def __init__(self, config: dict):
        self.config = config
        self.setup_logging()
        self.setup_accelerator()
        self.adaptation_cfg = self.config.get('training', {}).get('unsupervised_adaptation', {}) or {}
        self.adaptation_enabled = bool(self.adaptation_cfg.get('enabled', False))
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_tracking()

        self.trace_mining_buffer: TraceMiningBuffer | None = None
        self.parent_checkpoint_sha256: str | None = None
        self.mining_verifier_config_hash: str | None = None
        if self.adaptation_enabled:
            self.trace_mining_buffer = TraceMiningBuffer(
                max_items=int(self.adaptation_cfg.get('max_buffer_items', 128)),
                round_id=str(self.adaptation_cfg.get('round_id', 'adaptation')),
                round_index=int(self.adaptation_cfg.get('round_index', 0)),
                manifest_path=self.adaptation_cfg.get('mining_manifest'),
            )
            self.mining_verifier_config_hash = self._verifier_config_hash(self.pipeline.physical_verifier)
            self._validate_adaptation_round()
            self._freeze_physical_verifier()

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
        adaptation_sources = dataset_cfg.get('adaptation_sources')
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
        self.adaptation_dataloader = None
        if self.adaptation_enabled and adaptation_sources:
            self._ensure_adaptation_is_not_test(dataset_cfg)
            self.adaptation_dataloader = create_dataloader(
                dataset_type=dataset_type,
                filelist_path=None,
                data_root=dataset_cfg.get('data_root', "./data"),
                batch_size=int(self.adaptation_cfg.get('replay_batch_size', self.config['training']['batch_size'])),
                num_workers=dataset_cfg['num_workers'],
                image_size=tuple(dataset_cfg['image_size']),
                focal_stack_size=dataset_cfg['focal_stack_size'],
                focal_range=tuple(dataset_cfg['focal_range']),
                resize_mode=dataset_cfg.get('resize_mode', 'letterbox'),
                augmentation=False,
                shuffle=True,
                max_samples=dataset_cfg.get('max_unsupervised_adaptation_samples'),
                sources=adaptation_sources,
                **base_dataset_kwargs,
            )
        logger.info(f"Train samples: {len(self.train_dataloader.dataset)}")
        logger.info(f"Val samples: {len(self.val_dataloader.dataset)}")
        if self.adaptation_dataloader is not None:
            logger.info("Unsupervised adaptation samples: %s", len(self.adaptation_dataloader.dataset))

    @staticmethod
    def _source_filelists(sources: Iterable[dict[str, Any]] | None) -> set[str]:
        return {str(source.get("filelist")) for source in (sources or []) if source.get("filelist")}

    def _ensure_adaptation_is_not_test(self, dataset_cfg: dict[str, Any]) -> None:
        adapt_filelists = self._source_filelists(dataset_cfg.get("adaptation_sources"))
        test_filelists = self._source_filelists(dataset_cfg.get("test_sources"))
        overlap = adapt_filelists & test_filelists
        if overlap:
            raise ValueError(
                "data.adaptation_sources must not overlap data.test_sources; "
                f"overlapping filelists: {sorted(overlap)}"
            )


    @staticmethod
    def _verifier_config_hash(verifier: torch.nn.Module) -> str:
        """Hash non-learned verifier configuration for manifest compatibility checks."""
        if hasattr(verifier, "config_dict"):
            payload = verifier.config_dict()
        else:
            payload = {"class": type(verifier).__name__, "repr": repr(verifier)}
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_adaptation_round(self) -> None:
        """Enforce unsupervised adaptation round provenance before training."""
        round_index = int(self.adaptation_cfg.get("round_index", 0))
        if round_index < 1:
            raise ValueError("training.unsupervised_adaptation.enabled=true requires round_index >= 1; source training uses enabled=false.")
        parent_checkpoint = self.adaptation_cfg.get("parent_checkpoint")
        if not parent_checkpoint:
            raise ValueError("training.unsupervised_adaptation.enabled=true requires parent_checkpoint.")
        parent_path = Path(parent_checkpoint)
        if not parent_path.exists():
            raise FileNotFoundError(f"training.unsupervised_adaptation.parent_checkpoint does not exist: {parent_path}")
        self.parent_checkpoint_sha256 = checkpoint_sha256(parent_path)
        if self.trace_mining_buffer is None:
            raise ValueError("training.unsupervised_adaptation.enabled=true requires TraceMiningBuffer initialization.")
        manifest_path = self.trace_mining_buffer.manifest_path
        if manifest_path is None:
            raise ValueError("training.unsupervised_adaptation.enabled=true requires mining_manifest for checkpoint/verifier provenance.")
        if manifest_path and self.trace_mining_buffer.metadata:
            manifest_parent = self.trace_mining_buffer.metadata.get("parent_checkpoint_sha256")
            manifest_verifier = self.trace_mining_buffer.metadata.get("verifier_config_hash")
            if manifest_parent != self.parent_checkpoint_sha256:
                raise ValueError("mining manifest parent checkpoint SHA256 does not match training.unsupervised_adaptation.parent_checkpoint")
            if manifest_verifier != self.mining_verifier_config_hash:
                raise ValueError("mining manifest verifier config hash does not match current mining verifier")
        elif manifest_path and manifest_path.exists():
            raise ValueError("existing mining manifest is missing parent checkpoint/verifier metadata")
        self.trace_mining_buffer.set_metadata(
            parent_checkpoint_sha256=self.parent_checkpoint_sha256,
            verifier_config_hash=self.mining_verifier_config_hash,
        )

    def _freeze_physical_verifier(self) -> None:
        """Freeze the non-learned verifier during verifier-guided adaptation."""
        verifier = getattr(self.pipeline, "physical_verifier", None)
        if verifier is None:
            return
        verifier.eval()
        for parameter in verifier.parameters():
            parameter.requires_grad_(False)

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
        self.pipeline.physical_verifier.requires_grad_(False)
        self.pipeline.physical_verifier.eval()
        evaluation_cfg = self.config.get("validation", {}).get("evaluation_verifier", {}) or {}
        eval_focus_operator = str(evaluation_cfg.get("focus_operator", "gradient_variance"))
        self.evaluation_verifier = FocalPhysicalVerifier(focus_operator=eval_focus_operator).to(target_device)
        self.evaluation_verifier.requires_grad_(False)
        self.evaluation_verifier.eval()

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

    def train_epoch(self, epoch: int, global_step: int | None = None) -> float:
        from ..training.losses import FocalStackGenerationLoss

        self.pipeline.train()
        self.pipeline.physical_verifier.eval()
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
            lambda_trace=self.config["losses"].get("lambda_trace", 0.05),
            lambda_violation=self.config["losses"].get("lambda_violation", 0.05),
            lambda_invalid=self.config["losses"].get("lambda_invalid", 0.05),
            supervision_mode=supervision_mode,
        )
        loss_fn = loss_fn.to(self.accelerator.device)
        adaptation_enabled = self.adaptation_enabled
        mining_conflict_threshold = float(self.adaptation_cfg.get('conflict_threshold', 0.5))
        mining_confidence_threshold = float(self.adaptation_cfg.get('confidence_threshold', 0.8))
        trace_replay_weight = float(self.config['losses'].get('lambda_violation', 0.02))
        if adaptation_enabled and self.trace_mining_buffer is not None and len(self.trace_mining_buffer) == 0:
            self._mine_unsupervised_adaptation_manifest(
                max_batches=int(self.adaptation_cfg.get("max_mining_batches", 8)),
                conflict_threshold=mining_conflict_threshold,
                confidence_threshold=mining_confidence_threshold,
            )

        prompt_embeds, pooled_prompt_embeds = self._get_empty_prompt_embeddings()
        replay_iterator = iter(self.adaptation_dataloader) if (adaptation_enabled and self.adaptation_dataloader is not None) else None
        replay_batch_count = 0
        replay_loss_running = 0.0
        source_loss_running = 0.0
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )

        if global_step is not None:
            self.global_step = global_step

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

                # Normalize inputs without clamping before range detection.
                focal_stack_unit, focal_stack = (to_unit_range(focal_stack.float()), to_model_range(focal_stack.float()))
                rgb_target = None
                if rgb_gt is not None:
                    _, rgb_target = (to_unit_range(rgb_gt.float()), to_model_range(rgb_gt.float()))

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

                with self.accelerator.autocast():
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
                    focal_posterior = resize_probability_volume(focal_evidence["focal_posterior"], generated_depth_canonical.shape[-2:])
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
                    rgb_recon = rgb_recon.float()
                    rgb_target_fp32 = rgb_target.float()
                    focal_stack_for_trace = focal_stack_unit.float()
                    rgb_recon_for_trace, _ = (to_unit_range(rgb_recon.float()), to_model_range(rgb_recon.float()))
                    with torch.no_grad():
                        physical_verification_trace = self.pipeline.physical_verifier(
                            focal_stack=focal_stack_for_trace,
                            focal_plane_distances=focal_plane_distances.float(),
                            depth_canonical=final_depth_canonical.float(),
                            all_in_focus=rgb_recon_for_trace,
                            generated_depth_canonical=generated_depth_canonical.float(),
                        )

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
                        physical_verification_trace=physical_verification_trace,
                        predicted_verification_support=support_outputs["physical_evidence_support"].float(),
                        predicted_verification_invalid=uncertainty_final.float(),
                        camera_params=camera_params,
                    )
                    mined_trace_stats = {
                        "mined_trace_items": 0,
                        "mined_sample_count": 0,
                        "accepted_conflict_count": 0,
                        "accepted_invalid_count": 0,
                        "accepted_reliable_count": 0,
                        "buffer_size": len(self.trace_mining_buffer) if self.trace_mining_buffer is not None else 0,
                        "mined_conflict_mean": 0.0,
                        "mined_invalid_mean": 0.0,
                    }
                    if adaptation_enabled:
                        replay_loss = None
                        if replay_iterator is not None:
                            try:
                                replay_batch = next(replay_iterator)
                            except StopIteration:
                                replay_iterator = iter(self.adaptation_dataloader)
                                replay_batch = next(replay_iterator)
                            replay_loss = self._compute_adaptation_replay_loss(
                                replay_batch,
                                max_items=int(self.adaptation_cfg.get("max_replay_items_per_batch", 4)),
                            )
                        if replay_loss is not None:
                            replay_loss = trace_replay_weight * replay_loss
                            loss_dict["loss_violation"] = loss_dict["loss_violation"] + replay_loss
                            loss_dict["loss_trace_replay"] = replay_loss
                            loss_dict["total"] = loss_dict["total"] + replay_loss
                            replay_batch_count += 1
                            replay_loss_running += float(replay_loss.detach().item())
                    loss = loss_dict['total']
                source_loss_running += float((loss_dict['total'] - loss_dict.get('loss_trace_replay', torch.zeros_like(loss_dict['total']))).detach().item())

                self.accelerator.backward(loss)
                gradient_norm = 0.0
                if self.accelerator.sync_gradients:
                    if self.config['training'].get('max_grad_norm'):
                        clip_params = list(self._iter_pipeline_parameters(only_trainable=True))
                        if clip_params:
                            clipped_norm = self.accelerator.clip_grad_norm_(clip_params, self.config['training']['max_grad_norm'])
                            gradient_norm = float(clipped_norm.detach().item() if isinstance(clipped_norm, torch.Tensor) else clipped_norm)
                    else:
                        grad_sq = 0.0
                        for param in self._iter_pipeline_parameters(only_trainable=True):
                            if param.grad is not None:
                                grad_sq += float(param.grad.detach().float().norm(2).item() ** 2)
                        gradient_norm = math.sqrt(grad_sq)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if self.ema is not None:
                        self.ema.step(self._ema_parameters())
                    self.global_step += 1
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
                            **{f'train_{k}': v for k, v in mined_trace_stats.items()},
                            'train_replay_batch_count': replay_batch_count,
                            'train_replay_loss': replay_loss_running / max(replay_batch_count, 1),
                            'train_source_loss': source_loss_running / max(effective_steps + 1, 1),
                            'train_gradient_norm': gradient_norm,
                        }, step=self.global_step)

        if effective_steps == 0:
            return 0.0
        return epoch_loss / effective_steps


    def _forward_adaptation_generation(self, batch: dict[str, Any], *, requires_grad: bool) -> dict[str, torch.Tensor | object]:
        """Run the focal-conditioned generation path used for mining/replay."""
        device = self.accelerator.device
        focal_stack = batch["focal_stack"].to(device)
        focal_plane_distances = batch["focal_plane_distances"].to(device)
        focal_stack_unit, focal_stack_signed = (to_unit_range(focal_stack.float()), to_model_range(focal_stack.float()))
        prompt_embeds, pooled_prompt_embeds = self._get_empty_prompt_embeddings()
        batch_size = focal_stack.shape[0]
        height, width = focal_stack.shape[-2:]
        latent_h = max(1, height // int(self.pipeline.vae_scale_factor))
        latent_w = max(1, width // int(self.pipeline.vae_scale_factor))
        dtype = self.pipeline.transformer.dtype
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            focal_features = self.focal_processor(focal_stack_signed, focal_plane_distances.float())
            focal_features = {
                key: value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor) and value is not None else value
                for key, value in focal_features.items()
            }
            focal_evidence = self.focal_evidence_head(focal_stack_signed.float(), focal_plane_distances.float())
            latent_channels = int(getattr(self.pipeline.transformer.config, "in_channels", self.pipeline.vae.config.latent_channels))
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.adaptation_cfg.get("replay_noise_seed", 0)))
            latents = torch.randn((batch_size, latent_channels, latent_h, latent_w), device=device, dtype=dtype, generator=generator)
            self.pipeline.scheduler.set_timesteps(max(1, int(self.adaptation_cfg.get("adapt_inference_steps", 1))), device=device)
            timestep = self.pipeline.scheduler.timesteps[0]
            model_input = self.pipeline.scheduler.scale_model_input(latents, timestep) if hasattr(self.pipeline.scheduler, "scale_model_input") else latents
            batch_prompt_embeds, batch_pooled_prompt_embeds = self._repeat_prompt_embeddings(prompt_embeds, pooled_prompt_embeds, batch_size=batch_size)
            diffusion_pred = self.pipeline.transformer(
                hidden_states=model_input,
                timestep=timestep.expand(batch_size) if timestep.dim() == 0 else timestep,
                encoder_hidden_states=batch_prompt_embeds,
                pooled_projections=batch_pooled_prompt_embeds,
                focal_features=focal_features,
                return_dict=False,
            )[0]
            sigmas = torch.as_tensor(self.pipeline.scheduler.sigmas, device=device, dtype=latents.dtype)[:1].expand(batch_size)
            clean_latent_pred = predict_clean_latents_from_flow(
                noisy_latents=latents,
                model_pred=diffusion_pred,
                sigmas=sigmas,
            )
            decoder_outputs = self.task_output_decoder(clean_latent_pred.to(dtype=next(self.task_output_decoder.parameters()).dtype))
            generated_depth_canonical = F.interpolate(decoder_outputs["generated_depth_canonical"].float(), size=(height, width), mode="bilinear", align_corners=False)
            generative_uncertainty = F.interpolate(decoder_outputs["uncertainty"].float(), size=(height, width), mode="bilinear", align_corners=False)
            focal_depth_canonical = F.interpolate(focal_evidence["focal_depth_canonical"].float(), size=(height, width), mode="bilinear", align_corners=False)
            focal_entropy = F.interpolate(focal_evidence["focal_entropy"].float(), size=(height, width), mode="bilinear", align_corners=False)
            focal_posterior = resize_probability_volume(focal_evidence["focal_posterior"].float(), (height, width))
            support_inputs, _ = build_physical_evidence_features(
                focal_posterior=focal_posterior,
                focal_entropy=focal_entropy,
                focal_depth_canonical=focal_depth_canonical,
                generated_depth_canonical=generated_depth_canonical,
                generative_uncertainty=generative_uncertainty,
            )
            support_outputs = self.physical_evidence_support_head(support_inputs)
            focal_gate = support_outputs["focal_evidence_weight"]
            generative_gate = support_outputs["generative_prior_weight"]
            abstention = support_outputs["abstention_weight"]
            gate_sum = (focal_gate + generative_gate).clamp(min=1e-6)
            focal_gate_norm = focal_gate / gate_sum
            generative_gate_norm = generative_gate / gate_sum
            final_depth_canonical = focal_gate_norm * focal_depth_canonical + generative_gate_norm * generated_depth_canonical
            uncertainty_final = torch.maximum(support_outputs["uncertainty_final"], abstention).clamp(0.0, 1.0)
            rgb_latents = decoder_outputs["all_in_focus_latents"]
            rgb_recon = self.pipeline.vae.decode(rgb_latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0].float().clamp(-1, 1)
            rgb_recon_unit, _ = (to_unit_range(rgb_recon), to_model_range(rgb_recon))
            trace = self.pipeline.physical_verifier(
                focal_stack=focal_stack_unit,
                focal_plane_distances=focal_plane_distances.float(),
                depth_canonical=final_depth_canonical,
                all_in_focus=rgb_recon_unit,
                generated_depth_canonical=generated_depth_canonical,
            )
        return {
            "trace": trace,
            "uncertainty_final": uncertainty_final,
            "focal_depth_canonical": focal_depth_canonical,
            "generated_depth_canonical": generated_depth_canonical,
            "final_depth_canonical": final_depth_canonical,
            "focal_gate": focal_gate_norm,
            "generative_gate": generative_gate_norm,
            "abstention": abstention,
            "physical_evidence_support": support_outputs["physical_evidence_support"],
            "focal_plane_distances": focal_plane_distances,
            "focal_stack_unit": focal_stack_unit,
            "all_in_focus_unit": rgb_recon_unit,
        }

    def _mine_unsupervised_adaptation_manifest(
        self,
        *,
        max_batches: int,
        conflict_threshold: float,
        confidence_threshold: float,
    ) -> dict[str, float]:
        """Mine verifier targets from U_adapt with a frozen verifier.

        This is the source-model mining pass. It reads only
        ``data.adaptation_sources`` and stores a lightweight manifest
        for replay; no detached prediction tensors are saved.
        """
        if self.adaptation_dataloader is None:
            raise ValueError("unsupervised adaptation is enabled but data.adaptation_sources is not configured")
        self._freeze_physical_verifier()
        stats = {
            "mined_sample_count": 0.0,
            "accepted_conflict_count": 0.0,
            "accepted_invalid_count": 0.0,
            "accepted_reliable_count": 0.0,
            "buffer_size": float(len(self.trace_mining_buffer)) if self.trace_mining_buffer is not None else 0.0,
        }
        was_training = {
            "focal_evidence_head": self.focal_evidence_head.training,
            "physical_evidence_support_head": self.physical_evidence_support_head.training,
        }
        self.focal_evidence_head.eval()
        self.physical_evidence_support_head.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(self.adaptation_dataloader):
                if batch_index >= max_batches:
                    break
                outputs = self._forward_adaptation_generation(batch, requires_grad=False)
                split_seed = int(self.adaptation_cfg.get("focal_split_seed", 0)) + batch_index
                train_idx, val_idx = self.pipeline._split_refinement_planes(outputs["focal_stack_unit"].shape[1], outputs["focal_stack_unit"].device, split_seed)
                before_risk = self.pipeline._heldout_measurement_loss(
                    outputs["focal_stack_unit"],
                    outputs["all_in_focus_unit"],
                    outputs["final_depth_canonical"],
                    outputs["focal_plane_distances"],
                    val_idx,
                )
                cand_depth, cand_uncertainty, cand_aif = self.pipeline._selective_test_time_refinement(
                    focal_stack_unit=outputs["focal_stack_unit"],
                    focal_plane_distances=outputs["focal_plane_distances"],
                    final_depth_canonical=outputs["final_depth_canonical"],
                    focus_depth_canonical=outputs["focal_depth_canonical"],
                    prior_depth_canonical=outputs["generated_depth_canonical"],
                    all_in_focus_unit=outputs["all_in_focus_unit"],
                    uncertainty_final=outputs["uncertainty_final"],
                    trace=outputs["trace"],
                    seed=split_seed,
                    inner_steps=int(self.adaptation_cfg.get("mining_refinement_steps", 2)),
                )
                after_risk = self.pipeline._heldout_measurement_loss(
                    outputs["focal_stack_unit"], cand_aif, cand_depth, outputs["focal_plane_distances"], val_idx
                )
                accepted = self.pipeline._should_accept_refinement(before_risk, after_risk, float(self.adaptation_cfg.get("heldout_acceptance_epsilon", 1e-4)))
                stable_focus = float(outputs["trace"].texture_confidence.detach().mean().item()) >= float(self.adaptation_cfg.get("min_focus_evidence", 0.05))
                if self.trace_mining_buffer is None:
                    raise RuntimeError("TraceMiningBuffer is not initialized for unsupervised adaptation.")
                mined = self.trace_mining_buffer.mine(
                    trace=outputs["trace"],
                    uncertainty=cand_uncertainty,
                    sample_ids=batch.get("sample_path"),
                    batch_index=batch_index,
                    focal_plane_coordinates=outputs["focal_plane_distances"].float().detach().cpu(),
                    conflict_threshold=conflict_threshold,
                    confidence_threshold=confidence_threshold,
                    generated_depth=outputs["generated_depth_canonical"],
                    focal_depth=outputs["focal_depth_canonical"],
                    final_depth=cand_depth if accepted and stable_focus else None,
                    focal_gate=outputs["focal_gate"],
                    generative_gate=outputs["generative_gate"],
                    abstention=outputs["abstention"],
                    accepted_refinement=bool(accepted and stable_focus),
                    focal_split_seed=split_seed,
                    heldout_risk_before=float(before_risk.detach().item()),
                    heldout_risk_after=float(after_risk.detach().item()),
                    dataset_split="adaptation",
                )
                for key in ("mined_sample_count", "accepted_conflict_count", "accepted_invalid_count", "accepted_reliable_count"):
                    stats[key] += float(mined.get(key, 0.0))
                stats["buffer_size"] = float(len(self.trace_mining_buffer)) if self.trace_mining_buffer is not None else 0.0
        if was_training["focal_evidence_head"]:
            self.focal_evidence_head.train()
        if was_training["physical_evidence_support_head"]:
            self.physical_evidence_support_head.train()
        logger.info("Mined unsupervised adaptation manifest: %s", stats)
        return stats

    def _compute_adaptation_replay_loss(self, batch: dict[str, Any], max_items: int) -> torch.Tensor | None:
        """Re-read an adaptation batch and update full adapted-model outputs from replay targets."""
        if self.trace_mining_buffer is None or not self.trace_mining_buffer.items:
            return None
        outputs = self._forward_adaptation_generation(batch, requires_grad=True)
        trace = outputs["trace"]
        return self.trace_mining_buffer.replay_loss(
            predicted_support=outputs["physical_evidence_support"].float(),
            predicted_invalid=outputs["uncertainty_final"].float(),
            predicted_conflict=trace.conflict_score.float(),
            predicted_final_depth=outputs["final_depth_canonical"].float(),
            predicted_generated_depth=outputs["generated_depth_canonical"].float(),
            predicted_focal_gate=outputs["focal_gate"].float(),
            predicted_generative_gate=outputs["generative_gate"].float(),
            predicted_abstention=outputs["abstention"].float(),
            sample_ids=batch.get("sample_path"),
            max_items=max_items,
        )

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


class TraceMiningBuffer:
    """Lightweight manifest of mined physical-verification trace patches.

    The manifest stores source identity, crop geometry, focal-plane coordinates
    and scalar supervision targets. It intentionally does not store detached
    model predictions; replay must re-read the source focal stack and run the
    current model forward so gradients update adapted-model parameters.
    """

    def __init__(
        self,
        max_items: int = 128,
        round_id: str = "adaptation",
        patch_size: int = 16,
        round_index: int = 0,
        manifest_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.max_items = max(1, int(max_items))
        self.round_id = round_id
        self.patch_size = max(1, int(patch_size))
        self.round_index = int(round_index)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.metadata: dict[str, Any] = {}
        self.items: list[dict[str, Any]] = []
        if self.manifest_path and self.manifest_path.exists():
            self.load(self.manifest_path)

    def __len__(self) -> int:
        return len(self.items)

    def set_metadata(self, *, parent_checkpoint_sha256: str, verifier_config_hash: str) -> None:
        """Attach provenance required to replay a mined manifest."""
        self.metadata.update({
            "record_type": "metadata",
            "round_id": self.round_id,
            "round_index": self.round_index,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "verifier_config_hash": verifier_config_hash,
        })

    @staticmethod
    def _as_bchw(value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 3:
            return value.unsqueeze(1)
        if value.dim() == 4:
            return value
        raise ValueError(f"Expected [B,H,W] or [B,C,H,W], got {tuple(value.shape)}")

    @staticmethod
    def _sample_id(sample_ids: Any, batch_idx: int, fallback_index: int) -> str:
        if isinstance(sample_ids, (list, tuple)) and batch_idx < len(sample_ids):
            return str(sample_ids[batch_idx])
        if isinstance(sample_ids, str):
            return sample_ids
        return f"batch_{fallback_index}_item_{batch_idx}"

    def _crop_bounds(self, y: int, x: int, height: int, width: int) -> tuple[int, int, int, int]:
        half = self.patch_size // 2
        y0 = max(0, min(height - 1, y) - half)
        x0 = max(0, min(width - 1, x) - half)
        y1 = min(height, y0 + self.patch_size)
        x1 = min(width, x0 + self.patch_size)
        y0 = max(0, y1 - self.patch_size)
        x0 = max(0, x1 - self.patch_size)
        return y0, y1, x0, x1

    def _append_item(self, item: dict[str, Any]) -> None:
        self.items.append(item)
        overflow = len(self.items) - self.max_items
        if overflow > 0:
            del self.items[:overflow]

    def _mine_one(
        self,
        *,
        verdict_type: str,
        score_map: torch.Tensor,
        confidence: torch.Tensor,
        conflict: torch.Tensor,
        invalid: torch.Tensor,
        uncertainty: torch.Tensor,
        discrepancy: torch.Tensor,
        stack_reprojection: torch.Tensor,
        focus_support: torch.Tensor,
        generation_support: torch.Tensor,
        generated_depth: torch.Tensor | None,
        focal_depth: torch.Tensor | None,
        final_depth: torch.Tensor | None,
        focal_gate: torch.Tensor | None,
        generative_gate: torch.Tensor | None,
        abstention: torch.Tensor | None,
        sample_id: str,
        batch_index: int,
        focal_plane_coordinates: torch.Tensor | None,
        conflict_threshold: float,
        confidence_threshold: float,
        accepted_refinement: bool,
        focal_split_seed: int | None,
        heldout_risk_before: float | None,
        heldout_risk_after: float | None,
        dataset_split: str,
    ) -> dict[str, Any] | None:
        eligible = (score_map >= conflict_threshold) & (confidence >= confidence_threshold)
        if not bool(eligible.any()):
            return None
        height, width = score_map.shape[-2:]
        ranked = torch.where(eligible, score_map * confidence, score_map.new_tensor(-1.0))
        flat_index = int(torch.argmax(ranked).item())
        y = flat_index // width
        x = flat_index % width
        y0, y1, x0, x1 = self._crop_bounds(y, x, height, width)
        conflict_patch = conflict[..., y0:y1, x0:x1]
        invalid_patch = invalid[..., y0:y1, x0:x1]
        support_patch = (1.0 - torch.maximum(conflict_patch, invalid_patch)).clamp(0.0, 1.0)
        if verdict_type in {"conflict", "focus_discrepancy"}:
            support_patch = support_patch * (1.0 - discrepancy[..., y0:y1, x0:x1]).clamp(0.0, 1.0)
        focal_plane_coordinates_list = None
        if focal_plane_coordinates is not None:
            focal_plane_coordinates_list = [float(value) for value in focal_plane_coordinates.detach().cpu().flatten().tolist()]
        item = {
            "round_id": self.round_id,
            "round_index": self.round_index,
            "sample_id": sample_id,
            "sample_path": sample_id,
            "batch_index": int(batch_index),
            "dataset_split": dataset_split,
            "verdict_type": verdict_type,
            "accepted_refinement": bool(accepted_refinement),
            "focal_split_seed": None if focal_split_seed is None else int(focal_split_seed),
            "heldout_risk_before": heldout_risk_before,
            "heldout_risk_after": heldout_risk_after,
            "crop": {"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1)},
            "source_shape": {"height": int(height), "width": int(width)},
            "focal_plane_coordinates": focal_plane_coordinates_list,
            "conflict_target": float(conflict_patch.mean().item()),
            "invalid_target": float(invalid_patch.mean().item()),
            "support_target": float(support_patch.mean().item()),
            "verifier_confidence": float(confidence[..., y0:y1, x0:x1].mean().item()),
            "depth_focus_discrepancy_target": float(discrepancy[..., y0:y1, x0:x1].mean().item()),
            "stack_reprojection_residual_target": float(stack_reprojection[..., y0:y1, x0:x1].mean().item()),
            "focus_support_target": float(focus_support[..., y0:y1, x0:x1].mean().item()),
            "generation_support_target": float(generation_support[..., y0:y1, x0:x1].mean().item()),
            "conflict_mean": float(conflict_patch.mean().item()),
            "invalid_mean": float(invalid_patch.mean().item()),
            "uncertainty_mean": float(uncertainty[..., y0:y1, x0:x1].mean().item()),
        }
        optional_maps = {
            "generated_depth_target": generated_depth,
            "focal_depth_target": focal_depth,
            "final_depth_target": final_depth,
            "focal_gate_target": focal_gate,
            "generative_gate_target": generative_gate,
            "abstention_target": abstention,
        }
        for key, value in optional_maps.items():
            if value is not None:
                item[key] = float(value[..., y0:y1, x0:x1].mean().item())
        self._append_item(item)
        return item

    def mine(
        self,
        *,
        trace: object,
        uncertainty: torch.Tensor,
        sample_ids: Any,
        batch_index: int,
        focal_plane_coordinates: torch.Tensor | None = None,
        conflict_threshold: float,
        confidence_threshold: float,
        invalid_threshold: float | None = None,
        support_threshold: float = 0.5,
        discrepancy_threshold: float | None = None,
        generated_depth: torch.Tensor | None = None,
        focal_depth: torch.Tensor | None = None,
        final_depth: torch.Tensor | None = None,
        focal_gate: torch.Tensor | None = None,
        generative_gate: torch.Tensor | None = None,
        abstention: torch.Tensor | None = None,
        accepted_refinement: bool = False,
        focal_split_seed: int | None = None,
        heldout_risk_before: float | None = None,
        heldout_risk_after: float | None = None,
        dataset_split: str = "adaptation",
    ) -> dict[str, float]:
        if not accepted_refinement:
            return {
                "mined_trace_items": 0,
                "mined_sample_count": 0,
                "accepted_conflict_count": 0,
                "accepted_invalid_count": 0,
                "accepted_reliable_count": 0,
                "buffer_size": len(self),
                "mined_conflict_mean": 0.0,
                "mined_invalid_mean": 0.0,
            }
        conflict = self._as_bchw(getattr(trace, "conflict_score")).detach().float().clamp(0.0, 1.0)
        invalid = self._as_bchw(getattr(trace, "invalid_score")).detach().float().clamp(0.0, 1.0)
        discrepancy = self._as_bchw(getattr(trace, "depth_focus_discrepancy")).detach().float().clamp(0.0, 1.0)
        stack_reprojection = self._as_bchw(getattr(trace, "stack_reprojection_residual", torch.zeros_like(conflict))).detach().float().clamp(0.0, 1.0)
        focus_support = self._as_bchw(getattr(trace, "focus_support")).detach().float().clamp(0.0, 1.0)
        generation_support = self._as_bchw(getattr(trace, "generation_support", torch.ones_like(conflict))).detach().float().clamp(0.0, 1.0)
        optional_maps = {
            "generated_depth": generated_depth,
            "focal_depth": focal_depth,
            "final_depth": final_depth,
            "focal_gate": focal_gate,
            "generative_gate": generative_gate,
            "abstention": abstention,
        }
        optional_maps = {key: (self._as_bchw(value).detach().float() if value is not None else None) for key, value in optional_maps.items()}
        uncertainty = self._as_bchw(uncertainty).detach().float().clamp(0.0, 1.0)
        if uncertainty.shape[-2:] != conflict.shape[-2:]:
            uncertainty = F.interpolate(uncertainty, size=conflict.shape[-2:], mode="bilinear", align_corners=False)
        confidence = (1.0 - uncertainty).clamp(0.0, 1.0)
        invalid_threshold = conflict_threshold if invalid_threshold is None else invalid_threshold
        discrepancy_threshold = conflict_threshold if discrepancy_threshold is None else discrepancy_threshold

        mined: list[dict[str, Any]] = []
        batch = conflict.shape[0]
        for batch_idx in range(batch):
            sample_id = self._sample_id(sample_ids, batch_idx, batch_index)
            maps = {
                "conflict": conflict[batch_idx : batch_idx + 1],
                "invalid": invalid[batch_idx : batch_idx + 1] * confidence[batch_idx : batch_idx + 1],
                "focus_discrepancy": discrepancy[batch_idx : batch_idx + 1] * focus_support[batch_idx : batch_idx + 1],
                "reliable_non_conflict": (focus_support[batch_idx : batch_idx + 1] * (1.0 - conflict[batch_idx : batch_idx + 1]) * (1.0 - invalid[batch_idx : batch_idx + 1])),
            }
            for verdict_type, score_map in maps.items():
                threshold = conflict_threshold
                if verdict_type == "invalid":
                    threshold = invalid_threshold
                elif verdict_type == "focus_discrepancy":
                    threshold = discrepancy_threshold
                elif verdict_type == "reliable_non_conflict":
                    threshold = support_threshold
                focal_coords_one = None
                if focal_plane_coordinates is not None:
                    focal_coords_one = focal_plane_coordinates[batch_idx] if focal_plane_coordinates.dim() > 1 else focal_plane_coordinates
                item = self._mine_one(
                    verdict_type=verdict_type,
                    score_map=score_map,
                    confidence=confidence[batch_idx : batch_idx + 1],
                    conflict=conflict[batch_idx : batch_idx + 1],
                    invalid=invalid[batch_idx : batch_idx + 1],
                    uncertainty=uncertainty[batch_idx : batch_idx + 1],
                    discrepancy=discrepancy[batch_idx : batch_idx + 1],
                    stack_reprojection=stack_reprojection[batch_idx : batch_idx + 1],
                    focus_support=focus_support[batch_idx : batch_idx + 1],
                    generation_support=generation_support[batch_idx : batch_idx + 1],
                    generated_depth=optional_maps["generated_depth"][batch_idx : batch_idx + 1] if optional_maps["generated_depth"] is not None else None,
                    focal_depth=optional_maps["focal_depth"][batch_idx : batch_idx + 1] if optional_maps["focal_depth"] is not None else None,
                    final_depth=optional_maps["final_depth"][batch_idx : batch_idx + 1] if optional_maps["final_depth"] is not None else None,
                    focal_gate=optional_maps["focal_gate"][batch_idx : batch_idx + 1] if optional_maps["focal_gate"] is not None else None,
                    generative_gate=optional_maps["generative_gate"][batch_idx : batch_idx + 1] if optional_maps["generative_gate"] is not None else None,
                    abstention=optional_maps["abstention"][batch_idx : batch_idx + 1] if optional_maps["abstention"] is not None else None,
                    sample_id=sample_id,
                    batch_index=batch_index,
                    focal_plane_coordinates=focal_coords_one,
                    conflict_threshold=threshold,
                    confidence_threshold=confidence_threshold,
                    accepted_refinement=accepted_refinement,
                    focal_split_seed=focal_split_seed,
                    heldout_risk_before=heldout_risk_before,
                    heldout_risk_after=heldout_risk_after,
                    dataset_split=dataset_split,
                )
                if item is not None:
                    mined.append(item)

        if not mined:
            return {
                "mined_trace_items": 0,
                "mined_sample_count": 0,
                "accepted_conflict_count": 0,
                "accepted_invalid_count": 0,
                "accepted_reliable_count": 0,
                "buffer_size": len(self),
                "mined_conflict_mean": 0.0,
                "mined_invalid_mean": 0.0,
            }
        if self.manifest_path:
            self.save(self.manifest_path)
        return {
            "mined_trace_items": float(len(mined)),
            "mined_sample_count": float(len(mined)),
            "accepted_conflict_count": float(sum(1 for item in mined if item["verdict_type"] == "conflict")),
            "accepted_invalid_count": float(sum(1 for item in mined if item["verdict_type"] == "invalid")),
            "accepted_reliable_count": float(sum(1 for item in mined if item["verdict_type"] == "reliable_non_conflict")),
            "buffer_size": float(len(self)),
            "mined_conflict_mean": sum(item["conflict_mean"] for item in mined) / len(mined),
            "mined_invalid_mean": sum(item["invalid_mean"] for item in mined) / len(mined),
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            if self.metadata:
                handle.write(json.dumps(self.metadata, sort_keys=True) + "\n")
            for item in self.items:
                handle.write(json.dumps(item, sort_keys=True) + "\n")

    def load(self, path: str | os.PathLike[str]) -> None:
        loaded: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("record_type") == "metadata":
                        self.metadata = record
                    else:
                        loaded.append(record)
        self.items = loaded[-self.max_items :]

    @staticmethod
    def _crop_from_item(item: dict[str, Any]) -> tuple[int, int, int, int]:
        crop = item["crop"]
        if isinstance(crop, dict):
            return int(crop["y0"]), int(crop["y1"]), int(crop["x0"]), int(crop["x1"])
        y0, y1, x0, x1 = crop
        return int(y0), int(y1), int(x0), int(x1)

    @staticmethod
    def _shape_from_item(item: dict[str, Any]) -> tuple[int, int]:
        shape = item["source_shape"]
        if isinstance(shape, dict):
            return int(shape["height"]), int(shape["width"])
        height, width = shape
        return int(height), int(width)

    @staticmethod
    def _allowed_sample_ids(sample_ids: Any) -> set[str] | None:
        if sample_ids is None:
            return None
        if isinstance(sample_ids, str):
            return {sample_ids}
        if isinstance(sample_ids, (list, tuple)):
            return {str(item) for item in sample_ids}
        return None

    @staticmethod
    def _batch_index_for_item(item: dict[str, Any], sample_ids: Any, batch_size: int) -> int:
        if isinstance(sample_ids, (list, tuple)):
            item_ids = {str(item.get("sample_id")), str(item.get("sample_path"))}
            for index, sample_id in enumerate(sample_ids):
                if str(sample_id) in item_ids:
                    return min(index, batch_size - 1)
        return 0

    def replay_loss(
        self,
        *,
        predicted_support: torch.Tensor,
        predicted_invalid: torch.Tensor,
        predicted_conflict: torch.Tensor | None = None,
        predicted_final_depth: torch.Tensor | None = None,
        predicted_generated_depth: torch.Tensor | None = None,
        predicted_focal_gate: torch.Tensor | None = None,
        predicted_generative_gate: torch.Tensor | None = None,
        predicted_abstention: torch.Tensor | None = None,
        sample_ids: Any = None,
        max_items: int = 4,
    ) -> torch.Tensor | None:
        if not self.items:
            return None
        allowed_sample_ids = self._allowed_sample_ids(sample_ids)
        support_prediction = self._as_bchw(predicted_support).float().clamp(1e-6, 1.0 - 1e-6)
        invalid_prediction = self._as_bchw(predicted_invalid).float().clamp(1e-6, 1.0 - 1e-6)
        conflict_prediction = self._as_bchw(predicted_conflict).float().clamp(1e-6, 1.0 - 1e-6) if predicted_conflict is not None else None
        final_depth_prediction = self._as_bchw(predicted_final_depth).float() if predicted_final_depth is not None else None
        generated_depth_prediction = self._as_bchw(predicted_generated_depth).float() if predicted_generated_depth is not None else None
        focal_gate_prediction = self._as_bchw(predicted_focal_gate).float().clamp(1e-6, 1.0 - 1e-6) if predicted_focal_gate is not None else None
        generative_gate_prediction = self._as_bchw(predicted_generative_gate).float().clamp(1e-6, 1.0 - 1e-6) if predicted_generative_gate is not None else None
        abstention_prediction = self._as_bchw(predicted_abstention).float().clamp(1e-6, 1.0 - 1e-6) if predicted_abstention is not None else None
        losses: list[torch.Tensor] = []
        candidate_items = self.items
        if allowed_sample_ids is not None:
            candidate_items = [item for item in self.items if str(item.get("sample_id")) in allowed_sample_ids or str(item.get("sample_path")) in allowed_sample_ids]
        for item in candidate_items[-max_items:]:
            height, width = self._shape_from_item(item)
            y0, y1, x0, x1 = self._crop_from_item(item)
            pred_h, pred_w = invalid_prediction.shape[-2:]
            py0 = max(0, min(pred_h - 1, math.floor(y0 * pred_h / height)))
            py1 = max(py0 + 1, min(pred_h, math.ceil(y1 * pred_h / height)))
            px0 = max(0, min(pred_w - 1, math.floor(x0 * pred_w / width)))
            px1 = max(px0 + 1, min(pred_w, math.ceil(x1 * pred_w / width)))
            batch_index = self._batch_index_for_item(item, sample_ids, invalid_prediction.shape[0])
            invalid_patch = invalid_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
            support_patch = support_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
            invalid_target_value = float(item["invalid_target"])
            support_target_value = float(item["support_target"])
            invalid_target = torch.full_like(invalid_patch, invalid_target_value)
            support_target = torch.full_like(support_patch, support_target_value)
            losses.append(F.binary_cross_entropy(invalid_patch, invalid_target))
            losses.append(F.binary_cross_entropy(support_patch, support_target))
            if conflict_prediction is not None:
                conflict_patch = conflict_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
                conflict_target = torch.full_like(conflict_patch, float(item["conflict_target"]))
                losses.append(F.binary_cross_entropy(conflict_patch, conflict_target))

            verdict_type = str(item.get("verdict_type", ""))
            evidence_valid = invalid_target_value < 0.5
            is_conflict = verdict_type in {"conflict", "focus_discrepancy"} and evidence_valid
            is_invalid = verdict_type == "invalid" or invalid_target_value >= 0.5
            is_reliable = verdict_type == "reliable_non_conflict" and evidence_valid

            if abstention_prediction is not None:
                abstention_patch = abstention_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
                abstention_target_value = max(invalid_target_value, float(item.get("abstention_target", item.get("uncertainty_mean", invalid_target_value))))
                losses.append(F.binary_cross_entropy(abstention_patch, torch.full_like(abstention_patch, abstention_target_value)))

            if is_invalid:
                continue

            if is_conflict:
                # Conflict-only pseudo labels supervise reliability/abstention, not a depth correction.
                if abstention_prediction is not None:
                    abstention_patch = abstention_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
                    losses.append(F.binary_cross_entropy(abstention_patch, torch.ones_like(abstention_patch) * max(0.5, invalid_target_value)))

            if is_reliable:
                if final_depth_prediction is not None and "final_depth_target" in item:
                    final_patch = final_depth_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
                    losses.append(F.smooth_l1_loss(final_patch, torch.full_like(final_patch, float(item["final_depth_target"]))))
                if generated_depth_prediction is not None and "generated_depth_target" in item:
                    generated_patch = generated_depth_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
                    losses.append(F.smooth_l1_loss(generated_patch, torch.full_like(generated_patch, float(item["generated_depth_target"]))))
                if focal_gate_prediction is not None and "focal_gate_target" in item:
                    focal_gate_patch = focal_gate_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
                    losses.append(F.binary_cross_entropy(focal_gate_patch, torch.full_like(focal_gate_patch, float(item["focal_gate_target"]))))
                if generative_gate_prediction is not None and "generative_gate_target" in item:
                    generative_gate_patch = generative_gate_prediction[batch_index : batch_index + 1, :, py0:py1, px0:px1]
                    losses.append(F.binary_cross_entropy(generative_gate_patch, torch.full_like(generative_gate_patch, float(item["generative_gate_target"]))))
        if not losses:
            return None
        return torch.stack(losses).mean()


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    """Return a SHA256 digest for checkpoint identity comparisons."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
