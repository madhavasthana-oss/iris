#!/usr/bin/env python3
"""
Vision Transformer Training on ImageNet Subsets
Supports: ViT-Tiny/Small/Base/Large/Huge on ImageNet-1K/10K/21K subsets
Optimizers: AdamW, AdaBelief, Yogi, Adan, Sophia, Lion, IRIS
Uses HuggingFace transformers for flexible architecture construction

USAGE (set OPTIMIZER_NAME at the top, then):
    python vit_benchmark.py

OPTIMIZER INSTALL COMMANDS:
    pip install adabelief-pytorch          # AdaBelief
    pip install yogi                       # Yogi
    pip install adan                       # Adan (or use the standalone repo)
    # Sophia: clone https://github.com/liu-group/Sophia and place sophia.py locally
    # Lion:   included via optax or use the built-in class below
    # IRIS:   place iris.py locally or pip install from your source
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import time
import random
import numpy as np
from pathlib import Path
from typing import Optional
import datetime
import os
import shutil


# ============================================================================
# GLOBAL CONFIGURATION - MODIFY THESE
# ============================================================================

# Dataset & Model Configuration
DATASET = 'imagenet1k'          # 'imagenet1k' | 'imagenet10k' | 'imagenet21k'
MODEL_ARCH = 'vit_base_patch16_224'   # See MODEL_REGISTRY below

# Project settings
PROJECT_NAME = 'vit_imagenet_benchmark'
EXPERIMENT_NAME = 'baseline_adamw'

# Data paths
DATA_DIR = '/path/to/imagenet'
TRAIN_SUBDIR = 'train'
VAL_SUBDIR = 'val'

# Training hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 512
MAX_EPOCHS = 100
WARMUP_EPOCHS = 10

# --------------------------------------------------------------------------
# OPTIMIZER SELECTION
# --------------------------------------------------------------------------
# Options: 'adamw' | 'adabelief' | 'yogi' | 'adan' | 'sophia' | 'lion' | 'iris'
OPTIMIZER_NAME = 'adamw'

# Shared optimizer knobs
OPTIMIZER_BETAS  = (0.9, 0.999)
OPTIMIZER_EPS    = 1e-8
WEIGHT_DECAY     = 0.05           # Higher WD is typical for ViT

# AdaBelief extras
ADABELIEF_WEIGHT_DECOUPLE = True
ADABELIEF_RECTIFY         = False
ADABELIEF_AMSGRAD         = False

# Yogi extras
YOGI_EPS = 1e-3                   # Yogi prefers a larger eps

# Adan extras  (uses 3 betas)
ADAN_BETAS          = (0.98, 0.92, 0.99)
ADAN_MAX_GRAD_NORM  = 1.0
ADAN_NO_PROX        = False
ADAN_FOREACH        = False

# Sophia extras
SOPHIA_BETAS       = (0.965, 0.99)
SOPHIA_RHO         = 0.04         # Hessian clipping ratio
SOPHIA_GAMMA       = 2.0          # Hessian update interval (epochs)
SOPHIA_UPDATE_FREQ = 10           # Steps between Hessian estimates

# Lion extras
LION_BETAS      = (0.9, 0.99)
LION_WD         = 0.05

# IRIS extras
IRIS_SNR_THRESHOLD = 4.0          # rho: trust-region clipping. Higher = more conservative
IRIS_BETA_RES      = None         # None = standard mode. Set e.g. 0.92 to enable innovation residual
IRIS_AMSGRAD       = False        # Use max innovation variance for more stable updates

# Layer-wise learning rate decay
USE_LAYER_DECAY     = True
LAYER_DECAY_RATE    = 0.75

# Data augmentation
USE_RANDAUGMENT          = True
RANDAUGMENT_NUM_OPS      = 2
RANDAUGMENT_MAGNITUDE    = 9
USE_RANDOM_ERASING       = True
RANDOM_ERASING_PROB      = 0.25
LABEL_SMOOTHING          = 0.1

# Reproducibility
SEED = 42

# System
NUM_WORKERS        = 8
PERSISTENT_WORKERS = True

# Multi-GPU
STRATEGY         = 'ddp'
SYNC_BATCHNORM   = True

# WandB
FINISH_PREVIOUS_RUN = True
LOG_EVERY_N_STEPS   = 100

# Training
GRADIENT_CLIP_VAL        = 1.0
PRECISION                = '16-mixed'   # '16-mixed' | '32' | 'bf16-mixed'
ACCUMULATE_GRAD_BATCHES  = 1

# Checkpoints
SAVE_TOP_K          = 3
CHECKPOINT_MONITOR  = 'val/top1_acc'
CHECKPOINT_MODE     = 'max'


# ============================================================================
# DATASET & MODEL REGISTRY
# ============================================================================

DATASET_CONFIG = {
    'imagenet1k': {
        'num_classes': 1000,
        'subset_size': None,
        'mean': [0.485, 0.456, 0.406],
        'std':  [0.229, 0.224, 0.225],
    },
    'imagenet10k': {
        'num_classes': 10000,
        'subset_size': 10000,
        'mean': [0.485, 0.456, 0.406],
        'std':  [0.229, 0.224, 0.225],
    },
    'imagenet21k': {
        'num_classes': 21841,
        'subset_size': None,
        'mean': [0.485, 0.456, 0.406],
        'std':  [0.229, 0.224, 0.225],
    }
}

# Format: 'name': (size_tag, hidden, layers, heads, intermediate)
MODEL_REGISTRY = {
    'vit_tiny_patch16_224':  ('tiny',   192,  12,  3,  768),
    'vit_tiny_patch16_384':  ('tiny',   192,  12,  3,  768),
    'vit_small_patch16_224': ('small',  384,  12,  6, 1536),
    'vit_small_patch16_384': ('small',  384,  12,  6, 1536),
    'vit_small_patch32_224': ('small',  384,  12,  6, 1536),
    'vit_base_patch16_224':  ('base',   768,  12, 12, 3072),
    'vit_base_patch16_384':  ('base',   768,  12, 12, 3072),
    'vit_base_patch32_224':  ('base',   768,  12, 12, 3072),
    'vit_base_patch32_384':  ('base',   768,  12, 12, 3072),
    'vit_large_patch16_224': ('large', 1024,  24, 16, 4096),
    'vit_large_patch16_384': ('large', 1024,  24, 16, 4096),
    'vit_large_patch32_224': ('large', 1024,  24, 16, 4096),
    'vit_large_patch32_384': ('large', 1024,  24, 16, 4096),
    'vit_huge_patch14_224':  ('huge',  1280,  32, 16, 5120),
    'vit_huge_patch16_224':  ('huge',  1280,  32, 16, 5120),
}


# ============================================================================
# LION OPTIMIZER (self-contained fallback -- no external dep needed)
# ============================================================================

class Lion(torch.optim.Optimizer):
    """
    Lion optimizer (EvoLved Sign Momentum).
    Reference: https://arxiv.org/abs/2302.06675

    Uses sign of the interpolation between momentum and gradient for the
    update, making every step uniform-magnitude. Typically requires a
    smaller learning rate and higher weight decay than Adam-family optimizers.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]

                # State initialisation
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])

                # Update: sign of interpolation between momentum and gradient
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign(), alpha=-group['lr'])

                # Decay the momentum running average
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[dataset_name]


def get_model_config(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def extract_image_size_and_patch(model_name):
    parts = model_name.split('_')
    patch_size = 16
    for part in parts:
        if 'patch' in part:
            patch_size = int(part.replace('patch', ''))
    image_size = 224
    if parts[-1].isdigit():
        image_size = int(parts[-1])
    return image_size, patch_size


def create_vit_model(model_name, num_classes):
    """
    Create ViT via HuggingFace; fall back to torchvision for the four
    standard sizes if transformers is not installed.
    """
    try:
        from transformers import ViTConfig, ViTForImageClassification

        size_name, hidden_size, num_layers, num_heads, intermediate_size = get_model_config(model_name)
        image_size, patch_size = extract_image_size_and_patch(model_name)

        config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            image_size=image_size,
            patch_size=patch_size,
            num_labels=num_classes,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        model = ViTForImageClassification(config)
        print(f"✓ Created {model_name} using HuggingFace transformers")
        return model, True

    except ImportError:
        print("! HuggingFace transformers not available, falling back to torchvision")
        from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

        torchvision_models = {
            'vit_base_patch16_224':  vit_b_16,
            'vit_base_patch32_224':  vit_b_32,
            'vit_large_patch16_224': vit_l_16,
            'vit_large_patch32_224': vit_l_32,
        }
        if model_name not in torchvision_models:
            raise ValueError(
                f"Model {model_name} not available in torchvision. "
                "Install transformers: pip install transformers"
            )
        model = torchvision_models[model_name](weights=None)
        model.heads = nn.Linear(model.heads.head.in_features, num_classes)
        return model, False


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


def cleanup_wandb():
    try:
        wandb.finish()
    except:
        pass
    wandb_dir = './wandb'
    if os.path.exists(wandb_dir):
        shutil.rmtree(wandb_dir)
        print("✓ Cleaned up previous WandB runs")


# ============================================================================
# PYTORCH LIGHTNING MODULE
# ============================================================================

class ImageNetViT(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 optimizer_class,
                 optimizer_kwargs: dict,
                 lr: float = 0.001,
                 max_epochs: int = 100,
                 warmup_epochs: int = 10,
                 batch_size: int = 512,
                 label_smoothing: float = 0.1,
                 use_layer_decay: bool = True,
                 layer_decay_rate: float = 0.75,
                 optimizer_name: str = 'adamw'):
        super().__init__()
        self.save_hyperparameters(ignore=['optimizer_class'])

        self.optimizer_class   = optimizer_class
        self.optimizer_kwargs  = optimizer_kwargs
        self.optimizer_name    = optimizer_name
        self.lr                = lr
        self.max_epochs        = max_epochs
        self.warmup_epochs     = warmup_epochs
        self.use_layer_decay   = use_layer_decay
        self.layer_decay_rate  = layer_decay_rate

        self.model, self.is_hf_model = create_vit_model(model_name, num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.epoch_start_time = None

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x):
        if self.is_hf_model:
            return self.model(x).logits
        return self.model(x)

    # ------------------------------------------------------------------
    # steps
    # ------------------------------------------------------------------
    def _accuracy(self, logits, y):
        _, pred = logits.topk(5, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(y.view(1, -1).expand_as(pred))
        top1    = correct[:1].float().sum() / y.size(0) * 100.0
        top5    = correct[:5].float().sum() / y.size(0) * 100.0
        return top1, top5

    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        top1, top5 = self._accuracy(logits, y)

        self.log('train/loss',     loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/top1_acc', top1,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/top5_acc', top5,  on_step=False, on_epoch=True,                sync_dist=True)

        # -- Sophia needs the loss for Hessian estimation --------------
        if self.optimizer_name == 'sophia':
            return {'loss': loss, 'logits': logits, 'labels': y}
        return loss

    def training_step_end(self, step_output):
        """
        Called after training_step on every GPU.  Used to feed the Hessian
        closure into Sophia at the configured update frequency.
        """
        if self.optimizer_name != 'sophia':
            # For non-Sophia optimizers step_output is already the scalar loss;
            # just return it untouched.
            return step_output

        # Unpack Sophia's richer dict
        loss    = step_output['loss']
        logits  = step_output['logits']
        labels  = step_output['labels']

        optimizer = self.optimizers()
        step_idx  = self.global_step

        # Only compute Hessian every SOPHIA_UPDATE_FREQ steps
        if step_idx % SOPHIA_UPDATE_FREQ == 0:
            def closure():
                # Re-derive a fresh loss so autograd can build the graph
                fresh_logits = self(step_output.get('x', logits))  # fallback
                return self.criterion(fresh_logits, labels)

            # Sophia's .step() accepts an optional closure for Hessian
            # The underlying Sophia implementation handles the Gauss-Newton
            # diagonal approximation internally when closure is provided.
            optimizer.step(closure=closure)
        else:
            optimizer.step()

        # Return the plain scalar for PL bookkeeping
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        top1, top5 = self._accuracy(logits, y)

        self.log('val/loss',     loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/top1_acc', top1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/top5_acc', top5, on_step=False, on_epoch=True,                sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y   = batch
        logits = self(x)
        loss   = self.criterion(logits, y)
        top1, top5 = self._accuracy(logits, y)

        self.log('test/loss',     loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/top1_acc', top1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/top5_acc', top5, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # optimizer & scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # -- Build parameter groups with optional layer-wise LR decay --
        if self.use_layer_decay and self.is_hf_model:
            param_groups = self._get_layer_wise_param_groups_hf()
        elif self.use_layer_decay:
            param_groups = self._get_layer_wise_param_groups_torchvision()
        else:
            param_groups = [{'params': self.parameters(), 'lr': self.lr}]

        # -- Instantiate optimizer -------------------------------------
        # Adan does NOT accept a top-level `lr` kwarg when param_groups
        # already carry their own `lr`; we handle it the same way for all
        # optimizers by injecting lr into every group that lacks it.
        for g in param_groups:
            g.setdefault('lr', self.lr)

        # Sophia is special: it must NOT receive `lr` as a keyword arg when
        # groups already carry it, same as Adan.  We pass only the extra
        # kwargs, relying on per-group lr.
        optimizer = self.optimizer_class(param_groups, **self.optimizer_kwargs)

        # -- Cosine schedule with linear warmup ------------------------
        def warmup_cosine_schedule(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
        }

    # ------------------------------------------------------------------
    # layer-wise LR decay helpers
    # ------------------------------------------------------------------
    def _get_layer_wise_param_groups_hf(self):
        param_groups = []
        num_layers   = len(self.model.vit.encoder.layer)

        # Embeddings get the most aggressive decay
        param_groups.append({
            'params': list(self.model.vit.embeddings.parameters()),
            'lr': self.lr * (self.layer_decay_rate ** num_layers),
            'name': 'embeddings'
        })
        # Each transformer layer
        for i, layer in enumerate(self.model.vit.encoder.layer):
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': self.lr * (self.layer_decay_rate ** (num_layers - i - 1)),
                'name': f'layer_{i}'
            })
        # Head (no decay)
        param_groups.append({
            'params': (list(self.model.vit.layernorm.parameters()) +
                       list(self.model.classifier.parameters())),
            'lr': self.lr,
            'name': 'head'
        })
        return param_groups

    def _get_layer_wise_param_groups_torchvision(self):
        param_groups = []
        num_layers   = len(self.model.encoder.layers)

        param_groups.append({
            'params': (list(self.model.conv_proj.parameters()) +
                       [self.model.encoder.pos_embedding]),
            'lr': self.lr * (self.layer_decay_rate ** num_layers),
            'name': 'embeddings'
        })
        for i, layer in enumerate(self.model.encoder.layers):
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': self.lr * (self.layer_decay_rate ** (num_layers - i - 1)),
                'name': f'layer_{i}'
            })
        param_groups.append({
            'params': (list(self.model.encoder.ln.parameters()) +
                       list(self.model.heads.parameters())),
            'lr': self.lr,
            'name': 'head'
        })
        return param_groups

    # ------------------------------------------------------------------
    # timing hooks
    # ------------------------------------------------------------------
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.epoch_start_time is not None:
            self.log('timing/epoch_time', time.time() - self.epoch_start_time,
                     on_epoch=True, sync_dist=True)


# ============================================================================
# DATA MODULE  (unchanged from original)
# ============================================================================

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 dataset_name: str = 'imagenet1k',
                 batch_size: int = 512,
                 num_workers: int = 8,
                 image_size: int = 224,
                 persistent_workers: bool = True,
                 use_randaugment: bool = True,
                 randaugment_num_ops: int = 2,
                 randaugment_magnitude: int = 9,
                 use_random_erasing: bool = True,
                 random_erasing_prob: float = 0.25):
        super().__init__()
        self.data_dir              = Path(data_dir)
        self.batch_size            = batch_size
        self.num_workers           = num_workers
        self.image_size            = image_size
        self.persistent_workers    = persistent_workers
        self.use_randaugment       = use_randaugment
        self.randaugment_num_ops   = randaugment_num_ops
        self.randaugment_magnitude = randaugment_magnitude
        self.use_random_erasing    = use_random_erasing
        self.random_erasing_prob   = random_erasing_prob

        self.config      = get_dataset_config(dataset_name)
        self.subset_size = self.config['subset_size']
        self.normalize   = transforms.Normalize(mean=self.config['mean'], std=self.config['std'])

    def setup(self, stage=None):
        train_transform_list = [
            transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if self.use_randaugment:
            train_transform_list.append(
                transforms.RandAugment(num_ops=self.randaugment_num_ops,
                                       magnitude=self.randaugment_magnitude)
            )
        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(self.normalize)
        if self.use_random_erasing:
            train_transform_list.append(transforms.RandomErasing(p=self.random_erasing_prob))

        train_transform = transforms.Compose(train_transform_list)
        val_transform   = transforms.Compose([
            transforms.Resize(int(self.image_size * 256 / 224)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            self.normalize
        ])

        if stage == 'fit' or stage is None:
            full_train = torchvision.datasets.ImageFolder(
                root=self.data_dir / TRAIN_SUBDIR, transform=train_transform)
            full_val   = torchvision.datasets.ImageFolder(
                root=self.data_dir / VAL_SUBDIR,   transform=val_transform)

            if self.subset_size:
                train_idx = [i for i, (_, l) in enumerate(full_train.samples) if l < self.subset_size]
                val_idx   = [i for i, (_, l) in enumerate(full_val.samples)   if l < self.subset_size]
                self.train_dataset = Subset(full_train, train_idx)
                self.val_dataset   = Subset(full_val,   val_idx)
                print(f"✓ Subset: {len(train_idx)} train, {len(val_idx)} val")
            else:
                self.train_dataset = full_train
                self.val_dataset   = full_val
                print(f"✓ Full dataset: {len(full_train)} train, {len(full_val)} val")

        if stage == 'test' or stage is None:
            full_test = torchvision.datasets.ImageFolder(
                root=self.data_dir / VAL_SUBDIR, transform=val_transform)
            if self.subset_size:
                test_idx = [i for i, (_, l) in enumerate(full_test.samples) if l < self.subset_size]
                self.test_dataset = Subset(full_test, test_idx)
            else:
                self.test_dataset = full_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.persistent_workers if self.num_workers > 0 else False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================

def build_optimizer_class_and_kwargs(optimizer_name: str):
    """
    Returns (optimizer_class, kwargs_dict).

    kwargs does NOT include `lr` -- that is injected per-group by
    configure_optimizers so layer-wise decay works correctly for every
    optimizer.
    """

    if optimizer_name == 'adamw':
        return torch.optim.AdamW, {
            'betas':        OPTIMIZER_BETAS,
            'eps':          OPTIMIZER_EPS,
            'weight_decay': WEIGHT_DECAY,
        }

    # -- AdaBelief -----------------------------------------------------
    elif optimizer_name == 'adabelief':
        try:
            from adabelief_pytorch import AdaBelief
        except ImportError:
            raise ImportError(
                "AdaBelief not found. Install: pip install adabelief-pytorch"
            )
        return AdaBelief, {
            'betas':            OPTIMIZER_BETAS,
            'eps':              OPTIMIZER_EPS,
            'weight_decay':     WEIGHT_DECAY,
            'weight_decouple':  ADABELIEF_WEIGHT_DECOUPLE,
            'rectify':          ADABELIEF_RECTIFY,
            'amsgrad':          ADABELIEF_AMSGRAD,
        }

    # -- Yogi ----------------------------------------------------------
    elif optimizer_name == 'yogi':
        try:
            from yogi import Yogi
        except ImportError:
            raise ImportError(
                "Yogi not found. Install: pip install yogi"
            )
        return Yogi, {
            'betas':        OPTIMIZER_BETAS,
            'eps':          YOGI_EPS,
            'weight_decay': WEIGHT_DECAY,
        }

    # -- Adan ----------------------------------------------------------
    elif optimizer_name == 'adan':
        try:
            from adan import Adan
        except ImportError:
            raise ImportError(
                "Adan not found. Install: pip install adan"
            )
        return Adan, {
            'betas':          ADAN_BETAS,
            'eps':            OPTIMIZER_EPS,
            'weight_decay':   WEIGHT_DECAY,
            'max_grad_norm':  ADAN_MAX_GRAD_NORM,
            'no_prox':        ADAN_NO_PROX,
            'foreach':        ADAN_FOREACH,
        }

    # -- Sophia --------------------------------------------------------
    elif optimizer_name == 'sophia':
        try:
            from sophia import SophiaG as Sophia   # liu-group/Sophia repo
        except ImportError:
            try:
                from sophia import Sophia          # alternate packaging
            except ImportError:
                raise ImportError(
                    "Sophia not found. Clone https://github.com/liu-group/Sophia "
                    "and place sophia.py in your working directory, or install "
                    "from your local copy."
                )
        return Sophia, {
            'betas':        SOPHIA_BETAS,
            'eps':          OPTIMIZER_EPS,
            'weight_decay': WEIGHT_DECAY,
            'rho':          SOPHIA_RHO,
            'gamma':        SOPHIA_GAMMA,
        }

    # -- Lion ----------------------------------------------------------
    elif optimizer_name == 'lion':
        # Use the self-contained Lion class defined above -- no external dep.
        return Lion, {
            'betas':        LION_BETAS,
            'weight_decay': LION_WD,
        }

    # -- IRIS ----------------------------------------------------------
    elif optimizer_name == 'iris':
        try:
            from iris import IRIS
        except ImportError:
            raise ImportError(
                "IRIS not found. Place iris.py locally or install from your source."
            )

        kwargs = {
            'betas':         OPTIMIZER_BETAS,
            'eps':           OPTIMIZER_EPS,
            'weight_decay':  WEIGHT_DECAY,
            'snr_threshold': IRIS_SNR_THRESHOLD,
            'amsgrad':       IRIS_AMSGRAD,
        }
        # beta_res is optional: None = standard mode, float enables innovation residual
        if IRIS_BETA_RES is not None:
            kwargs['beta_res'] = IRIS_BETA_RES

        return IRIS, kwargs

    else:
        valid = ['adamw', 'adabelief', 'yogi', 'adan', 'sophia', 'lion', 'iris']
        raise ValueError(f"Unknown optimizer: '{optimizer_name}'. Choose from {valid}")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_vit_imagenet(optimizer_class, optimizer_kwargs: dict, config: dict = None):
    """Train Vision Transformer on ImageNet with the specified optimizer."""

    if config is None:
        image_size, _  = extract_image_size_and_patch(MODEL_ARCH)
        dataset_config = get_dataset_config(DATASET)
        config = {
            'dataset':         DATASET,
            'model_arch':      MODEL_ARCH,
            'num_classes':     dataset_config['num_classes'],
            'lr':              LEARNING_RATE,
            'batch_size':      BATCH_SIZE,
            'max_epochs':      MAX_EPOCHS,
            'warmup_epochs':   WARMUP_EPOCHS,
            'seed':            SEED,
            'num_workers':     NUM_WORKERS,
            'data_dir':        DATA_DIR,
            'image_size':      image_size,
            'label_smoothing': LABEL_SMOOTHING,
            'use_layer_decay': USE_LAYER_DECAY,
            'layer_decay_rate':LAYER_DECAY_RATE,
        }

    set_seed(config['seed'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = (f"{EXPERIMENT_NAME}_{config['model_arch']}_{config['dataset']}_"
                 f"lr{config['lr']}_bs{config['batch_size']}_"
                 f"warmup{config['warmup_epochs']}_{timestamp}")

    wandb_logger = WandbLogger(
        project=PROJECT_NAME, name=run_name,
        config={**config, 'optimizer': OPTIMIZER_NAME, **optimizer_kwargs},
        reinit=FINISH_PREVIOUS_RUN, force=True
    )

    data_module = ImageNetDataModule(
        data_dir=config['data_dir'], dataset_name=config['dataset'],
        batch_size=config['batch_size'], num_workers=config['num_workers'],
        image_size=config['image_size'], persistent_workers=PERSISTENT_WORKERS,
        use_randaugment=USE_RANDAUGMENT, randaugment_num_ops=RANDAUGMENT_NUM_OPS,
        randaugment_magnitude=RANDAUGMENT_MAGNITUDE,
        use_random_erasing=USE_RANDOM_ERASING, random_erasing_prob=RANDOM_ERASING_PROB
    )

    model = ImageNetViT(
        model_name=config['model_arch'], num_classes=config['num_classes'],
        optimizer_class=optimizer_class, optimizer_kwargs=optimizer_kwargs,
        lr=config['lr'], max_epochs=config['max_epochs'],
        warmup_epochs=config['warmup_epochs'], batch_size=config['batch_size'],
        label_smoothing=config['label_smoothing'],
        use_layer_decay=config['use_layer_decay'],
        layer_decay_rate=config['layer_decay_rate'],
        optimizer_name=OPTIMIZER_NAME
    )

    checkpoint_cb = ModelCheckpoint(
        monitor=CHECKPOINT_MONITOR, mode=CHECKPOINT_MODE, save_top_k=SAVE_TOP_K,
        filename=f'{config["model_arch"]}-{{epoch:02d}}-{{val/top1_acc:.2f}}'
    )
    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

    num_gpus = torch.cuda.device_count()
    strategy = STRATEGY if num_gpus > 1 else 'auto'
    sync_bn  = SYNC_BATCHNORM if num_gpus > 1 else False
    if num_gpus > 1:
        print(f"Multi-GPU training with {num_gpus} GPUs using {strategy}")

    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto', devices='auto', strategy=strategy,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor_cb],
        deterministic=True,
        precision=PRECISION if torch.cuda.is_available() else 32,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        sync_batchnorm=sync_bn,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    try:
        trainer.fit(model, data_module)
        trainer.test(model, data_module)

        best_score = (checkpoint_cb.best_model_score.item()
                      if checkpoint_cb.best_model_score is not None else None)
        if best_score is not None:
            wandb_logger.experiment.summary['best_val_top1_acc'] = best_score

        return {'best_val_top1_acc': best_score, 'best_model_path': checkpoint_cb.best_model_path}

    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise
    finally:
        wandb.finish()


# ============================================================================
# MAIN
# ============================================================================

def main():
    cleanup_wandb()

    # -- Pretty-print config -------------------------------------------
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Project:    {PROJECT_NAME}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Dataset:    {DATASET.upper()}")
    print(f"Model:      {MODEL_ARCH}")

    try:
        size_name, hidden_size, num_layers, num_heads, intermediate_size = get_model_config(MODEL_ARCH)
        image_size, patch_size = extract_image_size_and_patch(MODEL_ARCH)
        print(f"\n  Size: {size_name} | Hidden: {hidden_size} | Layers: {num_layers} | "
              f"Heads: {num_heads} | Image: {image_size} | Patch: {patch_size}")
    except:
        pass

    print(f"\nOptimizer:      {OPTIMIZER_NAME.upper()}")
    print(f"Learning Rate:  {LEARNING_RATE}")
    print(f"Batch Size:     {BATCH_SIZE}")
    print(f"Max Epochs:     {MAX_EPOCHS}")
    print(f"Warmup Epochs:  {WARMUP_EPOCHS}")
    print(f"Weight Decay:   {WEIGHT_DECAY}")
    print(f"Label Smoothing:{LABEL_SMOOTHING}")
    print(f"Seed:           {SEED}")

    if USE_LAYER_DECAY:
        print(f"\nLayer-wise LR Decay: rate={LAYER_DECAY_RATE}")

    # Per-optimizer detail block
    if OPTIMIZER_NAME == 'adabelief':
        print(f"\nAdaBelief: weight_decouple={ADABELIEF_WEIGHT_DECOUPLE}, "
              f"rectify={ADABELIEF_RECTIFY}, amsgrad={ADABELIEF_AMSGRAD}")
    elif OPTIMIZER_NAME == 'yogi':
        print(f"\nYogi: eps={YOGI_EPS}")
    elif OPTIMIZER_NAME == 'adan':
        print(f"\nAdan: betas={ADAN_BETAS}, max_grad_norm={ADAN_MAX_GRAD_NORM}, "
              f"no_prox={ADAN_NO_PROX}")
    elif OPTIMIZER_NAME == 'sophia':
        print(f"\nSophia: betas={SOPHIA_BETAS}, rho={SOPHIA_RHO}, "
              f"gamma={SOPHIA_GAMMA}, hessian_update_freq={SOPHIA_UPDATE_FREQ}")
    elif OPTIMIZER_NAME == 'lion':
        print(f"\nLion: betas={LION_BETAS}, weight_decay={LION_WD}")
    elif OPTIMIZER_NAME == 'iris':
        print(f"\nIRIS: snr_threshold={IRIS_SNR_THRESHOLD}, "
              f"beta_res={IRIS_BETA_RES}, amsgrad={IRIS_AMSGRAD}")

    print(f"\nAugmentation: RandAugment={USE_RANDAUGMENT}, RandomErasing={USE_RANDOM_ERASING}")
    print(f"System:       GPUs={torch.cuda.device_count()}, Workers={NUM_WORKERS}, "
          f"Precision={PRECISION}, GradClip={GRADIENT_CLIP_VAL}")
    print("=" * 80)

    # -- Build optimizer and kick off training -------------------------
    optimizer_class, optimizer_kwargs = build_optimizer_class_and_kwargs(OPTIMIZER_NAME)

    image_size, _  = extract_image_size_and_patch(MODEL_ARCH)
    dataset_config = get_dataset_config(DATASET)

    config = {
        'dataset':          DATASET,
        'model_arch':       MODEL_ARCH,
        'num_classes':      dataset_config['num_classes'],
        'lr':               LEARNING_RATE,
        'batch_size':       BATCH_SIZE,
        'max_epochs':       MAX_EPOCHS,
        'warmup_epochs':    WARMUP_EPOCHS,
        'seed':             SEED,
        'num_workers':      NUM_WORKERS,
        'data_dir':         DATA_DIR,
        'image_size':       image_size,
        'label_smoothing':  LABEL_SMOOTHING,
        'use_layer_decay':  USE_LAYER_DECAY,
        'layer_decay_rate': LAYER_DECAY_RATE,
    }

    print(f"\nStarting {MODEL_ARCH} on {DATASET.upper()} with {OPTIMIZER_NAME.upper()}...")
    results = train_vit_imagenet(optimizer_class, optimizer_kwargs, config)

    print(f"\n✓ Best val Top-1: {results['best_val_top1_acc']:.2f}%")
    print(f"Best model:    {results['best_model_path']}")


if __name__ == '__main__':
    main()