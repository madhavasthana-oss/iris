"""
Publication-Worthy LLM Training Benchmark Suite
Comprehensive training framework with reproducible seeding, multiple architectures, 
extensive optimizer comparison, checkpointing, and W&B artifact logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import random
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import wandb
from dataclasses import dataclass, asdict, field
import math
import argparse
from datasets import load_dataset
from transformers import (
    GPTNeoForCausalLM, GPTNeoConfig,
    GPT2LMHeadModel, GPT2Config,
    BertForMaskedLM, BertConfig,
    LlamaForCausalLM, LlamaConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask
)
from torch.optim import AdamW

# Import custom optimizers
from iris import IRIS
from adan import Adan
from adabelief_pytorch import AdaBelief

# Disable tokenizer parallelism to avoid deadlocks with DataLoader multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# SEEDING AND REPRODUCIBILITY


def set_seed(seed: int):
    """Set all random seeds for complete reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set worker seed for DataLoader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    return seed_worker



# LEARNING RATE SCHEDULER
class CosineDecayWithWarmup:
    """Cosine decay learning rate scheduler with linear warmup"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 lr_init: float, lr_min: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self) -> float:
        """Calculate current learning rate"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.lr_init * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.lr_min + (self.lr_init - self.lr_min) * cosine_decay
    
    def state_dict(self):
        return {
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'lr_init': self.lr_init,
            'lr_min': self.lr_min,
            'current_step': self.current_step
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.lr_init = state_dict['lr_init']
        self.lr_min = state_dict['lr_min']
        self.current_step = state_dict['current_step']


# CONFIGURATION MANAGEMENT
@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    # Model config
    model_type: str  # 'gpt2', 'gptneo', 'bert', 'llama'
    model_size: str  # 'nano', 'small', 'medium', 'large'
    
    # Training config
    optimizer: str
    batch_size: int
    num_epochs: int
    max_steps: int
    eval_steps: int
    
    # Optimizer config path
    optimizer_config_path: str
    
    # Learning rate schedule
    warmup_steps: int
    lr_init: float
    lr_min: float
    
    # Seeds
    seeds: Union[int, List[int]]
    
    # Dataset config
    dataset_name: str
    max_seq_length: int
    
    # All fields below have defaults
    gradient_accumulation_steps: int = 1
    dataset_config: Optional[str] = None
    streaming: bool = False
    num_workers: int = 4
    
    # Logging
    wandb_project: str = "llm-optimizer-benchmark"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 10
    save_steps: int = 1000
    output_dir: str = "./benchmark_results"
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    upload_to_wandb: bool = True  # Upload checkpoints as W&B artifacts
    keep_last_n_checkpoints: int = 3  # Keep only N most recent checkpoints locally
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Convert single seed to list"""
        if isinstance(self.seeds, int):
            self.seeds = [self.seeds]
    
    def save(self, path: str):
        """Save config to JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON"""
        with open(path, 'r') as f:
            return cls(**json.load(f))


# DATASET FACTORY
class DatasetFactory:
    """Factory for loading and preprocessing datasets"""
    
    # Dataset loading configurations
    DATASET_CONFIGS = {
        'pile': {
            'path': 'monology/pile-uncopyrighted',
            'config': None,
            'text_column': 'text',
            'streaming': True
        },
        'c4': {
            'path': 'c4',
            'config': 'en',
            'text_column': 'text',
            'streaming': True
        },
        'redpajama': {
            'path': 'togethercomputer/RedPajama-Data-1T',
            'config': 'default',
            'text_column': 'text',
            'streaming': True
        },
        'fineweb': {
            'path': 'HuggingFaceFW/fineweb',
            'config': None,
            'text_column': 'text',
            'streaming': True
        },
        'wikitext-103': {
            'path': 'wikitext',
            'config': 'wikitext-103-v1',
            'text_column': 'text',
            'streaming': False
        },
        'wikitext-2': {
            'path': 'wikitext',
            'config': 'wikitext-2-v1',
            'text_column': 'text',
            'streaming': False
        },
        'the-stack': {
            'path': 'bigcode/the-stack-dedup',
            'config': None,
            'text_column': 'content',
            'streaming': True
        },
        'wikipedia': {
            'path': 'wikipedia',
            'config': '20220301.en',
            'text_column': 'text',
            'streaming': False
        },
        'bookcorpus': {
            'path': 'bookcorpus',
            'config': None,
            'text_column': 'text',
            'streaming': False
        },
        'oscar': {
            'path': 'oscar',
            'config': 'unshuffled_deduplicated_en',
            'text_column': 'text',
            'streaming': True
        },
        'slimpajama': {
            'path': 'cerebras/SlimPajama-627B',
            'config': None,
            'text_column': 'text',
            'streaming': True
        },
        'dolma': {
            'path': 'allenai/dolma',
            'config': None,
            'text_column': 'text',
            'streaming': True
        }
    }
    
    @classmethod
    def load_dataset(cls, dataset_name: str, split: str = 'train', 
                    streaming: bool = False, seed: int = 42):
        """Load dataset from HuggingFace"""
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASET_CONFIGS.keys())}")
        
        config = cls.DATASET_CONFIGS[dataset_name]
        
        dataset = load_dataset(
            config['path'],
            config['config'],
            split=split,
            streaming=streaming or config['streaming']
        )
        
        # Shuffle with seed for reproducibility
        if hasattr(dataset, 'shuffle'):
            dataset = dataset.shuffle(seed=seed)
        
        return dataset, config['text_column']
    
    @classmethod
    def tokenize_function(cls, examples, tokenizer, text_column: str, max_length: int):
        """Tokenize text data"""
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_special_tokens_mask=True
        )
    
    @classmethod
    def prepare_dataset(cls, dataset_name: str, tokenizer, max_length: int,
                       split: str = 'train', streaming: bool = False, 
                       seed: int = 42, remove_columns: bool = True):
        """Load and tokenize dataset"""
        dataset, text_column = cls.load_dataset(dataset_name, split, streaming, seed)
        
        # Tokenize
        tokenized = dataset.map(
            lambda x: cls.tokenize_function(x, tokenizer, text_column, max_length),
            batched=True,
            remove_columns=[text_column] if remove_columns and not streaming else None
        )
        
        return tokenized

# MODEL FACTORY
class ModelFactory:
    """Factory for creating transformer models of different sizes"""
    
    MODEL_CONFIGS = {
        'gpt2': {
            'nano': GPT2Config(
                vocab_size=50257, 
                n_positions=512, 
                n_embd=128, 
                n_layer=2, 
                n_head=2,
                n_inner=512
            ),
            'small': GPT2Config(
                vocab_size=50257, 
                n_positions=1024, 
                n_embd=768, 
                n_layer=12, 
                n_head=12,
                n_inner=3072
            ),
            'medium': GPT2Config(
                vocab_size=50257, 
                n_positions=1024, 
                n_embd=1024, 
                n_layer=24, 
                n_head=16,
                n_inner=4096
            ),
            'large': GPT2Config(
                vocab_size=50257, 
                n_positions=1024, 
                n_embd=1280, 
                n_layer=36, 
                n_head=20,
                n_inner=5120
            ),
        },
        'gptneo': {
            'nano': GPTNeoConfig(
                vocab_size=50257, 
                max_position_embeddings=512, 
                hidden_size=128, 
                num_layers=2, 
                num_heads=2,
                intermediate_size=512,
                attention_types=[[["global", "local"], 1]]  # 2 layers
            ),
            'small': GPTNeoConfig(
                vocab_size=50257, 
                max_position_embeddings=2048, 
                hidden_size=768, 
                num_layers=12, 
                num_heads=12,
                intermediate_size=3072,
                attention_types=[[["global", "local"], 6]]  # 12 layers
            ),
            'medium': GPTNeoConfig(
                vocab_size=50257, 
                max_position_embeddings=2048, 
                hidden_size=1024, 
                num_layers=24, 
                num_heads=16,
                intermediate_size=4096,
                attention_types=[[["global", "local"], 12]]  # 24 layers
            ),
            'large': GPTNeoConfig(
                vocab_size=50257, 
                max_position_embeddings=2048, 
                hidden_size=2048, 
                num_layers=32, 
                num_heads=16,
                intermediate_size=8192,
                attention_types=[[["global", "local"], 16]]  # 32 layers
            ),
        },
        'bert': {
            'nano': BertConfig(
                vocab_size=30522, 
                max_position_embeddings=512, 
                hidden_size=128, 
                num_hidden_layers=2, 
                num_attention_heads=2,
                intermediate_size=512
            ),
            'small': BertConfig(
                vocab_size=30522, 
                max_position_embeddings=512, 
                hidden_size=768, 
                num_hidden_layers=12, 
                num_attention_heads=12,
                intermediate_size=3072
            ),
            'medium': BertConfig(
                vocab_size=30522, 
                max_position_embeddings=512, 
                hidden_size=1024, 
                num_hidden_layers=24, 
                num_attention_heads=16,
                intermediate_size=4096
            ),
            'large': BertConfig(
                vocab_size=30522, 
                max_position_embeddings=512, 
                hidden_size=1280, 
                num_hidden_layers=36, 
                num_attention_heads=20,
                intermediate_size=5120
            ),
        },
        'llama': {
            'nano': LlamaConfig(
                vocab_size=32000, 
                max_position_embeddings=512, 
                hidden_size=128, 
                intermediate_size=512,
                num_hidden_layers=2, 
                num_attention_heads=2,
                num_key_value_heads=2
            ),
            'small': LlamaConfig(
                vocab_size=32000, 
                max_position_embeddings=2048, 
                hidden_size=2048, 
                intermediate_size=5632,
                num_hidden_layers=16, 
                num_attention_heads=16,
                num_key_value_heads=16
            ),
            'medium': LlamaConfig(
                vocab_size=32000, 
                max_position_embeddings=4096, 
                hidden_size=4096, 
                intermediate_size=11008,
                num_hidden_layers=32, 
                num_attention_heads=32,
                num_key_value_heads=32
            ),
            'large': LlamaConfig(
                vocab_size=32000, 
                max_position_embeddings=4096, 
                hidden_size=8192, 
                intermediate_size=28672,
                num_hidden_layers=80, 
                num_attention_heads=64,
                num_key_value_heads=8  # GQA for large model
            ),
        },
    }
    
    MODEL_CLASSES = {
        'gpt2': GPT2LMHeadModel,
        'gptneo': GPTNeoForCausalLM,
        'bert': BertForMaskedLM,
        'llama': LlamaForCausalLM,
    }
    
    TOKENIZER_NAMES = {
        'gpt2': 'gpt2',
        'gptneo': 'EleutherAI/gpt-neo-125M',
        'bert': 'bert-base-uncased',
        'llama': 'meta-llama/Llama-2-7b-hf',  # Note: requires authentication
    }
    
    @classmethod
    def create_model(cls, model_type: str, model_size: str, seed: int):
        """Create a model with specified architecture and size"""
        set_seed(seed)  # Ensure model initialization is reproducible
        
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        if model_size not in cls.MODEL_CONFIGS[model_type]:
            raise ValueError(f"Unknown model size: {model_size}")
        
        config = cls.MODEL_CONFIGS[model_type][model_size]
        model_class = cls.MODEL_CLASSES[model_type]
        
        model = model_class(config)
        return model
    
    @classmethod
    def get_tokenizer(cls, model_type: str):
        """Get appropriate tokenizer for model type"""
        if model_type not in cls.TOKENIZER_NAMES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        tokenizer_name = cls.TOKENIZER_NAMES[model_type]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer



# OPTIMIZER FACTORY
class OptimizerFactory:
    """Factory for creating optimizers with config loading"""
    
    OPTIMIZER_CLASSES = {
        'adamw': AdamW,
        'iris': IRIS,
        'adabelief': AdaBelief,
        'adan': Adan
    }
    
    @classmethod
    def load_optimizer_config(cls, config_path: str) -> Dict[str, Any]:
        """Load optimizer hyperparameters from JSON config"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    @classmethod
    def create_optimizer(cls, optimizer_name: str, model_parameters, 
                        config_path: str, base_lr: float):
        """Create optimizer with config-specified hyperparameters"""
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name not in cls.OPTIMIZER_CLASSES:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(cls.OPTIMIZER_CLASSES.keys())}")
        
        # Load config
        config = cls.load_optimizer_config(config_path)
        
        # Override lr with base_lr from training config
        config['lr'] = base_lr
        
        # Create optimizer
        optimizer_class = cls.OPTIMIZER_CLASSES[optimizer_name]
        optimizer = optimizer_class(model_parameters, **config)
        
        return optimizer

# CHECKPOINT MANAGER
class CheckpointManager:
    """Manages saving and loading checkpoints with W&B integration"""
    
    def __init__(self, checkpoint_dir: str, run_name: str, keep_last_n: int = 3,
                 upload_to_wandb: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.upload_to_wandb = upload_to_wandb
        self.run_name = run_name
        self.saved_checkpoints = []
        
        print(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(self, model, optimizer, scheduler, global_step: int, 
                       epoch: int, train_loss: float, val_loss: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Save checkpoint locally and optionally to W&B"""
        checkpoint_name = f"checkpoint_step_{global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'global_step': global_step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metadata': metadata or {}
        }
        
        # Save locally
        model_path = checkpoint_path / "model.pt"
        optimizer_path = checkpoint_path / "optimizer.pt"
        checkpoint_info_path = checkpoint_path / "checkpoint_info.json"
        
        torch.save(checkpoint['model_state_dict'], model_path)
        torch.save({
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            'scheduler_state_dict': checkpoint['scheduler_state_dict']
        }, optimizer_path)
        
        # Save checkpoint info
        info = {
            'global_step': global_step,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metadata': metadata or {}
        }
        with open(checkpoint_info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        # Track saved checkpoints
        self.saved_checkpoints.append({
            'path': checkpoint_path,
            'step': global_step,
            'name': checkpoint_name
        })
        
        # Upload to W&B as artifact
        if self.upload_to_wandb:
            self._upload_to_wandb(checkpoint_path, checkpoint_name, global_step, info)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _upload_to_wandb(self, checkpoint_path: Path, checkpoint_name: str, 
                        global_step: int, info: Dict[str, Any]):
        """Upload checkpoint to W&B as artifact"""
        try:
            artifact = wandb.Artifact(
                name=f"{self.run_name}_{checkpoint_name}",
                type="model_checkpoint",
                metadata={
                    'global_step': global_step,
                    'epoch': info['epoch'],
                    'train_loss': info['train_loss'],
                    'val_loss': info['val_loss'],
                    **info['metadata']
                }
            )
            
            # Add all files in checkpoint directory
            artifact.add_dir(str(checkpoint_path))
            
            # Log artifact
            wandb.log_artifact(artifact, aliases=[f"step_{global_step}", "latest"])
            
            print(f"✓ Uploaded checkpoint to W&B: {artifact.name}")
                        
        except Exception as e:
            print(f"! Failed to upload checkpoint to W&B: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N"""
        if len(self.saved_checkpoints) > self.keep_last_n:
            # Sort by step
            self.saved_checkpoints.sort(key=lambda x: x['step'])
            
            # Remove oldest checkpoints
            to_remove = self.saved_checkpoints[:-self.keep_last_n]
            for checkpoint in to_remove:
                try:
                    import shutil
                    shutil.rmtree(checkpoint['path'])
                    print(f"✓ Removed old checkpoint: {checkpoint['name']}")
                except Exception as e:
                    print(f"! Failed to remove checkpoint {checkpoint['name']}: {e}")
            
            # Update saved checkpoints list
            self.saved_checkpoints = self.saved_checkpoints[-self.keep_last_n:]
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler):
        """Load checkpoint from disk"""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        model.load_state_dict(torch.load(model_path))
        
        # Load optimizer and scheduler
        optimizer_path = checkpoint_path / "optimizer.pt"
        checkpoint_data = torch.load(optimizer_path)
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Load checkpoint info
        info_path = checkpoint_path / "checkpoint_info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        return info


# TRAINER
class BenchmarkTrainer:
    """Trainer for LLM benchmark experiments"""
    
    def __init__(self, config: BenchmarkConfig, seed: int):
        self.config = config
        self.seed = seed
        self.global_step = 0
        self.local_step = 0
        self.epoch = 0
        
        # Set seed for this run
        self.seed_worker = set_seed(seed)
        
        # Initialize wandb with unique run per seed
        self.run_name = f"{config.model_type}_{config.model_size}_{config.optimizer}_seed{seed}"
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=self.run_name,
            config={
                **asdict(config),
                'seed': seed
            },
            reinit=True,
            tags=[config.model_type, config.model_size, config.optimizer, f"seed_{seed}"]
        )
        
        # Create model
        self.model = ModelFactory.create_model(
            config.model_type, 
            config.model_size, 
            seed
        ).to(config.device)
        
        # Get tokenizer
        self.tokenizer = ModelFactory.get_tokenizer(config.model_type)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {self.num_params:,} parameters")
        wandb.log({"num_parameters": self.num_params})
        
        # Create optimizer
        self.optimizer = OptimizerFactory.create_optimizer(
            config.optimizer,
            self.model.parameters(),
            config.optimizer_config_path,
            config.lr_init
        )
        
        # Prepare datasets
        self.train_dataset = DatasetFactory.prepare_dataset(
            config.dataset_name,
            self.tokenizer,
            config.max_seq_length,
            split='train',
            streaming=config.streaming,
            seed=seed
        )
        
        self.val_dataset = DatasetFactory.prepare_dataset(
            config.dataset_name,
            self.tokenizer,
            config.max_seq_length,
            split='validation' if 'validation' in ['train', 'validation', 'test'] else 'test',
            streaming=config.streaming,
            seed=seed + 1
        )
        
        # Create data collator based on model type
        if config.model_type == 'bert':
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=0.15
            )
        else:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            collate_fn=self.data_collator,
            num_workers=config.num_workers,
            worker_init_fn=self.seed_worker,
            generator=torch.Generator().manual_seed(seed),
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            collate_fn=self.data_collator,
            num_workers=config.num_workers,
            worker_init_fn=self.seed_worker,
            generator=torch.Generator().manual_seed(seed + 1)
        )
        
        # Calculate total steps
        if hasattr(self.train_dataset, '__len__'):
            num_train_examples = len(self.train_dataset)
        else:
            num_train_examples = 10000
        
        self.steps_per_epoch = num_train_examples // (config.batch_size * config.gradient_accumulation_steps)
        
        if config.max_steps > 0:
            self.total_steps = config.max_steps
            self.effective_num_epochs = math.ceil(config.max_steps / self.steps_per_epoch)
        else:
            self.total_steps = config.num_epochs * self.steps_per_epoch
            self.effective_num_epochs = config.num_epochs
        
        print(f"Dataset size: {num_train_examples:,} examples")
        print(f"Steps per epoch: {self.steps_per_epoch:,}")
        print(f"Total training steps: {self.total_steps:,}")
        
        # Create scheduler
        self.scheduler = CosineDecayWithWarmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=self.total_steps,
            lr_init=config.lr_init,
            lr_min=config.lr_min
        )
        
        # Initialize checkpoint manager
        if config.save_checkpoints:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=config.checkpoint_dir,
                run_name=self.run_name,
                keep_last_n=config.keep_last_n_checkpoints,
                upload_to_wandb=config.upload_to_wandb
            )
        else:
            self.checkpoint_manager = None
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Loss tracking
        self.train_losses = []
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
        
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Optimizer step with gradient clipping"""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        lr = self.scheduler.step()
        return lr
    
    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
            
            if num_batches >= 100:
                break
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint_if_needed(self, val_loss: Optional[float] = None):
        """Save checkpoint if checkpoint manager is enabled"""
        if self.checkpoint_manager is None:
            return
        
        avg_train_loss = np.mean(self.train_losses[-self.config.log_every_n_steps:])
        
        metadata = {
            'model_type': self.config.model_type,
            'model_size': self.config.model_size,
            'optimizer': self.config.optimizer,
            'num_parameters': self.num_params,
            'batch_size': self.config.batch_size,
            'lr': self.scheduler.get_lr()
        }
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            metadata=metadata
        )
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"Starting training: {self.run_name}")
        print(f"Total steps: {self.total_steps}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print(f"Effective epochs: {self.effective_num_epochs}")
        print(f"Checkpoint saving: {'Enabled' if self.config.save_checkpoints else 'Disabled'}")
        if self.config.save_checkpoints:
            print(f"Checkpoint frequency: every {self.config.log_every_n_steps} steps")
            print(f"W&B upload: {'Enabled' if self.config.upload_to_wandb else 'Disabled'}")
        print(f"{'='*80}\n")
        
        self.optimizer.zero_grad()
        
        for epoch in range(self.effective_num_epochs):
            self.epoch = epoch
            self.local_step = 0
            epoch_losses = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                self.train_losses.append(loss)
                epoch_losses.append(loss)
                self.local_step += 1
                
                # Optimizer step after gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    lr = self.optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        avg_loss = np.mean(self.train_losses[-self.config.log_every_n_steps:])
                        
                        log_dict = {
                            'train/loss': avg_loss,
                            'train/lr': lr,
                            'train/global_step': self.global_step,
                            'train/local_step': self.local_step,
                            'train/epoch': self.epoch,
                            'train/epoch_progress': self.local_step / (self.steps_per_epoch * self.config.gradient_accumulation_steps)
                        }
                        
                        wandb.log(log_dict, step=self.global_step)
                        
                        print(f"Epoch {epoch}/{self.effective_num_epochs} | "
                              f"Step {self.global_step}/{self.total_steps} "
                              f"({self.local_step}/{self.steps_per_epoch * self.config.gradient_accumulation_steps} in epoch) | "
                              f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                        
                        # Save checkpoint at log intervals
                        if self.config.save_checkpoints:
                            self.save_checkpoint_if_needed()
                    
                    # Validation
                    if self.global_step % self.config.eval_steps == 0:
                        val_loss = self.validate()
                        wandb.log({
                            'val/loss': val_loss,
                            'val/global_step': self.global_step,
                            'val/epoch': self.epoch
                        }, step=self.global_step)
                        print(f"  Validation Loss: {val_loss:.4f}")
                        
                        # Save checkpoint after validation
                        if self.config.save_checkpoints:
                            self.save_checkpoint_if_needed(val_loss=val_loss)
                    
                    # Check if max steps reached
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        print(f"\nReached max steps: {self.config.max_steps}")
                        self.finalize()
                        return
            
            # End of epoch validation
            epoch_avg_loss = np.mean(epoch_losses)
            val_loss = self.validate()
            wandb.log({
                'train/epoch_loss': epoch_avg_loss,
                'val/loss_epoch': val_loss,
                'val/epoch': self.epoch
            }, step=self.global_step)
            print(f"\n{'='*80}")
            print(f"End of Epoch {epoch}/{self.effective_num_epochs}")
            print(f"Train Loss: {epoch_avg_loss:.4f} | Validation Loss: {val_loss:.4f}")
            print(f"Completed {self.local_step} batches ({self.local_step // self.config.gradient_accumulation_steps} optimizer steps)")
            print(f"{'='*80}\n")
            
            # Save checkpoint at end of epoch
            if self.config.save_checkpoints:
                self.save_checkpoint_if_needed(val_loss=val_loss)
        
        self.finalize()
    
    def finalize(self):
        """Cleanup and final logging"""
        print(f"\nTraining completed: {self.run_name}")
        print(f"Total steps: {self.global_step}")
        
        # Final validation
        final_val_loss = self.validate()
        wandb.log({
            'final/val_loss': final_val_loss,
            'final/total_steps': self.global_step
        })
        
        # Save final checkpoint
        if self.config.save_checkpoints:
            print("\nSaving final checkpoint...")
            self.save_checkpoint_if_needed(val_loss=final_val_loss)
        
        # Close wandb run
        wandb.finish()

# MAIN BENCHMARK RUNNER
def run_benchmark(config_path: str):
    """Run benchmark across all seeds"""
    config = BenchmarkConfig.load(config_path)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model: {config.model_type}-{config.model_size}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Seeds: {config.seeds}")
    print(f"Checkpointing: {'Enabled' if config.save_checkpoints else 'Disabled'}")
    if config.save_checkpoints:
        print(f"Checkpoint dir: {config.checkpoint_dir}")
        print(f"Upload to W&B: {'Yes' if config.upload_to_wandb else 'No'}")
        print(f"Keep last N: {config.keep_last_n_checkpoints}")
    print(f"{'='*80}\n")
    
    for seed in config.seeds:
        print(f"\n{'#'*80}")
        print(f"# Running with seed: {seed}")
        print(f"{'#'*80}\n")
        
        trainer = BenchmarkTrainer(config, seed)
        trainer.train()
        
        print(f"\n{'#'*80}")
        print(f"# Completed seed: {seed}")
        print(f"{'#'*80}\n")
    
    print(f"\n{'='*80}")
    print(f"ALL SEEDS COMPLETED")
    print(f"{'='*80}\n")


# CLI
def main():
    parser = argparse.ArgumentParser(description='LLM Optimizer Benchmark')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to benchmark config JSON')
    
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == '__main__':
    main()