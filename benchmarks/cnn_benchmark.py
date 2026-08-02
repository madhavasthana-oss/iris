# import bitsandbytes as bnb  # For LARS
"""
ResNet18 CIFAR-100 Optimizer Benchmark
Train with AdamW, IRIS, Adan, AdaBelief, RAdam, Yogi, NAG, or LARS

USAGE:
from adan import Adan
from adabelief_pytorch import AdaBelief

train_single_optimizer("adamw")
train_single_optimizer("iris", IRIS)
train_single_optimizer("adan", Adan)
train_single_optimizer("adabelief", AdaBelief)
train_single_optimizer("radam")
train_single_optimizer("yogi")
train_single_optimizer("nag")
train_single_optimizer("lars", bnb.optim.LARS)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import wandb
from typing import Dict, Tuple, Optional
import random
import pickle
import os

# GLOBAL CONFIGURATION

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 200
BATCH_SIZE = 2048
NUM_WORKERS = 4

# Gradient norm clipping thresholds per optimizer.
# Set to None to disable clipping for a given optimizer.
GRAD_CLIP_MAX_NORM = {
    "iris":      1.0,
    "adamw":     1.0,
    "radam":     1.0,
    "yogi":      1.0,
    "nag":       5.0,   # SGD-based; typically tolerates larger gradients
    "lars":      None,  # LARS normalises layers internally — clipping is redundant
    "adan":      None,   # Adan already has max_grad_norm, but this clips before it sees them
    "adabelief": 1.0,
}

# Optimizer configurations
OPTIMIZER_CONFIGS = {
    "iris": {
        "lr": 0.002,
        "betas": (0.96, 0.92, 0.99),
        "eps": 1e-7,
        "weight_decay": 0.0005,
        "snr_threshold" : None
    },
    "adamw": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0005,
    },
    "radam": {
        "lr": 0.003,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.001,
    },
    "yogi": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-3,  # Yogi typically uses larger epsilon
        "weight_decay": 0.001,
    },
    "nag": {
        "lr": 0.1,  # NAG typically needs higher LR
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "dampening": 0,
        "nesterov": True,
    },
    "lars": {
        "lr": 0.4,  # LARS can use very high base LR
        "momentum": 0.9,
        "weight_decay": 0.0005,
        # eta and max_norm are LARS-specific (bitsandbytes)
    },
    "adan": {
        "lr": 0.001,
        "betas": (0.96, 0.92, 0.9995),
        "eps": 1e-8,
        "weight_decay": 0.0005,
        "max_grad_norm": 1.0,
        "no_prox": False,
        "foreach": False,
    },
    "adabelief": {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0005,
        "weight_decouple": True,
        "rectify": False,
        "amsgrad": False,
    }
}

# Learning rate schedule
LR_SCHEDULE = {
    "type": "cosine",
    "warmup_epochs": 10,
    "min_lr": 1e-6,
}

# Experiment tracking
WANDB_PROJECT = "RESNET18-CIFAR100-26.07.26"
WANDB_ENTITY = None

# SEEDING

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# MODEL DEFINITION (ResNet18)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# DATA LOADING

def get_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-100 dataloaders from cached Kaggle dataset"""
    import kagglehub
    from torch.utils.data import TensorDataset

    # Download/use cached dataset
    path = kagglehub.dataset_download("fedesoriano/cifar100")
    print("Path to dataset files:", path)

    def load_cifar100_batch(filepath):
        with open(filepath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        data = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        labels = np.array(d[b'fine_labels'])
        return data, labels

    train_data, train_labels = load_cifar100_batch(os.path.join(path, 'train'))
    test_data,  test_labels  = load_cifar100_batch(os.path.join(path, 'test'))

    # Normalize
    mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)[:, None, None]
    std  = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)[:, None, None]
    train_data = (train_data - mean) / std
    test_data  = (test_data  - mean) / std

    # Augmentation for training (random crop + horizontal flip)
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels, augment=False):
            self.data    = torch.tensor(data)
            self.labels  = torch.tensor(labels, dtype=torch.long)
            self.augment = augment

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            x, y = self.data[idx], self.labels[idx]
            if self.augment:
                # Random crop: pad 4, then crop back to 32x32
                x = F.pad(x, [4, 4, 4, 4], mode='reflect')
                i = random.randint(0, 8)
                j = random.randint(0, 8)
                x = x[:, i:i+32, j:j+32]
                if random.random() > 0.5:
                    x = x.flip(-1)  # horizontal flip
            return x, y

    trainset = AugmentedDataset(train_data, train_labels, augment=True)
    testset  = AugmentedDataset(test_data,  test_labels,  augment=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return trainloader, testloader

# LEARNING RATE SCHEDULING
def get_lr(epoch: int, base_lr: float) -> float:
    """Get learning rate with warmup and cosine annealing"""
    warmup_epochs = LR_SCHEDULE['warmup_epochs']
    min_lr = LR_SCHEDULE['min_lr']

    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (NUM_EPOCHS - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


# TRAINING AND EVALUATION
def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer,
                trainloader: DataLoader, criterion: nn.Module,
                epoch: int, grad_clip_max_norm: Optional[float]) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        grad_clip_max_norm: If not None, clip the global gradient norm to this
                            value before every optimizer.step().  The raw norm
                            *before* clipping is returned so it can be logged.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    grad_norm_sum = 0.0      # accumulate pre-clip norms for averaging
    clip_count = 0           # how many batches actually got clipped

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # --- gradient norm clipping (applied AFTER backward, BEFORE step) ---
        if grad_clip_max_norm is not None:
            # torch.nn.utils.clip_grad_norm_ returns the total norm BEFORE clipping
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_max_norm
            )
            grad_norm_sum += total_norm.item()
            if total_norm.item() > grad_clip_max_norm:
                clip_count += 1

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    num_batches = len(trainloader)
    metrics = {
        'train_loss': train_loss / num_batches,
        'train_acc': 100. * correct / total,
    }

    # Append grad-norm stats only when clipping is active
    if grad_clip_max_norm is not None:
        metrics['grad_norm_mean']   = grad_norm_sum / num_batches
        metrics['grad_clip_ratio']  = clip_count / num_batches  # fraction of batches clipped

    return metrics


def evaluate(model: nn.Module, testloader: DataLoader,
             criterion: nn.Module) -> Dict[str, float]:
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return {
        'test_loss': test_loss / len(testloader),
        'test_acc': 100. * correct / total
    }


# MAIN TRAINING FUNCTION

def train_single_optimizer(optimizer_name: str, optimizer_class=None,
                          custom_config: Optional[Dict] = None):
    """
    Train model with specified optimizer

    Args:
        optimizer_name: "iris", "adamw", "adan", "adabelief", "radam", "yogi", "nag", "lars"
        optimizer_class: Optimizer class (required for non-torch optimizers)
        custom_config: Optional dict to override default config

    Returns:
        best_test_acc: Best test accuracy achieved
    """

    # Validate optimizer name
    valid_optimizers = ["iris", "adamw", "adan", "adabelief", "radam", "yogi", "nag", "lars"]
    if optimizer_name not in valid_optimizers:
        raise ValueError(f"optimizer_name must be one of {valid_optimizers}, got '{optimizer_name}'")

    # Check optimizer class is provided for non-torch optimizers
    torch_optimizers = ["adamw", "nag", "radam"]
    if optimizer_name not in torch_optimizers and optimizer_class is None:
        raise ValueError(f"Must provide optimizer_class when using '{optimizer_name}'")

    # Resolve the per-optimizer gradient clipping threshold (None = disabled)
    grad_clip_max_norm = GRAD_CLIP_MAX_NORM.get(optimizer_name)

    # Get config
    config = OPTIMIZER_CONFIGS[optimizer_name].copy()
    if custom_config:
        config.update(custom_config)

    # Initialize wandb
    run_name = f"{optimizer_name}_lr{config['lr']}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "optimizer": optimizer_name,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "grad_clip_max_norm": grad_clip_max_norm,
            **config
        },
        reinit=True
    )

    # Set seed
    set_seed(SEED)

    # Create model
    model = ResNet18(num_classes=100).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Create optimizer
    if optimizer_name == "iris":
    # Don't filter out None values - pass them explicitly
        optimizer = optimizer_class(model.parameters(), **config)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **config)
    elif optimizer_name == "radam":
        optimizer = torch.optim.RAdam(model.parameters(), **config)
    elif optimizer_name == "nag":
        optimizer = torch.optim.SGD(model.parameters(), **config)
    elif optimizer_name == "lars":
        # LARS from bitsandbytes or provided class
        optimizer = optimizer_class(model.parameters(), **config)
    elif optimizer_name == "yogi":
        # Yogi from external implementation
        optimizer = optimizer_class(model.parameters(), **config)
    elif optimizer_name == "adan":
        optimizer = optimizer_class(model.parameters(), **config)
    elif optimizer_name == "adabelief":
        optimizer = optimizer_class(model.parameters(), **config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Get dataloaders
    trainloader, testloader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

    # Training loop
    best_acc = 0
    print(f"\n{'='*80}")
    print(f"Training with {optimizer_name.upper()}")
    print(f"Gradient clipping: max_norm = {grad_clip_max_norm}")
    print(f"{'='*80}")
    print(f"Config: {config}")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")

    for epoch in range(NUM_EPOCHS):
        # Update learning rate
        current_lr = get_lr(epoch, config['lr'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Train and evaluate
        train_metrics = train_epoch(
            model, optimizer, trainloader, criterion, epoch, grad_clip_max_norm
        )
        test_metrics = evaluate(model, testloader, criterion)

        # Track best accuracy
        if test_metrics['test_acc'] > best_acc:
            best_acc = test_metrics['test_acc']

        # Log to wandb
        log_dict = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics['train_loss'],
            'train_acc': train_metrics['train_acc'],
            'test_loss': test_metrics['test_loss'],
            'test_acc': test_metrics['test_acc'],
            'best_test_acc': best_acc,
        }

        # Include grad-norm metrics only when clipping is active
        if grad_clip_max_norm is not None:
            log_dict['grad_norm_mean']  = train_metrics['grad_norm_mean']
            log_dict['grad_clip_ratio'] = train_metrics['grad_clip_ratio']

        wandb.log(log_dict)

        # Print progress
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            clip_info = ""
            if grad_clip_max_norm is not None:
                clip_info = (
                    f" | ‖g‖={train_metrics['grad_norm_mean']:.2f} "
                    f"clip={train_metrics['grad_clip_ratio']*100:.1f}%"
                )
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | "
                  f"Train: {train_metrics['train_acc']:5.2f}% | "
                  f"Test: {test_metrics['test_acc']:5.2f}% | "
                  f"Best: {best_acc:5.2f}%"
                  f"{clip_info}")

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - {optimizer_name.upper()}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"{'='*80}\n")

    wandb.finish()
    return best_acc


if __name__ == "__main__":
    # Import your optimizers
    # from iris import IRIS
    # from adan import Adan
    # from adabelief_pytorch import AdaBelief
    # import bitsandbytes as bnb

    # Example 1: Train with standard AdamW (built-in PyTorch)
    # best_acc = train_single_optimizer("adamw")

    # Example 2: Train with RAdam (built-in PyTorch)
    # best_acc = train_single_optimizer("radam")

    # Example 3: Train with NAG/Nesterov SGD (built-in PyTorch)
    # best_acc = train_single_optimizer("nag")

    # Example 4: Train with IRIS
    # from iris import IRIS
    # best_acc = train_single_optimizer("iris", IRIS)

    # Example 5: Train with Adan
    # from adan.adan import Adan
    best_acc = train_single_optimizer("adan", Adan)

    # Example 6: Train with AdaBelief
    # from adabelief_pytorch import AdaBelief
    # best_acc = train_single_optimizer("adabelief", AdaBelief)

    # Example 7: Train with LARS (from bitsandbytes)
    # import bitsandbytes as bnb
    # best_acc = train_single_optimizer("lars", bnb.optim.LARS)

    # Example 8: Train with Yogi (requires external implementation)
    # from yogi import Yogi  # You'll need to import from wherever you have it
    # best_acc = train_single_optimizer("yogi", Yogi)

    # Example 9: Train with custom config
    # custom_config = {"lr": 0.002, "weight_decay": 0.05}
    # best_acc = train_single_optimizer("iris", IRIS, custom_config)

    print("Import optimizers and run train_single_optimizer() to start training!")