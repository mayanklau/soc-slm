"""
SOC-SLM Training Pipeline
Production-ready training system for security-focused SLM.

Features:
- Distributed training support (DDP)
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Learning rate scheduling
- Checkpointing and resumption
- Logging and metrics
- Evaluation during training
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model
    model_name: str = "soc-slm-125m"
    vocab_size: int = 32000
    max_length: int = 2048
    
    # Training
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Precision
    fp16: bool = False
    bf16: bool = True
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    output_dir: str = "./output"
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_steps: int = 500
    eval_batch_size: int = 16
    
    # Logging
    logging_steps: int = 100
    log_dir: str = "./logs"
    
    # Distributed
    distributed: bool = False
    local_rank: int = -1
    
    # Data
    train_file: str = None
    eval_file: str = None
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            return cls(**json.load(f))


@dataclass
class TrainingMetrics:
    """Training metrics tracker."""
    
    epoch: int = 0
    global_step: int = 0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    best_eval_loss: float = float('inf')
    samples_seen: int = 0
    tokens_seen: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    from torch.optim import AdamW, Adam, SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR
    from torch.cuda.amp import GradScaler, autocast
    
    TORCH_AVAILABLE = True
    
    
    class SecurityDataset(Dataset):
        """PyTorch dataset for security training data."""
        
        def __init__(
            self,
            data: List[Dict],
            tokenizer,
            max_length: int = 2048,
            task: str = "causal_lm"
        ):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.task = task
        
        def __len__(self) -> int:
            return len(self.data)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            sample = self.data[idx]
            
            if self.task == "causal_lm":
                # Combine input and output for causal LM
                text = f"{sample['input_text']}\n\n{sample['output_text']}"
                encoded = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True
                )
                
                input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
                attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
                
                # Labels are input_ids shifted
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # Ignore padding
                
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
            
            elif self.task == "classification":
                encoded = self.tokenizer.encode(
                    sample['input_text'],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True
                )
                
                # Convert intent to label
                intent_map = {
                    "triage": 0, "query": 1, "threat_intel": 2,
                    "incident_response": 3, "statistics": 4,
                    "timeline": 5, "search": 6, "help": 7
                }
                label = intent_map.get(sample.get('intent', 'help'), 7)
                
                return {
                    'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
            
            else:
                raise ValueError(f"Unknown task: {self.task}")
    
    
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function with dynamic padding."""
        # Find max length in batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            pad_len = max_len - len(item['input_ids'])
            
            input_ids.append(
                torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
            )
            
            if 'labels' in item:
                if item['labels'].dim() == 0:  # Classification
                    labels.append(item['labels'])
                else:  # Causal LM
                    labels.append(
                        torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
                    )
        
        result = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
        }
        
        if labels:
            if labels[0].dim() == 0:
                result['labels'] = torch.stack(labels)
            else:
                result['labels'] = torch.stack(labels)
        
        return result
    
    
    class SOCTrainer:
        """Production trainer for SOC-SLM."""
        
        def __init__(
            self,
            model: nn.Module,
            config: TrainingConfig,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            tokenizer = None,
            callbacks: List[Callable] = None
        ):
            self.model = model
            self.config = config
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.tokenizer = tokenizer
            self.callbacks = callbacks or []
            
            self.metrics = TrainingMetrics()
            self.device = self._setup_device()
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Setup optimizer
            self.optimizer = self._setup_optimizer()
            
            # Setup scheduler
            self.scheduler = None  # Will be set after dataloader is created
            
            # Setup mixed precision
            self.scaler = GradScaler() if config.fp16 else None
            self.use_amp = config.fp16 or config.bf16
            self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
            
            # Create output directory
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.log_dir, exist_ok=True)
            
            # Save config
            config.save(os.path.join(config.output_dir, 'training_config.json'))
        
        def _setup_device(self) -> torch.device:
            """Setup compute device."""
            if torch.cuda.is_available():
                if self.config.local_rank >= 0:
                    torch.cuda.set_device(self.config.local_rank)
                    return torch.device(f'cuda:{self.config.local_rank}')
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        
        def _setup_optimizer(self) -> torch.optim.Optimizer:
            """Setup optimizer with weight decay."""
            # Separate parameters that should/shouldn't have weight decay
            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'bias' in name or 'norm' in name or 'layernorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            
            param_groups = [
                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            
            if self.config.optimizer == 'adamw':
                return AdamW(param_groups, lr=self.config.learning_rate, betas=(0.9, 0.95))
            elif self.config.optimizer == 'adam':
                return Adam(param_groups, lr=self.config.learning_rate)
            elif self.config.optimizer == 'sgd':
                return SGD(param_groups, lr=self.config.learning_rate, momentum=0.9)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        def _setup_scheduler(self, num_training_steps: int):
            """Setup learning rate scheduler."""
            num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
            
            if self.config.lr_scheduler == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_training_steps - num_warmup_steps,
                    eta_min=self.config.learning_rate * 0.1
                )
            elif self.config.lr_scheduler == 'linear':
                self.scheduler = LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=num_training_steps
                )
            else:
                self.scheduler = ConstantLR(self.optimizer, factor=1.0)
        
        def _create_dataloader(self, dataset: Dataset, is_training: bool = True) -> DataLoader:
            """Create dataloader with optional distributed sampler."""
            batch_size = self.config.batch_size if is_training else self.config.eval_batch_size
            
            sampler = None
            if self.config.distributed and is_training:
                sampler = DistributedSampler(dataset)
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(sampler is None and is_training),
                sampler=sampler,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        def train(self) -> TrainingMetrics:
            """Main training loop."""
            train_dataloader = self._create_dataloader(self.train_dataset, is_training=True)
            
            # Calculate total steps
            num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
            num_training_steps = num_update_steps_per_epoch * self.config.num_epochs
            
            # Setup scheduler
            self._setup_scheduler(num_training_steps)
            
            # Resume from checkpoint if specified
            start_epoch = 0
            if self.config.resume_from_checkpoint:
                start_epoch = self._load_checkpoint(self.config.resume_from_checkpoint)
            
            logger.info(f"Starting training for {self.config.num_epochs} epochs")
            logger.info(f"Total optimization steps: {num_training_steps}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            # Training loop
            self.model.train()
            early_stopping_counter = 0
            
            for epoch in range(start_epoch, self.config.num_epochs):
                self.metrics.epoch = epoch
                epoch_loss = 0.0
                
                if self.config.distributed:
                    train_dataloader.sampler.set_epoch(epoch)
                
                for step, batch in enumerate(train_dataloader):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                        loss = outputs['loss']
                        loss = loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    
                    # Optimizer step
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        self.metrics.global_step += 1
                        self.metrics.train_loss = epoch_loss / (step + 1)
                        self.metrics.learning_rate = self.scheduler.get_last_lr()[0]
                        self.metrics.samples_seen += self.config.batch_size
                        
                        # Logging
                        if self.metrics.global_step % self.config.logging_steps == 0:
                            self._log_metrics()
                        
                        # Evaluation
                        if self.eval_dataset and self.metrics.global_step % self.config.eval_steps == 0:
                            eval_loss = self.evaluate()
                            self.metrics.eval_loss = eval_loss
                            
                            # Early stopping check
                            if eval_loss < self.metrics.best_eval_loss - self.config.early_stopping_threshold:
                                self.metrics.best_eval_loss = eval_loss
                                early_stopping_counter = 0
                                self._save_checkpoint(is_best=True)
                            else:
                                early_stopping_counter += 1
                            
                            if early_stopping_counter >= self.config.early_stopping_patience:
                                logger.info(f"Early stopping triggered after {self.metrics.global_step} steps")
                                return self.metrics
                            
                            self.model.train()
                        
                        # Save checkpoint
                        if self.metrics.global_step % self.config.save_steps == 0:
                            self._save_checkpoint()
                
                # End of epoch
                avg_epoch_loss = epoch_loss / len(train_dataloader)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
                
                # Run callbacks
                for callback in self.callbacks:
                    callback(self, self.metrics)
            
            # Final save
            self._save_checkpoint(is_final=True)
            
            return self.metrics
        
        @torch.no_grad()
        def evaluate(self) -> float:
            """Evaluate on validation set."""
            if self.eval_dataset is None:
                return 0.0
            
            self.model.eval()
            eval_dataloader = self._create_dataloader(self.eval_dataset, is_training=False)
            
            total_loss = 0.0
            total_steps = 0
            
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                
                total_loss += outputs['loss'].item()
                total_steps += 1
            
            avg_loss = total_loss / total_steps
            logger.info(f"Evaluation Loss: {avg_loss:.4f}")
            
            return avg_loss
        
        def _log_metrics(self):
            """Log training metrics."""
            metrics_str = (
                f"Step {self.metrics.global_step} | "
                f"Loss: {self.metrics.train_loss:.4f} | "
                f"LR: {self.metrics.learning_rate:.2e}"
            )
            logger.info(metrics_str)
            
            # Save metrics to file
            metrics_file = os.path.join(self.config.log_dir, 'metrics.jsonl')
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(self.metrics.to_dict()) + '\n')
        
        def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
            """Save training checkpoint."""
            if is_final:
                checkpoint_dir = os.path.join(self.config.output_dir, 'final')
            elif is_best:
                checkpoint_dir = os.path.join(self.config.output_dir, 'best')
            else:
                checkpoint_dir = os.path.join(
                    self.config.output_dir,
                    f'checkpoint-{self.metrics.global_step}'
                )
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(checkpoint_dir)
            else:
                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'pytorch_model.bin'))
            
            # Save tokenizer
            if self.tokenizer:
                self.tokenizer.save(os.path.join(checkpoint_dir, 'tokenizer'))
            
            # Save training state
            training_state = {
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'metrics': self.metrics.to_dict(),
                'config': asdict(self.config)
            }
            
            if self.scaler:
                training_state['scaler_state_dict'] = self.scaler.state_dict()
            
            torch.save(training_state, os.path.join(checkpoint_dir, 'training_state.pt'))
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Cleanup old checkpoints
            if not is_best and not is_final:
                self._cleanup_checkpoints()
        
        def _load_checkpoint(self, checkpoint_path: str) -> int:
            """Load training checkpoint."""
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load model
            model_path = os.path.join(checkpoint_path, 'pytorch_model.bin')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Load training state
            state_path = os.path.join(checkpoint_path, 'training_state.pt')
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=self.device)
                
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                
                if state['scheduler_state_dict'] and self.scheduler:
                    self.scheduler.load_state_dict(state['scheduler_state_dict'])
                
                if self.scaler and 'scaler_state_dict' in state:
                    self.scaler.load_state_dict(state['scaler_state_dict'])
                
                self.metrics = TrainingMetrics(**state['metrics'])
            
            return self.metrics.epoch
        
        def _cleanup_checkpoints(self):
            """Remove old checkpoints, keeping only the most recent."""
            checkpoints = []
            for name in os.listdir(self.config.output_dir):
                if name.startswith('checkpoint-'):
                    step = int(name.split('-')[1])
                    checkpoints.append((step, os.path.join(self.config.output_dir, name)))
            
            checkpoints.sort(reverse=True)
            
            for _, path in checkpoints[self.config.save_total_limit:]:
                logger.info(f"Removing old checkpoint: {path}")
                shutil.rmtree(path)


except ImportError:
    TORCH_AVAILABLE = False
    
    class SOCTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for training")
    
    class SecurityDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")


# ==============================================================================
# Training Script Helpers
# ==============================================================================

def load_training_data(path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Flatten if nested
    if isinstance(data, dict):
        samples = []
        for key, values in data.items():
            if isinstance(values, list):
                samples.extend(values)
        return samples
    
    return data


def prepare_training(
    model_name: str = "soc-slm-125m",
    train_data_path: str = None,
    eval_data_path: str = None,
    output_dir: str = "./output",
    **config_kwargs
) -> Tuple['SOCTrainer', TrainingConfig]:
    """Prepare trainer with data and model."""
    from .architecture import create_soc_model, get_model_config
    from ..tokenizer.security_tokenizer import SecurityTokenizer
    
    # Create model
    model = create_soc_model(model_name)
    logger.info(f"Created model {model_name} with {model.num_parameters():,} parameters")
    
    # Create tokenizer
    tokenizer = SecurityTokenizer()
    
    # Load data
    train_data = load_training_data(train_data_path) if train_data_path else []
    eval_data = load_training_data(eval_data_path) if eval_data_path else None
    
    # Create datasets
    model_config = get_model_config(model_name)
    train_dataset = SecurityDataset(train_data, tokenizer, model_config.max_position_embeddings)
    eval_dataset = SecurityDataset(eval_data, tokenizer, model_config.max_position_embeddings) if eval_data else None
    
    # Create config
    config = TrainingConfig(
        model_name=model_name,
        output_dir=output_dir,
        train_file=train_data_path,
        eval_file=eval_data_path,
        **config_kwargs
    )
    
    # Create trainer
    trainer = SOCTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    return trainer, config


# -------------------------------------------------------------------
# Compatibility: EarlyStopping
# Simple early stopping helper for training loops
# -------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = metric
            return False

        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# -------------------------------------------------------------------
# Compatibility: CheckpointManager
# Lightweight checkpoint manager for saving/loading model + optimizer state
# -------------------------------------------------------------------
import os
import torch
from typing import Optional, Dict, Any

class CheckpointManager:
    def __init__(self, directory: str = 'checkpoints', keep: int = 5):
        self.directory = directory
        self.keep = keep
        os.makedirs(self.directory, exist_ok=True)
        # list of saved checkpoint filenames (newest last)
        self._saved = sorted(os.listdir(self.directory))

    def _checkpoint_path(self, name: str) -> str:
        return os.path.join(self.directory, f"{name}.pt")

    def save(self, name: str, model: Optional[torch.nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             extra: Optional[Dict[str, Any]] = None) -> str:
        """Save states. Returns path to saved checkpoint."""
        state = {}
        if model is not None:
            # prefer .state_dict() to make checkpoints lightweight
            state['model_state'] = model.state_dict()
        if optimizer is not None:
            state['optimizer_state'] = optimizer.state_dict()
        if extra is not None:
            state['extra'] = extra

        path = self._checkpoint_path(name)
        torch.save(state, path)

        # maintain saved list and prune older files if needed
        self._saved = sorted([f for f in os.listdir(self.directory) if f.endswith('.pt')])
        while len(self._saved) > self.keep:
            oldest = self._saved.pop(0)
            try:
                os.remove(os.path.join(self.directory, oldest))
            except Exception:
                pass

        return path

    def load(self, name: str, model: Optional[torch.nn.Module] = None,
             optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load checkpoint by name; will restore into provided model/optimizer if given.
        Returns the raw checkpoint dict.
        """
        path = self._checkpoint_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ckpt = torch.load(path, map_location='cpu')
        if model is not None and 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt

    def latest(self) -> Optional[str]:
        """Return the newest checkpoint filename without path (or None)."""
        self._saved = sorted([f for f in os.listdir(self.directory) if f.endswith('.pt')])
        if not self._saved:
            return None
        return os.path.splitext(self._saved[-1])[0]
