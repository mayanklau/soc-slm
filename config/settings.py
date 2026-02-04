"""
SOC SLM Configuration - Production Defaults
Centralized configuration for all SLM components.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml


@dataclass
class TokenizerSettings:
    """Tokenizer configuration."""
    vocab_size: int = 32000
    max_sequence_length: int = 2048
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: [
        "[IP]", "[DOMAIN]", "[HASH]", "[CVE]", "[MITRE]",
        "[TIME]", "[USER]", "[HOST]", "[PORT]", "[SEV]"
    ])


@dataclass 
class ModelSettings:
    """Model architecture configuration."""
    # Architecture
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: Optional[int] = None  # For GQA, None = MHA
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    
    # Regularization
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Features
    use_rotary_embeddings: bool = True
    use_swiglu: bool = True
    use_rms_norm: bool = True
    tie_word_embeddings: bool = True
    
    # Security heads
    num_intent_classes: int = 8
    num_severity_classes: int = 5
    num_entity_types: int = 12

    @classmethod
    def from_preset(cls, preset: str) -> 'ModelSettings':
        """Load preset model configurations."""
        presets = {
            'soc-slm-125m': {
                'hidden_size': 768,
                'num_layers': 12,
                'num_heads': 12,
                'intermediate_size': 3072
            },
            'soc-slm-350m': {
                'hidden_size': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'intermediate_size': 4096
            },
            'soc-slm-760m': {
                'hidden_size': 1536,
                'num_layers': 24,
                'num_heads': 16,
                'num_kv_heads': 4,
                'intermediate_size': 6144
            },
            'soc-slm-1b': {
                'hidden_size': 2048,
                'num_layers': 24,
                'num_heads': 16,
                'num_kv_heads': 4,
                'intermediate_size': 8192
            }
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return cls(**presets[preset])


@dataclass
class TrainingSettings:
    """Training pipeline configuration."""
    # Basic
    output_dir: str = "./output"
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    
    # Optimizer
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # LR Schedule
    lr_scheduler_type: str = "cosine"
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Evaluation
    eval_steps: int = 100
    eval_batch_size: int = 16
    
    # Distributed
    local_rank: int = -1
    world_size: int = 1
    
    # Logging
    logging_steps: int = 10
    report_to: str = "none"
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001


@dataclass
class InferenceSettings:
    """Inference engine configuration."""
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Optimization
    use_cache: bool = True
    quantization: Optional[str] = None  # None, "int8", "int4"
    compile_model: bool = False
    
    # Batching
    max_batch_size: int = 8
    batch_timeout_ms: int = 50
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 1


@dataclass
class DataSettings:
    """Data generation configuration."""
    # Dataset sizes
    num_intent_samples: int = 2000
    num_qa_pairs: int = 2000
    num_alert_scenarios: int = 1000
    num_conversations: int = 500
    num_ocsf_events: int = 1000
    
    # Paths
    data_dir: str = "./data"
    train_file: str = "train.json"
    eval_file: str = "eval.json"
    
    # Processing
    train_split: float = 0.9
    shuffle: bool = True
    seed: int = 42


@dataclass
class SOCConfig:
    """Master configuration for SOC SLM."""
    tokenizer: TokenizerSettings = field(default_factory=TokenizerSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    data: DataSettings = field(default_factory=DataSettings)
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'tokenizer': self.tokenizer.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'data': self.data.__dict__
        }
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SOCConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                config_dict = json.load(f)
        
        return cls(
            tokenizer=TokenizerSettings(**config_dict.get('tokenizer', {})),
            model=ModelSettings(**config_dict.get('model', {})),
            training=TrainingSettings(**config_dict.get('training', {})),
            inference=InferenceSettings(**config_dict.get('inference', {})),
            data=DataSettings(**config_dict.get('data', {}))
        )
    
    @classmethod
    def from_env(cls) -> 'SOCConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override from environment
        if os.getenv('SOC_MODEL_PRESET'):
            config.model = ModelSettings.from_preset(os.getenv('SOC_MODEL_PRESET'))
        
        if os.getenv('SOC_BATCH_SIZE'):
            config.training.batch_size = int(os.getenv('SOC_BATCH_SIZE'))
        
        if os.getenv('SOC_LEARNING_RATE'):
            config.training.learning_rate = float(os.getenv('SOC_LEARNING_RATE'))
        
        if os.getenv('SOC_QUANTIZATION'):
            config.inference.quantization = os.getenv('SOC_QUANTIZATION')
        
        if os.getenv('SOC_OUTPUT_DIR'):
            config.training.output_dir = os.getenv('SOC_OUTPUT_DIR')
        
        return config


# Default configurations for different deployment scenarios
CONFIGS = {
    'development': SOCConfig(
        model=ModelSettings.from_preset('soc-slm-125m'),
        training=TrainingSettings(
            num_epochs=1,
            batch_size=4,
            fp16=True,
            bf16=False
        ),
        inference=InferenceSettings(
            max_batch_size=4
        )
    ),
    'production': SOCConfig(
        model=ModelSettings.from_preset('soc-slm-350m'),
        training=TrainingSettings(
            num_epochs=3,
            batch_size=8,
            bf16=True
        ),
        inference=InferenceSettings(
            quantization='int8',
            compile_model=True
        )
    ),
    'edge': SOCConfig(
        model=ModelSettings.from_preset('soc-slm-125m'),
        training=TrainingSettings(
            num_epochs=3,
            batch_size=4
        ),
        inference=InferenceSettings(
            quantization='int4',
            max_batch_size=2
        )
    )
}


def get_config(name: str = 'production') -> SOCConfig:
    """Get a predefined configuration."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]


if __name__ == "__main__":
    # Save example configs
    for name, config in CONFIGS.items():
        config.save(f"config_{name}.yaml")
        print(f"Saved {name} config")
