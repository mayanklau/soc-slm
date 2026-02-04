"""
SOC Small Language Model (SLM) - Production Ready
A custom-built language model optimized for Security Operations Center chatbots.

Components:
- tokenizer: Security-aware BPE tokenizer
- model: Transformer architecture with security-specific heads
- data: Training data generation for security use cases
- training: Distributed training pipeline
- inference: Optimized inference engine with quantization
"""

__version__ = "1.0.0"
__author__ = "SOC Platform Team"

from .tokenizer.security_tokenizer import (
    SecurityTokenizer,
    BPETokenizer,
    TokenizerConfig,
    SecurityPatternMatcher,
    create_security_tokenizer
)

from .model.architecture import (
    SOCModelConfig,
    SOCForCausalLM,
    SOCForSequenceClassification,
    SOCForTokenClassification,
    create_soc_model
)

from .data.data_generator import (
    SecurityDataGenerator,
    MITREGenerator,
    IOCGenerator
)

from .training.trainer import (
    SOCTrainer,
    TrainingConfig,
    SecurityDataset,
    prepare_training
)

from .inference.engine import (
    SOCInferenceEngine,
    GenerationConfig,
    ModelQuantizer,
    InferenceMetrics
)

from .inference.integration import (
    SOCChatbotIntegration,
    IntentClassifier,
    EntityExtractor,
    QueryInterpreter
)

__all__ = [
    # Tokenizer
    'SecurityTokenizer',
    'BPETokenizer', 
    'TokenizerConfig',
    'SecurityPatternMatcher',
    'create_security_tokenizer',
    # Model
    'SOCModelConfig',
    'SOCForCausalLM',
    'SOCForSequenceClassification',
    'SOCForTokenClassification',
    'create_soc_model',
    # Data
    'SecurityDataGenerator',
    'MITREGenerator',
    'IOCGenerator',
    # Training
    'SOCTrainer',
    'TrainingConfig',
    'SecurityDataset',
    'prepare_training',
    # Inference
    'SOCInferenceEngine',
    'GenerationConfig',
    'ModelQuantizer',
    'InferenceMetrics',
    # Integration
    'SOCChatbotIntegration',
    'IntentClassifier',
    'EntityExtractor',
    'QueryInterpreter',
]
