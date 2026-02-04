#!/usr/bin/env python3
"""
SOC SLM Training Script - Production Ready
End-to-end training pipeline for SOC Small Language Model.

Usage:
    python train.py --config config.yaml
    python train.py --preset production --output ./models/soc-slm
    python train.py --model-size 350m --epochs 3 --batch-size 8
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import SOCConfig, ModelSettings, get_config
from tokenizer.security_tokenizer import SecurityTokenizer, TokenizerConfig
from model.architecture import create_soc_model
from data.data_generator import SecurityDataGenerator
from training.trainer import SOCTrainer, TrainingConfig, SecurityDataset, prepare_training
from utils.helpers import (
    setup_logging, set_seed, get_device_info, timer, 
    count_parameters, format_number, ensure_dir
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SOC Small Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--preset', type=str, choices=['development', 'production', 'edge'],
                        default='production', help='Configuration preset')
    
    # Model
    parser.add_argument('--model-size', type=str, 
                        choices=['125m', '350m', '760m', '1b'],
                        default='350m', help='Model size preset')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--gradient-accumulation', type=int, default=4, 
                        help='Gradient accumulation steps')
    
    # Mixed precision
    parser.add_argument('--fp16', action='store_true', help='Use FP16 training')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 training')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--train-file', type=str, help='Training data file (JSON)')
    parser.add_argument('--generate-data', action='store_true', 
                        help='Generate synthetic training data')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples for synthetic data')
    
    # Output
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--save-steps', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--eval-steps', type=int, default=100, help='Evaluate every N steps')
    
    # Resume
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--dry-run', action='store_true', help='Validate setup without training')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(
        name='soc-slm-train',
        level=getattr(__import__('logging'), args.log_level),
        log_file=os.path.join(args.output, 'train.log')
    )
    
    logger.info("=" * 60)
    logger.info("SOC SLM Training Pipeline")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(args.seed)
    
    # Load or create configuration
    if args.config:
        config = SOCConfig.load(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = get_config(args.preset)
        logger.info(f"Using {args.preset} preset")
    
    # Override with command line arguments
    config.model = ModelSettings.from_preset(f'soc-slm-{args.model_size}')
    config.model.vocab_size = args.vocab_size
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.warmup_ratio = args.warmup_ratio
    config.training.gradient_accumulation_steps = args.gradient_accumulation
    config.training.output_dir = args.output
    config.training.save_steps = args.save_steps
    config.training.eval_steps = args.eval_steps
    
    if args.fp16:
        config.training.fp16 = True
        config.training.bf16 = False
    if args.bf16:
        config.training.bf16 = True
        config.training.fp16 = False
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
    
    # Create output directory
    output_dir = ensure_dir(args.output)
    
    # Save configuration
    config.save(output_dir / 'config.json')
    logger.info(f"Configuration saved to {output_dir / 'config.json'}")
    
    # Log device info
    device_info = get_device_info()
    logger.info(f"Device: {device_info['device']}")
    if 'gpu_name' in device_info:
        logger.info(f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory_gb']:.1f} GB)")
    
    # Step 1: Initialize or load tokenizer
    logger.info("\n[Step 1/4] Initializing tokenizer...")
    tokenizer_path = output_dir / 'tokenizer'
    
    if tokenizer_path.exists():
        tokenizer = SecurityTokenizer.load(str(tokenizer_path))
    else:
        tokenizer_config = TokenizerConfig(
            vocab_size=config.model.vocab_size,
            max_sequence_length=config.model.max_position_embeddings
        )
        tokenizer = SecurityTokenizer(tokenizer_config)
        logger.info("Training tokenizer on security corpus...")
        tokenizer.train_on_security_corpus(verbose=True)
        tokenizer.save(str(tokenizer_path))
    
    logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    # Step 2: Prepare training data
    logger.info("\n[Step 2/4] Preparing training data...")
    data_dir = ensure_dir(args.data_dir)
    
    if args.train_file:
        train_file = args.train_file
    elif args.generate_data or not (data_dir / 'train.json').exists():
        logger.info(f"Generating {args.num_samples} synthetic training samples...")
        generator = SecurityDataGenerator()
        
        with timer("Data generation"):
            dataset = generator.generate_full_dataset()
        
        train_file = data_dir / 'train.json'
        generator.save_dataset(dataset, str(train_file))
        logger.info(f"Generated dataset saved to {train_file}")
    else:
        train_file = data_dir / 'train.json'
    
    # Load training data
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    logger.info(f"Loaded {len(train_data)} training samples")
    
    # Create PyTorch dataset
    train_dataset = SecurityDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=config.model.max_position_embeddings
    )
    
    # Step 3: Initialize model
    logger.info("\n[Step 3/4] Initializing model...")
    
    model = create_soc_model(
        model_type='causal_lm',
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_kv_heads=config.model.num_kv_heads,
        intermediate_size=config.model.intermediate_size,
        max_position_embeddings=config.model.max_position_embeddings,
        hidden_dropout=config.model.hidden_dropout,
        attention_dropout=config.model.attention_dropout
    )
    
    params = count_parameters(model)
    logger.info(f"Model parameters: {format_number(params['total'])} total, "
                f"{format_number(params['trainable'])} trainable")
    
    # Step 4: Train
    logger.info("\n[Step 4/4] Starting training...")
    
    if args.dry_run:
        logger.info("Dry run complete. Setup validated successfully.")
        return
    
    training_config = TrainingConfig(
        output_dir=str(output_dir),
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        save_total_limit=config.training.save_total_limit,
        early_stopping_patience=config.training.early_stopping_patience
    )
    
    trainer = SOCTrainer(
        model=model,
        train_dataset=train_dataset,
        config=training_config,
        tokenizer=tokenizer
    )
    
    with timer("Training"):
        trainer.train()
    
    # Save final model
    final_dir = output_dir / 'final'
    trainer.save_model(str(final_dir))
    tokenizer.save(str(final_dir / 'tokenizer'))
    
    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Final model saved to: {final_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
