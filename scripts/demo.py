#!/usr/bin/env python3
"""
SOC-SLM Quick Start Demo
Demonstrates the complete workflow from data generation to inference.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_tokenizer():
    """Demonstrate the security tokenizer."""
    print("\n" + "="*60)
    print("🔤 TOKENIZER DEMO")
    print("="*60)
    
    from soc_slm.tokenizer import SecurityTokenizer, SecurityPatternMatcher
    
    # Pattern matching demo
    print("\n📍 Entity Extraction:")
    matcher = SecurityPatternMatcher()
    
    sample_text = """
    Security Alert: Suspicious activity detected from 192.168.1.100.
    The host attempted to connect to malicious.example.com.
    Associated CVE: CVE-2024-12345
    MITRE Technique: T1059 (Command and Scripting Interpreter)
    File hash: d41d8cd98f00b204e9800998ecf8427e
    """
    
    entities = matcher.extract_entities(sample_text)
    for entity in entities:
        print(f"  • {entity['type']}: {entity['value']}")
    
    # Tokenizer demo
    print("\n📝 Tokenizer Training:")
    tokenizer = SecurityTokenizer(vocab_size=5000)
    
    training_texts = [
        "Critical security alert detected on host workstation-01",
        "Malware signature matched: Emotet banking trojan",
        "Network connection to known C2 server blocked",
        "User admin@company.com failed authentication 5 times",
        "Suspicious PowerShell execution detected",
    ]
    
    tokenizer.train(training_texts)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {len(tokenizer.special_tokens)}")
    
    # Encoding demo
    print("\n🔢 Encoding Example:")
    test_text = "Alert from 192.168.1.100 regarding CVE-2024-1234"
    tokens = tokenizer.encode(test_text)
    print(f"  Input: {test_text}")
    print(f"  Tokens: {tokens[:10]}... ({len(tokens)} total)")
    
    return tokenizer


def demo_data_generator():
    """Demonstrate the data generator."""
    print("\n" + "="*60)
    print("📊 DATA GENERATOR DEMO")
    print("="*60)
    
    from soc_slm.data import SecurityDataGenerator, MITREGenerator, IOCGenerator
    
    # MITRE demo
    print("\n🎯 MITRE ATT&CK Generator:")
    mitre = MITREGenerator()
    print(f"  Tactics: {len(mitre.tactics)}")
    print(f"  Techniques: {len(mitre.techniques)}")
    
    technique = mitre.random_technique()
    print(f"  Sample technique: {technique['id']} - {technique['name']}")
    
    # IOC demo
    print("\n🔍 IOC Generator:")
    ioc = IOCGenerator()
    print(f"  Random IP: {ioc.random_ip()}")
    print(f"  Random Domain: {ioc.random_domain()}")
    print(f"  Random CVE: {ioc.random_cve()}")
    print(f"  Random MD5: {ioc.random_hash('md5')}")
    
    # Full dataset demo
    print("\n📦 Dataset Generation:")
    generator = SecurityDataGenerator()
    
    # Generate small sample
    intent_samples = generator.generate_intent_samples(5)
    print(f"  Intent samples: {len(intent_samples)}")
    print(f"    Example: '{intent_samples[0]['text'][:50]}...' → {intent_samples[0]['label']}")
    
    qa_samples = generator.generate_qa_pairs(3)
    print(f"  Q&A pairs: {len(qa_samples)}")
    print(f"    Q: {qa_samples[0]['question'][:50]}...")
    
    alert_samples = generator.generate_alert_scenarios(3)
    print(f"  Alert scenarios: {len(alert_samples)}")
    print(f"    Severity: {alert_samples[0]['severity']}")
    
    return generator


def demo_model_architecture():
    """Demonstrate the model architecture."""
    print("\n" + "="*60)
    print("🧠 MODEL ARCHITECTURE DEMO")
    print("="*60)
    
    import torch
    from soc_slm.model import SOCModelConfig, SOCForCausalLM, SOCForSequenceClassification
    
    # Show available presets
    print("\n📐 Available Model Sizes:")
    presets = ["soc-slm-125m", "soc-slm-350m", "soc-slm-760m", "soc-slm-1b"]
    for preset in presets:
        config = SOCModelConfig.from_preset(preset)
        params = (config.vocab_size * config.hidden_size + 
                  config.num_hidden_layers * (
                      4 * config.hidden_size * config.hidden_size +
                      config.hidden_size * config.intermediate_size * 2
                  ))
        print(f"  {preset}: ~{params/1e6:.0f}M params, {config.hidden_size} hidden, {config.num_hidden_layers} layers")
    
    # Create small model for demo
    print("\n🔧 Creating Demo Model (tiny):")
    config = SOCModelConfig(
        vocab_size=5000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512
    )
    model = SOCForCausalLM(config)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    
    # Forward pass demo
    print("\n⚡ Forward Pass:")
    model.eval()
    input_ids = torch.randint(0, 5000, (1, 32))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {outputs.logits.shape}")
    
    # Generation demo
    print("\n✨ Generation:")
    with torch.no_grad():
        generated = model.generate(input_ids[:, :10], max_length=20, temperature=1.0)
    print(f"  Generated sequence length: {generated.shape[1]}")
    
    return model, config


def demo_integration():
    """Demonstrate the chatbot integration."""
    print("\n" + "="*60)
    print("🔌 INTEGRATION DEMO")
    print("="*60)
    
    from soc_slm.inference.integration import IntentClassifier, EntityExtractor, QueryInterpreter
    
    # Intent classification
    print("\n🎯 Intent Classification:")
    classifier = IntentClassifier()
    
    test_queries = [
        "What is the severity of this alert?",
        "Search for events from the last hour",
        "Is 192.168.1.100 a known malicious IP?",
        "How do I respond to this ransomware incident?",
        "Show me statistics for failed logins today",
    ]
    
    for query in test_queries:
        intent = classifier.classify(query)
        print(f"  '{query[:40]}...' → {intent}")
    
    # Entity extraction
    print("\n🔍 Entity Extraction:")
    extractor = EntityExtractor()
    
    text = """
    Alert: Host workstation-01 (192.168.1.100) connected to evil.com.
    User john.doe attempted to exploit CVE-2024-5678 using T1059.
    File hash: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4
    """
    
    entities = extractor.extract(text)
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {values}")
    
    # Query interpretation
    print("\n📝 Query Interpretation:")
    interpreter = QueryInterpreter()
    
    queries = [
        "Show events from 10.0.0.1 in the last 24 hours",
        "Find all critical alerts from yesterday",
    ]
    
    for query in queries:
        interpreted = interpreter.interpret(query)
        print(f"  Query: '{query}'")
        print(f"    Filters: {interpreted.get('filters', {})}")
        print(f"    Time range: {interpreted.get('time_range', 'not specified')}")


def demo_inference():
    """Demonstrate the inference capabilities."""
    print("\n" + "="*60)
    print("🚀 INFERENCE DEMO")
    print("="*60)
    
    from soc_slm.inference import GenerationConfig, InferenceMetrics, SOCResponseTemplates
    
    # Generation config
    print("\n⚙️ Generation Config:")
    config = GenerationConfig(
        max_length=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1
    )
    print(f"  Max length: {config.max_length}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Top-p: {config.top_p}")
    print(f"  Top-k: {config.top_k}")
    
    # Metrics tracking
    print("\n📈 Metrics Tracking:")
    metrics = InferenceMetrics()
    
    # Simulate some requests
    metrics.record_request(latency_ms=120, tokens_generated=50)
    metrics.record_request(latency_ms=150, tokens_generated=75)
    metrics.record_request(latency_ms=100, tokens_generated=40)
    
    print(f"  Total requests: {metrics.total_requests}")
    print(f"  Total tokens: {metrics.total_tokens}")
    print(f"  Average latency: {metrics.average_latency_ms:.2f}ms")
    print(f"  Tokens/second: {metrics.tokens_per_second:.2f}")
    
    # Response templates
    print("\n📋 Response Templates:")
    templates = SOCResponseTemplates()
    
    triage_response = templates.triage_response(
        severity="high",
        alert_type="Malware Detection",
        summary="Emotet banking trojan detected on host",
        recommendations=["Isolate affected host", "Collect memory dump", "Block C2 domains"]
    )
    print(f"  Triage response preview:")
    print(f"    {triage_response[:100]}...")


def main():
    """Run all demos."""
    parser = argparse.ArgumentParser(description="SOC-SLM Quick Start Demo")
    parser.add_argument("--component", choices=["tokenizer", "data", "model", "integration", "inference", "all"],
                        default="all", help="Which component to demo")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🛡️  SOC-SLM: Small Language Model for Security Operations")
    print("="*60)
    print("This demo showcases the key components of the SOC-SLM package.")
    
    try:
        if args.component in ["tokenizer", "all"]:
            demo_tokenizer()
        
        if args.component in ["data", "all"]:
            demo_data_generator()
        
        if args.component in ["model", "all"]:
            demo_model_architecture()
        
        if args.component in ["integration", "all"]:
            demo_integration()
        
        if args.component in ["inference", "all"]:
            demo_inference()
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Generate full training dataset:")
        print("     python -c \"from soc_slm.data import SecurityDataGenerator; g = SecurityDataGenerator(); g.save_dataset(g.generate_full_dataset(), 'data.json')\"")
        print("  2. Train the model:")
        print("     python -m soc_slm.scripts.train --train-data data.json --model-size soc-slm-125m")
        print("  3. Run inference:")
        print("     See README.md for inference examples")
        print()
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
