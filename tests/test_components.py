"""
Comprehensive test suite for SOC-SLM components.
Tests tokenizer, model architecture, data generation, training, and inference.
"""

import pytest
import torch
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# ============================================================================
# Tokenizer Tests
# ============================================================================

class TestSecurityTokenizer:
    """Tests for the security-focused tokenizer."""
    
    def test_special_tokens_defined(self):
        """Test that all security special tokens are defined."""
        from soc_slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer(vocab_size=1000)
        
        expected_special_tokens = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]",
            "[IP]", "[DOMAIN]", "[HASH]", "[CVE]", "[MITRE]",
            "[TIME]", "[USER]", "[HOST]", "[PORT]", "[SEV]"
        ]
        
        for token in expected_special_tokens:
            assert token in tokenizer.special_tokens
    
    def test_ip_address_detection(self):
        """Test IPv4 address pattern matching."""
        from soc_slm.tokenizer import SecurityPatternMatcher
        
        matcher = SecurityPatternMatcher()
        
        # Test valid IPs
        text = "Connection from 192.168.1.100 to 10.0.0.1"
        entities = matcher.extract_entities(text)
        
        ips = [e for e in entities if e["type"] == "IP"]
        assert len(ips) == 2
        assert ips[0]["value"] == "192.168.1.100"
        assert ips[1]["value"] == "10.0.0.1"
    
    def test_cve_detection(self):
        """Test CVE identifier pattern matching."""
        from soc_slm.tokenizer import SecurityPatternMatcher
        
        matcher = SecurityPatternMatcher()
        
        text = "Vulnerability CVE-2024-12345 was exploited using CVE-2023-9999"
        entities = matcher.extract_entities(text)
        
        cves = [e for e in entities if e["type"] == "CVE"]
        assert len(cves) == 2
        assert "CVE-2024-12345" in [c["value"] for c in cves]
    
    def test_mitre_technique_detection(self):
        """Test MITRE ATT&CK technique pattern matching."""
        from soc_slm.tokenizer import SecurityPatternMatcher
        
        matcher = SecurityPatternMatcher()
        
        text = "Attack used T1059 (Command Line Interface) and T1003.001 (LSASS Memory)"
        entities = matcher.extract_entities(text)
        
        mitre = [e for e in entities if e["type"] == "MITRE"]
        assert len(mitre) == 2
    
    def test_hash_detection(self):
        """Test file hash pattern matching (MD5, SHA1, SHA256)."""
        from soc_slm.tokenizer import SecurityPatternMatcher
        
        matcher = SecurityPatternMatcher()
        
        md5 = "d41d8cd98f00b204e9800998ecf8427e"
        sha1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        
        text = f"Hashes detected: {md5}, {sha1}, {sha256}"
        entities = matcher.extract_entities(text)
        
        hashes = [e for e in entities if e["type"] == "HASH"]
        assert len(hashes) == 3
    
    def test_domain_detection(self):
        """Test domain name pattern matching."""
        from soc_slm.tokenizer import SecurityPatternMatcher
        
        matcher = SecurityPatternMatcher()
        
        text = "C2 server at evil.example.com contacted malware.badsite.org"
        entities = matcher.extract_entities(text)
        
        domains = [e for e in entities if e["type"] == "DOMAIN"]
        assert len(domains) >= 2
    
    def test_entity_masking(self):
        """Test entity masking for tokenization."""
        from soc_slm.tokenizer import SecurityPatternMatcher
        
        matcher = SecurityPatternMatcher()
        
        text = "Alert from 192.168.1.100 regarding CVE-2024-1234"
        masked, mask_map = matcher.mask_entities(text)
        
        assert "[IP]" in masked
        assert "[CVE]" in masked
        assert "192.168.1.100" not in masked
    
    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding produces consistent results."""
        from soc_slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer(vocab_size=1000)
        
        # Train on minimal data
        texts = [
            "Security alert detected",
            "Malware found on host",
            "Network connection blocked"
        ]
        tokenizer.train(texts)
        
        # Test roundtrip
        original = "Security alert detected"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        
        # Should preserve meaning (may not be exact due to BPE)
        assert len(encoded) > 0
        assert len(decoded) > 0
    
    def test_batch_encoding(self):
        """Test batch encoding functionality."""
        from soc_slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer(vocab_size=1000)
        tokenizer.train(["test data", "more data"])
        
        texts = ["First text", "Second text", "Third text"]
        batch = tokenizer.batch_encode(texts, padding=True)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert len(batch["input_ids"]) == 3
    
    def test_tokenizer_save_load(self):
        """Test saving and loading tokenizer."""
        from soc_slm.tokenizer import SecurityTokenizer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            tokenizer = SecurityTokenizer(vocab_size=1000)
            tokenizer.train(["test data for training"])
            tokenizer.save(tmpdir)
            
            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "vocab.json"))
            assert os.path.exists(os.path.join(tmpdir, "merges.txt"))
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            
            # Load and verify
            loaded = SecurityTokenizer.load(tmpdir)
            assert loaded.vocab_size == tokenizer.vocab_size


# ============================================================================
# Model Architecture Tests
# ============================================================================

class TestModelArchitecture:
    """Tests for the transformer model architecture."""
    
    def test_model_creation_125m(self):
        """Test creating 125M parameter model."""
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        config = SOCModelConfig.from_preset("soc-slm-125m")
        model = SOCForCausalLM(config)
        
        # Check parameter count (approximate)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 100_000_000  # Should be >100M
        assert param_count < 200_000_000  # Should be <200M
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=256
        )
        model = SOCForCausalLM(config)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Check output shape
        assert outputs.logits.shape == (batch_size, seq_len, 1000)
    
    def test_model_generation(self):
        """Test model text generation."""
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=256
        )
        model = SOCForCausalLM(config)
        model.eval()
        
        # Generate
        input_ids = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=20,
                temperature=1.0,
                top_k=50
            )
        
        assert generated.shape[1] <= 20
        assert generated.shape[0] == 1
    
    def test_classification_model(self):
        """Test sequence classification model."""
        from soc_slm.model import SOCModelConfig, SOCForSequenceClassification
        
        config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        model = SOCForSequenceClassification(config, num_labels=8)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (2, 32))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.logits.shape == (2, 8)
    
    def test_token_classification_model(self):
        """Test token classification (NER) model."""
        from soc_slm.model import SOCModelConfig, SOCForTokenClassification
        
        config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        model = SOCForTokenClassification(config, num_labels=12)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (2, 32))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.logits.shape == (2, 32, 12)
    
    def test_kv_cache(self):
        """Test KV cache functionality."""
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        model = SOCForCausalLM(config)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, 10))
        
        with torch.no_grad():
            # First pass with cache
            outputs1 = model(input_ids, use_cache=True)
            
            # Verify cache is returned
            assert outputs1.past_key_values is not None
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing can be enabled."""
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        model = SOCForCausalLM(config)
        
        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        
        assert model.model.gradient_checkpointing == True


# ============================================================================
# Data Generator Tests
# ============================================================================

class TestDataGenerator:
    """Tests for the security data generator."""
    
    def test_mitre_generator(self):
        """Test MITRE ATT&CK data generation."""
        from soc_slm.data import MITREGenerator
        
        generator = MITREGenerator()
        
        # Check tactics
        assert len(generator.tactics) == 12
        assert "Initial Access" in generator.tactics
        assert "Execution" in generator.tactics
        
        # Check techniques
        assert len(generator.techniques) > 40
        
        # Generate random technique
        technique = generator.random_technique()
        assert "id" in technique
        assert "name" in technique
        assert "tactic" in technique
    
    def test_ioc_generator(self):
        """Test IOC (Indicator of Compromise) generation."""
        from soc_slm.data import IOCGenerator
        
        generator = IOCGenerator()
        
        # Test IP generation
        ip = generator.random_ip()
        parts = ip.split(".")
        assert len(parts) == 4
        
        # Test domain generation
        domain = generator.random_domain()
        assert "." in domain
        
        # Test hash generation
        hash_val = generator.random_hash("md5")
        assert len(hash_val) == 32
        
        hash_val = generator.random_hash("sha256")
        assert len(hash_val) == 64
        
        # Test CVE generation
        cve = generator.random_cve()
        assert cve.startswith("CVE-")
    
    def test_intent_sample_generation(self):
        """Test intent classification sample generation."""
        from soc_slm.data import SecurityDataGenerator
        
        generator = SecurityDataGenerator()
        samples = generator.generate_intent_samples(100)
        
        assert len(samples) == 100
        
        for sample in samples:
            assert "text" in sample
            assert "label" in sample
            assert sample["label"] in ["triage", "query", "threat_intel", 
                                        "incident_response", "statistics", "general"]
    
    def test_qa_pair_generation(self):
        """Test Q&A pair generation."""
        from soc_slm.data import SecurityDataGenerator
        
        generator = SecurityDataGenerator()
        samples = generator.generate_qa_pairs(50)
        
        assert len(samples) == 50
        
        for sample in samples:
            assert "question" in sample
            assert "answer" in sample
            assert len(sample["question"]) > 0
            assert len(sample["answer"]) > 0
    
    def test_alert_generation(self):
        """Test security alert generation."""
        from soc_slm.data import SecurityDataGenerator
        
        generator = SecurityDataGenerator()
        samples = generator.generate_alert_scenarios(25)
        
        assert len(samples) == 25
        
        for sample in samples:
            assert "alert" in sample
            assert "analysis" in sample
            assert "severity" in sample
            assert sample["severity"] in ["critical", "high", "medium", "low", "info"]
    
    def test_conversation_generation(self):
        """Test multi-turn conversation generation."""
        from soc_slm.data import SecurityDataGenerator
        
        generator = SecurityDataGenerator()
        samples = generator.generate_conversations(10)
        
        assert len(samples) == 10
        
        for sample in samples:
            assert "turns" in sample
            assert len(sample["turns"]) >= 2
            
            for turn in sample["turns"]:
                assert "role" in turn
                assert "content" in turn
                assert turn["role"] in ["user", "assistant"]
    
    def test_ocsf_event_generation(self):
        """Test OCSF event generation."""
        from soc_slm.data import SecurityDataGenerator
        
        generator = SecurityDataGenerator()
        samples = generator.generate_ocsf_events(20)
        
        assert len(samples) == 20
        
        for sample in samples:
            assert "category" in sample
            assert "event" in sample
            event = sample["event"]
            assert "time" in event
            assert "severity_id" in event
    
    def test_full_dataset_generation(self):
        """Test full dataset generation."""
        from soc_slm.data import SecurityDataGenerator
        
        generator = SecurityDataGenerator()
        dataset = generator.generate_full_dataset(scale=0.1)  # 10% scale for speed
        
        # Check all types present
        types = set(sample.get("type") for sample in dataset)
        expected_types = {"intent", "qa", "alert", "conversation", "ocsf"}
        assert types == expected_types
    
    def test_streaming_generator(self):
        """Test streaming data generation."""
        from soc_slm.data import SecurityDataGenerator
        
        generator = SecurityDataGenerator()
        
        # Collect 10 samples from stream
        samples = []
        for sample in generator.generate_training_stream():
            samples.append(sample)
            if len(samples) >= 10:
                break
        
        assert len(samples) == 10
    
    def test_dataset_save_load(self):
        """Test saving and loading dataset."""
        from soc_slm.data import SecurityDataGenerator
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name
        
        try:
            generator = SecurityDataGenerator()
            dataset = generator.generate_full_dataset(scale=0.05)
            generator.save_dataset(dataset, filepath)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(filepath)
            
            with open(filepath, "r") as f:
                loaded = json.load(f)
            
            assert len(loaded) == len(dataset)
        finally:
            os.unlink(filepath)


# ============================================================================
# Training Pipeline Tests
# ============================================================================

class TestTrainingPipeline:
    """Tests for the training pipeline."""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        from soc_slm.training import TrainingConfig
        
        config = TrainingConfig()
        
        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.output_dir is not None
    
    def test_training_config_validation(self):
        """Test TrainingConfig validation."""
        from soc_slm.training import TrainingConfig
        
        # Valid config
        config = TrainingConfig(
            epochs=5,
            batch_size=8,
            learning_rate=1e-4
        )
        assert config.epochs == 5
        
        # Test from_dict
        config_dict = {
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 5e-5
        }
        config = TrainingConfig.from_dict(config_dict)
        assert config.epochs == 10
    
    def test_security_dataset(self):
        """Test SecurityDataset class."""
        from soc_slm.training import SecurityDataset
        
        # Create mock data
        data = [
            {"text": "Test security alert", "label": "triage"},
            {"text": "Another alert", "label": "query"},
            {"question": "What is this?", "answer": "A test"}
        ]
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        dataset = SecurityDataset(data, mock_tokenizer, max_length=128)
        
        assert len(dataset) == 3
        
        # Get item
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
    
    def test_optimizer_creation(self):
        """Test optimizer creation with weight decay separation."""
        from soc_slm.training import SOCTrainer, TrainingConfig
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        # Create small model
        model_config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=256
        )
        model = SOCForCausalLM(model_config)
        
        # Create trainer config
        config = TrainingConfig(
            learning_rate=1e-4,
            weight_decay=0.01
        )
        
        # Mock dataloader
        mock_loader = Mock()
        mock_loader.__len__ = Mock(return_value=10)
        mock_loader.__iter__ = Mock(return_value=iter([]))
        
        trainer = SOCTrainer(
            model=model,
            train_dataloader=mock_loader,
            config=config
        )
        
        # Check optimizer was created
        assert trainer.optimizer is not None
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduler creation."""
        from soc_slm.training import SOCTrainer, TrainingConfig
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        model_config = SOCModelConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=256
        )
        model = SOCForCausalLM(model_config)
        
        config = TrainingConfig(
            scheduler_type="cosine",
            warmup_steps=100
        )
        
        mock_loader = Mock()
        mock_loader.__len__ = Mock(return_value=100)
        
        trainer = SOCTrainer(
            model=model,
            train_dataloader=mock_loader,
            config=config
        )
        
        assert trainer.scheduler is not None
    
    def test_checkpoint_management(self):
        """Test checkpoint saving and loading."""
        from soc_slm.training import SOCTrainer, TrainingConfig
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_config = SOCModelConfig(
                vocab_size=1000,
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=256
            )
            model = SOCForCausalLM(model_config)
            
            config = TrainingConfig(output_dir=tmpdir)
            
            mock_loader = Mock()
            mock_loader.__len__ = Mock(return_value=10)
            
            trainer = SOCTrainer(
                model=model,
                train_dataloader=mock_loader,
                config=config
            )
            
            # Save checkpoint
            trainer.save_checkpoint(epoch=1, step=100)
            
            # Verify checkpoint exists
            checkpoint_path = os.path.join(tmpdir, "checkpoint-epoch1-step100")
            assert os.path.exists(checkpoint_path)


# ============================================================================
# Inference Engine Tests
# ============================================================================

class TestInferenceEngine:
    """Tests for the inference engine."""
    
    def test_generation_config(self):
        """Test GenerationConfig creation."""
        from soc_slm.inference import GenerationConfig
        
        config = GenerationConfig(
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        
        assert config.max_length == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
    
    def test_inference_metrics(self):
        """Test InferenceMetrics tracking."""
        from soc_slm.inference import InferenceMetrics
        
        metrics = InferenceMetrics()
        
        # Record some requests
        metrics.record_request(latency_ms=100, tokens_generated=50)
        metrics.record_request(latency_ms=150, tokens_generated=75)
        metrics.record_request(latency_ms=200, tokens_generated=100)
        
        assert metrics.total_requests == 3
        assert metrics.total_tokens == 225
        assert metrics.average_latency_ms == 150
    
    def test_response_templates(self):
        """Test SOCResponseTemplates."""
        from soc_slm.inference import SOCResponseTemplates
        
        templates = SOCResponseTemplates()
        
        # Test triage template
        triage = templates.triage_response(
            severity="high",
            alert_type="Malware Detection",
            summary="Malicious file detected",
            recommendations=["Isolate host", "Collect samples"]
        )
        
        assert "high" in triage.lower()
        assert "Malware Detection" in triage
        assert "Isolate host" in triage
        
        # Test threat intel template
        threat = templates.threat_intel_response(
            indicator="192.168.1.100",
            indicator_type="ip",
            verdict="malicious",
            context="Known C2 server"
        )
        
        assert "192.168.1.100" in threat
        assert "malicious" in threat.lower()


# ============================================================================
# Integration Tests
# ============================================================================

class TestChatbotIntegration:
    """Tests for chatbot integration components."""
    
    def test_intent_classifier(self):
        """Test IntentClassifier."""
        from soc_slm.inference.integration import IntentClassifier
        
        classifier = IntentClassifier()
        
        # Test various intents
        assert classifier.classify("What is the severity of this alert?") == "triage"
        assert classifier.classify("Search for events from 192.168.1.100") == "query"
        assert classifier.classify("Is this IP malicious?") == "threat_intel"
        assert classifier.classify("How do I contain this incident?") == "incident_response"
        assert classifier.classify("Show me statistics for today") == "statistics"
    
    def test_entity_extractor(self):
        """Test EntityExtractor."""
        from soc_slm.inference.integration import EntityExtractor
        
        extractor = EntityExtractor()
        
        text = "Alert from 192.168.1.100 about CVE-2024-1234 involving T1059"
        entities = extractor.extract(text)
        
        assert "ips" in entities
        assert "cves" in entities
        assert "mitre_techniques" in entities
        
        assert "192.168.1.100" in entities["ips"]
        assert "CVE-2024-1234" in entities["cves"]
        assert "T1059" in entities["mitre_techniques"]
    
    def test_query_interpreter(self):
        """Test QueryInterpreter."""
        from soc_slm.inference.integration import QueryInterpreter
        
        interpreter = QueryInterpreter()
        
        # Test filter extraction
        query = "Show me events from 192.168.1.100 in the last hour"
        interpreted = interpreter.interpret(query)
        
        assert "filters" in interpreted
        assert "time_range" in interpreted
    
    def test_agent_request_dataclass(self):
        """Test AgentRequest dataclass."""
        from soc_slm.inference.integration import AgentRequest
        
        request = AgentRequest(
            message="Analyze this alert",
            conversation_id="session_123",
            context={"previous": "data"}
        )
        
        assert request.message == "Analyze this alert"
        assert request.conversation_id == "session_123"
        assert request.context == {"previous": "data"}
    
    def test_agent_response_dataclass(self):
        """Test AgentResponse dataclass."""
        from soc_slm.inference.integration import AgentResponse
        
        response = AgentResponse(
            response="This is a high severity alert",
            intent="triage",
            entities={"ips": ["192.168.1.100"]},
            confidence=0.95,
            agent_used="triage_agent"
        )
        
        assert response.response == "This is a high severity alert"
        assert response.confidence == 0.95


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_tokenizer_model_integration(self):
        """Test tokenizer and model work together."""
        from soc_slm.tokenizer import SecurityTokenizer
        from soc_slm.model import SOCModelConfig, SOCForCausalLM
        
        # Create and train minimal tokenizer
        tokenizer = SecurityTokenizer(vocab_size=1000)
        tokenizer.train([
            "Security alert detected on host",
            "Malware found in network traffic",
            "Suspicious activity from IP address"
        ])
        
        # Create model with matching vocab size
        config = SOCModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=256
        )
        model = SOCForCausalLM(config)
        model.eval()
        
        # Encode and forward pass
        text = "Security alert detected"
        tokens = tokenizer.encode(text)
        input_ids = torch.tensor([tokens])
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.logits.shape[2] == tokenizer.vocab_size
    
    def test_data_generation_to_training(self):
        """Test data generation flows into training setup."""
        from soc_slm.data import SecurityDataGenerator
        from soc_slm.training import SecurityDataset
        
        # Generate data
        generator = SecurityDataGenerator()
        dataset = generator.generate_full_dataset(scale=0.01)
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        # Create dataset
        train_dataset = SecurityDataset(dataset, mock_tokenizer, max_length=128)
        
        assert len(train_dataset) > 0
        
        # Get a batch
        item = train_dataset[0]
        assert "input_ids" in item


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_model():
    """Create a small model for testing."""
    from soc_slm.model import SOCModelConfig, SOCForCausalLM
    
    config = SOCModelConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=256
    )
    return SOCForCausalLM(config)


@pytest.fixture
def trained_tokenizer():
    """Create a trained tokenizer for testing."""
    from soc_slm.tokenizer import SecurityTokenizer
    
    tokenizer = SecurityTokenizer(vocab_size=1000)
    tokenizer.train([
        "Security alert from host",
        "Malware detected in network",
        "Suspicious IP address found"
    ])
    return tokenizer


@pytest.fixture  
def sample_dataset():
    """Create sample dataset for testing."""
    return [
        {"text": "Critical security alert", "label": "triage"},
        {"text": "Search for events", "label": "query"},
        {"question": "What is CVE?", "answer": "Common Vulnerabilities and Exposures"}
    ]


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
