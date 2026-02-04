# SOC-SLM: Small Language Model for Security Operations

A production-ready Small Language Model (SLM) designed specifically for Security Operations Center (SOC) chatbots. Built from scratch with security-first principles and optimized for real-world SOC workflows.

## 🎯 Features

- **Custom Security Tokenizer**: BPE tokenizer optimized for security terminology with special tokens for IPs, domains, CVEs, MITRE techniques, and more
- **Flexible Architecture**: Transformer decoder with RoPE, SwiGLU, and Multi-Query Attention (125M to 1B parameters)
- **Synthetic Data Generator**: Generate diverse security training data including Q&A, alerts, conversations, and OCSF events
- **Production Training Pipeline**: Distributed training, mixed precision, gradient accumulation, early stopping
- **Optimized Inference**: INT8/INT4 quantization, KV-cache, streaming generation, async processing
- **Chatbot Integration**: Ready-to-use FastAPI endpoints with intent classification and entity extraction

## 📁 Project Structure

```
soc-slm/
├── tokenizer/
│   └── security_tokenizer.py    # Custom BPE tokenizer for security text
├── model/
│   └── architecture.py          # Transformer architecture (125M-1B params)
├── data/
│   └── data_generator.py        # Synthetic security data generation
├── training/
│   └── trainer.py               # Distributed training pipeline
├── inference/
│   ├── engine.py                # Production inference with quantization
│   └── integration.py           # FastAPI integration for SOC chatbot
├── config/
│   └── settings.py              # Configuration management
├── utils/
│   └── helpers.py               # Utility functions
├── scripts/
│   └── train.py                 # Training CLI script
└── tests/
    └── test_components.py       # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 1. Generate Training Data

```python
from soc_slm.data import SecurityDataGenerator

generator = SecurityDataGenerator()
dataset = generator.generate_full_dataset()
generator.save_dataset(dataset, "training_data.json")

print(f"Generated {len(dataset)} training samples")
```

### 2. Train the Tokenizer

```python
from soc_slm.tokenizer import SecurityTokenizer

tokenizer = SecurityTokenizer(vocab_size=32000)

# Train on your security data
texts = [sample["text"] for sample in dataset]
tokenizer.train(texts)
tokenizer.save("./tokenizer_output")
```

### 3. Train the Model

```python
from soc_slm.training import prepare_training, TrainingConfig

config = TrainingConfig(
    model_size="soc-slm-125m",
    epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    output_dir="./model_output"
)

trainer, _ = prepare_training(
    train_data=dataset,
    tokenizer_path="./tokenizer_output",
    config=config
)

trainer.train()
```

Or use the CLI:

```bash
python -m soc_slm.scripts.train \
    --train-data training_data.json \
    --tokenizer-path ./tokenizer_output \
    --model-size soc-slm-125m \
    --epochs 10 \
    --output-dir ./model_output
```

### 4. Run Inference

```python
from soc_slm.inference import SOCInferenceEngine, GenerationConfig

engine = SOCInferenceEngine(
    model_path="./model_output/final",
    quantization="int8"  # Options: int8, int4, fp16, bf16, none
)

response = engine.generate(
    "What is the severity of CVE-2024-1234?",
    config=GenerationConfig(max_length=256, temperature=0.7)
)
print(response)
```

### 5. Integrate with SOC Chatbot

```python
from soc_slm.inference import SOCChatbotIntegration, AgentRequest

integration = SOCChatbotIntegration(model_path="./model_output/final")

request = AgentRequest(
    message="Analyze suspicious activity from 192.168.1.100",
    conversation_id="session_123"
)

response = integration.process_request(request)
print(f"Intent: {response.intent}")
print(f"Response: {response.response}")
print(f"Entities: {response.entities}")
```

## 🔧 Model Configurations

| Model | Parameters | Hidden | Layers | Heads | Use Case |
|-------|------------|--------|--------|-------|----------|
| soc-slm-125m | 125M | 768 | 12 | 12 | Development, testing |
| soc-slm-350m | 350M | 1024 | 24 | 16 | Small deployments |
| soc-slm-760m | 760M | 1536 | 24 | 12 | Production |
| soc-slm-1b | 1B | 2048 | 24 | 16 | High accuracy |

## 🛡️ Security-Specific Features

### Entity Recognition

The tokenizer automatically recognizes:
- IPv4/IPv6 addresses
- Domain names and URLs
- CVE identifiers (CVE-YYYY-NNNNN)
- MITRE ATT&CK techniques (T####)
- File hashes (MD5, SHA1, SHA256)
- Windows registry keys
- Base64-encoded data
- MAC addresses
- File paths

### Intent Classification

Built-in classification for SOC operations:
- `triage` - Alert assessment and prioritization
- `query` - Data lake queries
- `threat_intel` - Threat intelligence lookups
- `incident_response` - IR playbook guidance
- `statistics` - Metrics and analytics
- `timeline` - Event timeline analysis

### MITRE ATT&CK Integration

Full coverage of MITRE ATT&CK framework:
- 12 tactics with technique mappings
- Technique descriptions and examples
- Attack chain correlation support

## 📊 Training Data Types

The data generator creates:
- **Intent Classification** (2000 samples): Labeled queries for intent detection
- **Q&A Pairs** (2000 samples): Security knowledge Q&A
- **Alert Scenarios** (1000 samples): Realistic security alerts
- **Conversations** (500 samples): Multi-turn SOC dialogues
- **OCSF Events** (1000 samples): Normalized security events

## ⚡ Performance Optimization

### Quantization

```python
from soc_slm.inference import ModelQuantizer

quantizer = ModelQuantizer(model, tokenizer)

# INT8 quantization (2x smaller, minimal quality loss)
int8_model = quantizer.quantize_int8()

# INT4 quantization (4x smaller, some quality loss)
int4_model = quantizer.quantize_int4()
```

### Streaming Generation

```python
for token in engine.generate_stream("Analyze this alert...", config):
    print(token, end="", flush=True)
```

### Async Processing

```python
from soc_slm.inference import AsyncInferenceServer

server = AsyncInferenceServer(engine, num_workers=4)
server.start()

future = server.submit_request("Your prompt here")
result = future.result(timeout=30)
```

## 🔌 FastAPI Integration

The integration module provides ready-to-use endpoints:

```python
from fastapi import FastAPI
from soc_slm.inference.integration import create_slm_router

app = FastAPI()
app.include_router(create_slm_router("./model_output/final"), prefix="/api/v1")
```

Available endpoints:
- `POST /slm/generate` - Text generation
- `POST /slm/classify` - Intent classification
- `POST /slm/extract` - Entity extraction
- `POST /slm/interpret` - Query interpretation
- `GET /slm/health` - Health check

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=soc_slm --cov-report=html
```

## 📈 Monitoring

The inference engine tracks metrics automatically:

```python
metrics = engine.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.average_latency_ms:.2f}ms")
print(f"Throughput: {metrics.tokens_per_second:.2f} tokens/sec")
```

## 🔗 Integration with SOC Chatbot

This SLM is designed to integrate with the [SOC Chatbot](https://github.com/mayanklau/soc-chatbot):

1. Copy the `soc-slm` directory to your chatbot project
2. Add the FastAPI router to your main application
3. Configure the model path in your settings
4. The chatbot agents will automatically use the SLM for enhanced responses

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

Built for production SOC environments. Optimized for real-world security operations.
