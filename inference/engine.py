"""
SOC-SLM Inference Engine
Production-ready inference with optimizations.

Features:
- Batched inference
- KV-Cache for efficient generation
- Quantization (INT8, INT4)
- Model serving API
- Streaming generation
- Response templates
- Security-specific post-processing
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Generator, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceMode(str, Enum):
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    min_new_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    num_beams: int = 1
    early_stopping: bool = True
    
    # SOC-specific
    include_reasoning: bool = False
    format_as_json: bool = False
    extract_entities: bool = False


@dataclass
class InferenceResult:
    """Result from inference."""
    text: str
    tokens: int
    latency_ms: float
    finish_reason: str = "stop"  # stop, length, error
    entities: Optional[Dict] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class InferenceMetrics:
    """Metrics for inference monitoring."""
    total_requests: int = 0
    total_tokens_generated: int = 0
    total_latency_ms: float = 0.0
    avg_tokens_per_request: float = 0.0
    avg_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    errors: int = 0
    
    def update(self, tokens: int, latency_ms: float, error: bool = False):
        self.total_requests += 1
        self.total_tokens_generated += tokens
        self.total_latency_ms += latency_ms
        self.avg_tokens_per_request = self.total_tokens_generated / self.total_requests
        self.avg_latency_ms = self.total_latency_ms / self.total_requests
        if latency_ms > 0:
            self.tokens_per_second = tokens / (latency_ms / 1000)
        if error:
            self.errors += 1


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    
    TORCH_AVAILABLE = True
    
    
    class QuantizationMode(str, Enum):
        NONE = "none"
        INT8 = "int8"
        INT4 = "int4"
        FP16 = "fp16"
        BF16 = "bf16"
    
    
    class ModelQuantizer:
        """Quantize models for efficient inference."""
        
        @staticmethod
        def quantize_int8(model: nn.Module) -> nn.Module:
            """Apply INT8 dynamic quantization."""
            return torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        
        @staticmethod
        def quantize_fp16(model: nn.Module) -> nn.Module:
            """Convert to FP16."""
            return model.half()
        
        @staticmethod
        def quantize_bf16(model: nn.Module) -> nn.Module:
            """Convert to BF16."""
            return model.to(torch.bfloat16)
        
        @staticmethod
        def apply_quantization(model: nn.Module, mode: QuantizationMode) -> nn.Module:
            """Apply specified quantization mode."""
            if mode == QuantizationMode.INT8:
                return ModelQuantizer.quantize_int8(model)
            elif mode == QuantizationMode.FP16:
                return ModelQuantizer.quantize_fp16(model)
            elif mode == QuantizationMode.BF16:
                return ModelQuantizer.quantize_bf16(model)
            return model
    
    
    class KVCache:
        """Efficient KV-Cache for autoregressive generation."""
        
        def __init__(self, num_layers: int, max_batch_size: int = 32, max_seq_len: int = 2048):
            self.num_layers = num_layers
            self.max_batch_size = max_batch_size
            self.max_seq_len = max_seq_len
            self.cache: List[Optional[Tuple[Tensor, Tensor]]] = [None] * num_layers
            self.seq_len = 0
        
        def update(self, layer_idx: int, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
            """Update cache for a layer and return full K, V."""
            if self.cache[layer_idx] is None:
                self.cache[layer_idx] = (key, value)
            else:
                prev_k, prev_v = self.cache[layer_idx]
                self.cache[layer_idx] = (
                    torch.cat([prev_k, key], dim=2),
                    torch.cat([prev_v, value], dim=2)
                )
            return self.cache[layer_idx]
        
        def get(self, layer_idx: int) -> Optional[Tuple[Tensor, Tensor]]:
            """Get cached K, V for a layer."""
            return self.cache[layer_idx]
        
        def clear(self):
            """Clear all cached values."""
            self.cache = [None] * self.num_layers
            self.seq_len = 0
    
    
    class SOCInferenceEngine:
        """Production inference engine for SOC-SLM."""
        
        def __init__(
            self,
            model_path: str,
            tokenizer_path: Optional[str] = None,
            device: str = "auto",
            quantization: QuantizationMode = QuantizationMode.NONE,
            max_batch_size: int = 32,
            num_threads: int = 4
        ):
            self.model_path = Path(model_path)
            self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path / "tokenizer"
            self.device = self._setup_device(device)
            self.quantization = quantization
            self.max_batch_size = max_batch_size
            
            # Load components
            self.model = self._load_model()
            self.tokenizer = self._load_tokenizer()
            self.config = self._load_config()
            
            # Metrics
            self.metrics = InferenceMetrics()
            
            # Thread pool for async inference
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            
            # Request queue for batching
            self.request_queue = queue.Queue()
            self.batch_timeout = 0.01  # 10ms
            
            logger.info(f"Loaded model on {self.device} with {quantization.value} quantization")
        
        def _setup_device(self, device: str) -> torch.device:
            """Determine compute device."""
            if device == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(device)
        
        def _load_model(self) -> nn.Module:
            """Load and optimize model."""
            from ..model.architecture import SOCForCausalLM, SOCModelConfig
            
            config_path = self.model_path / "config.json"
            model_file = self.model_path / "pytorch_model.bin"
            
            if config_path.exists():
                config = SOCModelConfig.load(str(config_path))
                model = SOCForCausalLM(config)
                
                if model_file.exists():
                    state_dict = torch.load(model_file, map_location=self.device)
                    model.load_state_dict(state_dict)
            else:
                # Create default model
                from ..model.architecture import create_soc_model
                model = create_soc_model("soc-slm-125m")
            
            # Apply quantization
            model = ModelQuantizer.apply_quantization(model, self.quantization)
            
            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            # Compile with torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device.type == 'cuda':
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}")
            
            return model
        
        def _load_tokenizer(self):
            """Load tokenizer."""
            from ..tokenizer.security_tokenizer import SecurityTokenizer
            
            if self.tokenizer_path.exists():
                return SecurityTokenizer.load(str(self.tokenizer_path))
            return SecurityTokenizer()
        
        def _load_config(self) -> Dict:
            """Load model config."""
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
            return {}
        
        @torch.no_grad()
        def generate(
            self,
            prompt: str,
            config: Optional[GenerationConfig] = None,
            stop_sequences: Optional[List[str]] = None
        ) -> InferenceResult:
            """Generate response for a single prompt."""
            config = config or GenerationConfig()
            start_time = time.time()
            
            try:
                # Encode prompt
                encoded = self.tokenizer.encode(
                    prompt,
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.device)
                
                # Generate
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode
                new_tokens = output_ids[0][input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
                
                # Check for stop sequences
                finish_reason = "stop"
                if stop_sequences:
                    for seq in stop_sequences:
                        if seq in generated_text:
                            generated_text = generated_text[:generated_text.index(seq)]
                            finish_reason = "stop_sequence"
                            break
                
                if len(new_tokens) >= config.max_new_tokens:
                    finish_reason = "length"
                
                # Extract entities if requested
                entities = None
                if config.extract_entities:
                    from ..tokenizer.security_tokenizer import SecurityPatternMatcher
                    entities = SecurityPatternMatcher.extract_entities(generated_text)
                
                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                num_tokens = len(new_tokens)
                self.metrics.update(num_tokens, latency_ms)
                
                return InferenceResult(
                    text=generated_text,
                    tokens=num_tokens,
                    latency_ms=latency_ms,
                    finish_reason=finish_reason,
                    entities=entities
                )
            
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.update(0, latency_ms, error=True)
                logger.error(f"Generation error: {e}")
                return InferenceResult(
                    text=f"Error: {str(e)}",
                    tokens=0,
                    latency_ms=latency_ms,
                    finish_reason="error"
                )
        
        @torch.no_grad()
        def generate_batch(
            self,
            prompts: List[str],
            config: Optional[GenerationConfig] = None
        ) -> List[InferenceResult]:
            """Generate responses for multiple prompts."""
            config = config or GenerationConfig()
            results = []
            
            # Process in batches
            for i in range(0, len(prompts), self.max_batch_size):
                batch_prompts = prompts[i:i + self.max_batch_size]
                batch_results = self._generate_batch_internal(batch_prompts, config)
                results.extend(batch_results)
            
            return results
        
        def _generate_batch_internal(
            self,
            prompts: List[str],
            config: GenerationConfig
        ) -> List[InferenceResult]:
            """Internal batch generation."""
            start_time = time.time()
            
            # Encode all prompts
            encodings = self.tokenizer.batch_encode(
                prompts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = torch.tensor(encodings['input_ids']).to(self.device)
            attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)
            
            results = []
            
            # Generate for each (simplified - could be optimized with batched generation)
            for i in range(len(prompts)):
                single_input = input_ids[i:i+1]
                single_mask = attention_mask[i:i+1]
                
                output_ids = self.model.generate(
                    single_input,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.do_sample,
                    repetition_penalty=config.repetition_penalty
                )
                
                new_tokens = output_ids[0][single_input.shape[1]:]
                generated_text = self.tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
                
                results.append(InferenceResult(
                    text=generated_text,
                    tokens=len(new_tokens),
                    latency_ms=0,  # Updated after
                    finish_reason="stop"
                ))
            
            # Update latency for all results
            total_latency = (time.time() - start_time) * 1000
            avg_latency = total_latency / len(results)
            for r in results:
                r.latency_ms = avg_latency
            
            return results
        
        def generate_stream(
            self,
            prompt: str,
            config: Optional[GenerationConfig] = None
        ) -> Generator[str, None, None]:
            """Stream generation token by token."""
            config = config or GenerationConfig()
            
            # Encode prompt
            encoded = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(self.device)
            
            past_key_values = None
            
            for _ in range(config.max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                logits = outputs['logits'][:, -1, :]
                past_key_values = outputs['past_key_values']
                
                # Sample next token
                if config.do_sample:
                    logits = logits / config.temperature
                    
                    if config.top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode and yield
                token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                yield token_text
                
                # Update input for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        @torch.no_grad()
        def classify_intent(self, text: str) -> Dict[str, float]:
            """Classify the intent of a security query."""
            from ..model.architecture import SOCForSequenceClassification, get_model_config
            
            intent_labels = [
                "triage", "query", "threat_intel", "incident_response",
                "statistics", "timeline", "search", "help"
            ]
            
            # Use simple keyword-based classification if no classifier model
            # This can be replaced with a fine-tuned classifier
            keywords = {
                "triage": ["triage", "prioritize", "priority", "urgent", "critical alert"],
                "query": ["show", "find", "search", "get", "list", "count"],
                "threat_intel": ["threat", "ioc", "indicator", "malicious", "reputation", "enrich"],
                "incident_response": ["respond", "playbook", "contain", "remediate", "isolate"],
                "statistics": ["statistics", "stats", "overview", "summary", "dashboard", "metrics"],
                "timeline": ["timeline", "history", "when", "sequence"],
                "search": ["search for", "look for", "find all"],
                "help": ["help", "how do i", "what can you", "explain"]
            }
            
            text_lower = text.lower()
            scores = {}
            
            for intent, kws in keywords.items():
                score = sum(1 for kw in kws if kw in text_lower)
                scores[intent] = score / len(kws) if kws else 0
            
            # Normalize
            total = sum(scores.values()) or 1
            return {k: v / total for k, v in scores.items()}
        
        @torch.no_grad()
        def get_embeddings(self, texts: List[str]) -> Tensor:
            """Get embeddings for texts."""
            encodings = self.tokenizer.batch_encode(texts, padding=True, truncation=True)
            input_ids = torch.tensor(encodings['input_ids']).to(self.device)
            attention_mask = torch.tensor(encodings['attention_mask']).to(self.device)
            
            # Forward pass
            outputs = self.model.model(input_ids, attention_mask)
            hidden_states = outputs[0]
            
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            return embeddings
        
        def get_metrics(self) -> Dict:
            """Get inference metrics."""
            return asdict(self.metrics)
        
        def reset_metrics(self):
            """Reset metrics."""
            self.metrics = InferenceMetrics()


except ImportError:
    TORCH_AVAILABLE = False
    
    class SOCInferenceEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for inference")


# ==============================================================================
# Response Templates for SOC Operations
# ==============================================================================

class SOCResponseTemplates:
    """Pre-defined response templates for common SOC operations."""
    
    @staticmethod
    def triage_response(alerts: List[Dict]) -> str:
        """Format triage results."""
        if not alerts:
            return "No alerts require triage at this time."
        
        response = "**Prioritized Alert Queue:**\n\n"
        for i, alert in enumerate(alerts, 1):
            severity = alert.get('severity', 'unknown').upper()
            name = alert.get('name', 'Unknown Alert')
            score = alert.get('risk_score', 0)
            response += f"{i}. [{severity}] {name} (Risk Score: {score})\n"
        
        return response
    
    @staticmethod
    def threat_intel_response(ioc: str, intel: Dict) -> str:
        """Format threat intelligence results."""
        response = f"**Threat Intelligence for {ioc}:**\n\n"
        
        response += f"- **Type**: {intel.get('type', 'Unknown')}\n"
        response += f"- **Reputation**: {intel.get('reputation', 'Unknown')}\n"
        response += f"- **Confidence**: {intel.get('confidence', 0)}%\n"
        
        if intel.get('threat_actors'):
            response += f"- **Associated Actors**: {', '.join(intel['threat_actors'])}\n"
        
        if intel.get('mitre_techniques'):
            response += f"- **MITRE Techniques**: {', '.join(intel['mitre_techniques'])}\n"
        
        if intel.get('recommendations'):
            response += f"\n**Recommendations**: {intel['recommendations']}\n"
        
        return response
    
    @staticmethod
    def incident_response_playbook(incident_type: str, steps: List[str]) -> str:
        """Format incident response playbook."""
        response = f"**Incident Response Playbook: {incident_type}**\n\n"
        
        for i, step in enumerate(steps, 1):
            response += f"{i}. {step}\n"
        
        response += "\n*Remember to document all actions taken.*"
        return response
    
    @staticmethod
    def query_results(results: List[Dict], query: str) -> str:
        """Format query results."""
        response = f"**Query Results** ({len(results)} matches)\n"
        response += f"Query: `{query}`\n\n"
        
        if not results:
            return response + "No results found."
        
        for i, result in enumerate(results[:10], 1):
            response += f"{i}. {json.dumps(result, indent=2)[:200]}...\n"
        
        if len(results) > 10:
            response += f"\n... and {len(results) - 10} more results."
        
        return response


# ==============================================================================
# Async Inference Server
# ==============================================================================

class AsyncInferenceServer:
    """Async server for handling inference requests."""
    
    def __init__(self, engine: 'SOCInferenceEngine'):
        self.engine = engine
        self.request_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """Start the async server."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()
        logger.info("Async inference server started")
    
    def stop(self):
        """Stop the async server."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Async inference server stopped")
    
    def submit(self, prompt: str, callback: Callable[[InferenceResult], None]):
        """Submit a request for async processing."""
        self.request_queue.put((prompt, callback))
    
    def _worker_loop(self):
        """Worker loop processing requests."""
        while self.running:
            try:
                prompt, callback = self.request_queue.get(timeout=0.1)
                result = self.engine.generate(prompt)
                callback(result)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
