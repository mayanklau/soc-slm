"""
SOC-SLM Model Architecture
Production-ready Small Language Model optimized for Security Operations.

Features:
- Efficient transformer architecture (125M-1B parameters)
- RoPE (Rotary Position Embeddings) for better position encoding
- SwiGLU activation for improved performance
- Multi-Query Attention (MQA) option for efficient inference
- Grouped Query Attention (GQA) option
- Flash Attention compatible
- KV-Cache for efficient generation
- Security-specific output heads
"""

import math
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class AttentionType(str, Enum):
    MHA = "mha"  # Multi-Head Attention
    MQA = "mqa"  # Multi-Query Attention  
    GQA = "gqa"  # Grouped Query Attention


@dataclass
class SOCModelConfig:
    """Configuration for SOC-SLM model."""
    
    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: Optional[int] = None  # For GQA/MQA
    
    # Architecture options
    attention_type: str = "mha"
    hidden_act: str = "swiglu"  # swiglu, gelu, relu
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    
    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    
    # Security-specific heads
    num_intent_classes: int = 8  # triage, query, threat_intel, ir, etc.
    num_severity_classes: int = 5  # critical, high, medium, low, info
    num_entity_types: int = 12  # ip, domain, hash, cve, mitre, etc.
    
    # Inference optimization
    use_cache: bool = True
    tie_word_embeddings: bool = True
    
    # Quantization hints
    quantization_config: Optional[Dict] = None
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attention_heads
    
    def save(self, path: str):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SOCModelConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            return cls(**json.load(f))


# Model size presets
MODEL_CONFIGS = {
    "soc-slm-125m": SOCModelConfig(
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
    "soc-slm-350m": SOCModelConfig(
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
    ),
    "soc-slm-760m": SOCModelConfig(
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,  # GQA
        attention_type="gqa",
    ),
    "soc-slm-1b": SOCModelConfig(
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_kv_heads=4,
        attention_type="gqa",
    ),
}


def get_model_config(name: str) -> SOCModelConfig:
    """Get predefined model configuration."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]


# ==============================================================================
# PyTorch Implementation
# ==============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    
    TORCH_AVAILABLE = True
    
    
    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization."""
        
        def __init__(self, hidden_size: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
        
        def forward(self, x: Tensor) -> Tensor:
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x
    
    
    class RotaryEmbedding(nn.Module):
        """Rotary Position Embedding (RoPE)."""
        
        def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
            super().__init__()
            self.dim = dim
            self.max_seq_len = max_seq_len
            self.theta = theta
            
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)
            
            self._set_cos_sin_cache(max_seq_len)
        
        def _set_cos_sin_cache(self, seq_len: int):
            t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cos_cached', emb.cos(), persistent=False)
            self.register_buffer('sin_cached', emb.sin(), persistent=False)
        
        def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
            if seq_len > self.max_seq_len:
                self._set_cos_sin_cache(seq_len)
            return (
                self.cos_cached[:seq_len],
                self.sin_cached[:seq_len]
            )
    
    
    def rotate_half(x: Tensor) -> Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    
    def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to query and key."""
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
    
    class SwiGLU(nn.Module):
        """SwiGLU activation function."""
        
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        def forward(self, x: Tensor) -> Tensor:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    
    class SOCAttention(nn.Module):
        """Multi-Head/Multi-Query/Grouped-Query Attention."""
        
        def __init__(self, config: SOCModelConfig, layer_idx: int):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_kv_heads or config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_groups = self.num_heads // self.num_kv_heads
            
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
            
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta
            )
            
            self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        
        def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
            use_cache: bool = False,
        ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
            batch_size, seq_len, _ = hidden_states.shape
            
            # Project Q, K, V
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            # Reshape
            query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # Position IDs
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            
            # Apply RoPE
            cos, sin = self.rotary_emb(hidden_states, seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            
            # KV Cache
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
            past_key_value = (key_states, value_states) if use_cache else None
            
            # Repeat KV for GQA
            if self.num_kv_groups > 1:
                key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
                value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)
            
            # Attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = self.attention_dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            return attn_output, past_key_value
    
    
    class SOCMLP(nn.Module):
        """MLP layer with configurable activation."""
        
        def __init__(self, config: SOCModelConfig):
            super().__init__()
            self.config = config
            
            if config.hidden_act == "swiglu":
                self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
            else:
                self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
                self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
                self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
                
                if config.hidden_act == "gelu":
                    self.act_fn = F.gelu
                elif config.hidden_act == "relu":
                    self.act_fn = F.relu
                else:
                    self.act_fn = F.silu
        
        def forward(self, x: Tensor) -> Tensor:
            if self.config.hidden_act == "swiglu":
                return self.mlp(x)
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    
    class SOCDecoderLayer(nn.Module):
        """Single transformer decoder layer."""
        
        def __init__(self, config: SOCModelConfig, layer_idx: int):
            super().__init__()
            self.self_attn = SOCAttention(config, layer_idx)
            self.mlp = SOCMLP(config)
            self.input_layernorm = RMSNorm(config.hidden_size)
            self.post_attention_layernorm = RMSNorm(config.hidden_size)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
            use_cache: bool = False,
        ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
            # Self-attention with residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, present_key_value = self.self_attn(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            
            # MLP with residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = residual + hidden_states
            
            return hidden_states, present_key_value
    
    
    class SOCModel(nn.Module):
        """Base SOC-SLM transformer model."""
        
        def __init__(self, config: SOCModelConfig):
            super().__init__()
            self.config = config
            
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList([
                SOCDecoderLayer(config, i) for i in range(config.num_hidden_layers)
            ])
            self.norm = RMSNorm(config.hidden_size)
            
            self.gradient_checkpointing = False
        
        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
            use_cache: bool = False,
        ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
            batch_size, seq_len = input_ids.shape
            
            # Embed tokens
            hidden_states = self.embed_tokens(input_ids)
            
            # Create causal mask
            if attention_mask is not None:
                # Expand attention mask to 4D
                attention_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            
            # Add causal mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            if attention_mask is not None:
                attention_mask = attention_mask + causal_mask
            else:
                attention_mask = causal_mask
            
            # Process layers
            present_key_values = [] if use_cache else None
            for i, layer in enumerate(self.layers):
                past_kv = past_key_values[i] if past_key_values else None
                
                hidden_states, present_kv = layer(
                    hidden_states, attention_mask, position_ids, past_kv, use_cache
                )
                
                if use_cache:
                    present_key_values.append(present_kv)
            
            hidden_states = self.norm(hidden_states)
            
            return hidden_states, present_key_values
    
    
    class SOCForCausalLM(nn.Module):
        """SOC-SLM for causal language modeling (text generation)."""
        
        def __init__(self, config: SOCModelConfig):
            super().__init__()
            self.config = config
            self.model = SOCModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
            use_cache: bool = False,
        ) -> Dict[str, Tensor]:
            hidden_states, present_key_values = self.model(
                input_ids, attention_mask, None, past_key_values, use_cache
            )
            
            logits = self.lm_head(hidden_states)
            
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
            
            return {
                'loss': loss,
                'logits': logits,
                'past_key_values': present_key_values,
                'hidden_states': hidden_states,
            }
        
        @torch.no_grad()
        def generate(
            self,
            input_ids: Tensor,
            max_new_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
            do_sample: bool = True,
            repetition_penalty: float = 1.1,
            eos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
        ) -> Tensor:
            """Generate text autoregressively."""
            eos_token_id = eos_token_id or self.config.eos_token_id
            pad_token_id = pad_token_id or self.config.pad_token_id
            
            batch_size = input_ids.shape[0]
            past_key_values = None
            
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids, use_cache=True, past_key_values=past_key_values)
                logits = outputs['logits'][:, -1, :]
                past_key_values = outputs['past_key_values']
                
                # Repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(input_ids[i].tolist()):
                            if logits[i, token_id] > 0:
                                logits[i, token_id] /= repetition_penalty
                            else:
                                logits[i, token_id] *= repetition_penalty
                
                if do_sample:
                    # Temperature
                    logits = logits / temperature
                    
                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')
                    
                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if (next_token == eos_token_id).all():
                    break
            
            return input_ids
        
        def save_pretrained(self, path: str):
            """Save model and config."""
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            self.config.save(path / 'config.json')
            torch.save(self.state_dict(), path / 'pytorch_model.bin')
        
        @classmethod
        def from_pretrained(cls, path: str, device: str = 'cpu') -> 'SOCForCausalLM':
            """Load model from disk."""
            path = Path(path)
            config = SOCModelConfig.load(path / 'config.json')
            model = cls(config)
            model.load_state_dict(torch.load(path / 'pytorch_model.bin', map_location=device))
            return model
        
        def num_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    class SOCForSequenceClassification(nn.Module):
        """SOC-SLM for sequence classification (intent, severity)."""
        
        def __init__(self, config: SOCModelConfig, num_labels: int):
            super().__init__()
            self.config = config
            self.num_labels = num_labels
            
            self.model = SOCModel(config)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, num_labels)
            )
            
            self._init_weights()
        
        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
        ) -> Dict[str, Tensor]:
            hidden_states, _ = self.model(input_ids, attention_mask)
            
            # Pool: use last non-padding token
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = input_ids.shape[0]
                pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
            else:
                pooled = hidden_states[:, -1, :]
            
            logits = self.classifier(pooled)
            
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
            
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': hidden_states,
            }
    
    
    class SOCForTokenClassification(nn.Module):
        """SOC-SLM for token classification (NER for security entities)."""
        
        def __init__(self, config: SOCModelConfig, num_labels: int):
            super().__init__()
            self.config = config
            self.num_labels = num_labels
            
            self.model = SOCModel(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, num_labels)
            
            self._init_weights()
        
        def _init_weights(self):
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.classifier.bias)
        
        def forward(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
        ) -> Dict[str, Tensor]:
            hidden_states, _ = self.model(input_ids, attention_mask)
            hidden_states = self.dropout(hidden_states)
            logits = self.classifier(hidden_states)
            
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100
                )
            
            return {
                'loss': loss,
                'logits': logits,
            }


except ImportError:
    TORCH_AVAILABLE = False
    
    class SOCForCausalLM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for model training/inference")
    
    class SOCForSequenceClassification:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")
    
    class SOCForTokenClassification:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")


# ==============================================================================
# Model Factory
# ==============================================================================

def create_soc_model(
    model_name: str = "soc-slm-125m",
    task: str = "causal_lm",
    num_labels: int = None,
    **config_overrides
) -> Union['SOCForCausalLM', 'SOCForSequenceClassification', 'SOCForTokenClassification']:
    """
    Factory function to create SOC-SLM models.
    
    Args:
        model_name: One of soc-slm-125m, soc-slm-350m, soc-slm-760m, soc-slm-1b
        task: causal_lm, sequence_classification, token_classification
        num_labels: Number of labels for classification tasks
        **config_overrides: Override specific config parameters
    
    Returns:
        Initialized model
    """
    config = get_model_config(model_name)
    
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    if task == "causal_lm":
        return SOCForCausalLM(config)
    elif task == "sequence_classification":
        if num_labels is None:
            num_labels = config.num_intent_classes
        return SOCForSequenceClassification(config, num_labels)
    elif task == "token_classification":
        if num_labels is None:
            num_labels = config.num_entity_types
        return SOCForTokenClassification(config, num_labels)
    else:
        raise ValueError(f"Unknown task: {task}")
