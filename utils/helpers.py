"""
SOC SLM Utilities - Production Ready
Common utilities for logging, metrics, device management, and security operations.
"""

import os
import sys
import json
import time
import logging
import hashlib
import functools
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import threading


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    name: str = "soc-slm",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """Configure logging for SOC SLM."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(file_handler)
    
    return logger


# Global logger
logger = setup_logging()


# =============================================================================
# Device Management
# =============================================================================

def get_device(prefer_gpu: bool = True) -> str:
    """Get the best available device."""
    try:
        import torch
        if prefer_gpu:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    info = {
        'device': get_device(),
        'cpu_count': os.cpu_count()
    }
    
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        pass
    
    return info


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =============================================================================
# Timing Utilities
# =============================================================================

@contextmanager
def timer(name: str = "Operation", log: bool = True):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if log:
            logger.info(f"{name} completed in {elapsed:.3f}s")


def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.tags is None:
            self.tags = {}


class MetricsCollector:
    """Thread-safe metrics collection."""
    
    def __init__(self, flush_interval: int = 60):
        self._metrics: List[Metric] = []
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._last_flush = time.time()
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric."""
        with self._lock:
            self._metrics.append(Metric(name=name, value=value, tags=tags or {}))
    
    def get_metrics(self, name: Optional[str] = None) -> List[Metric]:
        """Get recorded metrics."""
        with self._lock:
            if name:
                return [m for m in self._metrics if m.name == name]
            return self._metrics.copy()
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        import statistics
        
        summary = {}
        with self._lock:
            metrics_by_name = {}
            for m in self._metrics:
                if m.name not in metrics_by_name:
                    metrics_by_name[m.name] = []
                metrics_by_name[m.name].append(m.value)
            
            for name, values in metrics_by_name.items():
                summary[name] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'sum': sum(values)
                }
                if len(values) > 1:
                    summary[name]['stdev'] = statistics.stdev(values)
        
        return summary
    
    def flush(self, filepath: Optional[str] = None) -> List[Dict]:
        """Flush metrics to file or return as list."""
        with self._lock:
            data = [asdict(m) for m in self._metrics]
            self._metrics.clear()
            self._last_flush = time.time()
        
        if filepath:
            with open(filepath, 'a') as f:
                for d in data:
                    f.write(json.dumps(d) + '\n')
        
        return data
    
    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()


# Global metrics collector
metrics = MetricsCollector()


# =============================================================================
# Security Utilities
# =============================================================================

def compute_hash(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """Compute hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    if algorithm == 'md5':
        return hashlib.md5(data).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(data).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(data).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def mask_sensitive_data(text: str) -> str:
    """Mask potentially sensitive data in text."""
    import re
    
    # Mask credit cards
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
    
    # Mask SSN
    text = re.sub(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[SSN]', text)
    
    # Mask API keys (generic pattern)
    text = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[API_KEY]', text)
    
    # Mask passwords in common formats
    text = re.sub(r'password[:\s]*\S+', 'password: [REDACTED]', text, flags=re.IGNORECASE)
    
    return text


def validate_ip(ip: str) -> bool:
    """Validate an IP address."""
    import re
    ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return bool(re.match(ipv4_pattern, ip))


def is_private_ip(ip: str) -> bool:
    """Check if IP is private/internal."""
    if not validate_ip(ip):
        return False
    
    octets = [int(x) for x in ip.split('.')]
    
    # 10.0.0.0/8
    if octets[0] == 10:
        return True
    # 172.16.0.0/12
    if octets[0] == 172 and 16 <= octets[1] <= 31:
        return True
    # 192.168.0.0/16
    if octets[0] == 192 and octets[1] == 168:
        return True
    # 127.0.0.0/8
    if octets[0] == 127:
        return True
    
    return False


# =============================================================================
# File Utilities
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_load(filepath: str, default: Any = None) -> Any:
    """Safely load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return default


def safe_json_save(data: Any, filepath: str) -> bool:
    """Safely save data to JSON file."""
    try:
        ensure_dir(Path(filepath).parent)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        return False


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6
    }


def estimate_model_size_mb(model) -> float:
    """Estimate model size in MB (assuming FP32)."""
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * 4) / (1024 * 1024)


def format_number(n: Union[int, float]) -> str:
    """Format large numbers for display."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


# =============================================================================
# Initialization
# =============================================================================

__all__ = [
    'setup_logging',
    'logger',
    'get_device',
    'get_device_info',
    'set_seed',
    'timer',
    'timed',
    'Metric',
    'MetricsCollector',
    'metrics',
    'compute_hash',
    'mask_sensitive_data',
    'validate_ip',
    'is_private_ip',
    'ensure_dir',
    'safe_json_load',
    'safe_json_save',
    'get_file_size_mb',
    'count_parameters',
    'estimate_model_size_mb',
    'format_number'
]
