"""
SOC-SLM Security Tokenizer
Production-ready tokenizer optimized for cybersecurity terminology.

Features:
- MITRE ATT&CK technique IDs (T####, TA####)
- CVE identifiers (CVE-YYYY-NNNNN)
- IP addresses (IPv4/IPv6)
- Domain patterns and URLs
- Hash values (MD5, SHA1, SHA256)
- OCSF field names
- Security-specific vocabulary
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import struct


@dataclass
class TokenizerConfig:
    """Configuration for security tokenizer."""
    vocab_size: int = 32000
    max_length: int = 2048
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    mask_token: str = "[MASK]"
    sep_token: str = "[SEP]"
    cls_token: str = "[CLS]"
    
    # Security-specific special tokens
    ip_token: str = "[IP]"
    domain_token: str = "[DOMAIN]"
    hash_token: str = "[HASH]"
    cve_token: str = "[CVE]"
    mitre_token: str = "[MITRE]"
    timestamp_token: str = "[TIME]"
    user_token: str = "[USER]"
    host_token: str = "[HOST]"
    port_token: str = "[PORT]"
    severity_token: str = "[SEV]"
    
    min_frequency: int = 2
    num_merges: int = 30000
    lowercase: bool = False  # Preserve case for security terms


class SecurityPatternMatcher:
    """Regex patterns for security-specific entities."""
    
    PATTERNS = {
        'ipv4': re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        ),
        'ipv6': re.compile(
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|'
            r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|'
            r'\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b'
        ),
        'domain': re.compile(
            r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        ),
        'cve': re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE),
        'mitre_technique': re.compile(r'\bT\d{4}(?:\.\d{3})?\b'),
        'mitre_tactic': re.compile(r'\bTA\d{4}\b'),
        'md5': re.compile(r'\b[a-fA-F0-9]{32}\b'),
        'sha1': re.compile(r'\b[a-fA-F0-9]{40}\b'),
        'sha256': re.compile(r'\b[a-fA-F0-9]{64}\b'),
        'port': re.compile(r'\b(?:port\s*)?(?:tcp|udp)?[:/]?\s*(\d{1,5})\b', re.IGNORECASE),
        'timestamp_iso': re.compile(
            r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b'
        ),
        'email': re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        'mac_address': re.compile(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b'),
        'registry_key': re.compile(r'\bHK(?:EY_)?(?:LOCAL_MACHINE|CURRENT_USER|CLASSES_ROOT|USERS|CURRENT_CONFIG)\\[^\s]+'),
        'file_path_windows': re.compile(r'\b[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*'),
        'file_path_unix': re.compile(r'\b/(?:[^/\0\s]+/)*[^/\0\s]+'),
        'base64': re.compile(r'\b[A-Za-z0-9+/]{20,}={0,2}\b'),
    }
    
    @classmethod
    def extract_entities(cls, text: str) -> Dict[str, List[str]]:
        """Extract all security entities from text."""
        entities = {}
        for name, pattern in cls.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                entities[name] = list(set(matches))
        return entities
    
    @classmethod
    def mask_entities(cls, text: str, token_map: Dict[str, str]) -> Tuple[str, Dict[str, List[str]]]:
        """Replace entities with special tokens, return masked text and entity map."""
        entity_store = defaultdict(list)
        masked_text = text
        
        replacements = [
            ('ipv4', token_map.get('ip', '[IP]')),
            ('ipv6', token_map.get('ip', '[IP]')),
            ('domain', token_map.get('domain', '[DOMAIN]')),
            ('cve', token_map.get('cve', '[CVE]')),
            ('mitre_technique', token_map.get('mitre', '[MITRE]')),
            ('mitre_tactic', token_map.get('mitre', '[MITRE]')),
            ('md5', token_map.get('hash', '[HASH]')),
            ('sha1', token_map.get('hash', '[HASH]')),
            ('sha256', token_map.get('hash', '[HASH]')),
        ]
        
        for pattern_name, token in replacements:
            pattern = cls.PATTERNS[pattern_name]
            for match in pattern.finditer(masked_text):
                entity_store[pattern_name].append(match.group())
            masked_text = pattern.sub(token, masked_text)
        
        return masked_text, dict(entity_store)


class BPETrainer:
    """Byte-Pair Encoding trainer for security vocabulary."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.word_freqs: Dict[str, int] = defaultdict(int)
        
    def _get_stats(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """Count pair frequencies across all words."""
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge the most frequent pair in all words."""
        new_splits = {}
        for word, split in splits.items():
            if len(split) < 2:
                new_splits[word] = split
                continue
                
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits
    
    def train(self, texts: List[str], progress_callback=None) -> None:
        """Train BPE on corpus."""
        # Count word frequencies
        for text in texts:
            words = text.split()
            for word in words:
                self.word_freqs[word] += 1
        
        # Filter by minimum frequency
        self.word_freqs = {
            word: freq for word, freq in self.word_freqs.items() 
            if freq >= self.config.min_frequency
        }
        
        # Initialize splits as characters
        splits = {word: list(word) for word in self.word_freqs}
        
        # Build initial vocab from characters
        chars = set()
        for word in self.word_freqs:
            chars.update(word)
        
        # Add special tokens
        special_tokens = [
            self.config.pad_token, self.config.unk_token, self.config.bos_token,
            self.config.eos_token, self.config.mask_token, self.config.sep_token,
            self.config.cls_token, self.config.ip_token, self.config.domain_token,
            self.config.hash_token, self.config.cve_token, self.config.mitre_token,
            self.config.timestamp_token, self.config.user_token, self.config.host_token,
            self.config.port_token, self.config.severity_token
        ]
        
        self.vocab = {token: idx for idx, token in enumerate(special_tokens)}
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # BPE merges
        num_merges = min(self.config.num_merges, self.config.vocab_size - len(self.vocab))
        
        for i in range(num_merges):
            pair_freqs = self._get_stats(splits)
            if not pair_freqs:
                break
                
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]
            self.merges.append(best_pair)
            
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
            
            splits = self._merge_pair(best_pair, splits)
            
            if progress_callback and (i + 1) % 1000 == 0:
                progress_callback(i + 1, num_merges)
        
        # Truncate to vocab_size
        if len(self.vocab) > self.config.vocab_size:
            sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])[:self.config.vocab_size]
            self.vocab = {k: i for i, (k, _) in enumerate(sorted_vocab)}


class SecurityTokenizer:
    """Production-ready tokenizer for SOC-SLM."""
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.pattern_matcher = SecurityPatternMatcher()
        self._init_vocab()
        
        # Security vocabulary additions
        self._security_vocab = self._build_security_vocab()
        
    def _init_vocab(self):
        """Initialize vocabulary with special tokens."""
        special_tokens = [
            self.config.pad_token, self.config.unk_token, self.config.bos_token,
            self.config.eos_token, self.config.mask_token, self.config.sep_token,
            self.config.cls_token, self.config.ip_token, self.config.domain_token,
            self.config.hash_token, self.config.cve_token, self.config.mitre_token,
            self.config.timestamp_token, self.config.user_token, self.config.host_token,
            self.config.port_token, self.config.severity_token
        ]
        self.vocab = {token: idx for idx, token in enumerate(special_tokens)}
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
    
    def _build_security_vocab(self) -> Set[str]:
        """Build security-specific vocabulary."""
        vocab = set()
        
        # MITRE ATT&CK tactics
        tactics = [
            "reconnaissance", "resource_development", "initial_access", "execution",
            "persistence", "privilege_escalation", "defense_evasion", "credential_access",
            "discovery", "lateral_movement", "collection", "command_and_control",
            "exfiltration", "impact"
        ]
        vocab.update(tactics)
        
        # Common security terms
        security_terms = [
            "malware", "ransomware", "phishing", "exploit", "vulnerability",
            "authentication", "authorization", "encryption", "firewall", "intrusion",
            "incident", "alert", "threat", "attack", "compromise", "breach",
            "payload", "shellcode", "backdoor", "trojan", "worm", "botnet",
            "c2", "c&c", "command_control", "exfil", "lateral", "privilege",
            "escalation", "persistence", "evasion", "detection", "response",
            "forensics", "investigation", "artifact", "indicator", "ioc",
            "hash", "signature", "rule", "yara", "sigma", "snort", "suricata",
            "siem", "edr", "xdr", "ndr", "soar", "soc", "ciso", "analyst",
            "triage", "severity", "critical", "high", "medium", "low", "info",
            "dns", "http", "https", "tcp", "udp", "icmp", "smtp", "ftp", "ssh",
            "rdp", "smb", "ldap", "kerberos", "ntlm", "oauth", "saml", "jwt"
        ]
        vocab.update(security_terms)
        
        # OCSF categories
        ocsf_categories = [
            "security_finding", "detection_finding", "vulnerability_finding",
            "compliance_finding", "file_activity", "kernel_activity",
            "process_activity", "network_activity", "dns_activity",
            "http_activity", "authentication", "account_change", "api_activity",
            "web_resource_activity", "email_activity", "scheduled_job_activity"
        ]
        vocab.update(ocsf_categories)
        
        # Common event types
        event_types = [
            "login", "logout", "failed_login", "password_change", "privilege_grant",
            "process_start", "process_terminate", "file_create", "file_modify",
            "file_delete", "file_read", "network_connect", "network_disconnect",
            "dns_query", "dns_response", "http_request", "http_response",
            "alert_triggered", "alert_closed", "incident_created", "incident_resolved"
        ]
        vocab.update(event_types)
        
        return vocab
    
    def train(self, texts: List[str], progress_callback=None) -> None:
        """Train tokenizer on security corpus."""
        trainer = BPETrainer(self.config)
        
        # Add security vocabulary to training texts
        security_text = " ".join(self._security_vocab)
        texts = [security_text] + texts
        
        trainer.train(texts, progress_callback)
        
        self.vocab = trainer.vocab
        self.merges = trainer.merges
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Apply BPE to a single word."""
        if word in self.vocab:
            return [word]
        
        tokens = list(word)
        
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens = tokens[:i] + [merge[0] + merge[1]] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        mask_entities: bool = False
    ) -> Dict[str, List[int]]:
        """Encode text to token IDs."""
        max_length = max_length or self.config.max_length
        
        # Optional entity masking
        entity_map = {}
        if mask_entities:
            token_map = {
                'ip': self.config.ip_token,
                'domain': self.config.domain_token,
                'hash': self.config.hash_token,
                'cve': self.config.cve_token,
                'mitre': self.config.mitre_token,
            }
            text, entity_map = self.pattern_matcher.mask_entities(text, token_map)
        
        # Tokenize
        tokens = []
        if add_special_tokens:
            tokens.append(self.config.bos_token)
        
        words = text.split()
        for word in words:
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        
        if add_special_tokens:
            tokens.append(self.config.eos_token)
        
        # Convert to IDs
        input_ids = []
        for token in tokens:
            if token in self.vocab:
                input_ids.append(self.vocab[token])
            else:
                input_ids.append(self.vocab[self.config.unk_token])
        
        # Truncation
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            if add_special_tokens:
                input_ids[-1] = self.vocab[self.config.eos_token]
        
        # Attention mask
        attention_mask = [1] * len(input_ids)
        
        # Padding
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids.extend([self.vocab[self.config.pad_token]] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if mask_entities:
            result['entity_map'] = entity_map
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            try:
                import torch
                result['input_ids'] = torch.tensor([result['input_ids']])
                result['attention_mask'] = torch.tensor([result['attention_mask']])
            except ImportError:
                pass
        elif return_tensors == 'np':
            try:
                import numpy as np
                result['input_ids'] = np.array([result['input_ids']])
                result['attention_mask'] = np.array([result['attention_mask']])
            except ImportError:
                pass
        
        return result
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        special_token_ids = set()
        if skip_special_tokens:
            special_tokens = [
                self.config.pad_token, self.config.bos_token, self.config.eos_token,
                self.config.mask_token, self.config.sep_token, self.config.cls_token
            ]
            special_token_ids = {self.vocab.get(t, -1) for t in special_tokens}
        
        tokens = []
        for token_id in token_ids:
            if token_id in special_token_ids:
                continue
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.config.unk_token)
        
        # Reconstruct text
        text = "".join(tokens)
        
        # Clean up BPE artifacts (basic heuristic)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> Dict[str, List[List[int]]]:
        """Batch encode multiple texts."""
        results = {'input_ids': [], 'attention_mask': []}
        
        for text in texts:
            encoded = self.encode(text, **kwargs)
            results['input_ids'].append(encoded['input_ids'])
            results['attention_mask'].append(encoded['attention_mask'])
        
        return results
    
    def save(self, path: str) -> None:
        """Save tokenizer to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'max_length': self.config.max_length,
            'pad_token': self.config.pad_token,
            'unk_token': self.config.unk_token,
            'bos_token': self.config.bos_token,
            'eos_token': self.config.eos_token,
            'mask_token': self.config.mask_token,
            'sep_token': self.config.sep_token,
            'cls_token': self.config.cls_token,
            'ip_token': self.config.ip_token,
            'domain_token': self.config.domain_token,
            'hash_token': self.config.hash_token,
            'cve_token': self.config.cve_token,
            'mitre_token': self.config.mitre_token,
            'timestamp_token': self.config.timestamp_token,
            'user_token': self.config.user_token,
            'host_token': self.config.host_token,
            'port_token': self.config.port_token,
            'severity_token': self.config.severity_token,
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save vocab
        with open(path / 'vocab.json', 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        # Save merges
        merges_str = [f"{m[0]} {m[1]}" for m in self.merges]
        with open(path / 'merges.txt', 'w') as f:
            f.write('\n'.join(merges_str))
    
    @classmethod
    def load(cls, path: str) -> 'SecurityTokenizer':
        """Load tokenizer from disk."""
        path = Path(path)
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        config = TokenizerConfig(**config_dict)
        tokenizer = cls(config)
        
        # Load vocab
        with open(path / 'vocab.json', 'r') as f:
            tokenizer.vocab = json.load(f)
        tokenizer.id_to_token = {int(idx): token for token, idx in tokenizer.vocab.items()}
        
        # Load merges
        with open(path / 'merges.txt', 'r') as f:
            merges_str = f.read().strip().split('\n')
        tokenizer.merges = [tuple(m.split()) for m in merges_str if m]
        
        return tokenizer
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.config.pad_token]
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.config.unk_token]
    
    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.config.bos_token]
    
    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.config.eos_token]


# Pre-built security tokenizer factory
def create_soc_tokenizer(vocab_size: int = 32000, max_length: int = 2048) -> SecurityTokenizer:
    """Factory function to create a SOC-optimized tokenizer."""
    config = TokenizerConfig(vocab_size=vocab_size, max_length=max_length)
    return SecurityTokenizer(config)
