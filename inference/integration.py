"""
SOC-SLM Integration Layer
Integrates SOC-SLM with the multi-agent SOC chatbot.

Provides:
- Agent response generation
- Intent classification
- Entity extraction
- Threat intelligence enrichment
- Query interpretation
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """SOC Agent types matching the chatbot architecture."""
    ORCHESTRATOR = "orchestrator"
    QUERY = "query"
    TRIAGE = "triage"
    THREAT_INTEL = "threat_intel"
    INCIDENT_RESPONSE = "ir"


@dataclass
class AgentRequest:
    """Request to an agent."""
    message: str
    context: Optional[Dict] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class AgentResponse:
    """Response from an agent."""
    message: str
    agent_type: AgentType
    confidence: float = 1.0
    entities: Optional[Dict] = None
    actions: Optional[List[Dict]] = None
    data: Optional[Dict] = None
    metadata: Optional[Dict] = None


class IntentClassifier:
    """Classify user intent for agent routing."""
    
    # Intent patterns and keywords
    INTENT_PATTERNS = {
        "triage": {
            "keywords": ["triage", "prioritize", "priority", "urgent", "critical", "alert queue", "open alerts"],
            "patterns": [
                r"triage\s+(my\s+)?alerts?",
                r"prioritize\s+(my\s+)?alerts?",
                r"what\s+(alerts?\s+)?need\s+attention",
                r"critical\s+alerts?",
                r"urgent\s+alerts?"
            ]
        },
        "query": {
            "keywords": ["show", "find", "search", "get", "list", "count", "display", "events"],
            "patterns": [
                r"show\s+(me\s+)?(all\s+)?events?",
                r"find\s+(all\s+)?",
                r"search\s+for",
                r"get\s+(me\s+)?",
                r"count\s+(of\s+)?",
                r"list\s+(all\s+)?"
            ]
        },
        "threat_intel": {
            "keywords": ["threat", "intelligence", "ioc", "indicator", "malicious", "reputation", 
                        "enrich", "check", "lookup", "what do we know"],
            "patterns": [
                r"what\s+(do\s+)?(we\s+)?know\s+about",
                r"(is|check|lookup)\s+.*(malicious|suspicious)",
                r"enrich\s+(this\s+)?",
                r"threat\s+intel",
                r"reputation\s+(of|for|check)"
            ]
        },
        "incident_response": {
            "keywords": ["respond", "playbook", "contain", "remediate", "isolate", "investigate",
                        "how do i", "steps", "procedure", "response"],
            "patterns": [
                r"how\s+(do\s+i|should\s+i|to)\s+respond",
                r"(playbook|procedure)\s+for",
                r"(contain|isolate|remediate)\s+",
                r"response\s+(steps|procedure|playbook)",
                r"how\s+to\s+(investigate|handle)"
            ]
        },
        "statistics": {
            "keywords": ["statistics", "stats", "overview", "summary", "dashboard", "metrics", 
                        "numbers", "counts"],
            "patterns": [
                r"(show|give)\s+(me\s+)?stat(istic)?s",
                r"overview",
                r"summary",
                r"dashboard",
                r"how\s+many"
            ]
        },
        "timeline": {
            "keywords": ["timeline", "history", "sequence", "chronological", "when"],
            "patterns": [
                r"timeline\s+(of|for)",
                r"history\s+(of|for)",
                r"what\s+happened\s+(before|after|when)"
            ]
        }
    }
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent and return (intent, confidence)."""
        text_lower = text.lower()
        
        scores = {}
        for intent, config in self.INTENT_PATTERNS.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for kw in config["keywords"] if kw in text_lower)
            score += keyword_matches * 0.2
            
            # Pattern matching
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    score += 0.4
            
            scores[intent] = min(score, 1.0)  # Cap at 1.0
        
        if not scores or max(scores.values()) == 0:
            return "query", 0.3  # Default to query with low confidence
        
        best_intent = max(scores.items(), key=lambda x: x[1])
        return best_intent[0], best_intent[1]


class EntityExtractor:
    """Extract security entities from text."""
    
    PATTERNS = {
        "ipv4": re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        ),
        "domain": re.compile(
            r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        ),
        "cve": re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE),
        "mitre_technique": re.compile(r'\bT\d{4}(?:\.\d{3})?\b'),
        "md5": re.compile(r'\b[a-fA-F0-9]{32}\b'),
        "sha1": re.compile(r'\b[a-fA-F0-9]{40}\b'),
        "sha256": re.compile(r'\b[a-fA-F0-9]{64}\b'),
        "hostname": re.compile(r'\b[A-Z][A-Z0-9_-]{2,14}[0-9]\b'),
        "username": re.compile(r'\b(?:user|admin|root|guest|service)[a-z0-9_]*\b', re.IGNORECASE),
        "severity": re.compile(r'\b(critical|high|medium|low|info(?:rmational)?)\b', re.IGNORECASE),
        "port": re.compile(r'\b(?:port\s*)?(?:tcp|udp)?[:/]?\s*(\d{1,5})\b', re.IGNORECASE),
        "email": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
    }
    
    @classmethod
    def extract(cls, text: str) -> Dict[str, List[str]]:
        """Extract all security entities from text."""
        entities = {}
        for name, pattern in cls.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                entities[name] = list(set(matches))
        return entities
    
    @classmethod
    def extract_iocs(cls, text: str) -> List[Dict]:
        """Extract IOCs with type classification."""
        iocs = []
        
        for ip in cls.PATTERNS["ipv4"].findall(text):
            iocs.append({"type": "ip", "value": ip})
        
        for domain in cls.PATTERNS["domain"].findall(text):
            # Filter out common false positives
            if not domain.endswith(('.exe', '.dll', '.sys', '.txt', '.log')):
                iocs.append({"type": "domain", "value": domain})
        
        for cve in cls.PATTERNS["cve"].findall(text):
            iocs.append({"type": "cve", "value": cve.upper()})
        
        for hash_val in cls.PATTERNS["sha256"].findall(text):
            iocs.append({"type": "sha256", "value": hash_val.lower()})
        
        for hash_val in cls.PATTERNS["sha1"].findall(text):
            iocs.append({"type": "sha1", "value": hash_val.lower()})
        
        for hash_val in cls.PATTERNS["md5"].findall(text):
            iocs.append({"type": "md5", "value": hash_val.lower()})
        
        return iocs


class QueryInterpreter:
    """Interpret natural language queries for data lake."""
    
    # Query type patterns
    QUERY_PATTERNS = {
        "filter": {
            "patterns": [
                (r"from\s+(\w+)", "source"),
                (r"by\s+(user|host|source|ip)\s+(\w+)", "by"),
                (r"(today|yesterday|last\s+\d+\s+(?:hours?|days?|weeks?))", "time"),
                (r"severity\s+(critical|high|medium|low|info)", "severity"),
            ]
        },
        "aggregate": {
            "keywords": ["count", "sum", "average", "group by", "statistics"],
        },
        "timeline": {
            "keywords": ["timeline", "history", "sequence", "events for"],
        }
    }
    
    @classmethod
    def interpret(cls, query: str) -> Dict:
        """Interpret query and return structured representation."""
        result = {
            "type": "search",
            "filters": {},
            "aggregations": [],
            "time_range": None,
            "limit": 100,
            "raw_query": query
        }
        
        query_lower = query.lower()
        
        # Determine query type
        if any(kw in query_lower for kw in cls.QUERY_PATTERNS["aggregate"]["keywords"]):
            result["type"] = "aggregate"
        elif any(kw in query_lower for kw in cls.QUERY_PATTERNS["timeline"]["keywords"]):
            result["type"] = "timeline"
        
        # Extract filters
        entities = EntityExtractor.extract(query)
        
        if entities.get("ipv4"):
            result["filters"]["ip"] = entities["ipv4"]
        if entities.get("domain"):
            result["filters"]["domain"] = entities["domain"]
        if entities.get("hostname"):
            result["filters"]["hostname"] = entities["hostname"]
        if entities.get("username"):
            result["filters"]["user"] = entities["username"]
        if entities.get("severity"):
            result["filters"]["severity"] = entities["severity"][0].lower()
        
        # Time range extraction
        time_patterns = [
            (r"last\s+(\d+)\s+hours?", lambda m: {"hours": int(m.group(1))}),
            (r"last\s+(\d+)\s+days?", lambda m: {"days": int(m.group(1))}),
            (r"today", lambda m: {"days": 1}),
            (r"yesterday", lambda m: {"days": 2, "offset": 1}),
        ]
        
        for pattern, handler in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result["time_range"] = handler(match)
                break
        
        return result


class SOCChatbotIntegration:
    """Main integration class for SOC chatbot."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize integration with optional SLM model."""
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.query_interpreter = QueryInterpreter()
        
        # Load SLM if path provided
        self.slm_engine = None
        if model_path:
            try:
                from .engine import SOCInferenceEngine
                self.slm_engine = SOCInferenceEngine(model_path)
                logger.info(f"Loaded SLM from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load SLM: {e}")
    
    def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a chatbot request and return appropriate response."""
        message = request.message
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify(message)
        
        # Extract entities
        entities = self.entity_extractor.extract(message)
        
        # Route to appropriate handler
        agent_type = self._get_agent_type(intent)
        
        # Generate response
        if self.slm_engine:
            # Use SLM for response generation
            from .engine import GenerationConfig
            config = GenerationConfig(max_new_tokens=512, temperature=0.7)
            
            # Add context to prompt
            prompt = self._build_prompt(message, intent, entities, request.context)
            result = self.slm_engine.generate(prompt, config)
            
            response_text = result.text
        else:
            # Use template-based response
            response_text = self._generate_template_response(intent, message, entities)
        
        return AgentResponse(
            message=response_text,
            agent_type=agent_type,
            confidence=confidence,
            entities=entities,
            metadata={
                "intent": intent,
                "query_interpretation": self.query_interpreter.interpret(message) if intent == "query" else None
            }
        )
    
    def _get_agent_type(self, intent: str) -> AgentType:
        """Map intent to agent type."""
        mapping = {
            "triage": AgentType.TRIAGE,
            "query": AgentType.QUERY,
            "threat_intel": AgentType.THREAT_INTEL,
            "incident_response": AgentType.INCIDENT_RESPONSE,
            "statistics": AgentType.QUERY,
            "timeline": AgentType.QUERY,
        }
        return mapping.get(intent, AgentType.ORCHESTRATOR)
    
    def _build_prompt(
        self,
        message: str,
        intent: str,
        entities: Dict,
        context: Optional[Dict]
    ) -> str:
        """Build prompt for SLM generation."""
        prompt_parts = []
        
        # System context
        prompt_parts.append(
            "You are a Security Operations Center (SOC) AI assistant. "
            "Provide accurate, actionable security guidance."
        )
        
        # Add context if available
        if context:
            prompt_parts.append(f"\nContext: {json.dumps(context)}")
        
        # Add extracted entities
        if entities:
            prompt_parts.append(f"\nIdentified entities: {json.dumps(entities)}")
        
        # Add intent guidance
        intent_guidance = {
            "triage": "Prioritize and analyze security alerts.",
            "query": "Help search and analyze security data.",
            "threat_intel": "Provide threat intelligence and IOC analysis.",
            "incident_response": "Guide incident response and remediation.",
            "statistics": "Provide security metrics and summaries.",
        }
        
        if intent in intent_guidance:
            prompt_parts.append(f"\nTask: {intent_guidance[intent]}")
        
        # Add user message
        prompt_parts.append(f"\nUser: {message}")
        prompt_parts.append("\nAssistant:")
        
        return "\n".join(prompt_parts)
    
    def _generate_template_response(
        self,
        intent: str,
        message: str,
        entities: Dict
    ) -> str:
        """Generate response using templates (fallback when no SLM)."""
        
        if intent == "triage":
            return (
                "I'll help you triage alerts. To provide the best prioritization, I analyze:\n\n"
                "• Severity and risk score\n"
                "• MITRE ATT&CK techniques involved\n"
                "• Affected assets and users\n"
                "• Historical context\n\n"
                "Please wait while I fetch and prioritize your open alerts..."
            )
        
        elif intent == "query":
            interpretation = self.query_interpreter.interpret(message)
            filters = interpretation.get("filters", {})
            
            filter_str = ", ".join(f"{k}={v}" for k, v in filters.items()) if filters else "none"
            
            return (
                f"Executing query with filters: {filter_str}\n\n"
                "Searching the OCSF-normalized data lake..."
            )
        
        elif intent == "threat_intel":
            iocs = self.entity_extractor.extract_iocs(message)
            
            if iocs:
                ioc = iocs[0]
                return (
                    f"Analyzing {ioc['type'].upper()}: {ioc['value']}\n\n"
                    "Checking threat intelligence sources:\n"
                    "• Internal threat database\n"
                    "• MITRE ATT&CK mapping\n"
                    "• Historical incidents\n\n"
                    "Enrichment in progress..."
                )
            return "Please provide an IOC (IP, domain, hash, or CVE) to analyze."
        
        elif intent == "incident_response":
            return (
                "I can help with incident response. Please specify:\n\n"
                "• Type of incident (malware, phishing, credential theft, etc.)\n"
                "• Affected systems/users\n"
                "• Current status\n\n"
                "I'll provide a tailored response playbook."
            )
        
        elif intent == "statistics":
            return (
                "Fetching security statistics from the data lake...\n\n"
                "This includes:\n"
                "• Alert volume and distribution\n"
                "• Event counts by type\n"
                "• Top affected assets\n"
                "• MITRE technique coverage"
            )
        
        else:
            return (
                "I'm your SOC AI assistant. I can help you with:\n\n"
                "• **Triage**: Prioritize security alerts\n"
                "• **Query**: Search the security data lake\n"
                "• **Threat Intel**: Analyze IOCs and threats\n"
                "• **Incident Response**: Get response playbooks\n"
                "• **Statistics**: View security metrics\n\n"
                "What would you like to do?"
            )
    
    def classify_intent(self, message: str) -> Dict[str, Any]:
        """Public method for intent classification."""
        intent, confidence = self.intent_classifier.classify(message)
        return {
            "intent": intent,
            "confidence": confidence,
            "agent": self._get_agent_type(intent).value
        }
    
    def extract_entities(self, message: str) -> Dict[str, List[str]]:
        """Public method for entity extraction."""
        return self.entity_extractor.extract(message)
    
    def interpret_query(self, message: str) -> Dict:
        """Public method for query interpretation."""
        return self.query_interpreter.interpret(message)


# ==============================================================================
# FastAPI Integration
# ==============================================================================

def create_fastapi_router():
    """Create FastAPI router for SLM endpoints."""
    try:
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel
    except ImportError:
        logger.warning("FastAPI not available")
        return None
    
    router = APIRouter(prefix="/slm", tags=["SLM"])
    
    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.7
    
    class ClassifyRequest(BaseModel):
        text: str
    
    # Initialize integration (would be done at app startup in production)
    integration = SOCChatbotIntegration()
    
    @router.post("/generate")
    async def generate(request: GenerateRequest):
        """Generate response using SLM."""
        if not integration.slm_engine:
            raise HTTPException(status_code=503, detail="SLM not loaded")
        
        from .engine import GenerationConfig
        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        result = integration.slm_engine.generate(request.prompt, config)
        
        return {
            "text": result.text,
            "tokens": result.tokens,
            "latency_ms": result.latency_ms
        }
    
    @router.post("/classify")
    async def classify_intent(request: ClassifyRequest):
        """Classify intent of text."""
        return integration.classify_intent(request.text)
    
    @router.post("/extract")
    async def extract_entities(request: ClassifyRequest):
        """Extract security entities from text."""
        return integration.extract_entities(request.text)
    
    @router.post("/interpret")
    async def interpret_query(request: ClassifyRequest):
        """Interpret natural language query."""
        return integration.interpret_query(request.text)
    
    @router.get("/health")
    async def health():
        """Health check."""
        return {
            "status": "healthy",
            "slm_loaded": integration.slm_engine is not None
        }
    
    return router
