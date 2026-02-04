"""
SOC-SLM Training Data Generator
Generates synthetic security data for SLM training.

Includes:
- Intent classification data
- Security Q&A pairs
- Alert triage scenarios
- Incident response conversations
- Threat intelligence queries
- OCSF-formatted events
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class IntentType(str, Enum):
    TRIAGE = "triage"
    QUERY = "query"
    THREAT_INTEL = "threat_intel"
    INCIDENT_RESPONSE = "incident_response"
    STATISTICS = "statistics"
    TIMELINE = "timeline"
    SEARCH = "search"
    HELP = "help"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TrainingSample:
    """Single training sample."""
    input_text: str
    output_text: str
    intent: Optional[str] = None
    entities: Optional[Dict] = None
    metadata: Optional[Dict] = None


class MITREGenerator:
    """Generate MITRE ATT&CK related content."""
    
    TACTICS = {
        "TA0001": ("Initial Access", ["T1566", "T1190", "T1133", "T1078"]),
        "TA0002": ("Execution", ["T1059", "T1204", "T1203", "T1047"]),
        "TA0003": ("Persistence", ["T1547", "T1053", "T1136", "T1543"]),
        "TA0004": ("Privilege Escalation", ["T1068", "T1055", "T1134", "T1548"]),
        "TA0005": ("Defense Evasion", ["T1070", "T1036", "T1027", "T1562"]),
        "TA0006": ("Credential Access", ["T1003", "T1110", "T1555", "T1556"]),
        "TA0007": ("Discovery", ["T1082", "T1083", "T1057", "T1016"]),
        "TA0008": ("Lateral Movement", ["T1021", "T1570", "T1534", "T1563"]),
        "TA0009": ("Collection", ["T1560", "T1119", "T1005", "T1074"]),
        "TA0010": ("Exfiltration", ["T1041", "T1048", "T1567", "T1020"]),
        "TA0011": ("Command and Control", ["T1071", "T1105", "T1573", "T1095"]),
        "TA0040": ("Impact", ["T1486", "T1490", "T1489", "T1529"]),
    }
    
    TECHNIQUE_NAMES = {
        "T1566": "Phishing",
        "T1190": "Exploit Public-Facing Application",
        "T1133": "External Remote Services",
        "T1078": "Valid Accounts",
        "T1059": "Command and Scripting Interpreter",
        "T1204": "User Execution",
        "T1203": "Exploitation for Client Execution",
        "T1047": "Windows Management Instrumentation",
        "T1547": "Boot or Logon Autostart Execution",
        "T1053": "Scheduled Task/Job",
        "T1136": "Create Account",
        "T1543": "Create or Modify System Process",
        "T1068": "Exploitation for Privilege Escalation",
        "T1055": "Process Injection",
        "T1134": "Access Token Manipulation",
        "T1548": "Abuse Elevation Control Mechanism",
        "T1070": "Indicator Removal",
        "T1036": "Masquerading",
        "T1027": "Obfuscated Files or Information",
        "T1562": "Impair Defenses",
        "T1003": "OS Credential Dumping",
        "T1110": "Brute Force",
        "T1555": "Credentials from Password Stores",
        "T1556": "Modify Authentication Process",
        "T1082": "System Information Discovery",
        "T1083": "File and Directory Discovery",
        "T1057": "Process Discovery",
        "T1016": "System Network Configuration Discovery",
        "T1021": "Remote Services",
        "T1570": "Lateral Tool Transfer",
        "T1534": "Internal Spearphishing",
        "T1563": "Remote Service Session Hijacking",
        "T1560": "Archive Collected Data",
        "T1119": "Automated Collection",
        "T1005": "Data from Local System",
        "T1074": "Data Staged",
        "T1041": "Exfiltration Over C2 Channel",
        "T1048": "Exfiltration Over Alternative Protocol",
        "T1567": "Exfiltration Over Web Service",
        "T1020": "Automated Exfiltration",
        "T1071": "Application Layer Protocol",
        "T1105": "Ingress Tool Transfer",
        "T1573": "Encrypted Channel",
        "T1095": "Non-Application Layer Protocol",
        "T1486": "Data Encrypted for Impact",
        "T1490": "Inhibit System Recovery",
        "T1489": "Service Stop",
        "T1529": "System Shutdown/Reboot",
    }
    
    @classmethod
    def random_technique(cls) -> Tuple[str, str]:
        """Return random technique ID and name."""
        tech_id = random.choice(list(cls.TECHNIQUE_NAMES.keys()))
        return tech_id, cls.TECHNIQUE_NAMES[tech_id]
    
    @classmethod
    def random_tactic(cls) -> Tuple[str, str, List[str]]:
        """Return random tactic ID, name, and associated techniques."""
        tactic_id = random.choice(list(cls.TACTICS.keys()))
        tactic_name, techniques = cls.TACTICS[tactic_id]
        return tactic_id, tactic_name, techniques


class IOCGenerator:
    """Generate realistic IOCs."""
    
    MALICIOUS_IP_RANGES = [
        ("185.220.101.", 1, 254),  # Known Tor exit
        ("45.33.32.", 1, 254),
        ("91.189.114.", 1, 254),
        ("192.42.116.", 1, 254),
    ]
    
    C2_DOMAINS = [
        "evil-c2.{}.com", "malware-drop.{}.net", "data-exfil.{}.org",
        "backdoor-{}.com", "rat-server-{}.net", "apt-c2-{}.com",
        "beacon-{}.net", "implant-{}.org", "stager-{}.com"
    ]
    
    LEGITIMATE_DOMAINS = [
        "google.com", "microsoft.com", "amazon.com", "cloudflare.com",
        "github.com", "gitlab.com", "slack.com", "office365.com"
    ]
    
    @classmethod
    def random_malicious_ip(cls) -> str:
        prefix, start, end = random.choice(cls.MALICIOUS_IP_RANGES)
        return f"{prefix}{random.randint(start, end)}"
    
    @classmethod
    def random_legitimate_ip(cls) -> str:
        return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    
    @classmethod
    def random_c2_domain(cls) -> str:
        template = random.choice(cls.C2_DOMAINS)
        return template.format(random.randint(1000, 9999))
    
    @classmethod
    def random_legitimate_domain(cls) -> str:
        return random.choice(cls.LEGITIMATE_DOMAINS)
    
    @classmethod
    def random_hash(cls, hash_type: str = "sha256") -> str:
        data = str(random.random()).encode()
        if hash_type == "md5":
            return hashlib.md5(data).hexdigest()
        elif hash_type == "sha1":
            return hashlib.sha1(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()
    
    @classmethod
    def random_cve(cls) -> str:
        year = random.randint(2018, 2024)
        num = random.randint(1000, 50000)
        return f"CVE-{year}-{num}"


class SecurityDataGenerator:
    """Main generator for security training data."""
    
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        
        self.mitre = MITREGenerator()
        self.ioc = IOCGenerator()
        
        # User names
        self.users = [
            "admin", "jsmith", "mwilliams", "analyst1", "soc_lead",
            "itadmin", "dbadmin", "webmaster", "guest", "service_account"
        ]
        
        # Hostnames
        self.hosts = [
            "WORKSTATION-01", "SERVER-DC01", "WEB-PROD-01", "DB-MASTER",
            "LAPTOP-JSMITH", "ENDPOINT-142", "MAIL-SERVER", "FILE-SERVER"
        ]
        
        # Data sources
        self.sources = [
            "CrowdStrike", "Palo Alto", "Carbon Black", "Cisco Umbrella",
            "SentinelOne", "Splunk", "Microsoft Defender", "Elastic SIEM"
        ]
    
    def generate_intent_samples(self, count: int = 1000) -> List[TrainingSample]:
        """Generate intent classification training data."""
        samples = []
        
        # Triage intents
        triage_templates = [
            "Triage my open alerts",
            "What alerts need immediate attention?",
            "Show me critical alerts",
            "Prioritize today's security alerts",
            "Which alerts should I look at first?",
            "Give me a prioritized alert queue",
            "What's the most urgent alert right now?",
            "Rank my alerts by severity",
            "Show high priority incidents",
            "What needs triage?",
        ]
        
        # Query intents
        query_templates = [
            "Show me events from {host}",
            "Find all authentication failures",
            "Search for {ip} in the logs",
            "What happened on {date}?",
            "Get network connections to {domain}",
            "Count events by source",
            "List all process executions today",
            "Find DNS queries to external domains",
            "Show file modifications on {host}",
            "Search for user {user}",
        ]
        
        # Threat intel intents
        threat_intel_templates = [
            "What do we know about IP {ip}?",
            "Is {domain} malicious?",
            "Check this hash: {hash}",
            "Enrich {ioc}",
            "Get threat intelligence on {ip}",
            "Is this IOC known bad?",
            "What threat actor uses {technique}?",
            "Look up CVE {cve}",
            "What's the reputation of {domain}?",
            "Check if {hash} is malware",
        ]
        
        # IR intents
        ir_templates = [
            "How do I respond to {attack}?",
            "What's the playbook for {incident}?",
            "Steps to contain {threat}",
            "How to remediate {malware}?",
            "Incident response for {attack_type}",
            "Containment steps for compromised {asset}",
            "How do I investigate {technique}?",
            "Response procedure for {alert}",
            "Isolation steps for {host}",
            "Recovery from {attack}",
        ]
        
        # Statistics intents
        stats_templates = [
            "Show me statistics",
            "Give me an overview of alerts",
            "What's our alert volume?",
            "Summary of security events",
            "Dashboard metrics please",
            "How many alerts today?",
            "Event counts by type",
            "Security posture overview",
            "Weekly alert summary",
            "Show security metrics",
        ]
        
        attack_types = [
            "ransomware", "phishing", "credential theft", "malware",
            "C2 communication", "data exfiltration", "lateral movement",
            "privilege escalation", "brute force", "SQL injection"
        ]
        
        for _ in range(count):
            intent = random.choice(list(IntentType))
            
            if intent == IntentType.TRIAGE:
                template = random.choice(triage_templates)
                text = template
                
            elif intent == IntentType.QUERY:
                template = random.choice(query_templates)
                text = template.format(
                    host=random.choice(self.hosts),
                    ip=self.ioc.random_legitimate_ip(),
                    domain=self.ioc.random_legitimate_domain(),
                    date="yesterday" if random.random() > 0.5 else "today",
                    user=random.choice(self.users)
                )
                
            elif intent == IntentType.THREAT_INTEL:
                template = random.choice(threat_intel_templates)
                tech_id, _ = self.mitre.random_technique()
                text = template.format(
                    ip=self.ioc.random_malicious_ip(),
                    domain=self.ioc.random_c2_domain(),
                    hash=self.ioc.random_hash(),
                    ioc=self.ioc.random_malicious_ip(),
                    technique=tech_id,
                    cve=self.ioc.random_cve()
                )
                
            elif intent == IntentType.INCIDENT_RESPONSE:
                template = random.choice(ir_templates)
                attack = random.choice(attack_types)
                text = template.format(
                    attack=attack,
                    incident=f"{attack} attack",
                    threat=attack,
                    malware=attack,
                    attack_type=attack,
                    asset=random.choice(["server", "endpoint", "workstation", "user account"]),
                    technique=self.mitre.random_technique()[0],
                    alert=f"{attack} detected",
                    host=random.choice(self.hosts)
                )
                
            else:  # Statistics
                template = random.choice(stats_templates)
                text = template
            
            samples.append(TrainingSample(
                input_text=text,
                output_text=intent.value,
                intent=intent.value
            ))
        
        return samples
    
    def generate_qa_pairs(self, count: int = 1000) -> List[TrainingSample]:
        """Generate security Q&A training pairs."""
        samples = []
        
        qa_templates = [
            # Threat Intel Q&A
            {
                "q": "What MITRE ATT&CK technique is {tech_id}?",
                "a": "{tech_id} is {tech_name}. This technique falls under the {tactic} tactic and involves {description}.",
                "fill": lambda: {
                    "tech_id": (t := MITREGenerator.random_technique())[0],
                    "tech_name": t[1],
                    "tactic": random.choice(["Initial Access", "Execution", "Persistence", "Defense Evasion"]),
                    "description": "adversary behavior to achieve their objectives"
                }
            },
            {
                "q": "Is {ip} a known malicious IP?",
                "a": "Based on threat intelligence, {ip} is associated with {actor}. It has been observed in {activity}. Recommended action: {action}.",
                "fill": lambda: {
                    "ip": IOCGenerator.random_malicious_ip(),
                    "actor": random.choice(["APT28", "APT29", "Lazarus Group", "unknown threat actor", "Cobalt Strike infrastructure"]),
                    "activity": random.choice(["C2 communications", "malware distribution", "scanning activity", "brute force attacks"]),
                    "action": random.choice(["block at firewall", "add to blocklist", "investigate related alerts", "isolate affected systems"])
                }
            },
            # Triage Q&A
            {
                "q": "How should I prioritize this {severity} alert about {alert_type}?",
                "a": "This {severity} severity alert should be {priority}. {alert_type} alerts typically require {response}. Consider: {factors}.",
                "fill": lambda: {
                    "severity": (s := random.choice(["critical", "high", "medium", "low"]))[0].upper() + s[1:],
                    "alert_type": random.choice(["malware detection", "authentication failure", "network anomaly", "policy violation"]),
                    "priority": "handled immediately" if s in ["critical", "high"] else "triaged within 4 hours",
                    "response": random.choice(["immediate investigation", "user notification", "system isolation", "log review"]),
                    "factors": "affected assets, user context, and related events"
                }
            },
            # IR Q&A
            {
                "q": "What are the steps to respond to {incident}?",
                "a": "Response steps for {incident}: 1) {step1}, 2) {step2}, 3) {step3}, 4) {step4}. Remember to document all actions.",
                "fill": lambda: {
                    "incident": (i := random.choice(["ransomware", "phishing", "credential compromise", "malware infection"])),
                    "step1": "Identify scope and affected systems",
                    "step2": "Contain the threat by isolating affected systems",
                    "step3": "Eradicate malicious artifacts and persistence",
                    "step4": "Recover systems and verify integrity"
                }
            },
            # OCSF Q&A
            {
                "q": "What OCSF category does {event_type} fall under?",
                "a": "{event_type} events are categorized as {category} in OCSF. The class_uid is {class_uid} and typical fields include: {fields}.",
                "fill": lambda: {
                    "event_type": (e := random.choice(["login failure", "file creation", "process start", "DNS query"])),
                    "category": random.choice(["Security Finding", "System Activity", "Network Activity"]),
                    "class_uid": random.randint(1001, 6005),
                    "fields": "time, severity_id, status_id, actor, device"
                }
            },
        ]
        
        for _ in range(count):
            template = random.choice(qa_templates)
            fills = template["fill"]()
            
            q = template["q"].format(**fills)
            a = template["a"].format(**fills)
            
            samples.append(TrainingSample(
                input_text=q,
                output_text=a,
                metadata={"type": "qa"}
            ))
        
        return samples
    
    def generate_alert_scenarios(self, count: int = 500) -> List[TrainingSample]:
        """Generate alert analysis scenarios."""
        samples = []
        
        alert_types = [
            {
                "name": "Malware Detection",
                "description": "Endpoint detected malicious file",
                "severity": "high",
                "technique": "T1204",
                "response": "Isolate endpoint, collect artifacts, scan for lateral movement"
            },
            {
                "name": "Brute Force Attack",
                "description": "Multiple failed authentication attempts",
                "severity": "medium",
                "technique": "T1110",
                "response": "Review source IPs, check for successful logins, consider blocking"
            },
            {
                "name": "Data Exfiltration",
                "description": "Large data transfer to external destination",
                "severity": "critical",
                "technique": "T1041",
                "response": "Immediate containment, identify data scope, preserve evidence"
            },
            {
                "name": "Suspicious PowerShell",
                "description": "Encoded PowerShell command execution",
                "severity": "high",
                "technique": "T1059.001",
                "response": "Analyze command, check parent process, review network connections"
            },
            {
                "name": "Credential Dumping",
                "description": "LSASS memory access detected",
                "severity": "critical",
                "technique": "T1003",
                "response": "Isolate system, force password reset, hunt for lateral movement"
            },
        ]
        
        for _ in range(count):
            alert = random.choice(alert_types)
            host = random.choice(self.hosts)
            user = random.choice(self.users)
            source = random.choice(self.sources)
            
            input_text = f"""Alert: {alert['name']}
Source: {source}
Host: {host}
User: {user}
Severity: {alert['severity']}
Description: {alert['description']}
MITRE Technique: {alert['technique']}

What should I do?"""
            
            output_text = f"""Analysis of {alert['name']} alert:

**Severity**: {alert['severity'].upper()} - This requires {'immediate attention' if alert['severity'] in ['critical', 'high'] else 'timely review'}.

**MITRE ATT&CK**: {alert['technique']} - {MITREGenerator.TECHNIQUE_NAMES.get(alert['technique'].split('.')[0], 'Unknown technique')}

**Recommended Response**:
{alert['response']}

**Investigation Steps**:
1. Verify the alert is not a false positive
2. Check for related events on {host}
3. Review {user}'s recent activity
4. Search for similar patterns across the environment
5. Document findings and escalate if needed"""
            
            samples.append(TrainingSample(
                input_text=input_text,
                output_text=output_text,
                intent="triage",
                entities={
                    "host": host,
                    "user": user,
                    "technique": alert["technique"],
                    "severity": alert["severity"]
                }
            ))
        
        return samples
    
    def generate_ocsf_events(self, count: int = 500) -> List[Dict]:
        """Generate OCSF-formatted security events."""
        events = []
        
        for _ in range(count):
            event_type = random.choice([
                "authentication",
                "network_activity",
                "process_activity",
                "file_activity",
                "dns_activity"
            ])
            
            base_event = {
                "metadata": {
                    "version": "1.0.0",
                    "product": {
                        "name": random.choice(self.sources),
                        "vendor_name": "Security Vendor"
                    },
                    "logged_time": (datetime.now() - timedelta(hours=random.randint(0, 72))).isoformat()
                },
                "time": datetime.now().isoformat(),
                "severity_id": random.randint(1, 5),
                "status_id": random.choice([1, 2]),  # Success/Failure
            }
            
            if event_type == "authentication":
                base_event.update({
                    "class_uid": 3002,
                    "class_name": "Authentication",
                    "activity_id": random.choice([1, 2]),  # Login/Logout
                    "actor": {
                        "user": {
                            "name": random.choice(self.users),
                            "uid": str(uuid.uuid4())[:8]
                        }
                    },
                    "src_endpoint": {
                        "ip": self.ioc.random_legitimate_ip(),
                        "hostname": random.choice(self.hosts)
                    },
                    "dst_endpoint": {
                        "ip": self.ioc.random_legitimate_ip(),
                        "hostname": random.choice(self.hosts)
                    }
                })
                
            elif event_type == "network_activity":
                base_event.update({
                    "class_uid": 4001,
                    "class_name": "Network Activity",
                    "activity_id": random.choice([1, 2, 3]),
                    "src_endpoint": {
                        "ip": self.ioc.random_legitimate_ip(),
                        "port": random.randint(1024, 65535)
                    },
                    "dst_endpoint": {
                        "ip": self.ioc.random_malicious_ip() if random.random() > 0.8 else self.ioc.random_legitimate_ip(),
                        "port": random.choice([80, 443, 8080, 22, 3389, 445])
                    },
                    "connection_info": {
                        "protocol_name": random.choice(["TCP", "UDP"]),
                        "direction": random.choice(["Inbound", "Outbound"])
                    }
                })
                
            elif event_type == "process_activity":
                base_event.update({
                    "class_uid": 1001,
                    "class_name": "Process Activity",
                    "activity_id": random.choice([1, 2]),  # Create/Terminate
                    "actor": {
                        "user": {"name": random.choice(self.users)}
                    },
                    "process": {
                        "name": random.choice(["powershell.exe", "cmd.exe", "python.exe", "svchost.exe"]),
                        "pid": random.randint(1000, 65535),
                        "cmd_line": random.choice([
                            "powershell.exe -enc SGVsbG8gV29ybGQ=",
                            "cmd.exe /c whoami",
                            "python.exe script.py",
                            "svchost.exe -k netsvcs"
                        ])
                    },
                    "device": {
                        "hostname": random.choice(self.hosts)
                    }
                })
                
            elif event_type == "file_activity":
                base_event.update({
                    "class_uid": 1003,
                    "class_name": "File Activity",
                    "activity_id": random.choice([1, 2, 3, 4]),  # Create/Read/Update/Delete
                    "file": {
                        "name": random.choice(["document.docx", "malware.exe", "config.ini", "passwords.txt"]),
                        "path": f"C:\\Users\\{random.choice(self.users)}\\Documents\\",
                        "hashes": [
                            {"algorithm": "SHA256", "value": self.ioc.random_hash("sha256")}
                        ]
                    },
                    "device": {
                        "hostname": random.choice(self.hosts)
                    }
                })
                
            elif event_type == "dns_activity":
                is_malicious = random.random() > 0.85
                base_event.update({
                    "class_uid": 4003,
                    "class_name": "DNS Activity",
                    "query": {
                        "hostname": self.ioc.random_c2_domain() if is_malicious else self.ioc.random_legitimate_domain(),
                        "type": random.choice(["A", "AAAA", "CNAME", "TXT"])
                    },
                    "answers": [
                        {"rdata": self.ioc.random_malicious_ip() if is_malicious else self.ioc.random_legitimate_ip()}
                    ],
                    "src_endpoint": {
                        "ip": self.ioc.random_legitimate_ip(),
                        "hostname": random.choice(self.hosts)
                    }
                })
            
            events.append(base_event)
        
        return events
    
    def generate_conversation_data(self, count: int = 200) -> List[TrainingSample]:
        """Generate multi-turn conversation training data."""
        samples = []
        
        conversation_starters = [
            {
                "turns": [
                    ("user", "I see a critical alert about ransomware on WORKSTATION-01"),
                    ("assistant", "I understand this is urgent. Let me help you respond to this ransomware alert on WORKSTATION-01. First, can you confirm if the workstation is still connected to the network?"),
                    ("user", "Yes it's still connected"),
                    ("assistant", "Immediately isolate WORKSTATION-01 from the network. This prevents the ransomware from spreading to other systems. You can do this by: 1) Disabling the network adapter, 2) Blocking at the switch port, or 3) Using your EDR to isolate. Once isolated, do not power off the system as we need to preserve evidence. What's the user associated with this workstation?"),
                    ("user", "It's assigned to jsmith"),
                    ("assistant", "Thank you. Here's your immediate action plan:\n\n1. **Isolation** - Disconnect WORKSTATION-01 (already underway)\n2. **User notification** - Contact jsmith, have them stop any work and not interact with the system\n3. **Credential reset** - Reset jsmith's password and revoke active sessions\n4. **Scope assessment** - Check for network shares jsmith accessed in the last 24 hours\n5. **Evidence preservation** - Capture memory dump before any further action\n\nI'm searching for related alerts and lateral movement indicators now. Would you like me to generate the full incident report?")
                ]
            },
            {
                "turns": [
                    ("user", "Can you analyze this IP for me: 185.220.101.42"),
                    ("assistant", "Analyzing 185.220.101.42...\n\nThis IP is associated with known malicious activity:\n- **Threat Category**: Tor Exit Node / C2 Infrastructure\n- **First Seen**: 2023-06-15\n- **Confidence**: High (92%)\n- **Associated Threat Actors**: APT28, Generic Commodity Malware\n\nWould you like me to search for any connections to this IP in your environment?"),
                    ("user", "Yes please check our logs"),
                    ("assistant", "Searching your data lake for connections to 185.220.101.42...\n\nI found 3 connections in the last 7 days:\n\n1. **WORKSTATION-05** - Outbound connection on port 443 - 2 days ago\n2. **LAPTOP-MWILLIAMS** - DNS resolution - 3 days ago  \n3. **SERVER-DC01** - Blocked firewall connection - 5 days ago\n\nThe connection from WORKSTATION-05 is concerning as it was successful. I recommend immediate investigation. Should I create a timeline of all activity from WORKSTATION-05?")
                ]
            }
        ]
        
        for _ in range(count):
            conv = random.choice(conversation_starters)
            
            # Create training samples from conversation turns
            context = ""
            for i, (role, message) in enumerate(conv["turns"]):
                if role == "user" and i + 1 < len(conv["turns"]):
                    next_role, next_message = conv["turns"][i + 1]
                    if next_role == "assistant":
                        samples.append(TrainingSample(
                            input_text=f"{context}User: {message}",
                            output_text=next_message,
                            metadata={"type": "conversation", "turn": i}
                        ))
                context += f"{role.capitalize()}: {message}\n"
        
        return samples
    
    def generate_full_dataset(
        self,
        intent_count: int = 2000,
        qa_count: int = 2000,
        alert_count: int = 1000,
        conversation_count: int = 500
    ) -> Dict[str, List]:
        """Generate complete training dataset."""
        return {
            "intent_classification": [asdict(s) for s in self.generate_intent_samples(intent_count)],
            "question_answering": [asdict(s) for s in self.generate_qa_pairs(qa_count)],
            "alert_scenarios": [asdict(s) for s in self.generate_alert_scenarios(alert_count)],
            "conversations": [asdict(s) for s in self.generate_conversation_data(conversation_count)],
            "ocsf_events": self.generate_ocsf_events(1000)
        }
    
    def save_dataset(self, path: str, **kwargs):
        """Generate and save dataset to disk."""
        dataset = self.generate_full_dataset(**kwargs)
        
        with open(path, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        return dataset


# Streaming generator for large datasets
def generate_training_stream(
    batch_size: int = 100,
    total_samples: int = 10000,
    seed: int = None
) -> Generator[List[TrainingSample], None, None]:
    """Stream training samples in batches."""
    generator = SecurityDataGenerator(seed=seed)
    
    samples_generated = 0
    while samples_generated < total_samples:
        batch = []
        
        # Mix of different sample types
        batch.extend(generator.generate_intent_samples(batch_size // 4))
        batch.extend(generator.generate_qa_pairs(batch_size // 4))
        batch.extend(generator.generate_alert_scenarios(batch_size // 4))
        batch.extend(generator.generate_conversation_data(batch_size // 4))
        
        random.shuffle(batch)
        
        yield batch
        samples_generated += len(batch)


# -------------------------------------------------------------------
# Compatibility: OCSFEventGenerator (expected by scripts/demo.py)
# Generates simple synthetic OCSF-like events for demos/tests.
# -------------------------------------------------------------------
import random
import time
from typing import Dict, List, Any

class OCSFEventGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate_event(self) -> Dict[str, Any]:
        now = int(time.time())
        return {
            "time": now,
            "severity": self.rng.choice([1, 2, 3, 4, 5]),
            "category": self.rng.choice(["authentication", "network", "endpoint", "cloud", "application"]),
            "action": self.rng.choice(["login", "logout", "blocked", "allowed", "alert", "scan"]),
            "src_ip": ".".join(str(self.rng.randint(1, 254)) for _ in range(4)),
            "dst_ip": ".".join(str(self.rng.randint(1, 254)) for _ in range(4)),
            "user": self.rng.choice(["alice", "bob", "charlie", "svc_ci", "svc_etl", "admin"]),
            "resource": self.rng.choice(["vpn", "okta", "aws", "k8s", "db", "app"]),
            "message": self.rng.choice([
                "Suspicious login attempt",
                "Multiple failed logins",
                "Malware signature matched",
                "Outbound connection to rare domain",
                "Privilege escalation attempt",
            ]),
        }

    def generate_batch(self, n: int = 10) -> List[Dict[str, Any]]:
        return [self.generate_event() for _ in range(n)]
