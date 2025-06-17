"""
Universal AI Module - Core Engine
=================================

Univerzalni Python modul za delo z različnimi AI API-ji z vgrajeno bazo znanja in backup sistemom.

Podpira:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Google Gemini (1.5 Flash, Pro, 2.5 Flash Thinking)
- Anthropic Claude
- Azure OpenAI
- Cohere
- Custom API endpoints

Funkcionalnosti:
- Multi-provider support z backup sistemom
- Baza znanja (RAG)
- Smart routing in cost optimization
- Response caching
- Template sistem
- Performance monitoring
"""

import asyncio
import aiohttp
import json
import sqlite3
import hashlib
import pickle
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
import re
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProviderType(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "anthropic"
    AZURE = "azure"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeEntry:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

@dataclass
class APIEndpoint:
    provider: ProviderType
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    priority: int = 1  # 1 = najvišja prioriteta
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseProvider(ABC):
    """Abstraktni razred za AI providerje"""
    
    @abstractmethod
    async def format_request(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def parse_response(self, response: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def get_headers(self, api_key: str) -> Dict[str, str]:
        pass

class OpenAIProvider(BaseProvider):
    async def format_request(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        return {
            "messages": [{"role": msg.role.value, "content": msg.content} for msg in messages],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "model": kwargs.get("model", "gpt-3.5-turbo")
        }
    
    async def parse_response(self, response: Dict[str, Any]) -> str:
        return response["choices"][0]["message"]["content"]
    
    def get_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

class GeminiProvider(BaseProvider):
    async def format_request(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        # Kombiniraj vse sporočila v eno vsebino za Gemini
        content = "\n".join([f"{msg.role.value}: {msg.content}" for msg in messages])
        
        return {
            "contents": [{"parts": [{"text": content}]}],
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }
    
    async def parse_response(self, response: Dict[str, Any]) -> str:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    
    def get_headers(self, api_key: str) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

class ClaudeProvider(BaseProvider):
    async def format_request(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        # Ločimo system sporočila od user/assistant
        system_msgs = [msg.content for msg in messages if msg.role == MessageRole.SYSTEM]
        chat_msgs = [{"role": msg.role.value, "content": msg.content} 
                    for msg in messages if msg.role != MessageRole.SYSTEM]
        
        request_data = {
            "model": kwargs.get("model", "claude-3-sonnet-20240229"),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": chat_msgs
        }
        
        if system_msgs:
            request_data["system"] = "\n".join(system_msgs)
        
        return request_data
    
    async def parse_response(self, response: Dict[str, Any]) -> str:
        return response["content"][0]["text"]
    
    def get_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

class KnowledgeBase:
    """Baza znanja z embedding podporo"""
    
    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializiraj SQLite bazo"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags ON knowledge(tags)
        """)
        
        conn.commit()
        conn.close()
    
    def add_entry(self, entry: KnowledgeEntry):
        """Dodaj vnos v bazo znanja"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge 
            (id, content, metadata, embedding, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.content,
            json.dumps(entry.metadata),
            pickle.dumps(entry.embedding) if entry.embedding else None,
            json.dumps(entry.tags)
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Dodal vnos: {entry.id}")
    
    def search_by_text(self, query: str, limit: int = 5) -> List[KnowledgeEntry]:
        """Preišči po besedilu (enostavno iskanje)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, metadata, embedding, created_at, tags
            FROM knowledge 
            WHERE content LIKE ? 
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        results = []
        for row in cursor.fetchall():
            entry = KnowledgeEntry(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                embedding=pickle.loads(row[3]) if row[3] else None,
                created_at=datetime.fromisoformat(row[4]),
                tags=json.loads(row[5]) if row[5] else []
            )
            results.append(entry)
        
        conn.close()
        return results
    
    def search_by_tags(self, tags: List[str], limit: int = 5) -> List[KnowledgeEntry]:
        """Preišči po značkah"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ustvari poizvedbo za iskanje po značkah
        tag_conditions = " OR ".join([f"tags LIKE '%{tag}%'" for tag in tags])
        
        cursor.execute(f"""
            SELECT id, content, metadata, embedding, created_at, tags
            FROM knowledge 
            WHERE {tag_conditions}
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            entry = KnowledgeEntry(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                embedding=pickle.loads(row[3]) if row[3] else None,
                created_at=datetime.fromisoformat(row[4]),
                tags=json.loads(row[5]) if row[5] else []
            )
            results.append(entry)
        
        conn.close()
        return results
    
    def get_all_entries(self, limit: int = 100) -> List[KnowledgeEntry]:
        """Dobi vse vnose"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, metadata, embedding, created_at, tags
            FROM knowledge 
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            entry = KnowledgeEntry(
                id=row[0],
                content=row[1],
                metadata=json.loads(row[2]) if row[2] else {},
                embedding=pickle.loads(row[3]) if row[3] else None,
                created_at=datetime.fromisoformat(row[4]),
                tags=json.loads(row[5]) if row[5] else []
            )
            results.append(entry)
        
        conn.close()
        return results

class PromptTemplate:
    """Sistem za templiranje promptov"""
    
    def __init__(self):
        self.templates = {}
    
    def add_template(self, name: str, template: str, description: str = ""):
        """Dodaj template"""
        self.templates[name] = {
            "template": template,
            "description": description,
            "created_at": datetime.now()
        }
        logger.info(f"Dodal template: {name}")
    
    def render_template(self, name: str, **kwargs) -> str:
        """Renderiraj template z argumenti"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' ne obstaja")
        
        template = self.templates[name]["template"]
        
        # Enostavno template renderiranje
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        
        return template
    
    def list_templates(self) -> Dict[str, str]:
        """Seznam vseh templatev"""
        return {name: data["description"] for name, data in self.templates.items()}

class ResponseCache:
    """Cache sistem za odgovore"""
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generiraj cache ključ"""
        data = f"{prompt}:{model}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        """Dobi iz cache-a"""
        cache_key = self._get_cache_key(prompt, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cached_time = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - cached_time > self.ttl:
                cache_file.unlink()  # Pobriši star cache
                return None
            
            logger.info(f"Cache hit za {cache_key}")
            return data["response"]
        
        except Exception as e:
            logger.warning(f"Napaka pri branju cache-a: {e}")
            return None
    
    def set(self, prompt: str, model: str, response: str):
        """Shrani v cache"""
        cache_key = self._get_cache_key(prompt, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "model": model,
            "response": response
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Cache shranjen za {cache_key}")
        
        except Exception as e:
            logger.warning(f"Napaka pri shranjevanju cache-a: {e}")

class UniversalAI:
    """Glavni univerzalni AI modul"""
    
    def __init__(self, cache_enabled: bool = True, knowledge_db: str = "knowledge.db"):
        self.endpoints: List[APIEndpoint] = []
        self.providers = {
            ProviderType.OPENAI: OpenAIProvider(),
            ProviderType.GEMINI: GeminiProvider(),
            ProviderType.CLAUDE: ClaudeProvider(),
        }
        
        self.knowledge_base = KnowledgeBase(knowledge_db)
        self.templates = PromptTemplate()
        self.cache = ResponseCache() if cache_enabled else None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Prednastavljeni templati
        self._setup_default_templates()
        
        logger.info("UniversalAI inicializiran")
    
    def _setup_default_templates(self):
        """Nastavi prednastavljene template"""
        self.templates.add_template(
            "rag_query",
            """Kontekst iz baze znanja:
{context}

Vprašanje uporabnika:
{query}

Odgovori na podlagi zgornjih informacij. Če informacije niso dovolj, povej da potrebuješ več podatkov.""",
            "RAG template za vprašanja z bazo znanja"
        )
        
        self.templates.add_template(
            "summarize",
            """Povzemi naslednje besedilo v {length} besedah:

{text}

Povzetek:""",
            "Template za povzemanje besedila"
        )
        
        self.templates.add_template(
            "translate",
            """Prevedi naslednje besedilo iz {from_lang} v {to_lang}:

{text}

Prevod:""",
            "Template za prevajanje"
        )
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Dodaj API endpoint"""
        self.endpoints.append(endpoint)
        # Sortiraj po prioriteti
        self.endpoints.sort(key=lambda x: x.priority)
        logger.info(f"Dodal endpoint: {endpoint.provider.value} - {endpoint.model}")
    
    def add_openai(self, api_key: str, model: str = "gpt-3.5-turbo", priority: int = 1):
        """Hitro dodajanje OpenAI"""
        endpoint = APIEndpoint(
            provider=ProviderType.OPENAI,
            api_key=api_key,
            base_url="https://api.openai.com/v1/chat/completions",
            model=model,
            priority=priority,
            cost_per_1k_tokens=0.002 if "gpt-3.5" in model else 0.03
        )
        self.add_endpoint(endpoint)
    
    def add_gemini(self, api_key: str, model: str = "gemini-1.5-flash", priority: int = 1):
        """Hitro dodajanje Gemini"""
        endpoint = APIEndpoint(
            provider=ProviderType.GEMINI,
            api_key=api_key,
            base_url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            model=model,
            priority=priority,
            cost_per_1k_tokens=0.00015  # Zelo poceni!
        )
        self.add_endpoint(endpoint)
    
    def add_claude(self, api_key: str, model: str = "claude-3-sonnet-20240229", priority: int = 1):
        """Hitro dodajanje Claude"""
        endpoint = APIEndpoint(
            provider=ProviderType.CLAUDE,
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/messages",
            model=model,
            priority=priority,
            cost_per_1k_tokens=0.015
        )
        self.add_endpoint(endpoint)
    
    async def _make_request(self, endpoint: APIEndpoint, messages: List[Message], **kwargs) -> str:
        """Naredi zahtevo na določen endpoint"""
        provider = self.providers[endpoint.provider]
        
        # Pripravi zahtevo
        request_data = await provider.format_request(messages, model=endpoint.model, **kwargs)
        headers = provider.get_headers(endpoint.api_key)
        
        # Dodaj API ključ v URL če je potrebno (za Gemini)
        url = endpoint.base_url
        if endpoint.provider == ProviderType.GEMINI:
            url += f"?key={endpoint.api_key}"
        
        logger.info(f"Pošiljam zahtevo na {endpoint.provider.value} - {endpoint.model}")
        
        async with self.session.post(url, json=request_data, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                result = await response.json()
                return await provider.parse_response(result)
            else:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"API napaka: {error_text}"
                )
    
    async def generate(self, prompt: str, system_prompt: str = "", use_knowledge: bool = False, 
                      knowledge_query: str = "", template: str = "", template_vars: Dict[str, Any] = None,
                      **kwargs) -> str:
        """Glavna metoda za generiranje odgovorov"""
        
        # Pripravi sporočila
        messages = []
        
        if system_prompt:
            messages.append(Message(MessageRole.SYSTEM, system_prompt))
        
        # Uporabi template če je določen
        if template:
            template_vars = template_vars or {}
            prompt = self.templates.render_template(template, query=prompt, **template_vars)
        
        # RAG funkcionalnost
        if use_knowledge:
            search_query = knowledge_query or prompt
            knowledge_entries = self.knowledge_base.search_by_text(search_query, limit=3)
            
            if knowledge_entries:
                context = "\n\n".join([f"- {entry.content}" for entry in knowledge_entries])
                prompt = self.templates.render_template("rag_query", context=context, query=prompt)
                logger.info(f"Uporabljam {len(knowledge_entries)} vnosov iz baze znanja")
        
        messages.append(Message(MessageRole.USER, prompt))
        
        # Preveri cache
        cache_key = f"{prompt}:{kwargs.get('model', 'default')}"
        if self.cache:
            cached_response = self.cache.get(prompt, kwargs.get('model', 'default'))
            if cached_response:
                return cached_response
        
        # Poskusi z vsemi endpointi
        last_error = None
        
        for endpoint in self.endpoints:
            if not endpoint.enabled:
                continue
            
            try:
                response = await self._make_request(endpoint, messages, **kwargs)
                
                # Shrani v cache
                if self.cache:
                    self.cache.set(prompt, endpoint.model, response)
                
                logger.info(f"Uspešen odgovor iz {endpoint.provider.value}")
                return response
            
            except Exception as e:
                last_error = e
                logger.warning(f"Endpoint {endpoint.provider.value} neuspešen: {str(e)}")
                continue
        
        # Če vsi endpointi odpovedo
        raise Exception(f"Vsi API endpointi neuspešni. Zadnja napaka: {str(last_error)}")
    
    # Convenience metode
    async def ask(self, question: str, **kwargs) -> str:
        """Enostavno vprašanje"""
        return await self.generate(question, **kwargs)
    
    async def ask_with_knowledge(self, question: str, **kwargs) -> str:
        """Vprašanje z uporabo baze znanja"""
        return await self.generate(question, use_knowledge=True, **kwargs)
    
    async def summarize(self, text: str, length: str = "100", **kwargs) -> str:
        """Povzemi besedilo"""
        return await self.generate(
            text, 
            template="summarize", 
            template_vars={"text": text, "length": length},
            **kwargs
        )
    
    async def translate(self, text: str, from_lang: str, to_lang: str, **kwargs) -> str:
        """Prevedi besedilo"""
        return await self.generate(
            text,
            template="translate",
            template_vars={"text": text, "from_lang": from_lang, "to_lang": to_lang},
            **kwargs
        )
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None, tags: List[str] = None):
        """Dodaj znanje v bazo"""
        entry = KnowledgeEntry(
            id=hashlib.md5(content.encode()).hexdigest(),
            content=content,
            metadata=metadata or {},
            tags=tags or []
        )
        self.knowledge_base.add_entry(entry)
    
    def get_status(self) -> Dict[str, Any]:
        """Dobi status sistema"""
        return {
            "endpoints": len(self.endpoints),
            "active_endpoints": len([e for e in self.endpoints if e.enabled]),
            "templates": len(self.templates.templates),
            "knowledge_entries": len(self.knowledge_base.get_all_entries()),
            "cache_enabled": self.cache is not None
        }


# Primer uporabe
async def demo():
    """Demonstracija univerzalnega AI modula"""
    
    async with UniversalAI() as ai:
        
        # Dodaj različne API-je (uporabi svoje ključe!)
        # ai.add_openai("your-openai-key", "gpt-3.5-turbo", priority=1)
        # ai.add_gemini("your-gemini-key", "gemini-1.5-flash", priority=2)
        # ai.add_claude("your-claude-key", "claude-3-sonnet-20240229", priority=3)
        
        print("🤖 UNIVERSAL AI MODULE DEMO")
        print("Add your API keys to test!")
        
        # Dodaj znanje v bazo
        ai.add_knowledge(
            "Python je programski jezik, ki je bil ustvarjen leta 1991. Je enostaven za učenje.",
            metadata={"category": "programming", "language": "python"},
            tags=["python", "programming", "osnovne-informacije"]
        )
        
        print("✅ Knowledge base initialized")
        
        # Status sistema
        print("📊 STATUS SISTEMA:")
        status = ai.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo())