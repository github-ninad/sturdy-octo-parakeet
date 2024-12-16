from pydantic_settings import BaseSettings
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = "v1"
    API_TITLE: str = "Health Claims Adjudication API"

    # Database Settings
    LANCEDB_URI: str = "tmp/lancedb"

    # Model Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Agent Settings
    AGENT_CONFIG: Dict[str, Any] = {
        "manager_model": OpenAIChat(id="gpt-4o"),
        "lead_model": OpenAIChat(id="gpt-4o"),
        "worker_model": OpenAIChat(id="gpt-4o"),
        "memory_db": "tmp/agent_memory.db",
        "storage_db": "tmp/agent_storage.db"
    }


settings = Settings()
