import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    fal_api_key: Optional[str] = os.getenv("FAL_API_KEY")
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    openai_audio_model: str = os.getenv("OPENAI_AUDIO_MODEL", "gpt-4o-mini-transcribe")


settings = Settings()


def default_model_for(provider: str) -> str:
    provider_lower = provider.lower()
    if provider_lower == "openai":
        return settings.openai_model
    return settings.openai_model

