"""
Configuration module for ISAC-MIMO-DRL.

Loads environment variables from .env and sets up an OpenAI-compatible
client for NVIDIA Build API endpoints.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# NVIDIA Build API Configuration
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"


def get_nvidia_client() -> OpenAI:
    """Return an OpenAI-compatible client configured for the NVIDIA Build API."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Please set it in the .env file or environment."
        )
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
    )
