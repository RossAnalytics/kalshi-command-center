import os
from dotenv import load_dotenv
import kalshi_python

# Load .env into environment for local dev
load_dotenv()

def get_kalshi_client():
    """
    Returns an authenticated KalshiClient using:
      - KALSHI_API_KEY
      - KALSHI_PRIVATE_KEY
    from environment variables / .env
    """
    api_key = os.getenv("KALSHI_API_KEY")
    private_key = os.getenv("KALSHI_PRIVATE_KEY")

    if not api_key or not private_key:
        raise RuntimeError(
            "Missing KALSHI_API_KEY or KALSHI_PRIVATE_KEY in environment/.env"
        )

    configuration = kalshi_python.Configuration(
        host="https://api.elections.kalshi.com/trade-api/v2"
    )
    # The SDK expects these names:
    configuration.api_key_id = api_key
    configuration.private_key_pem = private_key

    client = kalshi_python.KalshiClient(configuration)
    return client
