import os

# ==============================================================================
# RABBITMQ BROKER & TOPOLOGY CONFIGURATION (MEDGEMMA WORKER)
#
# Edit connection URL, exchange names, or queue names in this file only.
# All other worker modules import their settings from this file.
# ==============================================================================

# 1. Broker Connection Settings
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://localhost")
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "60"))
BLOCKED_CONNECTION_TIMEOUT = int(os.getenv("BLOCKED_CONNECTION_TIMEOUT", "300"))

# 2. Worker Identity
SERVICE_NAME = "medgemma"

# 3. Incoming AI Request Queue Topology (MedGemma Worker consumes from this)
REQUEST_EXCHANGE = os.getenv("RABBITMQ_AI_REQUEST_EXCHANGE", "ai.request.exchange")
REQUEST_EXCHANGE_TYPE = "direct"
REQUEST_QUEUE = os.getenv("RABBITMQ_MEDGEMMA_REQUEST_QUEUE", "ai.medgemma.request")
REQUEST_ROUTING_KEY = "medgemma"

# 4. Dead-Letter Queue (DLQ) Topology for Failed Requests
REQUEST_DLX_EXCHANGE = os.getenv("RABBITMQ_AI_REQUEST_DLX", "ai.request.dlx")
REQUEST_DLX_TYPE = "direct"
REQUEST_DLQ_NAME = os.getenv("RABBITMQ_MEDGEMMA_REQUEST_DLQ", "ai.medgemma.request.dlq")
REQUEST_DLQ_ROUTING_KEY = "ai.medgemma.request.dlq"

# 5. Delayed-Retry Topology (Exponential Backoff Tiers)
REQUEST_RETRY_EXCHANGE = os.getenv("RABBITMQ_AI_REQUEST_RETRY_EXCHANGE", "ai.request.retry.exchange")
REQUEST_RETRY_TYPE = "direct"
# Tier delays in milliseconds: 5s, 15s, 30s, 60s, 120s
RETRY_DELAYS_MS = [5_000, 15_000, 30_000, 60_000, 120_000]
MAX_RETRIES = len(RETRY_DELAYS_MS)

def get_retry_queue_name(tier: int) -> str:
    """Returns the dedicated queue name for a specific delayed retry tier."""
    return f"{REQUEST_QUEUE}.retry.{tier}"

def get_retry_routing_key(tier: int) -> str:
    """Returns the routing key used to publish a message into a retry tier."""
    return f"{SERVICE_NAME}.retry.{tier}"

# 6. Outgoing AI Result Queue Topology (MedGemma Worker publishes results here)
RESULT_EXCHANGE = os.getenv("RABBITMQ_AI_RESPONSE_EXCHANGE", "ai.response.exchange")
RESULT_EXCHANGE_TYPE = "direct"
RESULT_QUEUE = os.getenv("RABBITMQ_AI_RESULT_QUEUE", "ai.result.queue")
RESULT_ROUTING_KEY = "ai.result"

# 7. Worker Processing Settings
# Restrict prefetch capacity to 1 so the worker processes one study at a time
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", "1"))

