import pika
import logging
from config import RABBITMQ_URL, HEARTBEAT_SECONDS, BLOCKED_CONNECTION_TIMEOUT

logger = logging.getLogger(__name__)

class RabbitMQConnection:
    """
    Thread-safe connection and channel manager for RabbitMQ using Pika.
    Automatically reconnects if the connection or channel is closed.
    """
    _connection = None
    _channel = None

    @classmethod
    def get_connection_and_channel(cls):
        """
        Returns an open (connection, channel) tuple.
        Establishes a new connection if uninitialized or closed.
        """
        if cls._connection and cls._connection.is_open and cls._channel and cls._channel.is_open:
            return cls._connection, cls._channel

        cls.close_connection()

        # Mask password in logs if present
        safe_url = RABBITMQ_URL
        if "@" in safe_url and "://" in safe_url:
            proto, rest = safe_url.split("://", 1)
            creds, host = rest.split("@", 1)
            user = creds.split(":", 1)[0]
            safe_url = f"{proto}://{user}:*****@{host}"

        logger.info(f"[RabbitMQ Connection] Connecting to broker at: {safe_url}")
        
        parameters = pika.URLParameters(RABBITMQ_URL)
        parameters.heartbeat = HEARTBEAT_SECONDS
        parameters.blocked_connection_timeout = BLOCKED_CONNECTION_TIMEOUT

        cls._connection = pika.BlockingConnection(parameters)
        cls._channel = cls._connection.channel()
        logger.info("[RabbitMQ Connection] Connection and channel successfully initialized.")
        return cls._connection, cls._channel

    @classmethod
    def close_connection(cls):
        """
        Safely closes the active channel and connection if open.
        """
        try:
            if cls._channel and cls._channel.is_open:
                cls._channel.close()
        except Exception as e:
            logger.debug(f"[RabbitMQ Connection] Error closing channel: {e}")
        finally:
            cls._channel = None

        try:
            if cls._connection and cls._connection.is_open:
                cls._connection.close()
        except Exception as e:
            logger.debug(f"[RabbitMQ Connection] Error closing connection: {e}")
        finally:
            cls._connection = None

