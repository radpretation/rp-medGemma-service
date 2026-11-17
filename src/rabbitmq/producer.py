import json
import logging
import pika
from connection import RabbitMQConnection
from config import (
    SERVICE_NAME,
    RESULT_EXCHANGE,
    RESULT_EXCHANGE_TYPE,
    RESULT_QUEUE,
    RESULT_ROUTING_KEY,
)

logger = logging.getLogger(__name__)

def publish_result(
    study_id: str,
    result: dict,
    report_id: str = None,
    request_id: str = None,
) -> bool:
    """
    Publishes AI inference results back to the backend's shared result queue.
    
    :param study_id: The Orthanc study instance UID / study ID.
    :param result: The normalized findings dictionary (normal, abnormality, body_part, finding).
    :param report_id: Optional MongoDB ScanReport _id (if provided in the request).
    :param request_id: Unique request correlation ID.
    :return: True if published successfully, raises Exception otherwise.
    """
    try:
        connection, channel = RabbitMQConnection.get_connection_and_channel()

        # Assert direct response exchange and shared result queue
        channel.exchange_declare(
            exchange=RESULT_EXCHANGE,
            exchange_type=RESULT_EXCHANGE_TYPE,
            durable=True,
        )
        channel.queue_declare(queue=RESULT_QUEUE, durable=True)
        channel.queue_bind(
            queue=RESULT_QUEUE,
            exchange=RESULT_EXCHANGE,
            routing_key=RESULT_ROUTING_KEY,
        )

        message = {
            "service": SERVICE_NAME,
            "studyId": study_id,
            "result": result,
        }

        payload = json.dumps(message)

        channel.basic_publish(
            exchange=RESULT_EXCHANGE,
            routing_key=RESULT_ROUTING_KEY,
            body=payload.encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent message on disk
                content_type="application/json",
                message_id=request_id,
            ),
        )

        logger.info(
            f"[RabbitMQ Producer] Published {SERVICE_NAME} result for studyId '{study_id}' "
            f"(reportId: '{report_id}', requestId: '{request_id}') to '{RESULT_QUEUE}'"
        )
        return True
    except Exception as err:
        logger.error(
            f"[RabbitMQ Producer] Failed to publish {SERVICE_NAME} result for studyId '{study_id}': {err}",
            exc_info=True,
        )
        raise

