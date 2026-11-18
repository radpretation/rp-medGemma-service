import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
import pika
import os


# Add project root (rp-medGemma-service/) to sys.path so `src`, `utils`, and `model` resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.router.router import predict, InferencePayload
from src.configuration.config import DICOM_TEMP_PATH

from config import (
    SERVICE_NAME,
    REQUEST_EXCHANGE,
    REQUEST_EXCHANGE_TYPE,
    REQUEST_QUEUE,
    REQUEST_ROUTING_KEY,
    REQUEST_DLX_EXCHANGE,
    REQUEST_DLX_TYPE,
    REQUEST_DLQ_NAME,
    REQUEST_DLQ_ROUTING_KEY,
    REQUEST_RETRY_EXCHANGE,
    REQUEST_RETRY_TYPE,
    RETRY_DELAYS_MS,
    MAX_RETRIES,
    PREFETCH_COUNT,
    get_retry_queue_name,
    get_retry_routing_key,
)
from connection import RabbitMQConnection
from producer import publish_result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(f"rabbitmq.{SERVICE_NAME}.consumer")

_shutdown_requested = False


def get_retry_count(properties: pika.BasicProperties, payload: dict) -> int:
    """Extracts retry attempt count from message headers or payload."""
    headers = properties.headers or {}
    if "x-retry-count" in headers:
        try:
            return int(headers["x-retry-count"])
        except (ValueError, TypeError):
            pass
    return int(payload.get("retryCount", 0))


def schedule_retry(
    ch,
    method,
    properties: pika.BasicProperties,
    payload: dict,
    study_id: str,
    reason: str,
    err: Exception = None,
) -> None:
    """
    Schedules a delayed retry using RabbitMQ TTL and Dead-Letter Exchange.
    - If retry_count < MAX_RETRIES, routes message to dedicated delay queue tier.
    - If retry_count >= MAX_RETRIES, routes message to the DLQ.
    In both cases, ACKs the original message from the main queue to avoid blocking.
    """
    retry_count = get_retry_count(properties, payload)

    if retry_count >= MAX_RETRIES:
        logger.error(
            f"[{SERVICE_NAME} Consumer] Study '{study_id}' exceeded max retries ({MAX_RETRIES}). Routing to DLQ. Reason: {reason}"
        )
        send_to_dlq(ch, properties, payload, reason)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    tier = retry_count  # 0-indexed into RETRY_DELAYS_MS
    delay_ms = RETRY_DELAYS_MS[tier]
    target_routing_key = get_retry_routing_key(tier)

    logger.warning(
        f"[{SERVICE_NAME} Consumer] Transient failure for study '{study_id}'. "
        f"Scheduling retry tier {tier} (attempt {retry_count + 1}/{MAX_RETRIES}) in {delay_ms}ms. Reason: {reason}"
    )

    headers = dict(properties.headers or {})
    headers["x-retry-count"] = retry_count + 1
    headers["x-last-failure-reason"] = str(reason)
    payload["retryCount"] = retry_count + 1

    ch.basic_publish(
        exchange=REQUEST_RETRY_EXCHANGE,
        routing_key=target_routing_key,
        body=json.dumps(payload).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
            headers=headers,
            message_id=properties.message_id,
        ),
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)


def send_to_dlq_and_ack(
    ch,
    method,
    properties: pika.BasicProperties,
    payload: dict,
    study_id: str,
    reason: str,
    err: Exception = None,
) -> None:
    """Routes an unrecoverable message straight to the DLQ and ACKs the original."""
    logger.error(
        f"[{SERVICE_NAME} Consumer] Permanent failure for study '{study_id}'. Moving to DLQ. Reason: {reason}"
    )
    send_to_dlq(ch, properties, payload, reason)
    ch.basic_ack(delivery_tag=method.delivery_tag)


def send_to_dlq(
    ch,
    properties: pika.BasicProperties,
    payload: dict,
    reason: str,
) -> None:
    """Publishes a failed message to the Dead Letter Queue."""
    headers = dict(properties.headers or {}) if properties else {}
    headers["x-death-reason"] = str(reason)
    headers["x-death-at"] = datetime.now(timezone.utc).isoformat()

    dlq_payload = {
        "service": SERVICE_NAME,
        "failedAt": datetime.now(timezone.utc).isoformat(),
        "errorReason": str(reason),
        "originalPayload": payload,
    }

    ch.basic_publish(
        exchange=REQUEST_DLX_EXCHANGE,
        routing_key=REQUEST_DLQ_ROUTING_KEY,
        body=json.dumps(dlq_payload).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
            headers=headers,
        ),
    )


def process_message(ch, method, properties, body):
    """
    Message handler invoked for every incoming AI study request.
    """
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as json_err:
        logger.error(f"[{SERVICE_NAME} Consumer] Received malformed JSON payload. Discarding to DLQ: {json_err}")
        send_to_dlq_and_ack(ch, method, properties, {}, "malformed JSON", json_err)
        return

    # Extract standard fields dispatched from the Radpretation Backend
    study_id = payload.get("studyId") or payload.get("studyID")
    auth_token = payload.get("AI_AUTH_TOKEN") or payload.get("auth-token") or payload.get("authToken") or ""
    report_id = payload.get("reportId")
    request_id = payload.get("requestId") or getattr(properties, "message_id", None)

    if not study_id:
        logger.error(f"[{SERVICE_NAME} Consumer] Missing 'studyId' in payload. Discarding to DLQ.")
        send_to_dlq_and_ack(ch, method, properties, payload, "unknown", "missing studyId")
        return

    logger.info(
        f"[{SERVICE_NAME} Consumer] Received task for studyId: '{study_id}' "
        f"(reportId: '{report_id}', requestId: '{request_id}')"
    )

    try:
        logger.info(f"[{SERVICE_NAME} Consumer] Executing MedGemma inference for study '{study_id}'...")

        # Assemble inference payload and directly invoke router.py::predict()
        payload_obj = InferencePayload(
            studyId=study_id,
            pacsUrl=payload.get("pacsUrl"),
            authCred=auth_token or None,  # falls back to AUTH_CRED env var if None
        )
        response = predict(payload_obj)

        if response.status_code == 200:
            # Parse response to get the file_id
            res_data = json.loads(response.body.decode("utf-8"))
            file_id = res_data.get("file_id")

            if not file_id:
                raise ValueError("Prediction response did not return a valid file_id")

            # Load the predictions file written by router.py
            predictions_path = os.path.join(DICOM_TEMP_PATH, file_id, "predictions.json")
            if not os.path.exists(predictions_path):
                raise FileNotFoundError(f"Predictions file not found at path: {predictions_path}")

            with open(predictions_path, "r", encoding="utf-8") as f:
                result = json.load(f)

            # Cleanup the temp files and directories
            try:
                os.remove(predictions_path)
                dir_path = os.path.join(DICOM_TEMP_PATH, file_id)
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception as cleanup_err:
                logger.warning(f"[{SERVICE_NAME} Consumer] Temp prediction folder cleanup error: {cleanup_err}")

            # Validate that inference result is formatted properly
            if not isinstance(result, dict):
                raise ValueError(f"AI inference output must be a dictionary, got: {type(result)}")

            # Publish the results back to the backend's shared result queue
            publish_result(
                study_id=study_id,
                result=result,
                report_id=report_id,
                request_id=request_id,
            )

            # Successfully finished, acknowledge the incoming request
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"[{SERVICE_NAME} Consumer] Successfully completed and acknowledged study '{study_id}'")
        else:
            # Prediction failed (e.g., HTTP 400, 404/401 from PACS, or 500)
            error_msg = response.body.decode("utf-8")
            reason = f"Non-200 response status: {response.status_code} ({error_msg})"
            logger.error(f"[{SERVICE_NAME} Consumer] Inference failed for study '{study_id}': {reason}")
            schedule_retry(ch, method, properties, payload, study_id, reason)

    except Exception as err:
        logger.error(
            f"[{SERVICE_NAME} Consumer] Error processing study '{study_id}': {err}",
            exc_info=True,
        )
        schedule_retry(ch, method, properties, payload, study_id, str(err), err)

def setup_queues_and_exchanges(channel):
    """
    Declares and binds the complete topology (main, DLQ, and retry queues).
    Idempotent operation safe to run on startup.
    """
    # 1. Main request direct exchange & queue
    channel.exchange_declare(
        exchange=REQUEST_EXCHANGE,
        exchange_type=REQUEST_EXCHANGE_TYPE,
        durable=True,
    )
    channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
    channel.queue_bind(
        queue=REQUEST_QUEUE,
        exchange=REQUEST_EXCHANGE,
        routing_key=REQUEST_ROUTING_KEY,
    )

    # 2. Dead-letter exchange (DLX) & queue (DLQ)
    channel.exchange_declare(
        exchange=REQUEST_DLX_EXCHANGE,
        exchange_type=REQUEST_DLX_TYPE,
        durable=True,
    )
    channel.queue_declare(queue=REQUEST_DLQ_NAME, durable=True)
    channel.queue_bind(
        queue=REQUEST_DLQ_NAME,
        exchange=REQUEST_DLX_EXCHANGE,
        routing_key=REQUEST_DLQ_ROUTING_KEY,
    )

    # 3. Delayed-retry exchange & tiered queues
    channel.exchange_declare(
        exchange=REQUEST_RETRY_EXCHANGE,
        exchange_type=REQUEST_RETRY_TYPE,
        durable=True,
    )
    for tier, delay_ms in enumerate(RETRY_DELAYS_MS):
        queue_name = get_retry_queue_name(tier)
        routing_key = get_retry_routing_key(tier)

        channel.queue_declare(
            queue=queue_name,
            durable=True,
            arguments={
                "x-message-ttl": delay_ms,
                "x-dead-letter-exchange": REQUEST_EXCHANGE,
                "x-dead-letter-routing-key": REQUEST_ROUTING_KEY,
            },
        )
        channel.queue_bind(
            queue=queue_name,
            exchange=REQUEST_RETRY_EXCHANGE,
            routing_key=routing_key,
        )

    logger.info(f"[{SERVICE_NAME} Consumer] Topology asserted: main, DLQ, and {len(RETRY_DELAYS_MS)} retry tiers.")


def sig_handler(signum, frame):
    """Graceful shutdown handler for SIGINT and SIGTERM."""
    global _shutdown_requested
    logger.info(f"[{SERVICE_NAME} Consumer] Received shutdown signal ({signum}). Gracefully closing...")
    _shutdown_requested = True
    RabbitMQConnection.close_connection()
    sys.exit(0)


def start_consumer():
    """
    Runs the main worker loop with auto-reconnection and exponential retry backoff.
    """
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    logger.info(f"[{SERVICE_NAME} Consumer] Starting {SERVICE_NAME} worker...")

    reconnect_delay = 5

    while not _shutdown_requested:
        try:
            connection, channel = RabbitMQConnection.get_connection_and_channel()

            # Ensure all queues, exchanges and bindings are asserted
            setup_queues_and_exchanges(channel)

            # Restrict prefetch capacity for sequential processing per worker
            channel.basic_qos(prefetch_count=PREFETCH_COUNT)
            channel.basic_consume(
                queue=REQUEST_QUEUE,
                on_message_callback=process_message,
                auto_ack=False,
            )

            logger.info(f"[{SERVICE_NAME} Consumer] Listening on queue '{REQUEST_QUEUE}' (prefetch={PREFETCH_COUNT}). Waiting for tasks...")
            channel.start_consuming()

        except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as conn_err:
            if _shutdown_requested:
                break
            logger.warning(f"[{SERVICE_NAME} Consumer] RabbitMQ connection lost: {conn_err}. Reconnecting in {reconnect_delay}s...")
            RabbitMQConnection.close_connection()
            time.sleep(reconnect_delay)
        except Exception as unhandled:
            if _shutdown_requested:
                break
            logger.error(f"[{SERVICE_NAME} Consumer] Unexpected error in worker loop: {unhandled}. Reconnecting in {reconnect_delay}s...", exc_info=True)
            RabbitMQConnection.close_connection()
            time.sleep(reconnect_delay)


if __name__ == "__main__":
    start_consumer()

