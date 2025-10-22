"""
Logging Configuration

Setup and configure logging for the application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        format_string: Custom format string (optional)

    Returns:
        Configured root logger
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logging configured - Level: {level}, Console: {console}, File: {log_file}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for specific module

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_experiment_logging(config: dict) -> logging.Logger:
    """
    Setup logging from experiment configuration

    Args:
        config: Experiment configuration dictionary

    Returns:
        Configured logger
    """
    logging_config = config.get('logging', {})

    level = logging_config.get('level', 'INFO')
    log_file = logging_config.get('file', None)
    console = logging_config.get('console', True)

    return setup_logging(level=level, log_file=log_file, console=console)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for adding context to log messages

    Usage:
        logger = LoggerAdapter(logging.getLogger(__name__), {'patient_id': 'patient_001'})
        logger.info("Processing patient")  # Will include patient_id in log
    """

    def process(self, msg, kwargs):
        """Add extra context to log message"""
        context_str = " | ".join(f"{k}={v}" for k, v in self.extra.items())
        return f"[{context_str}] {msg}", kwargs


def create_patient_logger(patient_id: str) -> LoggerAdapter:
    """
    Create logger with patient context

    Args:
        patient_id: Patient identifier

    Returns:
        Logger adapter with patient context
    """
    base_logger = logging.getLogger('patient_simulation')
    return LoggerAdapter(base_logger, {'patient_id': patient_id})


def create_conversation_logger(patient_id: str, day: int, condition: str) -> LoggerAdapter:
    """
    Create logger with conversation context

    Args:
        patient_id: Patient identifier
        day: Simulation day
        condition: Experimental condition

    Returns:
        Logger adapter with conversation context
    """
    base_logger = logging.getLogger('conversation')
    return LoggerAdapter(base_logger, {
        'patient_id': patient_id,
        'day': day,
        'condition': condition
    })


def log_error_with_context(logger: logging.Logger, error: Exception, context: dict):
    """
    Log error with additional context

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context dictionary
    """
    context_str = " | ".join(f"{k}={v}" for k, v in context.items())
    logger.error(f"Error occurred: {str(error)} | Context: {context_str}", exc_info=True)


def log_performance(logger: logging.Logger, operation: str, duration_seconds: float):
    """
    Log performance metrics

    Args:
        logger: Logger instance
        operation: Name of operation
        duration_seconds: Duration in seconds
    """
    logger.info(f"Performance | Operation: {operation} | Duration: {duration_seconds:.2f}s")
