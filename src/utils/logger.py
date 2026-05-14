import logging
import os
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime


# ─────────────────────────────────────────────
# Log Configuration
# ─────────────────────────────────────────────
LOG_DIR = "logs"

LOG_FILE = (
    f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')}.log"
)

MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3


# ─────────────────────────────────────────────
# Log File Path
# ─────────────────────────────────────────────
log_dir_path = os.path.join(from_root(), LOG_DIR)

os.makedirs(log_dir_path, exist_ok=True)

log_file_path = os.path.join(
    log_dir_path,
    LOG_FILE
)


# ─────────────────────────────────────────────
# Configure Logger
# ─────────────────────────────────────────────
def get_logger(name=__name__):

    logger = logging.getLogger(name)

    if not logger.hasHandlers():

        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[ %(asctime)s ] "
            "[ %(levelname)s ] "
            "[ %(filename)s:%(lineno)d ] "
            "🚀 %(message)s ✅"
        )

        # File Handler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )

        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console Handler
        console_handler = logging.StreamHandler()

        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Add Handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info("Logger configured successfully...🏆")

    return logger


# ─────────────────────────────────────────────
# Initialize Logger
# ─────────────────────────────────────────────
logger = get_logger()