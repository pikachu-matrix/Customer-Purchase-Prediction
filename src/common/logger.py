"""
=========================================================
Logger Configuration
=========================================================

Creates a project-wide logger that writes logs to:

1. Console
2. logs/project.log

=========================================================
"""

import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/project.log"

logger = logging.getLogger("PurchasePrediction")

logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not logger.handlers:

    formatter = logging.Formatter(

        "%(asctime)s - %(levelname)s - %(message)s"

    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)