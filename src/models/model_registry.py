import sys
import os
import joblib

from src.common.logger import logger
from src.common.exceptions import CustomException


def save_model(model, model_path):

    try:

        logger.info("\nSaving Model....\n")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(
            model,
            model_path
        )

        logger.info(f"Model saved at: {model_path}")

    except Exception as e:

        logger.exception("Model Saving Failed.")

        raise CustomException(
            e,
            sys
        )


def load_model(model_path):

    try:

        logger.info("\nLoading Model....\n")

        model = joblib.load(
            model_path
        )

        logger.info("\nModel loaded successfully.\n")

        return model

    except Exception as e:

        logger.exception("Model Loading Failed.")

        raise CustomException(
            e,
            sys
        )