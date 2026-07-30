import sys

from src.pipeline import (
    run_pipeline,
    run_prediction_pipeline
)

from src.common.logger import logger
from src.common.exceptions import CustomException


if __name__ == "__main__":

    try:

        logger.info("\n=============================")
        logger.info("Purchase Prediction System")
        logger.info("=============================")

        logger.info("1. Train Model")
        logger.info("2. Predict using Saved Model")

        choice = input("\nEnter your choice (1 or 2): ")

        if choice == "1":

            run_pipeline()

        elif choice == "2":

            run_prediction_pipeline()

        else:

            logger.warning(
                "Invalid choice. Please enter 1 or 2."
            )

    except Exception as e:

        logger.exception("Application Execution Failed.")

        raise CustomException(
            e,
            sys
        )