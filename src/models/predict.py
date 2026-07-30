import sys
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


def predict(model, X):

    try:

        logger.info("\nGenerating Predictions...\n")

        predictions = model.predict(X)

        logger.info("Predictions generated successfully.")

        return predictions

    except Exception as e:

        logger.exception("Prediction Failed.")

        raise CustomException(
            e,
            sys
        )


def save_predictions(dataframe, predictions, output_path):

    try:

        logger.info("\nSaving Predictions...\n")

        result = dataframe[
            [
                "Customer_ID",
                "Transaction_Date",
                "Product_Category",
                "Purchase_Amount"
            ]
        ].copy()

        result["Predicted_Days_Until_Next_Purchase"] = predictions

        result.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Predictions saved to {output_path}"
        )

    except Exception as e:

        logger.exception("Saving Predictions Failed.")

        raise CustomException(
            e,
            sys
        )