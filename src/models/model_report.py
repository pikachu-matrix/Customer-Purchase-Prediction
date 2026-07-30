import sys
import os

from src.common.logger import logger
from src.common.exceptions import CustomException


def save_model_comparison_report(
    results_df,
    output_path
):

    try:

        logger.info("Generating Model Comparison Report...")

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        with open(output_path, "w") as file:

            file.write("MODEL COMPARISON REPORT\n")
            file.write("=" * 60)
            file.write("\n\n")

            file.write(
                results_df.to_string(index=False)
            )

            file.write("\n")

        logger.info(
            f"Model Comparison Report saved at: {output_path}"
        )

    except Exception as e:

        logger.exception(
            "Model Comparison Report Generation Failed."
        )

        raise CustomException(
            e,
            sys
        )