import sys
import pandas as pd

from src.common.exceptions import CustomException
from src.common.logger import logger


def load_data(file_path):

    try:

        logger.info(f"Loading dataset from: {file_path}")

        df = pd.read_csv(file_path)

        logger.info("Dataset loaded successfully.")

        return df

    except Exception as e:

        logger.exception("Failed to load dataset.")

        raise CustomException(
            e,
            sys
        )