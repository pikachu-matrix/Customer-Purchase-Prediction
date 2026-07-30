import sys

from sklearn.model_selection import train_test_split

from src.common.logger import logger
from src.common.exceptions import CustomException


def split_dataset(X, y):

    try:

        logger.info("\nSplitting Dataset...\n")

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            shuffle=True

        )

        logger.info(
            f"Train Shape : {X_train.shape}, Test Shape : {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:

        logger.exception("Dataset Splitting Failed.")

        raise CustomException(
            e,
            sys
        )