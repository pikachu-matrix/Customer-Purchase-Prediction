import sys

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from config.config import RANDOM_STATE
from config.config import N_ESTIMATORS

from src.common.logger import logger
from src.common.exceptions import CustomException


def get_models():

    try:

        logger.info("Creating model dictionary...")

        models = {

            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(

                n_estimators=N_ESTIMATORS,

                random_state=RANDOM_STATE,

                n_jobs=-1

            ),

            "Gradient Boosting": GradientBoostingRegressor(

                random_state=RANDOM_STATE

            )

        }

        logger.info(f"{len(models)} models loaded successfully.")

        return models

    except Exception as e:

        logger.exception("Model Factory Failed.")

        raise CustomException(
            e,
            sys
        )