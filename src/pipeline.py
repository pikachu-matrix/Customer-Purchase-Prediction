import sys

from src.data.data_loader import load_data
from src.data.validator import validate_dataset
from src.data.preprocessing import preprocess_data
from src.data.quality_checker import check_data_quality

from src.features.time_features import create_time_features
from src.features.customer_features import create_customer_features
from src.features.gap_features import create_gap_features
from src.features.target_creator import create_target
from src.features.feature_selector import select_features

from src.models.split_data import split_dataset
from src.models.model_comparison import compare_models
from src.models.model_registry import save_model, load_model
from src.models.predict import predict, save_predictions
from src.models.feature_importance import plot_feature_importance

from config.config import (
    TEST_DATA_PATH,
    MODEL_PATH,
    PREDICTION_OUTPUT_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_CSV_PATH,
)

from src.common.logger import logger
from src.common.exceptions import CustomException


def run_pipeline():

    try:

        logger.info("======= Training Pipeline Started =======")

        file_path = "data/raw/train.csv"

        df = load_data(file_path)

        validate_dataset(df)

        df = preprocess_data(df)

        check_data_quality(df)

        df = create_time_features(df)

        df = create_customer_features(df)

        df = create_gap_features(df)

        df = create_target(df)

        X, y = select_features(df)

        X_train, X_test, y_train, y_test = split_dataset(
            X,
            y
        )

        model, model_name, comparison_df = compare_models(
            X_train,
            X_test,
            y_train,
            y_test
        )

        logger.info(f"Best Model: {model_name}")

        save_model(
            model,
            MODEL_PATH
        )

        plot_feature_importance(
            model=model,
            plot_path=FEATURE_IMPORTANCE_PATH,
            csv_path=FEATURE_IMPORTANCE_CSV_PATH
        )

        logger.info("======= Training Pipeline Completed =======")

    except Exception as e:

        logger.exception("Training Pipeline Failed.")

        raise CustomException(
            e,
            sys
        )


def run_prediction_pipeline():

    try:

        logger.info("======= Prediction Pipeline Started =======")

        df = load_data(TEST_DATA_PATH)

        validate_dataset(df)

        df = preprocess_data(df)

        check_data_quality(df)

        df = create_time_features(df)

        df = create_customer_features(df)

        df = create_gap_features(df)

        X = select_features(
            df,
            training=False
        )

        model = load_model(MODEL_PATH)

        predictions = predict(
            model,
            X
        )

        save_predictions(
            df,
            predictions,
            PREDICTION_OUTPUT_PATH
        )

        logger.info("======= Prediction Pipeline Completed =======")

    except Exception as e:

        logger.exception("Prediction Pipeline Failed.")

        raise CustomException(
            e,
            sys
        )