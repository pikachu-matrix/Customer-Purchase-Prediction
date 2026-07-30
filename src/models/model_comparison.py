import sys
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.models.model_factory import get_models
from src.models.model_report import save_model_comparison_report
from src.common.logger import logger
from src.common.exceptions import CustomException

from config.config import MODEL_COMPARISON_REPORT_PATH


def compare_models(
    X_train,
    X_test,
    y_train,
    y_test
):

    try:

        logger.info("Starting Model Comparison...")

        categorical_columns = X_train.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        numerical_columns = [
            column
            for column in X_train.columns
            if column not in categorical_columns
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_columns
                ),
                (
                    "numerical",
                    "passthrough",
                    numerical_columns
                )
            ]
        )

        models = get_models()

        results = []

        best_model = None
        best_model_name = None
        best_rmse = float("inf")

        for model_name, model in models.items():

            logger.info(f"Training {model_name}...")

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            r2 = r2_score(
                y_test,
                predictions
            )

            results.append({
                "Model": model_name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2 Score": round(r2, 4)
            })

            logger.info(
                f"{model_name} | MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}"
            )

            if rmse < best_rmse:

                best_rmse = rmse
                best_model = pipeline
                best_model_name = model_name

        comparison_df = pd.DataFrame(results)

        comparison_df = comparison_df.sort_values(
            by="RMSE"
        ).reset_index(drop=True)

        os.makedirs(
            "outputs/reports",
            exist_ok=True
        )

        comparison_df.to_csv(
            "outputs/reports/model_comparison.csv",
            index=False
        )

        save_model_comparison_report(
            comparison_df,
            MODEL_COMPARISON_REPORT_PATH
        )

        logger.info(f"Best Model : {best_model_name}")
        logger.info("Model Comparison Completed.")

        return (
            best_model,
            best_model_name,
            comparison_df
        )

    except Exception as e:

        logger.exception("Model Comparison Failed.")

        raise CustomException(
            e,
            sys
        )