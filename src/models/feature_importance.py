import sys
import os

import matplotlib.pyplot as plt
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


def plot_feature_importance(
    model,
    plot_path,
    csv_path
):

    try:

        model_object = model.named_steps["model"]

        if not hasattr(model_object, "feature_importances_"):

            logger.info(
                "Feature importance not available for this model."
            )

            return

        feature_importance = model_object.feature_importances_

        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        importance_df = pd.DataFrame({

            "Feature": feature_names,
            "Importance": feature_importance

        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).reset_index(drop=True)

        importance_df.insert(
            0,
            "Rank",
            range(1, len(importance_df) + 1)
        )

        os.makedirs(
            os.path.dirname(csv_path),
            exist_ok=True
        )

        importance_df.to_csv(
            csv_path,
            index=False
        )

        top_features = importance_df.head(10)

        plt.figure(figsize=(10, 6))

        plt.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        plt.xlabel("Importance")

        plt.ylabel("Feature")

        plt.title("Top 10 Feature Importances")

        plt.gca().invert_yaxis()

        os.makedirs(
            os.path.dirname(plot_path),
            exist_ok=True
        )

        plt.tight_layout()

        plt.savefig(plot_path)

        plt.close()

        logger.info(
            "Feature importance report generated successfully."
        )

    except Exception as e:

        logger.exception(
            "Feature Importance Generation Failed."
        )

        raise CustomException(
            e,
            sys
        )