from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from config.config import RANDOM_STATE
from config.config import N_ESTIMATORS
from src.common.logger import logger

def train_model(X_train, y_train):

    logger.info("\nTraining Model...\n")

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

                OneHotEncoder(

                    handle_unknown="ignore"

                ),

                categorical_columns

            ),

            (

                "numerical",

                "passthrough",

                numerical_columns

            )

        ]

    )

    model = Pipeline(

        steps=[

            (

                "preprocessor",

                preprocessor

            ),

            (

                "model",
                
                RandomForestRegressor(

                    n_estimators = N_ESTIMATORS,

                    random_state = RANDOM_STATE,

                    n_jobs=-1

                )

            )

        ]

    )

    model.fit(

        X_train,

        y_train

    )

    logger.info("\nModel Training Completed.\n")

    return model