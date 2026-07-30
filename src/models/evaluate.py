"""
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from src.common.logger import logger
from config.config import REPORT_PATH
import os

def evaluate_model(model, X_test, y_test):
    logger.info("\nEvaluating Model...\n")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

   #mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test,predictions))

    r2 = r2_score(y_test, predictions)


    logger.info(
        f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}"
    )

    save_evaluation_report(
        mae,rmse,r2)
    
    #rint(f"MAE  : {mae:.2f}")
    #rint(f"MSE  : {mse:.2f}")
    #rint(f"RMSE : {rmse:.2f}")
    #rint(f"R²   : {r2:.4f}")

    return {
    "predictions": predictions,
    "mae": mae,
   #"mse": mse,
    "rmse": rmse,
    "r2": r2
}
def save_evaluation_report(mae, rmse, r2):

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    with open(REPORT_PATH, "w") as file:

        file.write("Model Evaluation Report\n")
        file.write("=======================\n\n")
        file.write("Model : Random Forest Regressor\n\n")
        file.write(f"MAE  : {mae:.2f}\n")
        file.write(f"RMSE : {rmse:.2f}\n")
        file.write(f"R2   : {r2:.4f}\n")

    logger.info("Evaluation report saved successfully.")
"""