import sys

from src.common.logger import logger
from src.common.exceptions import CustomException


def select_features(df, training=True):

    try:

        logger.info("\nSelecting Features...\n")

        feature_columns = [

            "Purchase_Amount",

            "Purchase_Year",
            "Purchase_Month",
            "Purchase_Day",
            "Day_Of_Week",
            "Week_Of_Year",
            "Purchase_Quarter",
            "Is_Weekend",

            "Historical_Purchase_Count",
            "Historical_Total_Spend",
            "Historical_Average_Spend",
            "Historical_Minimum_Spend",
            "Historical_Maximum_Spend",
            "Historical_Median_Spend",
            "Historical_Spend_Standard_Deviation",

            "Previous_Purchase_Amount",
            "Purchase_Growth",

            "Customer_Lifetime_Days",
            "Historical_Unique_Categories",
            "Historical_Most_Frequent_Category",

            "Previous_Purchase_Gap_Days",
            "Historical_Average_Gap_Days",
            "Historical_Minimum_Gap_Days",
            "Historical_Maximum_Gap_Days",
            "Historical_Gap_Standard_Deviation",

            "Product_Category"

        ]

        X = df[feature_columns]

        if training:

            y = df["Target_Days_Until_Next_Purchase"]

            logger.info("Feature Selection Completed.")

            return X, y

        logger.info("Feature Selection Completed.")

        return X

    except Exception as e:

        logger.exception("Feature Selection Failed.")

        raise CustomException(
            e,
            sys
        )