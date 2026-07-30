import sys
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


# ==========================================================
# Step 1 :- Calculate Previous Purchase Gap
# ==========================================================

def create_previous_purchase_gap(df):

    previous_purchase_date = (

        df
        .groupby("Customer_ID")["Transaction_Date"]
        .shift(1)

    )

    df["Previous_Purchase_Gap_Days"] = (

        df["Transaction_Date"] - previous_purchase_date

    ).dt.days

    df["Previous_Purchase_Gap_Days"] = (

        df["Previous_Purchase_Gap_Days"]
        .fillna(0)

    )

    return df


# ==========================================================
# Step 2 :- Helper Function
# ==========================================================

def create_gap_statistic(df, feature_name, statistic):

    grouped = df.groupby("Customer_ID")["Previous_Purchase_Gap_Days"]

    if statistic == "mean":

        values = grouped.expanding().mean()

    elif statistic == "min":

        values = grouped.expanding().min()

    elif statistic == "max":

        values = grouped.expanding().max()

    elif statistic == "std":

        values = grouped.expanding().std()

    values = values.reset_index(level=0, drop=True)

    values = values.groupby(df["Customer_ID"]).shift(1)

    values = values.fillna(0)

    df[feature_name] = values

    return df


# ==========================================================
# Step 3 :- Wrapper Functions
# ==========================================================

def create_average_gap(df):

    return create_gap_statistic(

        df,

        "Historical_Average_Gap_Days",

        "mean"

    )


def create_minimum_gap(df):

    return create_gap_statistic(

        df,

        "Historical_Minimum_Gap_Days",

        "min"

    )


def create_maximum_gap(df):

    return create_gap_statistic(

        df,

        "Historical_Maximum_Gap_Days",

        "max"

    )


def create_gap_std(df):

    return create_gap_statistic(

        df,

        "Historical_Gap_Standard_Deviation",

        "std"

    )


# ==========================================================
# Step 4 :- Main Function
# ==========================================================

def create_gap_features(df):

    try:

        logger.info("\nCreating Gap Features...\n")

        df = create_previous_purchase_gap(df)

        df = create_average_gap(df)

        df = create_minimum_gap(df)

        df = create_maximum_gap(df)

        df = create_gap_std(df)

        logger.info("\nGap Features Created Successfully.\n")

        return df

    except Exception as e:

        logger.exception("Gap Feature Engineering Failed.")

        raise CustomException(
            e,
            sys
        )