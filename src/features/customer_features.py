"""
=========================================================
Customer Feature Engineering
=========================================================

Purpose
-------
Create customer history features without using any
future information.

This section creates:

1. Historical Purchase Count
2. Historical Total Spend
3. Historical Average Spend
4. Historical Minimum Spend
5. Historical Maximum Spend
6. Historical Median Spend
7. Historical Spend Standard Deviation
8. Previous Purchase Amount
9. Purchase Growth
10. Customer Lifetime
11. Historical Unique Categories
12. Historical Most Frequent Category

=========================================================
"""

import sys
import pandas as pd
import numpy as np

from src.common.logger import logger
from src.common.exceptions import CustomException

"""
Creating Helper functions for reusable purposes
"""

def create_historical_statistic(df, feature_name, statistic):

    grouped_data = df.groupby("Customer_ID")["Purchase_Amount"]

    if statistic == "min":

        values = grouped_data.expanding().min()

    elif statistic == "max":

        values = grouped_data.expanding().max()

    elif statistic == "median":

        values = grouped_data.expanding().median()

    elif statistic == "std":

        values = grouped_data.expanding().std()

    values = values.reset_index(level=0, drop=True)

    values = values.groupby(df["Customer_ID"]).shift(1)

    values = values.fillna(0)

    df[feature_name] = values

    return df


def create_purchase_count(df):
    """
    Create the historical purchase count.
    """

    df["Historical_Purchase_Count"] = (

        df
        .groupby("Customer_ID")
        .cumcount()

    )

    return df


def create_total_spend(df):
    """
    Create historical total spending.
    """

    cumulative_spend = (

        df
        .groupby("Customer_ID")["Purchase_Amount"]
        .cumsum()

    )

    df["Historical_Total_Spend"] = (

        cumulative_spend
        .groupby(df["Customer_ID"])
        .shift(1)
        .fillna(0)

    )

    return df


def create_average_spend(df):
    """
    Create historical average spending.
    """

    df["Historical_Average_Spend"] = 0.0

    customers_with_history = (

        df["Historical_Purchase_Count"] > 0

    )

    df.loc[
        customers_with_history,
        "Historical_Average_Spend"
    ] = (

        df.loc[
            customers_with_history,
            "Historical_Total_Spend"
        ]

        /

        df.loc[
            customers_with_history,
            "Historical_Purchase_Count"
        ]

    )

    return df


def create_minimum_spend(df):

    return create_historical_statistic(

        df,

        "Historical_Minimum_Spend",

        "min"

    )


def create_maximum_spend(df):

    return create_historical_statistic(

        df,

        "Historical_Maximum_Spend",

        "max"

    )


def create_median_spend(df):

    return create_historical_statistic(

        df,

        "Historical_Median_Spend",

        "median"

    )


def create_spending_std(df):

    return create_historical_statistic(

        df,

        "Historical_Spend_Standard_Deviation",

        "std"

    )


def create_previous_purchase_amount(df):

    df["Previous_Purchase_Amount"] = (

        df
        .groupby("Customer_ID")["Purchase_Amount"]
        .shift(1)
        .fillna(0)

    )

    return df


def create_purchase_growth(df):

    df["Purchase_Growth"] = 0.0

    customers_with_history = (

        df["Previous_Purchase_Amount"] > 0

    )

    df.loc[
        customers_with_history,
        "Purchase_Growth"
    ] = (

        (
            df.loc[
                customers_with_history,
                "Purchase_Amount"
            ]

            -

            df.loc[
                customers_with_history,
                "Previous_Purchase_Amount"
            ]
        )

        /

        df.loc[
            customers_with_history,
            "Previous_Purchase_Amount"
        ]

    )

    return df
def create_customer_lifetime(df):

    first_purchase_date = (

        df
        .groupby("Customer_ID")["Transaction_Date"]
        .transform("min")

    )

    df["Customer_Lifetime_Days"] = (

        df["Transaction_Date"] - first_purchase_date

    ).dt.days

    return df


def create_unique_category_count(df):

    df["Historical_Unique_Categories"] = 0

    for customer_id, customer_data in df.groupby("Customer_ID"):

        unique_counts = []

        categories_seen = set()

        for category in customer_data["Product_Category"]:

            unique_counts.append(len(categories_seen))

            categories_seen.add(category)

        df.loc[
            customer_data.index,
            "Historical_Unique_Categories"
        ] = unique_counts

    return df


def create_most_frequent_category(df):

    df["Historical_Most_Frequent_Category"] = "Unknown"

    for customer_id, customer_data in df.groupby("Customer_ID"):

        category_counter = {}

        most_frequent = []

        for category in customer_data["Product_Category"]:

            if len(category_counter) == 0:

                most_frequent.append("Unknown")

            else:

                highest = max(

                    category_counter,

                    key=category_counter.get

                )

                most_frequent.append(highest)

            if category not in category_counter:

                category_counter[category] = 0

            category_counter[category] += 1

        df.loc[
            customer_data.index,
            "Historical_Most_Frequent_Category"
        ] = most_frequent

    return df


def create_customer_features(df):
    """
    Create all customer features.
    """

    try:

        logger.info("\nCreating Customer Features...\n")

        df = create_purchase_count(df)

        df = create_total_spend(df)

        df = create_average_spend(df)

        df = create_minimum_spend(df)

        df = create_maximum_spend(df)

        df = create_median_spend(df)

        df = create_spending_std(df)

        df = create_previous_purchase_amount(df)

        df = create_purchase_growth(df)

        df = create_customer_lifetime(df)

        df = create_unique_category_count(df)

        df = create_most_frequent_category(df)

        logger.info("\nCustomer Features Created Successfully.\n")

        return df

    except Exception as e:

        logger.exception("Customer Feature Engineering Failed.")

        raise CustomException(
            e,
            sys
        )