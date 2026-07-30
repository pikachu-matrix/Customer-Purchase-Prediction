import sys
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


def create_next_purchase_date(df):

    # print("Creating Next Purchase Date...")

    df["Next_Purchase_Date"] = (

        df
        .groupby("Customer_ID")["Transaction_Date"]
        .shift(-1)

    )

    return df


def create_target_days(df):

    # print("Creating Target Days Until Next Purchase...")

    df["Target_Days_Until_Next_Purchase"] = (

        df["Next_Purchase_Date"]

        -

        df["Transaction_Date"]

    ).dt.days

    return df


def remove_last_purchase(df):

    # print("Removing Last Purchase Of Every Customer...")

    df = df.dropna(

        subset=["Target_Days_Until_Next_Purchase"]

    )

    df = df.reset_index(

        drop=True

    )

    return df


def create_target(df):

    try:

        logger.info("\nCreating Target...\n")

        df = create_next_purchase_date(df)

        df = create_target_days(df)

        df = remove_last_purchase(df)

        logger.info("\nTarget Created Successfully.\n")

        return df

    except Exception as e:

        logger.exception("Target Creation Failed.")

        raise CustomException(
            e,
            sys
        )