"""
=========================================================
Time Feature Engineering
=========================================================

Purpose:
--------
Create calendar-based features from the Transaction_Date
column.

This module is responsible ONLY for time-related features.

Created Features
----------------
1. Purchase_Year
2. Purchase_Month
3. Purchase_Day
4. Day_Of_Week
5. Week_Of_Year
6. Purchase_Quarter
7. Is_Weekend

=========================================================
"""

import sys
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


def create_time_features(dataframe):
    """
    Create all calendar-based features.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    try:

        logger.info("\nCreating Time Features...")

        dataframe["Purchase_Year"] = dataframe["Transaction_Date"].dt.year

        dataframe["Purchase_Month"] = dataframe["Transaction_Date"].dt.month

        dataframe["Purchase_Day"] = dataframe["Transaction_Date"].dt.day

        dataframe["Day_Of_Week"] = dataframe["Transaction_Date"].dt.day_name()

        dataframe["Week_Of_Year"] = (
            dataframe["Transaction_Date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        dataframe["Purchase_Quarter"] = (
            dataframe["Transaction_Date"].dt.quarter
        )

        dataframe["Is_Weekend"] = (
            dataframe["Day_Of_Week"]
            .isin(["Saturday", "Sunday"])
        )

        logger.info("Time Features Created Successfully.")

        return dataframe

    except Exception as e:

        logger.exception("Time Feature Engineering Failed.")

        raise CustomException(
            e,
            sys
        )