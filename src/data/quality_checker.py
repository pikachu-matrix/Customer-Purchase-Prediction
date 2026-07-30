import sys
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


def check_data_quality(df):

    try:

        # ==========================================================
        # Store All Quality Issues
        # ==========================================================

        logger.info("\nRunning data quality checks...\n")

        quality_issues = []

        # ==========================================================
        # Check 1
        # Dataset Empty
        # ==========================================================

        if len(df) == 0:

            quality_issues.append(
                "Dataset contains zero rows."
            )

        # ==========================================================
        # Check 2
        # Missing Values
        # ==========================================================

        missing_values = df.isnull().sum()

        for column in df.columns:

            missing_count = missing_values[column]

            if missing_count > 0:

                quality_issues.append(
                    f"{column} contains {missing_count} missing value(s)."
                )

        # ==========================================================
        # Check 3
        # Duplicate Rows
        # ==========================================================

        duplicate_rows = df.duplicated().sum()

        if duplicate_rows > 0:

            quality_issues.append(
                f"{duplicate_rows} duplicate row(s) still exist."
            )

        # ==========================================================
        # Check 4
        # Customer ID
        # ==========================================================

        customer_id_is_numeric = pd.api.types.is_numeric_dtype(
            df["Customer_ID"]
        )

        if customer_id_is_numeric is False:

            quality_issues.append(
                "Customer ID is not numeric."
            )

        else:

            invalid_customer_ids = (
                df["Customer_ID"] <= 0
            ).sum()

            if invalid_customer_ids > 0:

                quality_issues.append(
                    f"{invalid_customer_ids} invalid Customer ID(s) detected."
                )

        # ==========================================================
        # Check 5
        # Purchase Amount
        # ==========================================================

        purchase_amount_is_numeric = pd.api.types.is_numeric_dtype(
            df["Purchase_Amount"]
        )

        if purchase_amount_is_numeric is False:

            quality_issues.append(
                "Purchase Amount is not numeric."
            )

        else:

            invalid_purchase_amount = (
                df["Purchase_Amount"] <= 0
            ).sum()

            if invalid_purchase_amount > 0:

                quality_issues.append(
                    f"{invalid_purchase_amount} invalid Purchase Amount(s) detected."
                )

        # ==========================================================
        # Check 6
        # Transaction Date
        # ==========================================================

        datetime_check = pd.api.types.is_datetime64_any_dtype(
            df["Transaction_Date"]
        )

        if datetime_check is False:

            quality_issues.append(
                "Transaction Date is not datetime datatype."
            )

        # ==========================================================
        # Check 7
        # Product Category
        # ==========================================================

        blank_categories = (
            df["Product_Category"]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if blank_categories > 0:

            quality_issues.append(
                f"{blank_categories} blank Product Category value(s) found."
            )

        # ==========================================================
        # Check 8
        # Dataset Size
        # ==========================================================

        if len(df) < 100:

            quality_issues.append(
                "Dataset contains very few records for training."
            )

        # ==========================================================
        # Final Report
        # ==========================================================

        if len(quality_issues) == 0:

            logger.info("\nAll quality checks passed.\n")

            return True

        else:

            for issue in quality_issues:

                logger.error(issue)

            raise ValueError(
                "Dataset failed Data Quality Checks."
            )

    except Exception as e:

        logger.exception("Data Quality Check Failed.")

        raise CustomException(
            e,
            sys
        )