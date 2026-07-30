import sys
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


def validate_dataset(df):

    """
    Validate the dataset before preprocessing.

    This function only validates the dataset.
    It does NOT modify or clean the data.

    Returns:
        True

    Raises:
        ValueError
    """

    try:

        # ==========================================================
        # Store all validation errors
        # ==========================================================

        validation_errors = []

        logger.info("\nValidating Dataset...\n")

        # ==========================================================
        # 1. Check whether dataset is empty
        # ==========================================================

        if df.empty:

            validation_errors.append(
                "Dataset is empty."
            )

        # ==========================================================
        # 2. Check Required Columns
        # ==========================================================

        required_columns = [

            "Customer ID",

            "Transaction Date",

            "Product Category",

            "Purchase Amount"

        ]

        missing_columns = []

        for column in required_columns:

            if column not in df.columns:

                missing_columns.append(column)

        if len(missing_columns) > 0:

            validation_errors.append(

                f"Missing Required Columns : {missing_columns}"

            )

        # ----------------------------------------------------------
        # If required columns are missing,
        # further validation is not possible.
        # ----------------------------------------------------------

        if len(missing_columns) > 0:

            error_message = "\n"
            error_message += "=" * 60
            error_message += "\nDATASET VALIDATION FAILED\n"
            error_message += "=" * 60
            error_message += "\n\n"

            for index, error in enumerate(validation_errors):

                error_message += f"{index + 1}. {error}\n"

            logger.error(error_message)

            raise ValueError(error_message)

        # ==========================================================
        # 3. Check Missing Values
        # ==========================================================

        missing_value_summary = df.isnull().sum()

        for column in required_columns:

            number_of_missing_values = missing_value_summary[column]

            if number_of_missing_values > 0:

                validation_errors.append(

                    f"{column} contains {number_of_missing_values} missing value(s)."

                )

        # ==========================================================
        # 4. Check Duplicate Rows
        # ==========================================================

        duplicate_row_count = df.duplicated().sum()

        if duplicate_row_count > 0:

            validation_errors.append(

                f"{duplicate_row_count} duplicate row(s) found."

            )

        # ==========================================================
        # 5. Customer ID Validation
        # ==========================================================

        customer_id_is_numeric = pd.api.types.is_numeric_dtype(

            df["Customer ID"]

        )

        if not customer_id_is_numeric:

            validation_errors.append(

                "Customer ID must be numeric."

            )

        else:

            invalid_customer_ids = df["Customer ID"] <= 0

            invalid_customer_id_count = invalid_customer_ids.sum()

            if invalid_customer_id_count > 0:

                validation_errors.append(

                    f"{invalid_customer_id_count} Customer ID(s) are less than or equal to zero."

                )

        # ==========================================================
        # 6. Purchase Amount Validation
        # ==========================================================

        purchase_amount_is_numeric = pd.api.types.is_numeric_dtype(

            df["Purchase Amount"]

        )

        if not purchase_amount_is_numeric:

            validation_errors.append(

                "Purchase Amount must be numeric."

            )

        else:

            invalid_purchase_amounts = (

                df["Purchase Amount"] <= 0

            )

            invalid_purchase_amount_count = (

                invalid_purchase_amounts.sum()

            )

            if invalid_purchase_amount_count > 0:

                validation_errors.append(

                    f"{invalid_purchase_amount_count} Purchase Amount(s) are less than or equal to zero."

                )

        # ==========================================================
        # 7. Transaction Date Validation
        # ==========================================================

        converted_dates = pd.to_datetime(

            df["Transaction Date"],

            errors="coerce"

        )

        invalid_date_count = converted_dates.isnull().sum()

        if invalid_date_count > 0:

            validation_errors.append(

                f"{invalid_date_count} invalid Transaction Date value(s) found."

            )

        # ==========================================================
        # 8. Product Category Validation
        # ==========================================================

        product_category_is_text = pd.api.types.is_string_dtype(

            df["Product Category"]

        )

        if not product_category_is_text:

            validation_errors.append(

                "Product Category must be a text column."

            )

        else:

            blank_category_values = (

                df["Product Category"]

                .astype(str)

                .str.strip()

                .eq("")

                .sum()

            )

            if blank_category_values > 0:

                validation_errors.append(

                    f"{blank_category_values} blank Product Category value(s) found."

                )

        # ==========================================================
        # Final Validation Result
        # ==========================================================

        if len(validation_errors) > 0:

            error_message = "\n"

            error_message += "=" * 60
            error_message += "\nDATASET VALIDATION FAILED\n"
            error_message += "=" * 60
            error_message += "\n\n"

            for index, error in enumerate(validation_errors):

                error_message += f"{index + 1}. {error}\n"

            logger.error(error_message)

            raise ValueError(error_message)

        logger.info("\nDataset Validation Passed.\n")

        return True

    except Exception as e:

        logger.exception("Dataset validation failed.")

        raise CustomException(
            e,
            sys
        )