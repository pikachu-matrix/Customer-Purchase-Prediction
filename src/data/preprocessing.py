import sys
import pandas as pd

from src.common.logger import logger
from src.common.exceptions import CustomException


# ==========================================
# Standardize Column Names
# ==========================================

def standardize_column_names(df):

    # print("Standardizing Column Names...")

    df = df.rename(
        columns={
            "Customer ID": "Customer_ID",
            "Transaction Date": "Transaction_Date",
            "Product Category": "Product_Category",
            "Purchase Amount": "Purchase_Amount"
        }
    )

    # print("Completed.")
    logger.info("\nStarted Pre-Processing...\n")

    return df


def preprocess_data(df):

    """
    -------------------------------------------------------------
    Function Name : preprocess_data()

    Purpose:
        Clean and standardize the dataset before feature engineering.

    Input:
        Pandas DataFrame

    Returns:
        Cleaned Pandas DataFrame
    -------------------------------------------------------------
    """

    try:

        # print("\n")
        # print("=" * 60)
        # print("PREPROCESSING STARTED")
        # print("=" * 60)

        df = standardize_column_names(df)

        # print("\nColumns after Standardization:")
        # print(df.columns)

        # ==========================================================
        # Store Original Dataset Information
        # ==========================================================

        original_row_count = len(df)

        # ==========================================================
        # Report Variables
        # ==========================================================

        duplicate_rows_removed = 0
        missing_rows_removed = 0
        invalid_customer_ids_removed = 0
        invalid_purchase_amount_removed = 0
        invalid_dates_removed = 0
        blank_categories_fixed = 0

        # ==========================================================
        # STEP 1
        # Remove Duplicate Rows
        # ==========================================================

        # print("\nRemoving Duplicate Rows...")

        duplicate_rows_removed = df.duplicated().sum()

        df = df.drop_duplicates()

        # ==========================================================
        # STEP 2
        # Remove Missing Values
        # ==========================================================

        # print("\nRemoving Rows Containing Missing Values...")

        row_count_before = len(df)

        df = df.dropna()

        row_count_after = len(df)

        missing_rows_removed = row_count_before - row_count_after

        # ==========================================================
        # STEP 3
        # Remove Invalid Customer IDs
        # ==========================================================

        # print("\nChecking Customer IDs...")

        valid_customer_rows = df["Customer_ID"] > 0

        invalid_customer_ids_removed = len(df) - valid_customer_rows.sum()

        df = df[valid_customer_rows]

        # ==========================================================
        # STEP 4
        # Remove Invalid Purchase Amounts
        # ==========================================================

        # print("\nChecking Purchase Amounts...")

        valid_purchase_rows = df["Purchase_Amount"] > 0

        invalid_purchase_amount_removed = len(df) - valid_purchase_rows.sum()

        df = df[valid_purchase_rows]

        # ==========================================================
        # STEP 5
        # Convert Transaction Date
        # ==========================================================

        # print("\nConverting Transaction Date...")

        df["Transaction_Date"] = pd.to_datetime(
            df["Transaction_Date"],
            errors="coerce"
        )

        invalid_dates_removed = df["Transaction_Date"].isnull().sum()

        df = df.dropna(subset=["Transaction_Date"])
                # ==========================================================
        # STEP 6
        # Clean Product Category
        # ==========================================================

        # print("\nCleaning Product Category...")

        original_category = df["Product_Category"].copy()

        df["Product_Category"] = df["Product_Category"].astype(str)

        df["Product_Category"] = df["Product_Category"].str.strip()

        df["Product_Category"] = df["Product_Category"].str.title()

        category_changed = original_category != df["Product_Category"]

        blank_categories_fixed = category_changed.sum()

        # ==========================================================
        # STEP 7
        # Sort Dataset
        # ==========================================================

        # print("\nSorting Dataset...")

        df = df.sort_values(
            by=[
                "Customer_ID",
                "Transaction_Date"
            ]
        )

        # ==========================================================
        # STEP 8
        # Reset Index
        # ==========================================================

        # print("\nResetting Index...")

        df = df.reset_index(
            drop=True
        )

        # ==========================================================
        # Final Dataset Information
        # ==========================================================

        final_row_count = len(df)

        rows_removed = original_row_count - final_row_count

        # ==========================================================
        # Print Report
        # ==========================================================

        """
        print(f"Original Rows                  : {original_row_count}")

        print(f"Final Rows                     : {final_row_count}")

        print(f"Total Rows Removed             : {rows_removed}")

        print()

        print(f"Duplicate Rows Removed         : {duplicate_rows_removed}")

        print(f"Rows Removed (Missing Values)  : {missing_rows_removed}")

        print(f"Invalid Customer IDs Removed   : {invalid_customer_ids_removed}")

        print(f"Invalid Purchase Amount Removed: {invalid_purchase_amount_removed}")

        print(f"Invalid Dates Removed          : {invalid_dates_removed}")

        print(f"Categories Standardized        : {blank_categories_fixed}")

        print()

        print("Final Dataset Shape            :", df.shape)

        print("=" * 60)
        """

        logger.info(
            f"Pre-Processing completed. Final Shape : {df.shape}"
        )

        logger.info(
            f"""
            Preprocessing Summary
            Original Rows : {original_row_count}
            Final Rows    : {final_row_count}
            Rows Removed  : {rows_removed}
            Duplicate Rows Removed         : {duplicate_rows_removed}
            Rows Removed (Missing Values)  : {missing_rows_removed}
            Invalid Customer IDs Removed   : {invalid_customer_ids_removed}
            Invalid Purchase Amount Removed: {invalid_purchase_amount_removed}
            Invalid Dates Removed          : {invalid_dates_removed}
            Categories Standardized        : {blank_categories_fixed}
                        """
                    )

        return df

    except Exception as e:

        logger.exception("Pre-Processing failed.")

        raise CustomException(
            e,
            sys
        )