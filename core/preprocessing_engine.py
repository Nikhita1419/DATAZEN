import pandas as pd
import numpy as np


class PreprocessingEngine:
    """
    This class handles all data cleaning operations.
    Each operation is applied only if the user selects it.
    """

    def __init__(self, dataframe):
        # Create a copy to avoid modifying original data
        self.data = dataframe.copy()
        self.actions_log = []
        self.target_column = None  # ✅ NEW

    # -------------------------------
    # Set Target Column (Optional)
    # -------------------------------
    def set_target_column(self, target):
        self.target_column = target

    # -------------------------------
    # Missing Value Handling (Default)
    # -------------------------------
    def handle_missing_values(self):
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if self.data[column].dtype in ["int64", "float64"]:
                    mean_value = self.data[column].mean()
                    self.data[column].fillna(mean_value, inplace=True)
                else:
                    mode_value = self.data[column].mode()[0]
                    self.data[column].fillna(mode_value, inplace=True)

        self.actions_log.append("Missing values were handled using statistical methods.")

    # -------------------------------
    # Duplicate Removal (Optional)
    # -------------------------------
    def remove_duplicates(self):
        before_count = len(self.data)
        self.data.drop_duplicates(inplace=True)
        after_count = len(self.data)

        removed = before_count - after_count
        self.actions_log.append(f"Duplicate rows removed: {removed}")

    # -------------------------------
    # Outlier Handling (Optional)
    # -------------------------------
    def handle_outliers(self):
        numeric_columns = self.data.select_dtypes(include=np.number).columns
        total_removed = 0

        for column in numeric_columns:
            q1 = self.data[column].quantile(0.25)
            q3 = self.data[column].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            before_rows = len(self.data)
            self.data = self.data[
                (self.data[column] >= lower_bound) &
                (self.data[column] <= upper_bound)
            ]
            total_removed += before_rows - len(self.data)

        self.actions_log.append(
            f"Outliers handled using IQR method. Rows removed: {total_removed}"
        )

    # -------------------------------
    # Correlation Handling (Optional)
    # -------------------------------
    def remove_correlated_features(self, threshold=0.85):
        numeric_data = self.data.select_dtypes(include=np.number)

        if numeric_data.shape[1] < 2:
            self.actions_log.append(
                "Not enough numeric columns for correlation analysis."
            )
            return

        corr_matrix = numeric_data.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # ✅ TARGET-SAFE COLUMN REMOVAL
        columns_to_drop = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > threshold)
            and col != self.target_column
        ]

        if columns_to_drop:
            self.data.drop(columns=columns_to_drop, inplace=True)
            self.actions_log.append(
                f"Highly correlated columns removed: {columns_to_drop}"
            )
        else:
            self.actions_log.append("No highly correlated columns found.")

    # -------------------------------
    # Remove Constant Columns (Optional)
    # -------------------------------
    def remove_constant_columns(self):
        constant_cols = [
            col for col in self.data.columns
            if self.data[col].nunique(dropna=False) <= 1
        ]

        if constant_cols:
            self.data.drop(columns=constant_cols, inplace=True)
            self.actions_log.append(
                f"Removed {len(constant_cols)} constant column(s): {constant_cols}"
            )
        else:
            self.actions_log.append("No constant columns found.")

    # -------------------------------
    # Clean Text Columns (Optional)
    # -------------------------------
    def clean_text_columns(self):
        text_cols = self.data.select_dtypes(include="object").columns

        for col in text_cols:
            self.data[col] = (
                self.data[col]
                .astype(str)
                .str.strip()
                .str.replace(r"[^\w\s]", "", regex=True)
            )

        self.actions_log.append(
            "Cleaned text columns by trimming spaces and removing special characters."
        )

    # -------------------------------
    # Final Output
    # -------------------------------
    def get_clean_data(self):
        return self.data, self.actions_log
