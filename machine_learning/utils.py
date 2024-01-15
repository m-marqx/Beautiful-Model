from typing import Literal
import re

import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.model_selection import (
    learning_curve,
    train_test_split,
)

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go


class DataHandler:
    """
    Class for handling data preprocessing tasks.

    Parameters
    ----------
    dataframe : pd.DataFrame or pd.Series
        The input DataFrame or Series to be processed.

    Attributes
    ----------
    data_frame : pd.DataFrame
        The processed DataFrame.

    Methods
    -------
    get_datasets(feature_columns, test_size=0.5, split_size=0.7)
        Splits the data into development and validation datasets.
    drop_zero_predictions(column)
        Drops rows where the specified column has all zero values.
    get_splits(target, features)
        Splits the DataFrame into training and testing sets.
    get_best_results(target_column)
        Gets the rows with the best accuracy for each unique value in
        the target column.
    result_metrics(result_column=None, is_percentage_data=False,
    output_format="DataFrame")
        Calculates result-related statistics like expected return and win rate.
    fill_outlier(column=None, iqr_scale=1.5, upper_quantile=0.75,
    down_quantile=0.25)
        Removes outliers from a specified column using the IQR method.
    quantile_split(target_input, column=None, method="ratio",
    quantiles=None, log_values=False)
        Splits data into quantiles and analyzes the relationship with a target.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | pd.Series | np.ndarray,
    ) -> None:
        """
        Initialize the DataHandler object.

        Parameters:
        -----------
        dataframe : pd.DataFrame, pd.Series, or np.ndarray
            The input data to be processed. It can be a pandas DataFrame,
            Series, or a numpy array.

        """
        self.data_frame = dataframe.copy()

        if isinstance(dataframe, np.ndarray):
            self.data_frame = pd.Series(dataframe)

    def get_datasets(
        self,
        feature_columns: list,
        test_size: float = 0.5,
        split_size: float = 0.7
    ) -> dict[dict[pd.DataFrame, pd.Series]]:
        """
        Splits the data into development and validation datasets.

        Separates the DataFrame into training and testing sets for
        development, and a separate validation set, based on the
        specified split and test sizes.

        Parameters
        ----------
        feature_columns : list
            List of column names to be used as features.
        test_size : float
            Proportion of the dataset to include in the test split.
            (default: 0.5)
        split_size : float
            Proportion of the dataset to include in the development
            split.
            (default: 0.7)

        Returns
        -------
        dict
            A dictionary containing the development and validation
            datasets, each itself a dictionary with DataFrames and
            Series for features and target values respectively.

        Raises
        ------
        ValueError
            If the provided data_frame is not a Pandas DataFrame.
        """
        if not isinstance(self.data_frame, pd.DataFrame):
            raise ValueError("The dataframe must be a Pandas DataFrame")

        split_index = int(self.data_frame.shape[0] * split_size)
        development_df = self.data_frame.iloc[:split_index].copy()
        validation_df = self.data_frame.iloc[split_index:].copy()

        features = development_df[feature_columns]
        target = development_df["Target_1_bin"]

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            shuffle=False
        )

        origin_datasets = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }
        validation_dataset = {
            "X_validation": validation_df[feature_columns],
            "y_validation": validation_df["Target_1_bin"]
        }

        return {
            "development": origin_datasets,
            "validation": validation_dataset
        }

