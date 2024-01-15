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

    def calculate_targets(self, length=1) -> pd.DataFrame:
        """
        Calculate target variables for binary classification.

        Adds target variables to the DataFrame based on the 'close'
        column:
        - 'Return': Percentage change in 'close' from the previous day.
        - 'Target_1': Shifted 'Return', representing the future day's
        return.
        - 'Target_1_bin': Binary classification of 'Target_1':
            - 1 if 'Target_1' > 1 (positive return)
            - 0 otherwise.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added target variables.

        """
        if isinstance(self.data_frame, pd.Series):
            self.data_frame = pd.DataFrame(self.data_frame)

        self.data_frame['Return'] = self.data_frame["close"].pct_change(length) + 1
        self.data_frame["Target_1"] = self.data_frame["Return"].shift(-length)
        self.data_frame["Target_1_bin"] = np.where(
            self.data_frame["Target_1"] > 1,
            1, 0)

        self.data_frame["Target_1_bin"] = np.where(
            self.data_frame['Target_1'].isna(),
            np.nan, self.data_frame['Target_1_bin']
        )
        return self.data_frame

    def model_pipeline(
        self,
        features_columns: list,
        target_column: str,
        estimator: object,
        return_series: pd.Series,
        split_location: float | int | str = 0.3,
    ) -> pd.DataFrame:
        """
        Execute a machine learning pipeline for model evaluation.

        This method performs a machine learning pipeline, including
        data splitting, training, validation, and evaluation.

        Parameters:
        -----------
        features_columns : list
            List of column names representing features used for training
            the model.
        target_column : str
            Name of the target variable column.
        estimator : object
            Machine learning model (estimator) to be trained and
            evaluated.
        return_series : pd.Series
            Series containing the target variable for the model.
        split_location : float, int, or str, optional
            Determines the location to split the dataset into training
            and validation sets.
            - Float: it represents the proportion of
            the dataset to include in the validation split.
            - Integer: it specifies the index to split the
            dataset.
            - String: it represents the label/index to split
            the dataset.

            (default: 0.3)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing model returns and validation date.

        Raises:
        -------
        ValueError
            If validation_size is outside the valid range (0.0 to 1.0).
        """
        if not isinstance(split_location, (str, float, int)):
            raise ValueError(
                "Wrong split_location type: "
                f"{split_location.__class__.__name__}"
            )
        is_percentage_location = 0 < split_location < 1

        if not (isinstance(split_location, float) and is_percentage_location):
            raise ValueError(
                "When split_location is a float, "
                "it should be between 0.0 and 1.0"
            )

        if is_percentage_location:
            split_factor = 1 - split_location
            split_index = int(self.data_frame.shape[0] * split_factor)
        else:
            split_index = split_location

        if isinstance(split_index, int):
            development = self.data_frame.iloc[:split_index].copy()
            validation = self.data_frame.iloc[split_index:].copy()
        elif isinstance(split_index, str):
            development = self.data_frame.loc[:split_index].copy()
            validation = self.data_frame.loc[split_index:].copy()

        features = development[features_columns]
        target = development[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=0.5,
            shuffle=False
        )

        estimator.fit(X_train, y_train)

        validation_x_test = validation[features_columns]
        validation_y_test = validation[target_column]

        x_series = pd.concat([X_test, validation_x_test], axis=0)
        y_series = pd.concat([y_test, validation_y_test], axis=0)

        model_returns = (
            ModelHandler(estimator, x_series, y_series)
            .model_returns(return_series)
        )
        model_returns['validation_date'] = str(validation.index[0])
        return model_returns

    def drop_zero_predictions(
        self,
        column: str,
    ) -> pd.Series:
        """
        Drop rows where the specified column has all zero values.

        Parameters:
        -----------
        column : str
            The column name in the DataFrame to check for zero values.

        Returns:
        --------
        pd.Series
            The Series with rows dropped where the specified column
            has all zero values.
        """

        def _is_all_zero(list_values: list) -> bool:
            return all(value == 0 for value in list_values)

        if column not in self.data_frame.columns:
            raise ValueError(
                f"Column '{column}' does not exist in the DataFrame."
            )

        mask = self.data_frame[column].apply(_is_all_zero)

        return self.data_frame[~mask]

    def get_splits(
        self,
        target: list | str,
        features: str | list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into training and testing sets.

        Parameters:
        -----------
        target : list or str
            The target column name(s) to use for generating the training
            and testing sets.
        features : str or list of str, optional
            The list of feature column names to use in the DataFrame.

        Returns
        -------
        tuple of pd.DataFrame
            The tuple containing training data, training target,
            testing data, and testing target.
        """
        end_train_index = int(self.data_frame.shape[0] / 2)

        x_train = self.data_frame.iloc[:end_train_index]
        y_train = pd.DataFrame()
        x_test = self.data_frame.iloc[end_train_index:]
        y_test = pd.DataFrame()

        df_train = x_train.loc[:, features]
        df_test = x_test.loc[:, features]

        for value in enumerate(target):
            y_train[f"target_{value[0]}"] = x_train[value[1]]

        for value in enumerate(target):
            y_test[f"target_{value[0]}"] = x_test[value[1]]

        return df_train, y_train, df_test, y_test

    def get_best_results(
        self,
        target_column: str,
    ) -> pd.DataFrame:
        """
        Get the rows in the DataFrame with the best accuracy for each
        unique value in the target_column.

        Parameters:
        -----------
        target_column : str
            The column name in the DataFrame containing target values.

        Returns:
        --------
        pd.DataFrame
            The rows with the best accuracy for each unique value in the
            target_column.
        """
        max_acc_targets = [
            (
                self.data_frame
                .query(f"{target_column} == @target")["acc_test"]
                .astype("float64")
                .idxmax(axis=0)
            )
            for target in self.data_frame[target_column].unique()
        ]

        return self.data_frame.loc[max_acc_targets]

    def result_metrics(
        self,
        result_column: str = None,
        is_percentage_data: bool = False,
        output_format: Literal["dict", "Series", "DataFrame"] = "DataFrame",
    ) -> dict[float, float, float, float] | pd.Series | pd.DataFrame:
        """
        Calculate various statistics related to results, including
        expected return, win rate, positive and negative means, and
        payoff ratio.

        Parameters:
        -----------
        result_column : str, optional
            The name of the column containing the results (returns) for
            analysis.
            If None, the instance's data_frame will be used as the
            result column.
            (default: None).
        is_percentage_data : bool, optional
            Indicates whether the data represents percentages.
            (default: False).
        output_format : Literal["dict", "Series", "DataFrame"],
        optional
            The format of the output. Choose from 'dict', 'Series', or
            'DataFrame'
            (default: 'DataFrame').

        Returns:
        --------
        dict or pd.Series or pd.DataFrame
            Returns the calculated statistics in the specified format:
            - If output_format is `'dict'`, a dictionary with keys:
                - 'Expected_Return': float
                    The expected return based on the provided result
                    column.
                - 'Win_Rate': float
                    The win rate (percentage of positive outcomes) of
                    the model.
                - 'Positive_Mean': float
                    The mean return of positive outcomes from the
                    model.
                - 'Negative_Mean': float
                    The mean return of negative outcomes from the
                    model.
                - 'Payoff': float
                    The payoff ratio, calculated as the positive mean
                    divided by the absolute value of the negative mean.
                - 'Observations': int
                    The total number of observations considered.
            - If output_format is `'Series'`, a pandas Series with
            appropriate index labels.
            - If output_format is `'DataFrame'`, a pandas DataFrame
            with statistics as rows and a 'Stats' column as the index.

        Raises:
        -------
        ValueError
            If output_format is not one of `'dict'`, `'Series'`, or
            `'DataFrame'`.
        ValueError
            If result_column is `None` and the input data_frame is not
            a Series.
        """
        data_frame = self.data_frame.copy()

        if is_percentage_data:
            data_frame = (data_frame - 1) * 100

        if output_format not in ["dict", "Series", "DataFrame"]:
            raise ValueError(
                "output_format must be one of 'dict', 'Series', or "
                "'DataFrame'."
            )

        if result_column is None:
            if isinstance(data_frame, pd.Series):
                positive = data_frame[data_frame > 0]
                negative = data_frame[data_frame < 0]
                positive_mean = positive.mean()
                negative_mean = negative.mean()
            else:
                raise ValueError(
                    "result_column must be provided for DataFrame input."
                )

        else:
            positive = data_frame.query(f"{result_column} > 0")
            negative = data_frame.query(f"{result_column} < 0")
            positive_mean = positive[result_column].mean()
            negative_mean = negative[result_column].mean()

        win_rate = (
            positive.shape[0]
            / (positive.shape[0] + negative.shape[0])
        )

        expected_return = (
            positive_mean
            * win_rate
            - negative_mean
            * (win_rate - 1)
        )

        payoff = positive_mean / abs(negative_mean)

        results = {
            "Expected_Return": expected_return,
            "Win_Rate": win_rate,
            "Positive_Mean": positive_mean,
            "Negative_Mean": negative_mean,
            "Payoff" : payoff,
            "Observations" : positive.shape[0] + negative.shape[0],
        }

        stats_str = "Stats %" if is_percentage_data else "Stats"
        if output_format == "Series":
            return pd.Series(results).rename(stats_str)
        if output_format == "DataFrame":
            return pd.DataFrame(
                results,
                index=["Value"]
            ).T.rename_axis(stats_str)

        return results

