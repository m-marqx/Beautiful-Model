from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
)


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
        target = development_df["Target_bin"]

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
            "y_validation": validation_df["Target_bin"]
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
        - 'Target': Shifted 'Return', representing the future day's
        return.
        - 'Target_bin': Binary classification of 'Target':
            - 1 if 'Target' > 1 (positive return)
            - 0 otherwise.

        Returns:
        --------
        pd.DataFrame
            DataFrame with added target variables.

        """
        if isinstance(self.data_frame, pd.Series):
            self.data_frame = pd.DataFrame(self.data_frame)

        self.data_frame['Return'] = self.data_frame["close"].pct_change(length) + 1
        self.data_frame["Target"] = self.data_frame["Return"].shift(-length)
        self.data_frame["Target_bin"] = np.where(
            self.data_frame["Target"] > 1,
            1, 0)

        self.data_frame["Target_bin"] = np.where(
            self.data_frame['Target'].isna(),
            np.nan, self.data_frame['Target_bin']
        )
        return self.data_frame

    def model_pipeline(
        self,
        features_columns: list,
        target_column: str,
        estimator: object,
        target_series: pd.Series,
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
        target_series : pd.Series
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
            .model_returns(target_series)
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

    def calculate_outlier_values(
        self,
        column: str = None,
        iqr_scale: float = 1.5,
        upper_quantile: float = 0.75,
        down_quantile: float = 0.25,
    ) -> tuple[float, float]:
        """
        Calculate upper and lower bounds for identifying outliers in the
        specified column.

        Parameters:
        -----------
        column : str, optional
            The name of the column in the DataFrame. If None, the
            instance's data_frame will be used as the target.
        iqr_scale : float, optional
            The scale factor for the Interquartile Range (IQR).
            (default: 1.5)
        upper_quantile : float, optional
            The upper quantile used to calculate the IQR.
            (default: 0.75 (75th percentile))
        down_quantile : float, optional
            The lower quantile used to calculate the IQR.
            (default: 0.25 (25th percentile))

        Returns:
        --------
        tuple[float, float]
            A tuple containing the upper and lower bounds for
            identifying outliers.

        Raises:
        -------
        ValueError
            If column is None and the input is a DataFrame.
        """
        if column is None:
            if isinstance(self.data_frame, pd.Series):
                outlier_array = self.data_frame.copy()
            else:
                raise ValueError(
                    "column must be provided for DataFrame input."
                )
        else:
            outlier_array = self.data_frame[column].copy()

        Q1 = outlier_array.quantile(down_quantile)
        Q3 = outlier_array.quantile(upper_quantile)

        IQR = Q3 - Q1
        upper_bound = Q3 + iqr_scale * IQR
        lower_bound = Q1 - iqr_scale * IQR

        return upper_bound, lower_bound

    def fill_outlier(
        self,
        column: str = None,
        iqr_scale: float = 1.5,
        upper_quantile: float = 0.75,
        down_quantile: float = 0.25,
    ) -> pd.Series:
        """
        Remove outliers from a given target column using the IQR
        (Interquartile Range) method.

        Parameters:
        -----------
        column : str, optional
            The name of the target column containing the data to be processed.
            If None, the instance's data_frame will be used as the target.
        iqr_scale : float, optional
            The scaling factor to determine the outlier range based on the IQR.
            (default: 1.5)
        upper_quantile : float, optional
            The upper quantile value for calculating the IQR.
            (default: 0.75 (75th percentile))
        down_quantile : float, optional
            The lower quantile value for calculating the IQR.
            (default: 0.25 (25th percentile))

        Returns:
        --------
        pd.Series
            A Series with outliers removed based on the specified
            criteria.
        """
        if column is None:
            if isinstance(self.data_frame, pd.Series):
                outlier_array = self.data_frame.copy()
            else:
                raise ValueError(
                    "column must be provided for DataFrame input."
                )
        else:
            outlier_array = self.data_frame[column].copy()

        outlier_args = [column, iqr_scale, upper_quantile, down_quantile]
        upper_bound, lower_bound = self.calculate_outlier_values(*outlier_args)

        outlier_array = np.where(
            outlier_array > upper_bound,
            upper_bound,
            outlier_array
        )

        outlier_array = np.where(
            outlier_array < lower_bound,
            lower_bound,
            outlier_array
        )

        return pd.Series(outlier_array, index=self.data_frame.index)

    def quantile_split(
        self,
        target_input: str | pd.Series | np.ndarray,
        column: str = None,
        method: Literal["simple", "ratio", "sum", "prod"] | None = "ratio",
        quantiles: np.ndarray | pd.Series | int = 10,
        log_values: bool = False,
    ) -> pd.DataFrame:
        """
        Split data into quantiles based on a specified column and
        analyze the relationship between these quantiles and a target
        variable.

        Parameters:
        -----------
        target_input : str, pd.Series, or np.ndarray
            The target variable for the analysis. It can be a column
            name, a pandas Series, or a numpy array.
        column : str, optional
            The name of the column used for quantile splitting.
        method : Literal["simple", "ratio", "sum", "prod"], optional
            The method used for calculating class proportions. 'simple'
            returns the raw class counts, 'ratio' returns the
            proportions of the target variable within each quantile.
            (default: "ratio")
        quantiles : np.ndarray or pd.Series or int, optional
            The quantile intervals used for splitting the 'feature' into
            quantiles. If an integer is provided, it represents the
            number of quantiles to create. If an array or series is
            provided, it specifies the quantile boundaries.
            (default: 10).
        log_values : bool, optional
            If True and 'method' is 'prod', the resulting values are
            computed using logarithmic aggregation.
            (default: False)

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the quantile split analysis. Rows
            correspond to quantile intervals based on the specified
            column, columns correspond to unique values of the target
            variable, and the values represent either counts or
            proportions, depending on the chosen method and split type.
        """
        if method in ["prod", "sum"]:
            split_type = 'data'
        elif method in ["simple","ratio"]:
            split_type = 'frequency'
        else:
            raise ValueError(
                "method must be prod, sum,"
                f" simple or ratio instead of {method}"
            )

        if isinstance(self.data_frame, pd.Series):
            feature = self.data_frame
        else:
            feature = self.data_frame[column]

        if feature.hasnans:
            feature = feature.dropna()

        if isinstance(quantiles, int):
            range_step = 1 / quantiles
            quantiles = np.quantile(
                feature,
                np.arange(0, 1.01, range_step)
            )

            quantiles = np.unique(quantiles)

        if isinstance(target_input, str):
            target_name = target_input
            target = self.data_frame[target_input]
        else:
            target_name = "target"
            target = pd.Series(target_input)

        if feature.index.dtype != target.index.dtype:
            feature = feature.reset_index(drop=True)
            target = target.reset_index(drop=True)

        if not target.index.equals(feature.index):
            target = target.reindex(feature.index)

        class_df = pd.cut(
            feature,
            quantiles,
            include_lowest=True,
        )

        feature_name = column if column else "feature"

        quantile_df = pd.DataFrame(
            {
                feature_name: class_df,
                target_name: target
            }
        )
        if split_type == 'data':
            quantile_df = quantile_df.groupby(feature_name)[target_name]

            if method == 'sum':
                quantile_df = quantile_df.sum()
            if method == 'prod':
                if log_values:
                    quantile_df = np.log(quantile_df.prod())
                else:
                    quantile_df = quantile_df.prod() - 1

        else:
            quantile_df = pd.crosstab(
                index=quantile_df[feature_name],
                columns=quantile_df[target_name],
            )

            if method == "ratio":
                quantile_df = (
                    quantile_df
                    .div(quantile_df.sum(axis=1), axis=0)
                )
        return quantile_df

    def get_split_variable(
        self,
        target_input: str,
        column: str,
        quantiles: np.ndarray | pd.Series | int = 10,
        method: Literal["simple", "ratio", "sum", "prod"] = "ratio",
        log_values: bool = False,
        threshold: float = 0.5,
        higher_than_threshold: bool = True,
    ) -> pd.Series:
        """
        Get a binary variable based on quantile analysis.

        This method performs quantile analysis on the specified column
        using the provided target variable and threshold. It creates a
        binary variable indicating whether the values in the column fall
        within specific quantile intervals.

        Parameters:
        -----------
        target_input : str, pd.Series, or np.ndarray
            The target variable for the analysis. It can be a column
            name, a pandas Series, or a numpy array.
        column : str, optional
            The name of the column used for quantile splitting.
        method : Literal["simple", "ratio", "sum", "prod"], optional
            The method used for calculating class proportions. 'simple'
            returns the raw class counts, 'ratio' returns the
            proportions of the target variable within each quantile.
            (default: "ratio")
        quantiles : np.ndarray or pd.Series or int, optional
            The quantile intervals used for splitting the 'feature' into
            quantiles. If an integer is provided, it represents the
            number of quantiles to create. If an array or series is
            provided, it specifies the quantile boundaries.
            (default: 10).
        log_values : bool, optional
            If True and 'method' is 'prod', the resulting values are
            computed using logarithmic aggregation.
            (default: False)
        threshold : float or int, optional
            The threshold value for determining the quantile intervals.
            Values above this threshold will be considered.
            (default: 0.5)

        Returns:
        --------
        pd.Series
            A binary variable indicating whether the values in the
            specified column fall within the determined quantile
            intervals.
        """
        split_data = self.quantile_split(
            target_input,
            column,
            method,
            quantiles,
            log_values,
        )

        split_data = (
            split_data.iloc[:, 1] if split_data.shape[1] == 2 else split_data
        )

        if higher_than_threshold:
            data = split_data[split_data > threshold]
        else:
            data = split_data[split_data < threshold]

        intervals = [[x[0].left, x[0].right] for x in data.items()]
        variable = pd.Series(False, index=self.data_frame.index)

        for x in intervals:
            variable |= self.data_frame[column].between(x[0], x[1])

        lower_bound = data.index[0].right if data.iloc[0] > threshold else None

        upper_bound = (
            data.index[-1].left if data.iloc[-1] > threshold
            else None
        )

        if upper_bound:
            variable |= self.data_frame[column] > upper_bound

        if lower_bound:
            variable |= self.data_frame[column] <= lower_bound
        return variable

    def get_split_variable_intervals(
        self,
        target_input: str,
        column: str,
        quantiles: np.ndarray | pd.Series | int = 10,
        method: Literal["simple", "ratio", "sum", "prod"] = "ratio",
        log_values: bool = False,
        threshold: float = 0.5,
        higher_than_threshold: bool = True,
    ) -> pd.Series:
        """
        Get intervals from quantile-split variable analysis.

        This method performs quantile analysis on the specified column
        using the provided target variable, method, and threshold. It
        returns intervals based on values higher or lower than the
        specified threshold.

        Parameters:
        -----------
        target_input : str, pd.Series, or np.ndarray
            The target variable for the analysis. It can be a column
            name, a pandas Series, or a numpy array.
        column : str, optional
            The name of the column used for quantile splitting.
        method : Literal["simple", "ratio", "sum", "prod"], optional
            The method used for calculating class proportions. 'simple'
            returns the raw class counts, 'ratio' returns the
            proportions of the target variable within each quantile.
            (default: "ratio")
        quantiles : np.ndarray or pd.Series or int, optional
            The quantile intervals used for splitting the 'feature' into
            quantiles. If an integer is provided, it represents the
            number of quantiles to create. If an array or series is
            provided, it specifies the quantile boundaries.
            (default: 10).
        log_values : bool, optional
            If True and 'method' is 'prod', the resulting values are
            computed using logarithmic aggregation.
            (default: False)
        threshold : float or int, optional
            The threshold value for determining the quantile intervals.
            Values above this threshold will be considered.
            (default: 0.5)
        higher_than_threshold : bool, optional
            If True, values higher than the threshold are considered
            for quantile intervals. If False, values lower than the
            threshold are considered.
            (default: True)

        Returns:
        --------
        pd.Series
            A dictionary containing the intervals based on quantile
            analysis.
        """
        split_data = self.quantile_split(
            target_input,
            column,
            method,
            quantiles,
            log_values,
        )

        split_data = (
            split_data.iloc[:, 1] if split_data.shape[1] == 2 else split_data
        )

        if higher_than_threshold:
            data = split_data[split_data > threshold]
        else:
            data = split_data[split_data < threshold]

        intervals = [[x[0].left, x[0].right] for x in data.items()]
        variable_intervals = {}

        for x in enumerate(intervals):
            variable_intervals[f"interval_{x[0]}"] = x[1]

        if data.shape[0] > 0:
            lower_bound = (
                data.index[0].right if data.iloc[0] > threshold else None
            )

            upper_bound = (
                data.index[-1].left if data.iloc[-1] > threshold else None
            )

            variable_intervals["upper_bound"] = upper_bound
            variable_intervals["lower_bound"] = lower_bound

        return variable_intervals

    def calculate_intervals_variables(
        self,
        column: str,
        intervals: dict,
    ) -> pd.Series:
        """
        Get a binary variable based on specified intervals.

        This method creates a binary variable indicating whether the
        values in the specified column fall within the given intervals.

        Parameters:
        -----------
        column : str
            The name of the column to analyze.
        intervals : dict
            A dictionary defining the intervals. The keys represent the
            names of the intervals, and the values can be:
            - A list [start, end] defining a closed interval.
            - A single value for open intervals.

        Returns:
        --------
        pd.Series
            A binary variable indicating whether the values in the
            specified column fall within the specified intervals.
        """
        variable = pd.Series(False, index=self.data_frame.index)
        interval_list = list(intervals.values())

        for x in intervals.values():
            if isinstance(x, list):
                variable |= self.data_frame[column].between(x[0], x[1])

            #This case will be handled by `get_split_variable_intervals` method
            elif x == interval_list[-2] and not isinstance(x, list):
                if x:
                    variable |= self.data_frame[column] > x

            elif x == interval_list[-1] and not isinstance(x, list):
                if x:
                    variable |= self.data_frame[column] <= x
        return variable


def get_recommendation(
    predict_series,
    return_dtype: Literal["string", "normal", "int", "bool"] = "string",
    add_span_tag: bool = False,
):
    """
    Generate trading recommendations based on the prediction series.

    This function takes a series of predictions and generates trading
    recommendations in different formats based on the specified return
    data type.

    Parameters
    ----------
    predict_series : pd.Series
        A pandas Series containing the prediction values.
    return_dtype : {'string', 'normal', 'int', 'bool'}, optional
        The desired return data type for the recommendations.
        - 'string': Returns recommendations as strings
        ('Long', 'Do Nothing', 'Short').
        - 'normal': Returns the original prediction series.
        - 'int': Returns the predictions as integers.
        - 'bool': Returns the predictions as boolean values
        (True for positive, False otherwise).

        (default:'string')
    add_span_tag : bool, optional
        If True, the recommendations are returned as HTML strings with
        span tags for color formatting. If False, the recommendations
        are returned as plain text. (compatible with itables)

        (default: False)

    Returns
    -------
    pd.Series
        A pandas Series containing the trading recommendations in the
        specified format.
    """
    predict = predict_series.copy()
    predict.index = predict.index.date
    predict = predict.rename_axis("date")

    confirmed_signals = pd.Series(predict.iloc[:-1], name=predict.name)

    unconfirmed_signal = pd.Series(
        predict.iloc[-1],
        index=["Unconfirmed"],
    )

    signals = pd.concat([confirmed_signals, unconfirmed_signal])

    if add_span_tag:
        long_color = "<b><span style='color: #00e676'>Open Position</span></b>"
        do_nothing_color = "——————"
        short_color = (
            "<b><span style='color: #ef5350'>Close Position</span></b>"
        )

    else:
        long_color = "Open Position"
        do_nothing_color = "——————"
        short_color = "Close Position"

    match return_dtype:
        case "string":
            recommendation = np.where(
                signals > 0,
                long_color,
                np.where(signals == 0, do_nothing_color, short_color),
            )
            recommendation_series = pd.Series(
                recommendation, index=signals.index
            )

            recommendation_array = np.where(
                (recommendation_series.shift(7) == long_color)
                & (recommendation_series == do_nothing_color),
                short_color,
                recommendation_series,
            )

            return pd.Series(
                recommendation_array,
                index=recommendation_series.index,
                name=predict.name,
            )

        case "normal":
            return signals

        case "int":
            return signals.astype(int)

        case "bool":
            return signals > 0

def get_recommendation_trades(predict_series):
    """
    Generate trading recommendations based on the signal and price series.

    This function generates trading recommendations based on the signal
    series and the price series. It returns a DataFrame containing the
    trading recommendations and the corresponding prices.

    Parameters
    ----------

    predict_series : pd.Series
        A pandas Series containing the prediction values.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the trading recommendations and the
        corresponding prices.
    """
    open_pos_name = "<b><span style='color: #00e676'>Open Position</span></b>"
    close_pos_name = (
        "<b><span style='color: #ef5350'>Close Position</span></b>"
    )

    str_recommendation = get_recommendation(predict_series, "string", True)
    int_recommentation = str_recommendation.to_frame()

    col = int_recommentation.columns[0]

    int_recommentation[col] = np.where(
        int_recommentation[col] == open_pos_name,
        1,
        int_recommentation[col],
    )

    int_recommentation[col] = np.where(
        int_recommentation[col] == close_pos_name,
        -1,
        int_recommentation[col],
    )

    int_recommentation[col] = np.where(
        int_recommentation[col] == "——————",
        0,
        int_recommentation[col],
    )

    int_recommentation.columns = [
        col + "_int" for col in int_recommentation.columns
    ]
    int_recommentation = int_recommentation.iloc[:-1]

    str_recommendation = str_recommendation.iloc[:-1]
    str_recommendation.index = pd.DatetimeIndex(
        str_recommendation.index.tolist()
    )
    int_recommentation.index = pd.DatetimeIndex(
        int_recommentation.index.tolist()
    )

    close_pos = (
        int_recommentation[int_recommentation < 0]
        .loc[datetime.date(2024, 9, 15) :]
        .cumsum()
    )
    open_pos = (
        int_recommentation[int_recommentation > 0]
        .loc[datetime.date(2024, 9, 15) :]
        .cumsum()
    )

    int_recommentation = close_pos.combine_first(open_pos)

    trade_recommendation = pd.concat(
        [str_recommendation, int_recommentation], axis=1
    )
    trade_recommendation = trade_recommendation.fillna("")

    col = trade_recommendation.columns[0]

    trade_recommendation[col] = np.where(
        trade_recommendation[col + "_int"] == "",
        "——————",
        trade_recommendation[col],
    )
    trade_recommendation[col] = np.where(
        trade_recommendation[col] == close_pos_name,
        trade_recommendation[col]
        + " ("
        + trade_recommendation[col + "_int"].astype(str)
        + ")",
        trade_recommendation[col],
    )
    trade_recommendation[col] = np.where(
        trade_recommendation[col] == open_pos_name,
        trade_recommendation[col]
        + " ("
        + trade_recommendation[col + "_int"].astype(str)
        + ")",
        trade_recommendation[col],
    )

    return trade_recommendation.iloc[:, 0]
