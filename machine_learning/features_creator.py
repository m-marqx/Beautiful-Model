from itertools import combinations
from typing import Literal
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import catboost
import tradingview_indicators as ta
from utils.exceptions import InvalidArgumentError

from machine_learning.ml_utils import DataHandler
from machine_learning.model_handler import ModelHandler
from machine_learning.feature_params import FeaturesParamsComplete

from utils.math_features import MathFeature


class FeaturesCreator:
    """
    A class for creating features and evaluating machine learning
    models.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        The input DataFrame containing financial data.
    target_series : pd.Series
        Series containing return values.
    source : pd.Series
        Series containing source data.
    validation_index : str | int, optional
        Index or column to split the data for training and development.
        (default: None)

    Attributes:
    -----------
    data_frame : pd.DataFrame
        Processed DataFrame using DataHandler.
    target_series : pd.Series
        Series containing return values.
    validation_index : int
        Index to split the data for training and development.
    train_development : pd.DataFrame
        Subset of the data for training and development.
    train_development_index : int
        Index used for splitting training and development data.
    source : pd.Series
        Series containing source data.
    split_params : dict
        Parameters for splitting the data.
    split_paramsH : dict
        Parameters for splitting the data with a higher threshold.
    split_paramsL : dict
        Parameters for splitting the data with a lower threshold.

    Methods:
    --------
    calculate_results(features_columns=None, model_params=None, fee=0.1) \
    -> pd.DataFrame:
        Calculate the results of the model pipeline.
    temp_indicator(value: int | list, \
    indicator: Literal['RSI', 'rolling_ratio'] = 'RSI') \
    -> pd.Series:
        Calculate a temporary indicator series.
    results_model_pipeline(value, indicator, model_params=None, \
    fee=0.1, train_end_index=None, results_column=None) -> dict:
        Calculate drawdown results for different variable combinations.

    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_series: pd.Series,
        source: pd.Series,
        feature_params: FeaturesParamsComplete,
        validation_index: str | int = None,
    ):
        """
        Initialize the FeaturesCreator instance.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input DataFrame containing financial data.
        target_series : pd.Series
            Series containing return values.
        source : pd.Series
            Series containing source data.
        validation_index : str | int, optional
            Index or column to split the data for training and
            development.
            (default: None)

        """
        self.data_frame = DataHandler(dataframe).calculate_targets()
        self.target_series = target_series
        self.validation_index = (
            validation_index
            or int(self.data_frame.shape[0] * 0.7)
        )
        self.temp_indicator_series = None

        self.train_development = (
            self.data_frame.loc[: self.validation_index]
            if isinstance(self.validation_index, str)
            else self.data_frame.iloc[: self.validation_index]
        )

        self.train_development_index = int(
            self.train_development.shape[0]
            * 0.5
        )

        self.source = source

        self.split_params = feature_params.split_features.dict()
        self.split_paramsH = feature_params.high_features.dict()
        self.split_paramsL = feature_params.low_features.dict()

    def calculate_results(
        self,
        features_columns: list | None=None,
        model_params =None,
        fee=0.13,
        test_size=0.5,
        model_algorithm: Literal["xgboost", "catboost"] = "xgboost",
        save_model: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculate the results of the model pipeline.

        Parameters:
        -----------
        features_columns : list | None, optional
            List of feature columns to use in the model.
            (default: None)
        model_params : dict | None, optional
            Parameters for the XGBoost classifier.
            (default: None)
        fee : float, optional
            Transaction fee percentage.
            (default: 0.1)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing model results.

        """
        development = (
            self.data_frame.iloc[: self.validation_index].copy()
            if isinstance(self.validation_index, int)
            else self.data_frame.loc[: self.validation_index].copy()
        )

        validation = (
            self.data_frame.iloc[self.validation_index :].copy()
            if isinstance(self.validation_index, int)
            else self.data_frame.loc[self.validation_index :].copy()
        )

        features = development[features_columns]
        target = development["Target_bin"]

        if not model_params:
            model_params = {
                "objective": "binary:logistic",
                "random_state": 42,
                "eval_metric": "auc",
            }

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=model_params["random_state"],
            shuffle=False,
        )

        if model_algorithm == "xgboost":
            model = xgb.XGBClassifier(**model_params)
        elif model_algorithm == "catboost":
            model = catboost.CatBoostClassifier(**model_params)

        model.fit(X_train, y_train)

        if save_model:
            with open("xgboost_model.pkl", "wb") as file:
                pickle.dump(model, file)

        validacao_X_test = validation[features_columns].iloc[:-1]
        validacao_y_test = validation["Target_bin"].iloc[:-1]

        x_series = pd.concat([X_test, validacao_X_test], axis=0)
        y_series = pd.concat([y_test, validacao_y_test], axis=0)

        mh2 = ModelHandler(model, x_series, y_series).model_returns(
            self.target_series, fee, **kwargs
        )
        mh2["validation_date"] = str(validation.index[0])
        mh2["Target"] = y_series
        return mh2

    def temp_indicator(
        self,
        value: int | list,
        indicator: Literal["RSI", "rolling_ratio", "wick_proportion"] = "RSI",
        source: None | pd.Series = None,
    ) -> pd.Series:
        """
        Calculate a temporary indicator series.

        Parameters:
        -----------
        value : int | list
            Parameter value for the indicator.
        indicator : Literal['RSI', 'rolling_ratio'], optional
            Type of indicator to calculate.
            (default: 'RSI')

        Returns:
        --------
        pd.Series
            Series containing the temporary indicator values.

        Raises:
        -------
        InvalidArgumentError
            If the specified indicator is not found.

        """
        if source is None:
            source = self.source

        match indicator:
            case "RSI":
                return ta.RSI(source, value)

            case "rolling_ratio":

                source_name = source.name if isinstance(source, pd.Series) else source
                source_name = source_name or self.source.name

                return MathFeature(
                    self.data_frame[source_name].to_frame(), source_name
                ).rolling_ratio(*value)
            case "wick_proportion":
                open_column = (
                    "open" if "open" in self.data_frame.columns else "Open"
                )
                open_price = self.data_frame[open_column].copy()

                close_column = (
                    "close" if "close" in self.data_frame.columns else "Close"
                )
                close_price = self.data_frame[close_column].copy()

                high_column = (
                    "high" if "high" in self.data_frame.columns else "High"
                )
                high_price = self.data_frame[high_column].copy()

                low_column = (
                    "low" if "low" in self.data_frame.columns else "Low"
                )
                low_price = self.data_frame[low_column].copy()

                candle_amplitude = high_price - low_price
                wick_proportion = np.where(
                    close_price > open_price,
                    (high_price - close_price) / candle_amplitude,
                    (close_price - low_price) / candle_amplitude,
                )

                return pd.Series(
                    wick_proportion,
                    index=self.data_frame.index
                ).fillna(0)

            case _:
                raise InvalidArgumentError(f"Indicator {indicator} not found")

    def get_features(
        self,
        based_on: str,
        train_end_index: int | None = None,
    ) -> dict:
        """
        Calculate features for the model pipeline.

        Parameters:
        -----------
        based_on : str
            Column to base the indicator on.
        train_end_index : int | None, optional
            Index for splitting the training and development data.
            (default: None)
        features : list[pd.Index] | None, optional
            List of features to calculate.
            (default: None)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the calculated features.

        """
        train_end_index = train_end_index or self.train_development_index

        train_development = (
            self.data_frame.iloc[:train_end_index]
            if isinstance(train_end_index, int)
            else self.data_frame.loc[:train_end_index]
        )

        self.data_frame["temp_indicator"] = self.data_frame[based_on]

        self.temp_indicator_series = (
            self.data_frame["temp_indicator"]
            .reindex(train_development.index)
        ).dropna()

        intervals = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_params)
        )

        intervals_h = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_paramsH)
        )

        intervals_l = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_paramsL)
        )

        return {"split": intervals, "high": intervals_h, "low": intervals_l}

    def calculate_features(
        self,
        based_on: str,
        train_end_index: int | None = None,
    ) -> dict:
        """
        Calculate features for the model pipeline.

        Parameters:
        -----------
        based_on : str
            Column to base the indicator on.
        train_end_index : int | None, optional
            Index for splitting the training and development data.
            (default: None)
        features : list[pd.Index] | None, optional
            List of features to calculate.
            (default: None)

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the calculated features.

        """
        train_end_index = train_end_index or self.train_development_index

        train_development = (
            self.data_frame.iloc[:train_end_index]
            if isinstance(train_end_index, int)
            else self.data_frame.loc[:train_end_index]
        )

        self.data_frame["temp_indicator"] = self.data_frame[based_on]

        self.temp_indicator_series = (
            self.data_frame["temp_indicator"]
            .reindex(train_development.index)
        ).dropna()

        intervals = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_params)
        )

        intervals_highs = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_paramsH)
        )

        intervals_lows = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_paramsL)
        )

        self.data_frame[f"{based_on}_split"] = (
            DataHandler(self.data_frame)
            .calculate_intervals_variables("temp_indicator", intervals)
        ).astype("int8")

        self.data_frame[f"{based_on}_high"] = (
            DataHandler(self.data_frame).calculate_intervals_variables(
                "temp_indicator", intervals_highs
            )
        ).astype("int8")

        self.data_frame[f"{based_on}_low"] = (
            DataHandler(self.data_frame)
            .calculate_intervals_variables("temp_indicator", intervals_lows)
        ).astype("int8")

        self.data_frame = self.data_frame.drop(columns="temp_indicator")

        return self.data_frame

    def calculate_model_returns(
        self,
        value: int,
        indicator: Literal["RSI", "rolling_ratio"] = "RSI",
        model_params: dict | None = None,
        fee: float = 0.1,
        train_end_index: int | None = 1526,
        results_column: str | list[pd.Index] | None = None,
        features: list[pd.Index] | None = None,
    ) -> dict:
        """
        Run the entire model pipeline and return drawdown results.

        Parameters:
        -----------
        value : int
            Parameter value for the indicator.
        indicator : Literal['RSI', 'rolling_ratio'], optional
            Type of indicator to generate.
            (default: 'RSI')
        model_params : dict | None, optional
            Parameters for the XGBoost model.
            (default: None)
        fee : float, optional
            Transaction fee for calculating returns.
            (default: 0.1)
        train_end_index : int | None, optional
            Index for splitting the training and development data.
            (default: None)
        results_column : str | None, optional
            Column to return in the results.
            (default: None)

        Returns:
        --------
        dict
            Dictionary containing drawdown results.
        """
        features = features or []

        train_end_index = train_end_index or self.train_development_index

        train_development = (
            self.data_frame.iloc[:train_end_index]
            if isinstance(train_end_index, int)
            else self.data_frame.loc[:train_end_index]
        )

        self.data_frame["temp_indicator"] = (
            self.temp_indicator(value, indicator)
        )

        self.temp_indicator_series = (
            self.temp_indicator(value, indicator)
            .reindex(train_development.index)
        ).dropna()

        intervals = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_params)
        )

        intervals_highs = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_paramsH)
        )

        intervals_lows = (
            DataHandler(self.temp_indicator_series)
            .get_split_variable_intervals(**self.split_paramsL)
        )

        self.data_frame["temp_variable"] = (
            DataHandler(self.data_frame).calculate_intervals_variables(
                "temp_indicator", intervals
            )
        ).astype("int8")

        self.data_frame["temp_variableH"] = (
            DataHandler(self.data_frame)
            .calculate_intervals_variables("temp_indicator", intervals_highs)
        ).astype("int8")

        self.data_frame["temp_variableL"] = (
            DataHandler(self.data_frame)
            .calculate_intervals_variables("temp_indicator", intervals_lows)
        ).astype("int8")

        temp_variables = ("temp_variable", "temp_variableH", "temp_variableL")

        all_combinations = []
        for items in range(1, 4):
            combinations_item = combinations(temp_variables, items)
            all_combinations.extend(list(combinations_item))

        params = {"model_params": model_params, "fee": fee}

        if results_column:
            results = {
                f"RSI{value}_{combination}": self.calculate_results(
                    features_columns=list(combination) + features, **params
                )[results_column]
                for combination in all_combinations
            }
            return results

        results = {
            f"RSI{value}_{combination}": self.calculate_results(
                features_columns=list(combination) + features, **params
            )
            for combination in all_combinations
        }
        return results
