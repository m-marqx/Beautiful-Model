from typing import Literal
import warnings
import logging
import time

import numpy as np
import pandas as pd
import tradingview_indicators as ta
from utils import DynamicTimeWarping

def feature_binning(
    feature: pd.Series,
    test_index: str | int,
    bins: int = 10,
) -> pd.Series:
    """
    Perform feature binning using quantiles.

    Parameters:
    -----------
    feature : pd.Series
        The input feature series to be binned.
    test_index : str or int
        The index or label up to which the training data is considered.
    bins : int, optional
        The number of bins to use for binning the feature.
        (default: 10)

    Returns:
    --------
    pd.Series
        The binned feature series.

    Raises:
    -------
    ValueError
        If the feature contains NaN or infinite values.
    """
    has_inf = np.sum(np.isinf(feature.dropna().to_numpy())) >= 1
    has_na = np.sum(np.isnan(feature.dropna().to_numpy())) >= 1

    if has_inf or has_na:
        raise ValueError(
            "Feature contains NaN or infinite values. "
            "Please clean the data before binning."
        )

    train_series = (
        feature.iloc[:test_index].copy() if isinstance(test_index, int)
        else feature.loc[:test_index].copy()
    )

    intervals = (
        pd.qcut(train_series, bins, duplicates='drop')
        .value_counts()
        .index
        .to_list()
    )

    lows = pd.Series([interval.left for interval in intervals])
    highs = pd.Series([interval.right for interval in intervals])
    lows.iloc[0] = -np.inf
    highs.iloc[-1] = np.inf

    intervals_range = (
        pd.concat([lows.rename("lowest"), highs.rename("highest")], axis=1)
        .sort_values("highest")
        .reset_index(drop=True)
    )

    return feature.dropna().apply(
        lambda x: intervals_range[
            (x >= intervals_range["lowest"])
            & (x <= intervals_range["highest"])
        ].index[0]
    )


class ModelFeatures:
    """
    Class for creating and manipulating features for a machine learning
    model.

    Parameters:
    -----------
    dataset : pd.DataFrame
        The dataset containing the features.
    test_index : int
        The index or label up to which the training data is considered.
    bins : int, optional
        The number of bins to use for binning the features.
        (default: 10)

    Attributes:
    -----------
    dataset : pd.DataFrame
        The dataset containing the features.
    test_index : int
        The index or label up to which the training data is considered.
    bins : int
        The number of bins to use for binning the features.
    logger : logging.Logger
        The logger for the ModelFeatures class.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        test_index: int,
        bins: int = 10,
        verbose: bool = True,
        normalize: bool = False,
    ):
        self.dataset = dataset.copy()
        self.test_index = test_index
        self.bins = bins
        self.normalize = normalize

        self.logger = logging.getLogger("Model_Features")
        formatter = logging.Formatter(
            '%(levelname)s %(asctime)s: %(message)s', datefmt='%H:%M:%S'
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(handler)
        self.logger.propagate = False

        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def set_normalize(self, normalize: bool):
        """
        Set value of normalize attribute.

        Parameters
        ----------
        normalize : bool
            The value to set the normalize attribute to.
        """
        self.normalize: bool = normalize
        return self

    def set_bins(self, bins: int):
        """
        Set the number of bins to use for binning the features.

        Parameters
        ----------
        bins : int
            The number of bins to use for binning the features.
        """
        self.bins: int = bins
        return self

    def create_rsi_feature(
        self,
        source: pd.Series,
        length: int,
        ma_method: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'sma'
    ):
        """
        Create the RSI (Relative Strength Index) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating RSI.
        length : int
            The length of the RSI calculation.

        Returns:
        --------
        pd.DataFrame
            The dataset with the RSI feature added.
        """
        self.logger.info("Calculating RSI...")
        start = time.perf_counter()

        self.dataset["RSI"] = ta.RSI(source, length, ma_method)
        if self.normalize:
            self.dataset["RSI"] = self.dataset["RSI"].rolling(2).std().diff()

        self.dataset.loc[:, "RSI_feat"] = feature_binning(
            self.dataset["RSI"],
            self.test_index,
            self.bins,
        )
        self.logger.info(
            "RSI calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_slow_stoch_feature(
        self,
        source_column: str,
        k_length: int = 14,
        k_smoothing: int = 1,
        d_smoothing: int = 3,
    ):
        """
        Create the slow stochastic feature.

        Parameters:
        -----------
        source_column : str
            The column name of the source data.
        k_length : int, optional
            The length of the %K calculation. (default: 14)
        k_smoothing : int, optional
            The smoothing factor for %K. (default: 1)
        d_smoothing : int, optional
            The smoothing factor for %D. (default: 3)

        Returns:
        --------
        pd.DataFrame
            The dataset with the slow stochastic feature added.
        """

        self.logger.info("Calculating slow stochastic...")
        start = time.perf_counter()

        stoch_k, stoch_d = (
            ta.slow_stoch(
                self.dataset[source_column],
                self.dataset['high'],
                self.dataset['low'],
                k_length,
                k_smoothing,
                d_smoothing,
                'sma'
            )
        )

        if self.normalize:
            stoch_k = stoch_k.rolling(2).std().diff()
            stoch_d = stoch_d.rolling(2).std().diff()

        self.dataset["stoch_k"] = stoch_k

        self.dataset.loc[:, "stoch_k_feat"] = feature_binning(
            self.dataset["stoch_k"],
            self.test_index,
            self.bins,
        )

        self.dataset["stoch_d"] = stoch_d
        self.dataset.loc[:, "stoch_d_feat"] = feature_binning(
            self.dataset["stoch_d"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "Slow stochastic calculated in %.2f seconds.",
            time.perf_counter() - start,
        )

        return self.dataset

    def create_dtw_distance_feature(
        self,
        source: pd.Series,
        feats: Literal["sma", "ema", "dema", "tema", "rma", "all"] | list,
        length: int,
    ) -> pd.DataFrame:
        """
        Create the DTW distance feature.

        Parameters
        ----------
        source : pd.Series
            The source series for calculating the DTW distance.
        feats : Literal["sma", "ema", "dema", "tema", "rma", "all"]
        | list
            The list of features to calculate the DTW distance for.
        length : int
            The length of the moving average calculation.

        Returns
        -------
        pd.DataFrame
            The dataset with the DTW distance features added.
        """
        if self.normalize:
            source = source.copy().pct_change().rolling(2).std().iloc[2:]

        if feats == "all":
            feats = ["sma", "ema", "rma", "dema", "tema"]

        self.logger.info("Calculating DTW distance for moving averages...\n")
        start_time_dtw: float = time.perf_counter()

        for ma in feats:
            dtw_distance_params = [source, length]
            MA = ma.upper()

            if ma == "dema":
                dtw_distance_params.append(2)
                ma = "sema"
            elif ma == "tema":
                dtw_distance_params.append(3)
                ma = "sema"

            self.logger.info("Calculating DTW distance for %s...", MA)
            start = time.perf_counter()
            method = getattr(ta, ma)
            moving_average = method(*dtw_distance_params)

            if self.normalize:
                self.dataset[f"{MA}_DTW"] = (
                    DynamicTimeWarping(source.dropna(), moving_average)
                    .calculate_dtw_distance("ratio", True)
                    .rolling(2)
                    .std()
                    .diff()
                )
            else:
                self.dataset[f"{MA}_DTW"] = (
                    DynamicTimeWarping(source, moving_average)
                    .calculate_dtw_distance("absolute", True)
                )

            self.dataset.loc[:, f"{MA}_DTW_feat"] = feature_binning(
                self.dataset[f"{MA}_DTW"],
                self.test_index,
                self.bins,
            )

            self.logger.info(
                "DTW distance for %s calculated in %.2f seconds.",
                MA,
                time.perf_counter() - start,
            )

        self.logger.info(
            "\nDTW distance for moving averages calculated in %.2f seconds.\n",
            time.perf_counter() - start_time_dtw,
        )

        return self.dataset

    def create_cci_feature(
        self,
        source: pd.Series,
        length: int = 20,
        method: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'sma',
    ):
        """
        Create the CCI (Commodity Channel Index) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating CCI.
        length : int, optional
            The length of the CCI calculation.
            (default: 20)
        method : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for CCI calculation.
            (default: 'sma')

        Returns:
        --------
        pd.DataFrame
            The dataset with the CCI feature added.
        """
        self.logger.info("Calculating CCI...")
        start = time.perf_counter()

        self.dataset['CCI'] =  ta.CCI(source, length, method=method)['CCI']

        self.dataset.loc[:, "CCI_feat"] = feature_binning(
            self.dataset["CCI"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "CCI calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_didi_index_feature(
        self,
        source: pd.Series,
        short_length: int = 3,
        medium_length: int = 18,
        long_length: int = 20,
        ma_type: Literal['sma', 'ema','dema','tema', 'rma'] = 'sma',
        method: Literal['absolute', 'ratio', 'dtw'] = 'absolute',
    ):
        """
        Create the Didi Index feature.

        Parameters:
        -----------

        source : pd.Series
            The source series for calculating the DIDI index.
        short_length : int, optional
            The length of the short EMA.
            (default: 3)
        medium_length : int, optional
            The length of the medium EMA.
            (default: 18)
        long_length : int, optional
            The length of the long EMA.
            (default: 20)

        Returns:
        --------
        pd.DataFrame
            The dataset with the DIDI index feature added.
        """
        if method not in ["absolute", "ratio", "dtw"]:
            raise ValueError(
                "Invalid method provided. Use 'absolute', 'ratio', or 'dtw'."
            )

        self.logger.info("Calculating new DIDI index...")
        start = time.perf_counter()

        if self.normalize:
            source = source.copy().pct_change().rolling(2).std().iloc[2:]

            self.dataset["DIDI"] = ta.didi_index(
                source,
                short_length,
                medium_length,
                long_length,
                ma_type,
                'ratio',
                False,
            ).rolling(2).std().diff()
        else:
            is_dtw_distance = False

            if method == "dtw":
                method = "absolute"
                is_dtw_distance = True

            self.dataset["DIDI"] = ta.didi_index(
                source,
                short_length,
                medium_length,
                long_length,
                ma_type,
                method,
                is_dtw_distance,
            )


        self.dataset.loc[:, "DIDI_feat"] = feature_binning(
            self.dataset["DIDI"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "DIDI index calculated in %.2f seconds.",
            time.perf_counter() - start
        )

        return self.dataset

    def create_macd_feature(
        self,
        source: pd.Series,
        fast_length: int = 12,
        slow_length: int = 26,
        signal_length: int = 9,
        diff_method: Literal['absolute', 'ratio', 'dtw'] = 'absolute',
        ma_method: Literal['sma', 'ema','dema','tema', 'rma'] = 'ema',
        signal_method: Literal['sma', 'ema','dema','tema', 'rma'] = 'ema',
        column: Literal['macd', 'signal', 'histogram'] = 'histogram',
    ):
        """
        Create the MACD index feature.

        Parameters
        ----------
        source : pd.Series
            The source series for calculating the MACD index.
        short_length : int, optional
            The length of the short EMA.
            (default: 12)
        long_length : int, optional
            The length of the long EMA.
            (default: 26)
        signal_length : int, optional
            The length of the signal line.
            (default: 9)
        diff_method : Literal['absolute', 'ratio', 'dtw'], optional
            The method to use for calculating the MACD index.
            (default: 'absolute')
        ma_method : Literal['sma', 'ema','dema','tema', 'rma'], optional
            The moving average method to use for MACD calculation.
            (default: 'ema')
        signal_method : Literal['sma', 'ema','dema','tema', 'rma'], optional
            The moving average method to use for signal line calculation.
            (default: 'ema')
        column : Literal['macd', 'signal', 'histogram'], optional
            The column to return from the MACD calculation.
            (default: 'histogram')

        Returns
        -------
        pd.DataFrame
            The dataset with the MACD index feature added.
        """
        self.logger.info("Calculating MACD index...")
        start = time.perf_counter()

        if self.normalize:
            if column != 'histogram':
                warnings.warn(
                    f"{column} isn't compatible with normalization"
                    + " and will be set to 'histogram'."
                )
            source = source.copy().pct_change().rolling(2).std().iloc[2:]

            self.dataset["MACD"] = ta.MACD(
                source=source,
                fast_length=fast_length,
                slow_length=slow_length,
                signal_length=signal_length,
                diff_method='ratio',
                ma_method=ma_method,
                signal_method=signal_method,
            )['histogram'].rolling(2).std().diff()

        else:
            self.dataset["MACD"] = ta.MACD(
                source=source,
                fast_length=fast_length,
                slow_length=slow_length,
                signal_length=signal_length,
                diff_method=diff_method,
                ma_method=ma_method,
                signal_method=signal_method,
            )[column]

        self.dataset.loc[:, "MACD_feat"] = feature_binning(
            self.dataset["MACD"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "MACD index calculated in %.2f seconds.",
            time.perf_counter() - start
        )

        return self.dataset

    def create_trix_feature(
        self,
        source: pd.Series,
        length: int = 15,
        signal_length: int = 1,
        method: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
    ):
        """
        Create the TRIX (Triple Exponential Moving Average) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the TRIX.
        length : int, optional
            The length of the TRIX calculation.
            (default: 15)
        signal_length : int, optional
            The length of the signal line.
            (default: 1)
        method : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for TRIX calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the TRIX feature added.
        """
        self.logger.info("Calculating TRIX...")
        start = time.perf_counter()

        if self.normalize:
            source = source.copy().pct_change().rolling(2).std().iloc[2:]
            self.dataset["TRIX"] = (
                ta.TRIX(source, length, signal_length, method)
                .rolling(2)
                .std()
                .diff()
            )
        else:
            self.dataset["TRIX"] = ta.TRIX(source, length, signal_length, method)

        self.dataset.loc[:, "TRIX_feat"] = feature_binning(
            self.dataset["TRIX"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "TRIX calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_smio_feature(
        self,
        source: pd.Series,
        short_length: int = 20,
        long_length: int = 5,
        signal_length: int = 5,
        ma_type: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
    ):
        """
        Create the SMIO (SMI Ergotic Oscillator) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the SMIO.
        short_length : int, optional
            The length of the faster moving average. (default: 3)
        long_length : int, optional
            The length of the slower moving average. (default: 18)
        ma_type : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for SMIO calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the SMIO feature added.
        """
        self.logger.info("Calculating SMIO...")
        start = time.perf_counter()

        self.dataset["SMIO"] = ta.SMIO(
            source=source,
            long_length=long_length,
            short_length=short_length,
            signal_length=signal_length,
            ma_method=ma_type,
        )

        if self.normalize:
            self.dataset["SMIO"] = self.dataset["SMIO"].rolling(2).std().diff()

        self.dataset.loc[:, "SMIO_feat"] = feature_binning(
            self.dataset["SMIO"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "SMIO calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_tsi_feature(
        self,
        source: pd.Series,
        short_length: int = 13,
        long_length: int = 25,
        ma_type: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
    ):
        """
        Create the TSI (True Strength Index) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the TSI.
        short_length : int, optional
            The length of the faster MA.
            (default: 13)
        long_length : int, optional
            The length of the slower MA.
            (default: 25)
        ma_type : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for TSI calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the TSI feature added.
        """
        self.logger.info("Calculating TSI...")
        start = time.perf_counter()

        self.dataset["TSI"] = ta.tsi(
            source=source,
            short_length=short_length,
            long_length=long_length,
            ma_method=ma_type,
        )

        if self.normalize:
            self.dataset["TSI"] = self.dataset["TSI"].rolling(2).std().diff()

        self.dataset.loc[:, "TSI_feat"] = feature_binning(
            self.dataset["TSI"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "TSI calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_tsi_opt_feature(
        self,
        source: pd.Series,
        short_length: int = 13,
        long_length: int = 25,
        ma_type: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
    ):
        """
        Create the TSI (True Strength Index) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the TSI.
        short_length : int, optional
            The length of the faster MA.
            (default: 13)
        long_length : int, optional
            The length of the slower MA.
            (default: 25)
        ma_type : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for TSI calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the TSI feature added.
        """
        self.logger.info("Calculating TSI...")
        start = time.perf_counter()

        self.dataset["TSI"] = ta.tsi(
            source=source,
            short_length=short_length,
            long_length=long_length,
            ma_method=ma_type,
        ).rolling(2).std().diff()

        self.dataset.loc[:, "TSI_feat"] = feature_binning(
            self.dataset["TSI"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "TSI calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_ichimoku_feature(
        self,
        conversion_periods: int,
        base_periods: int,
        lagging_span_2_periods: int,
        displacement: int,
        based_on: Literal["lead_line", "leading_span"] = "lagging_span",
        method: Literal["absolute", "ratio", "dtw"] = "absolute",
    ):
        """
        Create the Ichimoku Clouds feature.

        Parameters:
        -----------
        conversion_periods : int
            The conversion line period.
        base_periods : int
            The base line period.
        lagging_span_2_periods : int
            The lagging span 2 period.
        displacement : int
            The displacement period.
        based_on : Literal["lead_line", "lagging_span"], optional
            The line to base the distance calculation on.
            (default: "lagging_span")
        method : Literal["absolute", "ratio", "dtw"], optional
            The method to use for calculating the distance.
            (default: "absolute")
        """
        self.logger.info("Calculating Ichimoku Clouds...")
        start = time.perf_counter()

        ichimoku = ta.Ichimoku(
            dataframe=self.dataset,
            conversion_periods=conversion_periods,
            base_periods=base_periods,
            lagging_span_2_periods=lagging_span_2_periods,
            displacement=displacement,
        )[[
            "lead_line1",
            "lead_line2",
            "leading_span_a",
            "leading_span_b",
        ]]

        if method == "absolute":
            lead_line_distance = (
                ichimoku['lead_line1'] - ichimoku['lead_line2']
            )

            leading_span_distance = (
                ichimoku['leading_span_a'] - ichimoku['leading_span_b']
            )
        elif method == "ratio":
            lead_line_distance = (
                ichimoku['lead_line1'] / ichimoku['lead_line2']
            )

            leading_span_distance = (
                ichimoku['leading_span_a'] / ichimoku['leading_span_b']
            )

        elif method == "dtw":
            lead_line_distance = DynamicTimeWarping(
                ichimoku['lead_line1'].dropna(),
                ichimoku['lead_line2'].dropna(),
            ).calculate_dtw_distance(
                method="absolute", align_sequences=True,
            )

            leading_span_distance = DynamicTimeWarping(
                ichimoku['leading_span_a'].dropna(),
                ichimoku['leading_span_b'].dropna(),
            ).calculate_dtw_distance(
                method="absolute", align_sequences=True,
            )
        else:
            raise ValueError(f"method '{method}' not found.")

        if based_on == "lead_line":
            self.dataset['ichimoku_distance'] = lead_line_distance
        elif based_on == "leading_span":
            self.dataset['ichimoku_distance'] = leading_span_distance
        else:
            raise ValueError(f"'{based_on}' is a invalid parameter.")

        self.dataset.loc[:, "ichimoku_feat"] = feature_binning(
            self.dataset["ichimoku_distance"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "Ichimoku calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_ichimoku_price_distance_feature(
        self,
        source: pd.Series,
        conversion_periods: int,
        base_periods: int,
        lagging_span_2_periods: int,
        displacement: int,
        based_on: Literal["lead_line", "leading_span"] = "leading_span",
        method: Literal["absolute", "ratio", "dtw"] = "absolute",
        use_pct_change: bool = True,
    ):
        """
        Create the Ichimoku Clouds feature.

        Parameters:
        -----------
        conversion_periods : int
            The conversion line period.
        base_periods : int
            The base line period.
        lagging_span_2_periods : int
            The lagging span 2 period.
        displacement : int
            The displacement period.
        based_on : Literal["lead_line", "lagging_span"], optional
            The line to base the distance calculation on.
            (default: "lagging_span")
        method : Literal["absolute", "ratio", "dtw"], optional
            The method to use for calculating the distance.
            (default: "absolute")
        """
        self.logger.info("Calculating Ichimoku Clouds...")
        start = time.perf_counter()

        if not isinstance(source, pd.Series):
            raise ValueError("source must be a pandas Series.")

        if use_pct_change:
            dataset = self.dataset.copy().pct_change().iloc[1:]
            source = source.copy().pct_change().iloc[1:]

        ichimoku = ta.Ichimoku(
            dataframe=dataset,
            conversion_periods=conversion_periods,
            base_periods=base_periods,
            lagging_span_2_periods=lagging_span_2_periods,
            displacement=displacement,
        )[[
            "lead_line1",
            "lead_line2",
            "leading_span_a",
            "leading_span_b",
        ]]

        ichimoku['source'] = source

        if based_on == 'leading_span':
            line1 = ichimoku['leading_span_a']
            line2 = ichimoku['leading_span_b']
        elif based_on == 'lead_line':
            line1 = ichimoku['lead_line1']
            line2 = ichimoku['lead_line2']
        else:
            raise ValueError(
                "based_on parameter must be 'lead_line' or 'leading_span'"
            )

        if method == 'absolute':
            line1_distance = abs(ichimoku['source'] - line1)
            line2_distance = abs(ichimoku['source'] - line2)
        elif method == 'ratio':
            line1_distance = abs(ichimoku['source'] / line1)
            line2_distance = abs(ichimoku['source'] / line2)
        elif method == 'dtw':
            line1_distance = abs(
                DynamicTimeWarping(ichimoku['source'], line1.fillna(ichimoku['source']))
                .calculate_dtw_distance(method='absolute', align_sequences=True)
            )
            line2_distance = abs(
                DynamicTimeWarping(ichimoku['source'], line2.fillna(ichimoku['source']))
                .calculate_dtw_distance(method='absolute', align_sequences=True)
            )
        else:
            raise ValueError(
                "method parameter must be 'absolute', 'ratio', or 'dtw'"
            )

        ichimoku_df = pd.concat([
            line1_distance.rename('diff_line1'),
            line2_distance.rename('diff_line2'),
        ], axis=1)

        ichimoku_df['minimum_distance'] = np.where(
            ichimoku_df['diff_line1'] < ichimoku_df['diff_line2'],
            line1, line2
        )

        self.dataset['ichimoku_distance'] = (
            ichimoku['source'] - ichimoku_df['minimum_distance']
        )

        self.dataset.loc[:, "ichimoku_distance_feat"] = feature_binning(
            self.dataset["ichimoku_distance"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "Ichimoku distance calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_bb_trend_feature(
        self,
        source: pd.Series,
        short_length: int = 13,
        long_length: int = 25,
        stdev_multiplier: float = 2.0,
        ma_type: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
        stdev_method: Literal['absolute', 'ratio', 'dtw'] = 'absolute',
        diff_method: Literal['absolute', 'ratio', 'normal'] = 'normal',
        based_on: Literal['short_length', 'long_length'] = 'long_length',
    ):
        """
        Create the BB Trend (Bollinger Bands Trend) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the BB Trend.
        short_length : int, optional
            The length of the faster MA.
            (default: 13)
        long_length : int, optional
            The length of the slower MA.
            (default: 25)
        ma_type : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for BB Trend calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the BB Trend feature added.
        """
        self.logger.info("Calculating BB Trend...")
        start = time.perf_counter()

        if self.normalize:
            source = source.copy().pct_change().rolling(2).std().iloc[2:]

        self.dataset["bb_trend"] = ta.bollinger_trends(
            source=source,
            short_length=short_length,
            long_length=long_length,
            mult=stdev_multiplier,
            ma_method=ma_type,
            stdev_method=stdev_method,
            diff_method=diff_method,
            based_on=based_on,
        )

        if self.normalize:
            self.dataset["bb_trend"] = (
                self.dataset["bb_trend"]
                .rolling(2)
                .std()
                .diff()
            )

        self.dataset.loc[:, "bb_trend_feat"] = feature_binning(
            self.dataset["bb_trend"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "BB Trend calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset

    def create_bb_trend_feature_opt(
        self,
        source: pd.Series,
        short_length: int = 13,
        long_length: int = 25,
        stdev_multiplier: float = 2.0,
        ma_type: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
        stdev_method: Literal['sma', 'ema', 'dema', 'tema', 'rma'] = 'ema',
        diff_method: Literal['absolute', 'ratio', 'dtw'] = 'absolute',
        based_on: Literal['close', 'open', 'high', 'low'] = 'close',
    ):
        """
        Create the BB Trend (Bollinger Bands Trend) feature.

        Parameters:
        -----------
        source : pd.Series
            The source series for calculating the BB Trend.
        short_length : int, optional
            The length of the faster MA.
            (default: 13)
        long_length : int, optional
            The length of the slower MA.
            (default: 25)
        ma_type : Literal['sma', 'ema', 'dema', 'tema', 'rma'], optional
            The moving average method to use for BB Trend calculation.
            (default: 'ema')

        Returns:
        --------
        pd.DataFrame
            The dataset with the BB Trend feature added.
        """
        self.logger.info("Calculating BB Trend...")
        start = time.perf_counter()

        self.dataset["bb_trend"] = ta.bollinger_trends(
            source=source.copy().pct_change().rolling(2).std().iloc[2:],
            short_length=short_length,
            long_length=long_length,
            mult=stdev_multiplier,
            ma_method=ma_type,
            stdev_method=stdev_method,
            diff_method=diff_method,
            based_on=based_on,
        ).rolling(2).std().diff()

        self.dataset.loc[:, "bb_trend_feat"] = feature_binning(
            self.dataset["bb_trend"],
            self.test_index,
            self.bins,
        )

        self.logger.info(
            "BB Trend calculated in %.2f seconds.", time.perf_counter() - start
        )

        return self.dataset
