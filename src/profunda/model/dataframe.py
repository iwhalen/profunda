from typing import Any

import pandas as pd

from profunda.config import Settings
from profunda.model.pandas.dataframe_pandas import pandas_preprocess
from profunda.utils.backend import is_ibis_installed, is_pyspark_installed

if not is_pyspark_installed():

    class _sparkDataFrame:
        """Dummy class that that no object will ever type match."""

        pass

    sparkDataFrame = _sparkDataFrame

else:
    from pyspark.sql import DataFrame as sparkDataFrame  # type: ignore

    from profunda.model.spark.dataframe_spark import spark_preprocess

if not is_ibis_installed():

    class _ibisDataFrame:
        """Dummy class that that no object will ever type match."""

        pass

    ibisDataFrame = _ibisDataFrame
else:
    from ibis import Table as ibisDataFrame  # type: ignore

    from profunda.model.ibis.dataframe_ibis import ibis_preprocess


def preprocess(config: Settings, df: Any) -> Any:
    """
    Search for invalid columns datatypes as well as ensures column names follow the expected rules
    Args:
        config: ydataprofiling Settings class
        df: a pandas or spark dataframe

    Returns: a pandas or spark dataframe
    """
    if isinstance(df, pd.DataFrame):
        df = pandas_preprocess(config=config, df=df)
    elif isinstance(df, sparkDataFrame):  # type: ignore
        df = spark_preprocess(config=config, df=df)
    elif isinstance(df, ibisDataFrame):  # type: ignore
        df = ibis_preprocess(config=config, df=df)
    else:
        return NotImplementedError()
    return df
