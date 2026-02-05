import ibis
import pandas as pd
import pytest

from profunda.config import Settings
from profunda.model.ibis.describe_counts_ibis import describe_counts_ibis
from profunda.model.ibis.describe_generic_ibis import describe_generic_ibis
from profunda.model.ibis.describe_supported_ibis import describe_supported_ibis
from profunda.model.pandas.describe_counts_pandas import pandas_describe_counts
from profunda.model.pandas.describe_generic_pandas import pandas_describe_generic
from profunda.model.pandas.describe_supported_pandas import pandas_describe_supported


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"A": [1, 2, 2, 3, 3, 3, 4, 5, None, None]}),
        pd.DataFrame({"A": [1, 2, 3, 4, 5]}),
        pd.DataFrame({"A": [1, 1, 2, 2]}),
        pd.DataFrame({"A": []}),
    ],
)
def test_describe_supported_ibis_matches_pandas(data: pd.DataFrame):
    config = Settings()
    series = data["A"]
    ibis_series = ibis.memtable(data)

    pandas_summary = {}  # type: ignore
    for func in [
        pandas_describe_counts,
        pandas_describe_generic,
        pandas_describe_supported,
    ]:
        _, _, pandas_summary = func(config, series, pandas_summary)

    ibis_summary = {}  # type: ignore
    for func in [
        describe_counts_ibis,
        describe_generic_ibis,
        describe_supported_ibis,
    ]:
        _, _, ibis_summary = func(config, ibis_series, ibis_summary)

    # Assertions
    assert ibis_summary["n_distinct"] == pandas_summary["n_distinct"]
    assert ibis_summary["p_distinct"] == pytest.approx(pandas_summary["p_distinct"])
    assert ibis_summary["is_unique"] == pandas_summary["is_unique"]
    assert ibis_summary["n_unique"] == pandas_summary["n_unique"]
    assert ibis_summary["p_unique"] == pytest.approx(pandas_summary["p_unique"])
