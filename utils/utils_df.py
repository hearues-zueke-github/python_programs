import numpy as np
import pandas as pd

from typing import Tuple, List, Any

def check_df_index(df: pd.DataFrame) -> None:
    assert df.index.values[0] == 0
    assert np.all(np.diff(df.index.values) == 1)

def get_df_part_loc(
    df: pd.DataFrame,
    l_column_l_values: List[Tuple[str, List[Any]]],
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if len(l_column_l_values) == 0:
        arr_idx_bool = np.ones((df.shape[0], ), dtype=np.bool)
        return np.df

    column_name, l_values = l_column_l_values[0]
    arr_idx_bool = np.isin(df[column_name].values, l_values)
    for column_name, l_values in l_column_l_values[1:]:
        arr_idx_bool &= np.isin(df[column_name].values, l_values)

    return arr_idx_bool, df.loc[arr_idx_bool], df.loc[~arr_idx_bool]
