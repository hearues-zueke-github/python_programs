import numpy as np
import pandas as pd

from typing import Tuple, List, Any

def get_csv_lines_as_df(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        content = f.read()

    l_line = content.split('\n')
    df_line = pd.DataFrame(
        data=[(i, line) for i, line in enumerate(l_line, 1)],
        columns=['line_nr', 'line'],
        dtype=object,
    )

    return df_line
