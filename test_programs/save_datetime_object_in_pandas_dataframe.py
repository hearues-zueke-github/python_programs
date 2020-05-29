#! /usr/bin/python3

import datetime
import pandas as pd

if __name__=='__main__':
    t = pd.DataFrame({'date': [pd.to_datetime('2012-12-31')]})
    t.dtypes # date    datetime64[ns], as expected
    pure_python_datetime_array = t.date.dt.to_pydatetime() # works fine
    t['date'] = pd.Series(pure_python_datetime_array, dtype=object) # should do what you expect
    t.dtypes # object, but the type of the date column is now correct! datetime
    type(t.values[0, 0]) # datetime, now you can access the datetime object directly
