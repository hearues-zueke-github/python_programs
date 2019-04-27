import datetime

import numpy as np

all_symbols_16 = np.array(list("0123456789ABCDEF"))
def get_random_string_base_16(n):
    l = np.random.randint(0, 16, (n, ))
    return "".join(all_symbols_16[l])

def get_date_time_str():
    dt = datetime.datetime.now()
    return "Y{:04}_m{:02}_d{:02}_H{:02}_M{:02}_S{:02}_f{:06}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
