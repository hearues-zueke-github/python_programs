import datetime
import string

import numpy as np

all_symbols_16 = np.array(list("0123456789ABCDEF"))
def get_random_str_base_16(n):
    l = np.random.randint(0, 16, (n, ))
    return "".join(all_symbols_16[l])

all_symbols_64 = np.array(list(string.ascii_lowercase+string.ascii_uppercase+string.digits+"-_"))
def get_random_str_base_64(n):
    l = np.random.randint(0, 64, (n, ))
    return "".join(all_symbols_64[l])

def get_date_time_str_full():
    dt = datetime.datetime.now()
    dt_params = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
    return "Y{:04}_m{:02}_d{:02}_H{:02}_M{:02}_S{:02}_f{:06}".format(*dt_params)

def get_date_time_str_full_short():
    dt = datetime.datetime.now()
    dt_params = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond)
    return "{:04}_{:02}_{:02}_{:02}_{:02}_{:02}_{:06}".format(*dt_params)
