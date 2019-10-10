#! /usr/bin/python3

# -*- coding: utf-8 -*-

import datetime
import os
import sys

from copy import deepcopy

sys.path.append("../combinatorics/")
import different_combinations as combinations
sys.path.append("../math_numbers/")
from prime_numbers_fun import get_primes

from time import time
from functools import reduce

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
plt.close("all")

from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

sys.path.append("../")
from utils_serialization import get_pkl_gz_obj, save_pkl_gz_obj

PATH_ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))+"/"

def time_measure(f, args):
    start_time = time()
    ret = f(*args)
    end_time = time()
    diff_time = end_time-start_time
    return ret, diff_time


if __name__ == '__main__':
    print('Hello World!')
    with open('saved_rx_tx_2019.10.10_13:06:17.txt', 'r') as f:
    # with open('saved_rx_tx_2019.10.09_19:36:02.txt', 'r') as f:
        lines = f.readlines()

    lst_rows = list(map(lambda x: (lambda y: (datetime.datetime.strptime(y[0], '%Y.%m.%d-%H:%M:%S.%f'), int(y[1]), int(y[2])))(x.replace('\n', '').split(';')), lines))
    arr = np.array(lst_rows)

    dts, rxs, txs = arr.T

    diff_dts = np.array([(dt2-dt1).total_seconds() for dt1, dt2 in zip(dts[:-1], dts[1:])])
    diff_rxs = rxs[1:]-rxs[:-1]
    diff_txs = txs[1:]-txs[:-1]

    ys_speed_rxs = diff_rxs/diff_dts/2**20
    ys_speed_txs = diff_txs/diff_dts/2**20

    xs_sec = np.array(list(map(lambda x: (lambda y: y.hour*60*60+y.minute*60+y.second+y.microsecond/1000000)(x.time()), dts[1:])))

    # TODO: use this part later on!
    # xs_2 = []
    # ys_2_min = []
    # ys_2_q1 = []
    # ys_2_median = []
    # ys_2_q3 = []
    # ys_2_max = []

    # r = 1200 # seconds, range, previous time
    # # now calculate the mean, median, min and max from a range of values
    # modulo = 60*60*24
    # for i in range(0, 60*60*24):
    #     l = []
    #     for j in range(-r+1, 1):
    #         l.extend(dict_times[(i+j)%modulo])

    #     xs_2.append(i)

    #     if len(l)>0:
    #         ys_2_min.append(np.min(l))
    #         ys_2_median.append(np.median(l))
    #         ys_2_max.append(np.max(l))
    #     else:
    #         ys_2_min.append(0)
    #         ys_2_median.append(0)
    #         ys_2_max.append(0)

    dt_now = datetime.datetime.now()
    dt_str = '{:04}{:02}{:02}{:02}{:02}{:02}'.format(dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute, dt_now.second)


    dpi = 300

    fig = plt.figure(figsize=(13, 8))
    
    max_y = np.max((np.max(ys_speed_rxs), np.max(ys_speed_rxs)))
    plt.xlim([-1, 60*60*24])
    plt.ylim([0, max_y+0.5])

    plt.title('Date Time Range\nFrom: {}\nTo: {}'.format(dts[0], dts[-1]))
    
    plt.xlabel('Time of day')
    plt.ylabel('Download/Upload Speed [MiB/s]')

    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.grid(b=True, which='minor', axis='y', color='k', linewidth=0.1, linestyle='-')
    plt.grid(b=True, which='major', axis='x', color='k', linewidth=0.1, linestyle='-')

    plt.xticks([i*60*60 for i in range(0, 25)], ['{:02}:00:00'.format(i) for i in range(0, 25)], rotation=60)
    fig.subplots_adjust(left=0.06, bottom=0.14, right=0.98, top=0.91)
    
    p_rxs = plt.plot(xs_sec, ys_speed_rxs, 'b.', markersize=0.5)[0]
    p_txs = plt.plot(xs_sec, ys_speed_txs, 'g.', markersize=0.5)[0]

    lgnd = plt.legend((p_rxs, p_txs), ('Download Speed [MiB/s]', 'Upload Speed [MiB/s]'), loc='upper right')

    lgnd.legendHandles[0]._legmarker.set_markersize(8)
    lgnd.legendHandles[1]._legmarker.set_markersize(8)

    # plt.savefig('speedtest_average_all_values_{}.eps'.format(dt_str))
    
    # plt.show(block=False)
    plt.savefig('speedtest_average_all_values_{}.png'.format(dt_str), dpi=dpi)
