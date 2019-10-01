#! /usr/bin/python3

import time
import datetime
import dill
import os
import subprocess
import sys

PATH_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))+"/"

if __name__=='__main__':
    # print('Hello World!')
    while True:
        lines = subprocess.Popen('ifconfig', stdout=subprocess.PIPE).communicate()[0].decode('utf-8').split('\n')
        for i, l in enumerate(lines, 0):
            if 'RX bytes:' in l:
                break
        rx_bytes = l.split('bytes:')[1].lstrip().split('(')[0].rstrip()
        rx_bytes = int(rx_bytes)
        # print("rx_bytes: {}".format(rx_bytes))
        current_dt = datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')

        file_path = PATH_ROOT_DIR+'saved_rx_bytes.txt'
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass
        os.system('printf %s "{},{};" >> {}'.format(current_dt, rx_bytes, file_path))
        print("rx_bytes: {}".format(rx_bytes))
        time.sleep(5)
