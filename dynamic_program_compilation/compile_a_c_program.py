#! /usr/bin/python3.6

# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

if __name__ == "__main__":
    cmd = """echo '
#include<stdio.h>
int main()
{
  int a1 = {a1};
  int a2 = {a2};

  printf("a1: %d, a2: %d\\n", a1, a2);
  printf("Hello World!\\n");
  {innerpart}

  return 0;
}
' | gcc -Wall -o output.o -xc -"""

    a1, a2 = np.random.randint(0, 1000, [2]).tolist()
    print("a1: {}, a2: {}".format(a1, a2))

    cmd = cmd.replace("{a1}", "{}".format(a1))
    cmd = cmd.replace("{a2}", "{}".format(a2))

    '''cmd = cmd.replace("{innerpart}", """
printf("TEST!!!\\n");
{innerpart}
""")'''

    cmd = cmd.replace("{innerpart}", "")

    cmd = cmd.replace("\\n", "\\\\n")
    print("cmd:\n{}".format(cmd))

    os.system(cmd)
