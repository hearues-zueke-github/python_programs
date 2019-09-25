#! /usr/bin/python3

# -*- coding: utf-8 -*-

import tarfile

if __name__ == "__main__":
    print("Hello World!")

    with tarfile.open('images/pixabay_com.tar', 'r:*') as ftar:
        members = ftar.getmembers()

    print("len(members): {}".format(len(members)))
