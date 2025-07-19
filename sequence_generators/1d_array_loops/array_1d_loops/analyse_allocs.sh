#! /bin/bash
strace -f -e brk,mmap,mmap2 -- ./target/release/array_1d_loops 2>&1 | grep 'brk\|mmap' | sed -e 's/^\(.*\)(.*/\1/' | sort | uniq -c
