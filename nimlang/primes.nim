import os
import strutils
import streams

import std/strformat
import std/sequtils
import std/times

proc sqrt_int(v: uint64): uint64 =
    var v1: uint64 = v div 2
    var v2: uint64 = (v1 + v div v1) div 2

    for i in 1..10:
        let v3 = (v2 + v div v2) div 2

        if v1 == v3 or v2 == v3:
            break

        v1 = v2
        v2 = v3

    return v2

let args = commandLineParams()
echo(fmt"args: {args}")

let argc = len(args)
assert(argc >= 2, "At least needed 2 arguments!")

let max_p = cast[uint64](parseUInt(args[0]))
let amount = cast[uint64](parseUInt(args[1]))

var l_diff: seq[float64] = @[]
for i_round in 0..amount-1:
    let start_time = cpuTime()
    var l: seq[uint64]= @[uint64(2), uint64(3), uint64(5)]
    var l_jump: seq[uint64]= @[uint64(4), uint64(2)]
    
    var i_jump = uint64(0)
    var p = uint64(7)

    while p < max_p:
        let max_sqrt_p = sqrt_int(p) + 1

        # is p a prime number? let's test this
        var is_prime = true
        var i = uint64(0)
        while l[i] < max_sqrt_p:
            if p mod l[i] == 0:
                is_prime = false
                break
            i += 1

        if is_prime:
            l.add(p)

        p += l_jump[i_jump]
        i_jump = (i_jump + 1) mod 2

    let end_time = cpuTime()
    let elapsed_time = end_time - start_time
    l_diff.add(elapsed_time)

    if i_round == 0:
        var f = newFileStream(fmt"/tmp/primes_n_{max_p}_nim.txt", FileMode.fmWrite)

        for v in l:
            f.write(fmt"{v},")
        
        f.close()

let average_time = foldl(l_diff, a + b, float64(0)) / float64(len(l_diff))

echo(fmt"l_diff: {l_diff}")
echo(fmt"average_time: {average_time}")
