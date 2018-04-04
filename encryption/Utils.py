def pretty_block_printer(block, bits, length):
    digits = int((bits+3)/4)
    temp = length-1
    digits_cols = 0
    while temp > 0:
        temp >>= 4
        digits_cols += 1

    line = ""
    line += " {}\x1b[1;31;38m".format(" "*digits_cols)
    for i in range(16):
        line += " {{:0{}X}}".format(digits).format(i)
    print(line)

    rows = length // 16
    for i in range(rows):
        line = ""
        line += "\x1b[1;35;38m{{:0{}X}}:\x1b[0m".format(digits_cols).format(i*16)
        for j in range(16):
            line += " {{:0{}X}}".format(digits).format(block[i*16+j])
        print(line)

    cols = length%16
    if cols != 0:
        i = rows
        line = ""
        line += "\x1b[1;35;38m{{:0{}X}}:\x1b[0m".format(digits_cols).format(i*16)
        for j in range(cols):
            line += " {{:0{}X}}".format(digits).format(block[i*16+j])
        print(line)
