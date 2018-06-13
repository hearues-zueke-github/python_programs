from dotmap import DotMap

clrs = DotMap()

clrs.dgy = "\x1b[0;30;38m"
clrs.lr = "\x1b[0;31;38m"
clrs.lg = "\x1b[0;32;38m"
clrs.ly = "\x1b[0;33;38m"
clrs.lb = "\x1b[0;34;38m"
clrs.lm = "\x1b[0;35;38m"
clrs.lc = "\x1b[0;36;38m"
clrs.lw = "\x1b[0;38;38m"

clrs.dgyb = "\x1b[1;30;38m"
clrs.lrb = "\x1b[1;31;38m"
clrs.lgb = "\x1b[1;32;38m"
clrs.lyb = "\x1b[1;33;38m"
clrs.lbb = "\x1b[1;34;38m"
clrs.lmb = "\x1b[1;35;38m"
clrs.lcb = "\x1b[1;36;38m"
clrs.lwb = "\x1b[1;38;38m"

clrs.rst = "\x1b[0m"

def test_color_print():
    print("dgy: {}testTEST{}".format(clrs.dgy, clrs.rst))
    print("lr: {}testTEST{}".format(clrs.lr, clrs.rst))
    print("lg: {}testTEST{}".format(clrs.lg, clrs.rst))
    print("ly: {}testTEST{}".format(clrs.ly, clrs.rst))
    print("lb: {}testTEST{}".format(clrs.lb, clrs.rst))
    print("lm: {}testTEST{}".format(clrs.lm, clrs.rst))
    print("lc: {}testTEST{}".format(clrs.lc, clrs.rst))
    print("lw: {}testTEST{}".format(clrs.lw, clrs.rst))
    print("")
    print("dgyb: {}testTEST{}".format(clrs.dgyb, clrs.rst))
    print("lrb: {}testTEST{}".format(clrs.lrb, clrs.rst))
    print("lgb: {}testTEST{}".format(clrs.lgb, clrs.rst))
    print("lyb: {}testTEST{}".format(clrs.lyb, clrs.rst))
    print("lbb: {}testTEST{}".format(clrs.lbb, clrs.rst))
    print("lmb: {}testTEST{}".format(clrs.lmb, clrs.rst))
    print("lcb: {}testTEST{}".format(clrs.lcb, clrs.rst))
    print("lwb: {}testTEST{}".format(clrs.lwb, clrs.rst))

def pretty_block_printer(block, bits, length, per_line=16):
    digits = int((bits+3)/4)
    temp = length-1
    digits_cols = 0
    while temp > 0:
        temp >>= 4
        digits_cols += 1

    line = ""
    line += " {}{}".format(" "*digits_cols, clrs.lrb)
    for i in range(0, per_line):
        line += " {{:0{}X}}".format(digits).format(i)
    print(line+(clrs.rst if length == 0 else ""))

    rows = length // per_line
    for i in range(0, rows):
        line = ""
        line += "{}{{:0{}X}}:{}".format(clrs.lmb, digits_cols, clrs.rst).format(i*per_line)
        for j in range(0, per_line):
            line += " {{:0{}X}}".format(digits).format(block[i*per_line+j])
        print(line)

    cols = length%per_line
    if cols != 0:
        i = rows
        line = ""
        line += "{}{{:0{}X}}:{}".format(clrs.lmb, digits_cols, clrs.rst).format(i*per_line)
        for j in range(0, cols):
            line += " {{:0{}X}}".format(digits).format(block[i*per_line+j])
        print(line)
