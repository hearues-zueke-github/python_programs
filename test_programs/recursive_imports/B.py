with open("num.bin", "rb") as fin:
    num = int.from_bytes(fin.read(), byteorder="big")

num += 1
print("num: {}".format(num))

bytes_ = bytes([(num>>(i*8))&0xFF for i in range((lambda l: l//8)(len(bin(num))-2), -1, -1)])

with open("num.bin", "wb") as fout:
    fout.write(bytes_)

import C
