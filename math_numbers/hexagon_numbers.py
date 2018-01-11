#! /usr/bin/python3.5

hex_numbers = []
hex_numbers_sets = []

check_hex_number = lambda l: len(list(set(l))) == 9

rot_list = lambda l: l[1:]+l[:1]

def gcd(a, b):
    if b > a:
        a, b = b, a

    while b > 0:
        t = a % b
        a = b
        b = t

    return a

print("gcd(2, 8): {}".format(gcd(2, 8)))
print("gcd(12, 8): {}".format(gcd(12, 8)))
print("gcd(3, 8): {}".format(gcd(3, 8)))
print("gcd(15, 5): {}".format(gcd(15, 5)))

for x in range(2, 20):
    for y in range(x+1, 20):
        for a in range(1, x):
            for b in range(1, x):
                l = [x-a, a, y-a, x-b, b, y-b, 2*x-a-b, a+b, 2*y-a-b]

                if check_hex_number(l):
                    # hex_numbers.append(l)
                    
                    is_in_list = False
                    for i in range(0, 9):
                        l = rot_list(l)
                        # print("l: {}".format(l))
                        if l in hex_numbers:
                            is_in_list = True
                            break

                    if not is_in_list:
                        s = set(l)
                        if not s in hex_numbers_sets:
                            if gcd(l[-1], l[-2]) == 1 and \
                               gcd(l[-1], l[-3]) == 1 and \
                               gcd(l[-2], l[-3]) == 1:
                                hex_numbers.append(l)
                                hex_numbers_sets.append(s)

                # s = set(l)
                # if check_hex_number(l) and not s in hex_numbers:
                #     hex_numbers.append([])

for i, hex_num in enumerate(hex_numbers):
    print("i: {}, hex_num: {}".format(i, hex_num))
