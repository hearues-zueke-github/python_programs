#! /usr/bin/python3.5

import numpy as np

n = 20

lst_of_lsts = []
empty_lsts = [lst_of_lsts]

for i in range(0, n):
    # Take one list of empty_lsts
    lsts = empty_lsts.pop(np.random.randint(0, len(empty_lsts)))

    # Create a random amount of new lists into the list
    amount = np.random.randint(1, 6)
    lsts.extend([[] for _ in range(0, amount)])

    # Add the empty lists into the empty_lsts list
    for lst in lsts:
        empty_lsts.append(lst)
    
    print("i: {}, lst_of_lsts:\n{}".format(i, lst_of_lsts))
    print("len(empty_lsts): {}".format(len(empty_lsts)))

# Now fill all remained empty lists
for empty_lst in empty_lsts:
    amoount = np.random.randint(1, 6)
    empty_lst.extend(np.random.randint(0, 10, (amount, )))

print("lst_of_lsts:\n{}".format(lst_of_lsts))

def get_max_depth(lst):
    max_depth = 0
    for l in lst:
        if type(l) is list:
            d = get_max_depth(l)+1
            if d > max_depth:
                max_depth = d

    return max_depth

print("get_max_depth(lst_of_lsts): {}".format(get_max_depth(lst_of_lsts)))
