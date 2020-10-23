import numpy as np

from functools import reduce

def check_if_prime(n, primes):
    n_sqrt = int(np.sqrt(n))+1
    i = 0
    v = primes[i]
    while v<=n_sqrt:
        if n%v==0:
            return False
        i += 1
        v = primes[i]
    return True


def num_to_base(n, b):
    l = []
    while n>0:
        l.append(n%b)
        n = n//b
    return l


def find_max_match(arr):
    previous_pattern = None
    length = arr.shape[0]
    for i in range(1, length//2+1):
        if np.all(arr[:i]==arr[i:2*i]):
            if isinstance(previous_pattern, type(None)):
                previous_pattern = arr[:i]
            else:
                pattern = arr[:i]
                # Test if this previous pattern is really appliable on the new pattern or not!
                length_prev = previous_pattern.shape[0]
                length_now = pattern.shape[0]

                if (length_prev == 1) or \
                   (length_now % length_prev > 0) or \
                   (not np.all(pattern.reshape((-1, length_prev))==previous_pattern)):
                    previous_pattern = pattern

    return previous_pattern


def find_first_repeat_pattern(arr):
    arr_reverse = arr[::-1]

    pattern = find_max_match(arr_reverse)
    # print("pattern: {}".format(pattern))

    if isinstance(pattern, type(None)):
        return None, None

    length_arr = arr.shape[0]
    length_pattern = pattern.shape[0]

    last_possible_i = 0
    for i in range(1, length_arr//length_pattern+1):
        if np.all(arr_reverse[length_pattern*i:length_pattern*(i+1)]==pattern):
            last_possible_i = i
        else:
            break

    rest_arr = arr_reverse[length_pattern*(last_possible_i+1):]
    if rest_arr.shape[0] == 0:
        return pattern[::-1], 0

    # print("rest_arr: {}".format(rest_arr))

    same_length = np.min((pattern.shape[0], rest_arr.shape[0]))
    equal_numbers = pattern[:same_length]==rest_arr[:same_length]

    # print("same_length: {}".format(same_length))
    # print("equal_numbers: {}".format(equal_numbers))

    eql_nums_sum = np.cumsum(equal_numbers)
    rest_nums = eql_nums_sum[~equal_numbers]

    # print("eql_nums_sum: {}".format(eql_nums_sum))
    # print("rest_nums: {}".format(rest_nums))
    if rest_nums.shape[0] == 0:
        last_idx = equal_numbers.shape[0]
    else:
        last_idx = np.min(rest_nums)

    idx1 = length_pattern*last_possible_i+last_idx
    idx2 = length_pattern*(last_possible_i+1)+last_idx
    first_pattern = arr_reverse[idx1:idx2][::-1]

    return first_pattern, length_arr-idx2


def all_possibilities_changing_one_position(m, n):
    length = 2*(m-1)*m**(n-1)
    print("length: {}".format(length))

    arr = np.zeros((length, n), dtype=np.int)

    for i in range(1, n+1):
        if i == n:
            arr_resh = arr.reshape((2*(m-1), m**(i-1), n))
            arr_resh[:m, :, -i] = np.arange(0, m).reshape((-1, 1))
            arr_resh[m:m+m-2, :, -i] = np.arange(m-2, 0, -1).reshape((-1, 1))
        else:
            arr_resh = arr.reshape((-1, m, m**(i-1), n))
            idxs1 = np.arange(0, arr_resh.shape[0], 2)
            idxs2 = np.arange(1, arr_resh.shape[0], 2)
            arr_resh[idxs1, :, :, -i] = np.arange(0, m).reshape((-1, 1))
            arr_resh[idxs2, :, :, -i] = np.arange(m-1, -1, -1).reshape((-1, 1))

    return arr


def calc_gcd(lst):
    return reduce(lambda a, b: np.gcd(a, b), lst[1:], lst[0])


def calc_lsm(lst):
    return np.multiply.reduce(lst)//calc_gcd(lst)


def get_all_linear_coefficients(n):
    c = np.tile(np.arange(0, n), n)
    a = c.reshape((n, n)).T.flatten()
    x = np.zeros((n*n, ), dtype=np.int64)

    l = []
    for i in range(0, n):
        x = (a*x+c)%n
        l.append(x)
    X = np.vstack(l).T

    idxs = np.all(np.diff(np.sort(X, axis=1), axis=1)==1, axis=1)
    a_valid = a[idxs]
    c_valid = c[idxs]

    l_linear_coefficients = [(a, c) for a, c in zip(a_valid, c_valid)]
    return l_linear_coefficients


def mix_l1_with_l2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    acc_i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i1+l2[acc_i2])%len1
            if i2==i1:
                i2 = (i2+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def mix_l1_with_l2_method_2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    # acc_i1 = 0
    acc_i2 = 0
    i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i2+l1[i1]+l2[acc_i2]+1)%len1
            # i2 = (i2+l1[acc_i1]+l2[acc_i2]+1)%len1
            if i2==i1:
                i2 = (i2+1)%len1
            # acc_i1 = (acc_i1+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def create_sboxes(xs, mod):
    assert len(xs.shape)==1
    assert xs.shape[0]%mod==0

    u, c = np.unique(xs, return_counts=True)
    assert u.shape[0]==mod
    assert np.all(c==c[0])

    rows = xs.shape[0]//mod
    sboxes = np.zeros((rows, mod), dtype=np.int)
    sbox_num_index = np.zeros((mod, ), dtype=np.int)
    sbox_num_pos = np.zeros((rows, ), dtype=np.int)

    # create sboxes from the xs aka np.roll(xs, 1)
    for v in xs:
        sbox_num = sbox_num_index[v]
        sboxes[sbox_num, sbox_num_pos[sbox_num]] = v
        sbox_num_index[v] += 1
        sbox_num_pos[sbox_num] += 1

    # check, if for any i is s[i]==i true!
    idxs_nums = np.arange(0, mod)
    for sbox_nr, sbox in enumerate(sboxes, 0):
        idxs = sbox==idxs_nums
        amount_same_pos = np.sum(idxs)
        if amount_same_pos>0:
            # print("sbox_nr: {}".format(sbox_nr))
            if amount_same_pos==1:
                i = np.where(idxs)[0]
                j = 0
                if i==j and mod>1:
                    j = 1
                v1, v2 = sbox[i], sbox[j]
                sbox[j], sbox[i] = v1, v2
            else:
                sbox[idxs] = np.roll(sbox[idxs], 1)

    return sboxes


def calc_gcd(lst):
    return reduce(lambda a, b: np.gcd(a, b), lst[1:], lst[0])


def calc_lsm(lst):
    return np.multiply.reduce(lst)//calc_gcd(lst)


def get_all_linear_coefficients(n):
    c = np.tile(np.arange(0, n), n)
    a = c.reshape((n, n)).T.flatten()
    x = np.zeros((n*n, ), dtype=np.int64)

    l = []
    for i in range(0, n):
        x = (a*x+c)%n
        l.append(x)
    X = np.vstack(l).T

    idxs = np.all(np.diff(np.sort(X, axis=1), axis=1)==1, axis=1)
    a_valid = a[idxs]
    c_valid = c[idxs]

    l_linear_coefficients = [(a, c) for a, c in zip(a_valid, c_valid)]
    return l_linear_coefficients


def mix_l1_with_l2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    acc_i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i1+l2[acc_i2])%len1
            if i2==i1:
                i2 = (i2+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def mix_l1_with_l2_method_2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    # acc_i1 = 0
    acc_i2 = 0
    i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i2+l1[i1]+l2[acc_i2]+1)%len1
            # i2 = (i2+l1[acc_i1]+l2[acc_i2]+1)%len1
            if i2==i1:
                i2 = (i2+1)%len1
            # acc_i1 = (acc_i1+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def create_sboxes(xs, mod):
    assert len(xs.shape)==1
    assert xs.shape[0]%mod==0

    u, c = np.unique(xs, return_counts=True)
    assert u.shape[0]==mod
    assert np.all(c==c[0])

    rows = xs.shape[0]//mod
    sboxes = np.zeros((rows, mod), dtype=np.int)
    sbox_num_index = np.zeros((mod, ), dtype=np.int)
    sbox_num_pos = np.zeros((rows, ), dtype=np.int)

    # create sboxes from the xs aka np.roll(xs, 1)
    for v in xs:
        sbox_num = sbox_num_index[v]
        sboxes[sbox_num, sbox_num_pos[sbox_num]] = v
        sbox_num_index[v] += 1
        sbox_num_pos[sbox_num] += 1

    # check, if for any i is s[i]==i true!
    idxs_nums = np.arange(0, mod)
    for sbox_nr, sbox in enumerate(sboxes, 0):
        idxs = sbox==idxs_nums
        amount_same_pos = np.sum(idxs)
        if amount_same_pos>0:
            # print("sbox_nr: {}".format(sbox_nr))
            if amount_same_pos==1:
                i = np.where(idxs)[0]
                j = 0
                if i==j and mod>1:
                    j = 1
                v1, v2 = sbox[i], sbox[j]
                sbox[j], sbox[i] = v1, v2
            else:
                sbox[idxs] = np.roll(sbox[idxs], 1)

    return sboxes


def calc_gcd(lst):
    return reduce(lambda a, b: np.gcd(a, b), lst[1:], lst[0])


def calc_lsm(lst):
    return np.multiply.reduce(lst)//calc_gcd(lst)


def get_all_linear_coefficients(n):
    c = np.tile(np.arange(0, n), n)
    a = c.reshape((n, n)).T.flatten()
    x = np.zeros((n*n, ), dtype=np.int64)

    l = []
    for i in range(0, n):
        x = (a*x+c)%n
        l.append(x)
    X = np.vstack(l).T

    idxs = np.all(np.diff(np.sort(X, axis=1), axis=1)==1, axis=1)
    a_valid = a[idxs]
    c_valid = c[idxs]

    l_linear_coefficients = [(a, c) for a, c in zip(a_valid, c_valid)]
    return l_linear_coefficients


def mix_l1_with_l2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    acc_i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i1+l2[acc_i2])%len1
            if i2==i1:
                i2 = (i2+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def mix_l1_with_l2_method_2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    # acc_i1 = 0
    acc_i2 = 0
    i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i2+l1[i1]+l2[acc_i2]+1)%len1
            # i2 = (i2+l1[acc_i1]+l2[acc_i2]+1)%len1
            if i2==i1:
                i2 = (i2+1)%len1
            # acc_i1 = (acc_i1+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def create_sboxes(xs, mod):
    assert len(xs.shape)==1
    assert xs.shape[0]%mod==0

    u, c = np.unique(xs, return_counts=True)
    assert u.shape[0]==mod
    assert np.all(c==c[0])

    rows = xs.shape[0]//mod
    sboxes = np.zeros((rows, mod), dtype=np.int)
    sbox_num_index = np.zeros((mod, ), dtype=np.int)
    sbox_num_pos = np.zeros((rows, ), dtype=np.int)

    # create sboxes from the xs aka np.roll(xs, 1)
    for v in xs:
        sbox_num = sbox_num_index[v]
        sboxes[sbox_num, sbox_num_pos[sbox_num]] = v
        sbox_num_index[v] += 1
        sbox_num_pos[sbox_num] += 1

    # check, if for any i is s[i]==i true!
    idxs_nums = np.arange(0, mod)
    for sbox_nr, sbox in enumerate(sboxes, 0):
        idxs = sbox==idxs_nums
        amount_same_pos = np.sum(idxs)
        if amount_same_pos>0:
            # print("sbox_nr: {}".format(sbox_nr))
            if amount_same_pos==1:
                i = np.where(idxs)[0]
                j = 0
                if i==j and mod>1:
                    j = 1
                v1, v2 = sbox[i], sbox[j]
                sbox[j], sbox[i] = v1, v2
            else:
                sbox[idxs] = np.roll(sbox[idxs], 1)

    return sboxes


def calc_gcd(lst):
    return reduce(lambda a, b: np.gcd(a, b), lst[1:], lst[0])


def calc_lsm(lst):
    return np.multiply.reduce(lst)//calc_gcd(lst)


def get_all_linear_coefficients(n):
    c = np.tile(np.arange(0, n), n)
    a = c.reshape((n, n)).T.flatten()
    x = np.zeros((n*n, ), dtype=np.int64)

    l = []
    for i in range(0, n):
        x = (a*x+c)%n
        l.append(x)
    X = np.vstack(l).T

    idxs = np.all(np.diff(np.sort(X, axis=1), axis=1)==1, axis=1)
    a_valid = a[idxs]
    c_valid = c[idxs]

    l_linear_coefficients = [(a, c) for a, c in zip(a_valid, c_valid)]
    return l_linear_coefficients


def mix_l1_with_l2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    acc_i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i1+l2[acc_i2])%len1
            if i2==i1:
                i2 = (i2+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def mix_l1_with_l2_method_2(l1, l2, rounds=1):
    l1 = l1.copy()

    len1 = len(l1)
    len2 = len(l2)

    # acc_i1 = 0
    acc_i2 = 0
    i2 = 0
    for it_round in range(0, rounds):
        for i1 in range(0, len1):
            i2 = (i2+l1[i1]+l2[acc_i2]+1)%len1
            # i2 = (i2+l1[acc_i1]+l2[acc_i2]+1)%len1
            if i2==i1:
                i2 = (i2+1)%len1
            # acc_i1 = (acc_i1+1)%len1
            acc_i2 = (acc_i2+1)%len2
            l1[i1], l1[i2] = l1[i2], l1[i1]

    return l1


def create_sboxes(xs, mod):
    assert len(xs.shape)==1
    assert xs.shape[0]%mod==0

    u, c = np.unique(xs, return_counts=True)
    assert u.shape[0]==mod
    assert np.all(c==c[0])

    rows = xs.shape[0]//mod
    sboxes = np.zeros((rows, mod), dtype=np.int)
    sbox_num_index = np.zeros((mod, ), dtype=np.int)
    sbox_num_pos = np.zeros((rows, ), dtype=np.int)

    # create sboxes from the xs aka np.roll(xs, 1)
    for v in xs:
        sbox_num = sbox_num_index[v]
        sboxes[sbox_num, sbox_num_pos[sbox_num]] = v
        sbox_num_index[v] += 1
        sbox_num_pos[sbox_num] += 1

    # check, if for any i is s[i]==i true!
    idxs_nums = np.arange(0, mod)
    for sbox_nr, sbox in enumerate(sboxes, 0):
        idxs = sbox==idxs_nums
        amount_same_pos = np.sum(idxs)
        if amount_same_pos>0:
            # print("sbox_nr: {}".format(sbox_nr))
            if amount_same_pos==1:
                i = np.where(idxs)[0]
                j = 0
                if i==j and mod>1:
                    j = 1
                v1, v2 = sbox[i], sbox[j]
                sbox[j], sbox[i] = v1, v2
            else:
                sbox[idxs] = np.roll(sbox[idxs], 1)

    return sboxes
