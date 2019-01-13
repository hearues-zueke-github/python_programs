import numpy as np

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