#1 /usr/bin/python3.6

import string

import numpy as np

if __name__ == "__main__":
    strs = np.array(list(string.ascii_letters+string.digits+" -_"))
    get_random_str = lambda: "".join(np.random.choice(strs, (np.random.randint(4, 11), )))
    print("Hello World!")

    def get_random_arr():
        n = 10
        m = 2
        arr = np.zeros((n, m), dtype=object)
        arr[:, 0] = np.random.randint(0, 10000, (n, ))
        arr[:, 1] = [get_random_str() for _ in range(0, n)]
        return arr

    arr = get_random_arr()