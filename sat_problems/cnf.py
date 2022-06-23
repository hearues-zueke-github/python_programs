from typing import List, Tuple

class CNF():
    def __init__(self):
        self.last_free_variable = 1
        self.cnfs = []


    def extend_cnfs(self, l_t_v: List[Tuple[int]]):
        self.cnfs.extend(l_t_v)


    def get_tseytin_and(l_v: List[int], v_res: int):
        l = []
        for v in l_v:
            l.append((v, -v_res))
        l.append(tuple(-v for v in l_v) + (v_res, ))
        return l


    def get_tseytin_or(l_v: List[int], v_res: int):
        l = []
        for v in l_v:
            l.append((-v, v_res))
        l.append(tuple(v for v in l_v) + (-v_res, ))
        return l


    def get_tseytin_xor(v_1: int, v_2: int, v_res: int):
        return [(-v_1, -v_2, -v_res), (v_1, v_2, -v_res), (-v_1, v_2, v_res), (v_1, -v_2, v_res)]


    def get_tseytin_not(v_1: int, v_res: int):
        return [(-v_1, v_res), (v_1, -v_res)]


    def get_tseytin_only_one_true(l_v: List[int]):
        l = [tuple(l_v)]
        
        for i1, v1 in enumerate(l_v, 1):
            for v2 in l_v[i1:]:
                l.append((-v1, -v2))

        return l


    def get_new_variables(self, amount):
        l_v = list(range(self.last_free_variable, self.last_free_variable + amount))
        self.last_free_variable += amount
        return l_v


    def add_count_sum(self, l_v: List[int], bits: int, num: int):
        assert num >= 0 and num < len(l_v)
        assert 2**bits >= len(l_v)

        # calculate the sum of the l_v variables! (binary count for all varibale were v > 0 means that v is bit 1)

        cnfs_part = []

        l_v_sum_prev = self.get_new_variables(amount=bits)
        cnfs_part.extend([(-v, ) for v in l_v_sum_prev]) # set the first values to 0, for the sum accumulator!

        for v in l_v:
            l_v_rest_next = self.get_new_variables(amount=bits-1)
            l_v_sum_next = self.get_new_variables(amount=bits)

            cnfs_part.extend(CNF.get_tseytin_and(l_v=[l_v_sum_prev[0], v], v_res=l_v_rest_next[0]))
            cnfs_part.extend(CNF.get_tseytin_xor(v_1=l_v_sum_prev[0], v_2=v, v_res=l_v_sum_next[0]))
            
            for i in range(1, bits-1):
                cnfs_part.extend(CNF.get_tseytin_and(l_v=[l_v_sum_prev[i], l_v_rest_next[i-1]], v_res=l_v_rest_next[i]))
                
            for i in range(1, bits):
                cnfs_part.extend(CNF.get_tseytin_xor(v_1=l_v_sum_prev[i], v_2=l_v_rest_next[i-1], v_res=l_v_sum_next[i]))

            l_v_sum_prev = l_v_sum_next

        l_bit = list(map(int, bin(num)[2:].zfill(bits)))[::-1]
        cnfs_part.extend((v, ) if b == 1 else (-v, ) for v, b in zip(l_v_sum_prev, l_bit))

        return cnfs_part
