import numpy as np

class SequenceFunctions(Exception):
    def __init__(self, amount_args, modulo, max_power):
        self.amount_args = amount_args
        self.modulo = modulo
        self.max_power = max_power

        self.lss = None
        self.functions = None

        self.x = tuple(np.random.randint(0, self.modulo, (self.amount_args, )))

    def define_new_funcitons(self):
        self.lss = [self.get_random_factors() for _ in xrange(0, self.amount_args)]
        self.constz = [np.random.randint(0, self.modulo, (self.amount_args+1, )) for _ in xrange(0, self.amount_args)]
        functions_with_strings = [self.get_spec_func_pows(ls, consts) for ls, consts in zip(self.lss, self.constz)]
        self.func_strs, self.funcs = list(zip(*functions_with_strings))

    def get_random_factors(self):
        ls = np.random.randint(0, self.max_power, (np.random.randint(5, 15), self.amount_args))
        ls = np.delete(ls, np.where(np.sum(ls, axis=1)==0)[0], axis=0)

        ls_eliminated = np.array(list(set(list(map(tuple, ls)))))

        ls_sorted = np.array(sorted(ls_eliminated, key=lambda x: np.sum(x*self.max_power**np.arange(0, x.shape[0])[::-1])))

        return ls_sorted

    def get_spec_func_pows(self, ls, consts):
        def get_x_mults(l):
            expr = ""
            for i, j in enumerate(l):
                if j > 0:
                    if expr != "":
                        expr += "*"
                    expr += "x["+str(i)+"]"
                    if j > 1:
                        expr += "**"+str(j)

            return expr

        def get_const_linear_part():
            expr = ""

            for i, c in enumerate(consts[:-1]):
                if c > 0:
                    if expr != "":
                        expr += "+"
                    if c > 1:
                        expr += "{}*".format(c)
                    expr += "x[{}]".format(i)

            if consts[-1] > 0:
                expr += "+{}".format(consts[-1])
            # print("linear part expr: {}".format(expr))
            return expr

        str_expr = "lambda x, m: ("

        # mult_exprs = [get_x_mults(l) for l in ls]+\
        mult_exprs = [get_const_linear_part()]

        str_expr += mult_exprs[0]
        for mult_expr in mult_exprs[1:]:
            str_expr += "+"+mult_expr

        str_expr += ") % m"

        return str_expr, eval(str_expr)

    def nextX(self):
        x = self.x
        self.x = tuple(func(x, self.modulo) for i, func in enumerate(self.funcs))
        return self.x
