import unittest

import prime_factor_factorization as pff

from pysat.formula import CNF
from pysat.solvers import Solver

# cnf_logic_equal(a, b)
# cnf_logic_not(a, b)
# cnf_logic_and(a, b, c)
# cnf_logic_or(a, b, c)
# cnf_logic_xor(a, b, c)
# cnf_logic_and_many(l_a, b)
# cnf_logic_or_many(l_a, b)
# cnf_logic_adder_1_bit(a, b, c, r, s, l_temp)
# cnf_logic_adder(l_a, l_b, l_carry, l_result, l_temp)

sign = lambda x: (1, -1)[x<0]

class TestPrimeFactorFactorization(unittest.TestCase):

	def test_cnf_logic_equal(self):
		cnf_logic_equal = pff.cnf_logic_equal

		cnf = cnf_logic_equal(a=1, b=2)
		cnf.append((1, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)
		self.assertEqual(l_model[0], [1, 2])


	def test_cnf_logic_not(self):
		cnf_logic_not = pff.cnf_logic_not

		cnf = cnf_logic_not(a=1, b=2)
		cnf.append((1, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)
		self.assertEqual(l_model[0], [1, -2])


	def test_cnf_logic_and(self):
		cnf_logic_and = pff.cnf_logic_and

		cnf = cnf_logic_and(a=1, b=2, c=3)
		cnf.append((1, ))
		cnf.append((-3, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)
		self.assertEqual(l_model[0], [1, -2, -3])


	def test_cnf_logic_or(self):
		cnf_logic_or = pff.cnf_logic_or

		cnf = cnf_logic_or(a=1, b=2, c=3)
		cnf.append((1, ))
		cnf.append((3, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 2)
		self.assertEqual(l_model[0], [1, -2, 3])
		self.assertEqual(l_model[1], [1, 2, 3])


	def test_cnf_logic_and_many_1(self):
		cnf_logic_and_many = pff.cnf_logic_and_many

		cnf = cnf_logic_and_many(l_a=[1], b=2)
		cnf.append((1, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:

			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)
		self.assertEqual(l_model[0], [1, 2])


	def test_cnf_logic_and_many_2(self):
		cnf_logic_and_many = pff.cnf_logic_and_many

		cnf = cnf_logic_and_many(l_a=[1, 2], b=3)
		cnf.append((1, ))
		cnf.append((3, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)
		self.assertEqual(l_model[0], [1, 2, 3])


	def test_cnf_logic_and_many_3(self):
		cnf_logic_and_many = pff.cnf_logic_and_many

		cnf = cnf_logic_and_many(l_a=[1, 2, 3, 4, 5], b=6)
		cnf.append((6, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)


	def test_cnf_logic_or_many_1(self):
		cnf_logic_or_many = pff.cnf_logic_or_many

		cnf = cnf_logic_or_many(l_a=[1], b=2)
		cnf.append((1, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)
		self.assertEqual(l_model[0], [1, 2])


	def test_cnf_logic_or_many_2(self):
		cnf_logic_or_many = pff.cnf_logic_or_many

		cnf = cnf_logic_or_many(l_a=[1, 2], b=3)
		cnf.append((1, ))
		cnf.append((3, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 2)
		self.assertEqual(l_model[0], [1, -2, 3])
		self.assertEqual(l_model[1], [1, 2, 3])


	def test_cnf_logic_or_many_3(self):
		cnf_logic_or_many = pff.cnf_logic_or_many

		cnf = cnf_logic_or_many(l_a=[1, 2, 3, 4, 5], b=6)
		cnf.append((6, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 2**5 - 1)


	def test_cnf_logic_adder_1_bit(self):
		cnf_logic_adder_1_bit = pff.cnf_logic_adder_1_bit

		cnf = cnf_logic_adder_1_bit(a=1, b=2, c=3, r=4, s=5, l_temp=[i + 6 for i in range(0, 7)])
		cnf.append((1, ))
		cnf.append((-2, ))
		cnf.append((3, ))

		l_model = []
		with Solver(bootstrap_with=cnf) as solver:
			is_solveable = solver.solve()
			
			for  model in solver.enum_models():
				l_model.append(model)

		self.assertEqual(len(l_model), 1)
		model = l_model[0]
		self.assertEqual(sign(model[4 - 1]), 1)
		self.assertEqual(sign(model[5 - 1]), -1)


	def test_cnf_logic_adder(self):
		cnf_logic_adder = pff.cnf_logic_adder

		bits = 4
		var_num = 1

		l_a = [var_num + i for i in range(0, bits)]
		var_num += bits

		l_b = [var_num + i for i in range(0, bits)]
		var_num += bits

		l_result = [var_num + i for i in range(0, bits + 1)]
		var_num += bits + 1

		l_temp_all = [var_num + i for i in range(0, 7 * (bits - 1) + bits)]

		cnf = cnf_logic_adder(l_a=l_a, l_b=l_b, l_result=l_result, l_temp_all=l_temp_all)

		for num_a in range(0, 2**4):
			l_bit_a = [int(s) for s in bin(num_a)[2:][::-1]]
			if len(l_bit_a) < bits:
				l_bit_a.extend([0] * (bits - len(l_bit_a)))
			
			for num_b in range(0, 2**4):
				l_bit_b = [int(s) for s in bin(num_b)[2:][::-1]]	
				if len(l_bit_b) < bits:
					l_bit_b.extend([0] * (bits - len(l_bit_b)))

				for a, bit_a in zip(l_a, l_bit_a):
					cnf.append((-a, ) if bit_a == 0 else (a, ))

				for b, bit_b in zip(l_b, l_bit_b):
					cnf.append((-b, ) if bit_b == 0 else (b, ))

				l_model = []
				with Solver(bootstrap_with=cnf) as solver:
					is_solveable = solver.solve()
					
					for  model in solver.enum_models():
						l_model.append(model)

				self.assertEqual(len(l_model), 1)
				model = l_model[0]
				l_bit_result = [1 if model[result - 1] > 0 else 0 for result in l_result]
				num_result = sum([v * 2**i for i, v in enumerate(l_bit_result, 0)])

				self.assertEqual(num_result, num_a + num_b)

				cnf = cnf[:-bits*2] # remove the added fixed numbers


if __name__ == '__main__':
	unittest.main()
