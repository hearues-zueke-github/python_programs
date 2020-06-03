import numpy as np

from SimpleNeuralNetwork import SimpleNeuralNetwork
from TetrisGameField import TetrisGameField

class GeneticAlgorithmTetris(Exception):
    def __init__(self, d_basic_data_info, population_size, Xs, Ts, hidden_nodes=[200], is_single_instance_only=False):
        self.d_basic_data_info = d_basic_data_info
        self.population_size = population_size

        self.is_single_instance_only = is_single_instance_only
        
        self.using_parent_amount_percent = 0.5
        self.using_parent_amount = int(self.population_size*self.using_parent_amount_percent)
        self.crossover_percent = 0.40
        self.mutation_percent = 0.08
        self.mutation_add_percent = 0.05
        self.mutation_sub_percent = 0.05

        self.Xs = Xs
        self.Ts = Ts

        self.input_nodes = self.Xs[0].shape[1]

        self.hidden_nodes = hidden_nodes
        self.eta = 0.00001

        max_piece_nr = 1000
        using_pieces = d_basic_data_info['using_pieces']

        self.l_lsnn = [self.create_new_lsnn() for _ in range(0, population_size)]
        self.l_tgm = [TetrisGameField(d_basic_data_info=d_basic_data_info, l_snn=l_snn, max_piece_nr=max_piece_nr, using_pieces=using_pieces) for l_snn in self.l_lsnn]


    def create_new_lsnn(self):
        return [self.create_new_snn(output_nodes=T.shape[1]) for T in self.Ts]


    def create_new_snn(self, output_nodes):
        return SimpleNeuralNetwork(input_nodes=self.input_nodes, hidden_nodes=self.hidden_nodes, output_nodes=output_nodes)


    def simple_genetic_algorithm_training(self, epochs=100):
        self.l_cecf_all = []
        self.l_cecf_sum_all = []

        self.l_max_played_games = []
        self.l_best_tgm_piece_nr = []
        self.l_best_tgm_clear_lines = []
        # self.l_max_played_games_list = []

        l_cecf = [
            [snn.f_cecf(snn.calc_feed_forward(X), T)/T.shape[0]/T.shape[1] for snn, X, T in zip(l_snn, self.Xs, self.Ts)]
            for l_snn in self.l_lsnn
        ]
        l_cecf_sum = [np.sum(l) for l in l_cecf]

        self.l_cecf_all.append(l_cecf)
        self.l_cecf_sum_all.append(sorted(l_cecf_sum))

        # l_sorted = sorted([(i, v) for i, v in enumerate(l_cecf_sum, 0)], key=lambda x: x[1])
        # l_idx = [idx for idx, cecf_train in l_sorted]

        print("first: self.l_cecf_sum: {}".format(l_cecf_sum))

        for epoch in range(1, epochs):
            print("epoch: {}".format(epoch))

            # play the tetris games!
            l_pn_cl = []
            for i, tgm in enumerate(self.l_tgm, 0):
                tgm.main_game_loop()
                piece_nr = tgm.piece_nr
                clear_lines = tgm.clear_lines
                l_pn_cl.append((i, piece_nr, clear_lines))
                # print("i: {}, piece_nr: {}, clear_lines: {}".format(i, piece_nr, clear_lines))
            l_pn_cl_sorted = sorted(l_pn_cl, key=lambda x: (x[2], x[1]), reverse=True)
            print("l_pn_cl_sorted[0]: {}".format(l_pn_cl_sorted[0]))
            print()
            
            # l_idx = [idx for idx, pn, cl in l_pn_cl_sorted]
            l_sorted = sorted([(i, v) for i, v in enumerate(l_cecf_sum, 0)], key=lambda x: x[1])
            l_idx = [idx for idx, cecf_train in l_sorted]

            l_lsnn_new_parents = []
            l_lsnn_new_parents = [self.l_lsnn[i] for i in l_idx[:self.using_parent_amount]]
                
            for i in l_idx[self.using_parent_amount:]:
                self.l_tgm[i].games_played = 0

            l_max_played_games = [tgm.games_played for tgm in self.l_tgm]
            # self.l_max_played_games_list.append(l_max_played_games)
            self.l_max_played_games.append(max(l_max_played_games))

            best_tgm = self.l_tgm[l_idx[0]]
            self.l_best_tgm_piece_nr.append(best_tgm.piece_nr)
            self.l_best_tgm_clear_lines.append(best_tgm.clear_lines)

            # do the back_propagation only for the one best parent!
            lsnn = l_lsnn_new_parents[0]
            for snn, X, T in zip(lsnn, self.Xs, self.Ts):
                bwsd = snn.calc_backprop_own_bws(X, T)
                snn.bws = [w-wg*self.eta for w, wg in zip(snn.bws, bwsd)]

            amount_parents = len(l_lsnn_new_parents)
            amount_childs = self.population_size-amount_parents
            
            using_idx_delta = np.random.randint(1, amount_parents, (amount_childs, ))
            using_idx_delta[0] = np.random.randint(0, amount_parents)
            using_idx = np.cumsum(using_idx_delta)%amount_parents
            l_lsnn_new_childs = [self.create_new_lsnn() for _ in using_idx]

            for i1, i2, lsnn_child in zip(using_idx, np.roll(using_idx, 1), l_lsnn_new_childs):
                lsnn1, lsnn2 = l_lsnn_new_parents[i1], l_lsnn_new_parents[i2]
                for snn1, snn2, snn_child in zip(lsnn1, lsnn2, lsnn_child):
                    for w1, w2, w_c in zip(snn1.bws, snn2.bws, snn_child.bws):
                        shape = w1.shape
                        a1 = w1.reshape((-1, ))
                        a2 = w2.reshape((-1, ))
                        a_c = w_c.reshape((-1, ))

                        length = a1.shape[0]

                        # crossover step
                        idx_crossover = np.zeros((length, ), dtype=np.uint8)
                        idx_crossover[:int(length*self.crossover_percent)] = 1

                        a_c[:] = a1

                        # idx_crossover_1 = np.random.permutation(idx_crossover)==1
                        # assert np.all(a_c[idx_crossover_1]==a1[idx_crossover_1])
                        # a_c[idx_crossover_1] = a1[idx_crossover_1]
                        
                        idx_crossover_2 = np.random.permutation(idx_crossover)==1
                        a_c[idx_crossover_2] = a2[idx_crossover_2]

                        # mutation step
                        idx_mutation = np.zeros((length, ), dtype=np.uint8)
                        idx_mutation[:int(length*self.mutation_percent)] = 1

                        idx_mutation_c = np.random.permutation(idx_mutation)==1
                        amount_mutation_vals = np.sum(idx_mutation)
                        a_c[np.random.permutation(idx_mutation_c)] = (np.random.random((amount_mutation_vals, ))*2.-1.)/10.

                        # mutation step add
                        idx_mutation_add = np.zeros((length, ), dtype=np.uint8)
                        idx_mutation_add[:int(length*self.mutation_add_percent)] = 1

                        idx_mutation_add_c = np.random.permutation(idx_mutation_add)==1
                        amount_mutation_add_vals = np.sum(idx_mutation_add)
                        a_c[np.random.permutation(idx_mutation_add_c)] += (np.random.random((amount_mutation_add_vals, ))*2.-1.)/10000.

                        # mutation step sub
                        idx_mutation_sub = np.zeros((length, ), dtype=np.uint8)
                        idx_mutation_sub[:int(length*self.mutation_sub_percent)] = 1

                        idx_mutation_sub_c = np.random.permutation(idx_mutation_sub)==1# print("self.arr_x: {}".format(self.arr_x))
        # arr_field = self.arr_x[:self.rows*self.cols].reshape((-1, self.cols)).astype(object)
        # arr_field[arr_field==1.] = '0'
        # arr_field[arr_field==-1.] = ' '
        # print("self.arr_x:\n{}".format(arr_field))
        # input("ENTER...")
                        amount_mutation_sub_vals = np.sum(idx_mutation_sub)
                        a_c[np.random.permutation(idx_mutation_sub_c)] -= (np.random.random((amount_mutation_sub_vals, ))*2.-1.)/10000.


            l_lsnn_new = l_lsnn_new_parents+l_lsnn_new_childs
            assert len(l_lsnn_new)==self.population_size
            self.l_lsnn = l_lsnn_new

            for tgm, l_snn in zip(self.l_tgm, self.l_lsnn):
                tgm.l_snn = l_snn

            l_cecf = [
                [snn.f_cecf(snn.calc_feed_forward(X), T)/T.shape[0]/T.shape[1] for snn, X, T in zip(l_snn, self.Xs, self.Ts)]
                for l_snn in self.l_lsnn
            ]
            l_cecf_sum = [np.sum(l) for l in l_cecf]

            self.l_cecf_all.append(l_cecf)
            self.l_cecf_sum_all.append(sorted(l_cecf_sum))

            print("np.min(l_cecf_sum): {}".format(np.min(l_cecf_sum)))

        self.best_tgm = self.l_tgm[0]
