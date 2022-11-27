import numpy as np

from copy import deepcopy

class SimpleNeuralNetwork(Exception):
    def __init__(self, input_nodes=None, hidden_nodes=None, output_nodes=None, bws=None, rnd=None):
        if bws is not None:
            self.bws = deepcopy(bws)
            return

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.eta = 0.001
        self.learning_epochs = 100

        self.layers = [input_nodes]+hidden_nodes+[output_nodes]
        self.nl = self.layers

        self.bws = self.get_random_bws(rnd=rnd)


    def get_random_bws(self, rnd=None):
        if rnd is None:
            return np.array([np.random.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)) for m, n in zip(self.nl[:-1], self.nl[1:])], dtype=object)

        return np.array([rnd.uniform(-1./np.sqrt(n), 1./np.sqrt(n), (m+1, n)) for m, n in zip(self.nl[:-1], self.nl[1:])], dtype=object)
    

    def __repr__(self):
        return f"SimpleNeuralNetwork(ni={self.input_nodes}, nh={self.hidden_nodes}, on={self.output_nodes}, "+ \
            f"eta={self.eta}, learning_epochs={self.learning_epochs})"


    def calc_output_argmax(self, input_vec):
        x = np.hstack(([1], input_vec))
        for w in self.bws[:-1]:
            z = np.dot(x, w)
            x = self.f1(z)
            x = np.hstack(([1], x))
        z = np.dot(x, self.bws[-1])
        y = self.f2(z)
        return np.argmax(y)


    def calc_grad_numeric(self, X, T, epsilon=0.0001, softmax=False):
        ws_grad = [np.empty(w.shape) for w in self.bws]

        for bw, bw_g in zip(self.bws, ws_grad):
            rows, cols = bw.shape
            for j in range(0, rows):
                for i in range(0, cols):
                    v = bw[j, i]

                    bw[j, i] = v+epsilon
                    Y = self.calc_Y(X, softmax=softmax)
                    cecf_plus = self.f_cecf(Y, T) # /T.shape[0]/T.shape[1]

                    bw[j, i] = v-epsilon
                    Y = self.calc_Y(X, softmax=softmax)
                    cecf_minus = self.f_cecf(Y, T) # /T.shape[0]/T.shape[1]

                    v_grad = (cecf_plus-cecf_minus) / 2. / epsilon
                    bw[j, i] = v
                    bw_g[j, i] = v_grad

        return ws_grad


    def calc_Y(self, X, softmax=False):
        ones = np.ones((X.shape[0], 1))

        X = np.hstack((ones, X))
        for bw in self.bws[:-1]:
            Z = np.dot(X, bw)
            Y = self.f1(Z)
            X = np.hstack((ones, Y))
        Z = np.dot(X, self.bws[-1])
        Y = self.f2(Z)
        if softmax:
            Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        return Y


    def calc_feed_forward(self, X):
        ones = np.ones((X.shape[0], 1))
        Y = X
        for i, bw in zip(range(len(self.bws), 0, -1), self.bws[:-1]):
            Y = self.f1(np.hstack((ones, Y)).dot(bw)) # *self.hidden_multiplier**i)
        Y = self.f2(np.hstack((ones, Y)).dot(self.bws[-1]))
        return Y


    def calc_backprop(self, X, T, bws, softmax=False):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in bws[:-1]:
            A = np.hstack((ones, Y)).dot(bw)
            Xs.append(A)
            Y = self.f1(A)
            Ys.append(Y)
        A = np.hstack((ones, Y)).dot(bws[-1]); Xs.append(A)
        Y = self.f2(A)
        if softmax:
            Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        Ys.append(Y)

        d = (Y-T)
        bwds = np.array([np.zeros(bwsdi.shape) for bwsdi in bws], dtype=object)
        if softmax:
            bwds[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        else:
            bwds[-1] = np.hstack((ones, Ys[-2])).T.dot(d)

        length = len(bws)
        for i in range(2, length+1):
            d = d.dot(bws[-i+1][1:].T)*self.f1_d(Xs[-i])
            bwds[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return np.array(bwds)


    def calc_backprop_own_bws(self, X, T, softmax=False):
        Xs = []
        Ys = [X]
        ones = np.ones((X.shape[0], 1))
        Y = X
        for bw in self.bws[:-1]:
            A = np.hstack((ones, Y)).dot(bw)
            Xs.append(A)
            Y = self.f1(A)
            Ys.append(Y)
        A = np.hstack((ones, Y)).dot(self.bws[-1]); Xs.append(A)
        Y = self.f2(A)
        if softmax:
            Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        Ys.append(Y)

        d = (Y-T)
        bwds = np.array([np.zeros(bwsdi.shape) for bwsdi in self.bws])
        if softmax:
            bwds[-1] = np.hstack((ones, Ys[-2])).T.dot(d)
        else:
            bwds[-1] = np.hstack((ones, Ys[-2])).T.dot(d)

        length = len(self.bws)
        for i in range(2, length+1):
            d = d.dot(self.bws[-i+1][1:].T)*self.f1_d(Xs[-i])
            bwds[-i] = np.hstack((ones, Ys[-i-1])).T.dot(d)

        return np.array(bwds)


    def f_rmse(self, Y, T):
        return np.sqrt(np.mean(np.sum((Y-T)**2, axis=1)))
    def f_cecf(self, Y, T):
        return np.sum(np.sum(np.vectorize(lambda y, t: -np.log(y) if t==1. else -np.log(1-y))(Y, T), axis=1))


    def f1(self, x):
        return np.tanh(x)
    def f1_d(self, x):
        return 1 - np.tanh(x)**2

    def f2(self, x):
        return 1 / (1 + np.exp(-x))
    def f2_d(self, x):
        return self.f2(x)*(1-self.f2(x))


def simple_test_for_neural_network_class():
    # do some testing for the backpropagation function!

    input_nodes = 5
    output_nodes = 3

    snn = SimpleNeuralNetwork(input_nodes=input_nodes, hidden_nodes=[7, 6], output_nodes=output_nodes)

    amount = 2000

    X = np.random.randint(0, 2, (amount, input_nodes)).T
    # a somehow simple, but again complex, target array
    T = np.vstack(((X[0]^X[3]+X[4])%2, (X[1]^X[2])&(X[2]|X[3]), (X[3]|X[4])&X[0])).T
    X = X.T

    snn.ws = snn.get_random_bws()

    X[X==0] = -1

    X = X.astype(np.float)
    T = T.astype(np.float)

    X_part = X[:40]
    T_part = T[:40]

    bws_prev = deepcopy(snn.bws)
    bwds_grad_numeric_nosoftmax = snn.calc_grad_numeric(X_part, T_part, softmax=False)
    bws_now = deepcopy(snn.bws)
    # should not change, when the numerical gradiant calculation is done!
    assert all([np.all(w_p==w_n) for w_p, w_n in zip(bws_prev, bws_now)])

    bwds_grad_true_nosoftmax = snn.calc_backprop(X_part, T_part, snn.bws, softmax=False)

    # should be the same calculated gradient! should be the same for all biases+weights layers!
    assert np.sum([np.sum(np.abs(bwd_n-bwd_t)<0.001) for bwd_n, bwd_t in zip(bwds_grad_numeric_nosoftmax, bwds_grad_true_nosoftmax)])==np.sum([bw.shape[0]*bw.shape[1] for bw in snn.bws])

    # test for a simple backpropagation example!
    # add some noise to the X matrix!
    X += np.random.uniform(-1./100, 1./100, X.shape)

    split_n = int(X.shape[0]*0.7)
    X_train = X[:split_n]
    X_test = X[split_n:]
    T_train = T[:split_n]
    T_test = T[split_n:]

    l_cecf_train = [snn.f_cecf(snn.calc_feed_forward(X_train), T_train)/T_train.shape[0]/T_train.shape[1]]

    eta = 0.0001
    epochs = 100
    for i in range(1, epochs+1):
        ws_grad = snn.calc_backprop(X_train, T_train, snn.bws)
        ws_new = [w-wg*eta for w, wg in zip(snn.bws, ws_grad)]
        snn.bws = ws_new
        Y_train = snn.calc_feed_forward(X_train)
        cecf_train = snn.f_cecf(Y_train, T_train)/T_train.shape[0]/T_train.shape[1]
        # print("i: {:4}, cecf_train: {:06f}".format(i, cecf_train))
        if cecf_train >= l_cecf_train[-1]:
            l_cecf_train.append(cecf_train)
            # print("Stop!! cecf_train >= l_cecf_train[-1]!")
            break
        l_cecf_train.append(cecf_train)

    # here the train cecf should be smaller (hopefully)
    print("Last train cecf: l_cecf_train[-1]: {}".format(l_cecf_train[-1]))

    Y_test = snn.calc_feed_forward(X_test)
    cecf_test = snn.f_cecf(Y_test, T_test)/T_test.shape[0]/T_test.shape[1]

    print("cecf_test : {}".format(cecf_test))


simple_test_for_neural_network_class()
