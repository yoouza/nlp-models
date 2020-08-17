import numpy as np

class XOR:    
    def _init_weight(self, X, N_hidden):
        w_input = np.random.random_sample((len(X)+1, N_hidden))
        w_hidden = np.random.random_sample(N_hidden)
        w_b = np.random.random_sample()
        return w_input, w_hidden, w_b

    def _loss(self, Y, Y_hat):
        N = len(Y)
        loss = (-1/N) * np.sum( np.multiply(Y, np.log(Y_hat)) + np.multiply(1-Y, np.log(1-Y_hat)) )
        return loss
    
    def _sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def optimize(self, X, Y, N_hidden, alpha, epoch):
        
        w_input, w_hidden, w_b = self._init_weight(X, N_hidden)

        b = np.ones(len(X[0]))
        X_b = np.concatenate([X, b.reshape(1, len(b))])

        for i in range(epoch):
            h = self._sigmoid(np.dot(w_input.T, X_b))
            Y_hat = self._sigmoid(np.dot(w_hidden, h) + b * w_b)
            loss = self._loss(Y, Y_hat)

            # output -> hidden
            gradient_hidden = np.dot(h, Y_hat-Y)
            gradient_bias = Y_hat-Y

            # hidden -> input
            gradient_h = np.dot(w_hidden.reshape(len(w_hidden), 1), (Y_hat-Y).reshape(1, len(Y)))
            gradient_input = np.dot(X_b, np.multiply(gradient_h, h, 1-h).T)

            # update weights
            w_hidden = w_hidden - alpha * gradient_hidden
            b = b - alpha * gradient_bias
            w_input = w_input - alpha * gradient_input

        return (Y_hat > 0.5) * 1, Y_hat

if __name__ == "__main__":
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([0, 1, 1, 0])

    xor = XOR()
    pred, y_hat = xor.optimize(X, Y, 5, alpha = .01, epoch = 10000)
    print('pred:', pred)
    print('y_hat:', y_hat)