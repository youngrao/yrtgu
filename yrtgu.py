from datetime import datetime
from math import exp, log, sqrt
from copy import copy

# Workhorse Model
class ftrl_proximal(object):
    def __init__(self, alpha, beta, L1, L2, D, interaction=False):
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.D = D

        self.interaction = interaction
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):

        yield 0
        for index in x:
            yield index

        if self.interaction:
            D = self.D
            L = len(x)
            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        n = self.n
        z = self.z
        w = {}

        wTx = 0.
        for i in self._indices(x):
            sign = -1. if self.z[i] < 0 else 1.
            if sign * z[i] <= L1:
                w[i] = 0
            else:
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)
            wTx += w[i]

        self.w = w

        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        alpha = self.alpha
        n = self.n
        z = self.z
        w = self.w
        g = p - y

        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


# Training Binary Classification/Logistic Regression or Multiclass Classification (One Against All)

class yrtgu(object):
    def __init__(self, alpha, beta, L1, L2, D):
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2
        self.D = D
        self.model = None

    def fit(self, train_path, read_data, epoch, holdout):
        start = datetime.now()

        self.model = ftrl_proximal(self.alpha, self.beta, self.L1, self.L2, self.D)

        for e in xrange(epoch):
            loss = 0.
            count = 0

            for t, x, y in read_data(train_path):
                p = self.model.predict(x)
                if t % holdout == 0:
                    loss += logloss(p, y)
                    count += 1
                else:
                    self.model.update(x, p, y)

                if t % 1000 == 0 and t > 1:
                    print(' %s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), t, loss / count))

            print(
            'Epoch %d finished, holdout logloss: %f, elapsed time: %s' % (e+1, loss / count, str(datetime.now() - start)))

    def fit_multi(self, train_path, read_data, epoch, holdout, numclasses):
        start = datetime.now()

        self.model = []
        for k in range(numclasses):
            self.model.append(copy(ftrl_proximal(self.alpha, self.beta, self.L1, self.L2, self.D)))

        for e in xrange(epoch):
            loss = 0.
            count = 0

            for t, id, x, y in read_data(train_path, train=True):
                p = [[0] for k in range(numclasses)]
                loss_k = 0
                for k in range(numclasses):
                    p[k] = self.model[k].predict(x)
                    loss_k += logloss(p[k], y[k])

                    if t % holdout == 0:
                        if k == numclasses - 1:
                            loss += loss_k
                            count += 1
                    else:
                        self.model[k].update(x, p[k], y[k])

                if t % 10000 == 0 and t > 1:
                    print(' %s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), t, loss / count))

            print('Epoch %d finished, holdout logloss: %f, elapsed time: %s' % (e + 1, loss / count, str(datetime.now() - start)))