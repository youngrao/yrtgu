from csv import DictReader
from yrtgu import yrtgu

# A: Paths
train_path = 'trainOH.csv'
label = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
pred = 'yrtgusub.csv'
test = 'test.csv'

# B: Model Parameters
alpha = .1
beta = 1.
L1 = 1.
L2 = 1.
numclasses = 9

# C: Interaction/Hash Value
D = 2 ** 20

# D: Training and Validation
epoch = 1
holdout = 100

# Data Generator
def data(path, train = False):
    for t, row in enumerate(DictReader(open(path))):
        if train:
            y = []
            for target in label:
                y.append(float(row[target]))
                del row[target]
        id = row['id']
        x = [0]
        for key in sorted(row):
            value = row[key]
            index = abs(hash(key + '_' + value)) % D
            x.append(index)
        if train:
            yield t, id, x, y
        else:
            yield t, id, x

# Training - Multi-Class Classification (One Against All)

yrtgu = yrtgu(alpha, beta, L1, L2, D)

yrtgu.fit_multi(train_path, data, epoch, holdout, numclasses)

# Generate Predictions

with open(pred, 'w') as outfile:
    outfile.write('id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n')
    for t, id, x, in data(test, train=False):
        outfile.write('%s,' % id)
        p = [[0] for k in range(numclasses)]
        for k in range(numclasses):
            p[k] = yrtgu.model[k].predict(x)
        p = [x / sum(p) for x in p]
        for k in range(numclasses):
            if k < 8:
                outfile.write('%s,' % str(p[k]))
            else:
                outfile.write('%s\n' % str(p[k]))
