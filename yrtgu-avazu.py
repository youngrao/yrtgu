from csv import DictReader
from yrtgu import yrtgu, ftrl_proximal

# A: Paths
train_path = 'train.csv'
test = 'test.csv'

# B: Model Parameters
alpha = .1
beta = 1.
L1 = 1.
L2 = 1.

# C: Interaction/Hash Value
D = 2 ** 20

# D: Training and Validation
epoch = 1
holdout = 10

def data(path):
    for t, row in enumerate(DictReader(open(path))):
        del row['id']
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        row['hour'] = row['hour'][6:]
        x = []
        for key in row:
            value = row[key]
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield t, x, y


# Training
yrtgu = yrtgu(alpha, beta, L1, L2, D)

yrtgu.fit(train_path, data, epoch, holdout)

# Create submission file

def data_sub(path):
    for t, row in enumerate(DictReader(open(path))):
        id=row['id']
        del row['id']
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        row['hour'] = row['hour'][6:]
        x = []
        for key in row:
            value = row[key]
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield t, id, x, y

with open('avazupred.csv', 'w') as outfile:
    outfile.write('id,click\n')
    for t, id, x, y in data_sub(test):
        p = yrtgu.model.predict(x)
        outfile.write('%s,%s\n' % (id, str(p)))