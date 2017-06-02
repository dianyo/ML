import numpy as np
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neighbors import NearestNeighbors
import sys
test_path = sys.argv[1]
ans_path = sys.argv[2]
def get_eigenvalues(data):
    SAMPLE = 1 # sample some points to estimate
    NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

# Train a linear SVR
print('start load npz file')
npzfile = np.load('large_data.npz')
X = npzfile['X']
y = npzfile['y']

np.random.seed(1)
# tmp = np.arange(300)
# np.random.shuffle(tmp)

# X = X[tmp]
# y = y[tmp]

# val_X = X[:30]
# val_y = y[:30]
# train_X = X[30:]
# train_y = y[30:]

print('loaded train data')

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=1000, epsilon=0.2, gamma=5)
svr.fit(X, y)
# pred_val_y = np.rint(svr.predict(val_X))
# dif = val_y[val_y != pred_val_y]
# print(len(dif))
# sys.exit(0)
# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

# predict
testdata = np.load(test_path)
print('loaded data')
test_X = []
for i in range(200):
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)
pred_y = np.rint(pred_y)
print(pred_y)
with open(ans_path, 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
    	f.write(str(i) + ',' + str(np.log(d)) + '\n')