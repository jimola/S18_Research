import sklearn
np.random.seed(200)
num_pts = 240
nattr = 5
Xs = np.random.normal(0, 10, (num_pts, nattr))
intercept = np.random.normal(0, 1)
wts = np.random.normal(0, 1, 5)
ys = intercept + np.dot(Xs, wts) + np.random.normal(0, 30, num_pts)

cutoff = (3*num_pts)/4
Xs_train = Xs[:cutoff, :]
Xs_test = Xs[cutoff:, :]
ys_train = ys[:cutoff]
ys_test = ys[cutoff:]

def get_scores(L):
	for a in L:
		clf = sklearn.linear_model.Ridge(alpha=a)
		clf.fit(Xs_train, ys_train)
		print(clf.score(Xs_test, ys_test))