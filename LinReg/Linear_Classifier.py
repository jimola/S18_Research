from sklearn import linear_model
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation

svm_class = svm.SVC(kernel='linear', C=1)
logistic_class = linear_model.LogisticRegression(C=1)
ir = datasets.load_iris()
t0 = np.where(ir.target == 0, 1, 0)
t1 = np.where(ir.target == 2, 1, 0)
t2 = np.where(ir.target == 2, 1, 0)
cross_validation.cross_val_score(svm_class, ir.data, ir.target).mean()

def get_scores_SVC(l, X, y):
	c_scores = []
	for x in l:
		c1 = svm.SVC(kernel='linear', C=x)
		c_scores.append(get_cross_val_score(c1, X, y))
	return c_scores

def get_scores_logistic(l, X, y):
	c_scores = []
	for x in l:
		l1 = linear_model.LogisticRegression(C=x)
		c_scores.append(get_cross_val_score(l1, X, y))
	return c_scores

def get_cross_val_score(clf, X, y, windows=10, epsilon=None):
	s = 0
	w_size = len(X) / windows
	for x in range(0, len(X), w_size):
		X_train = np.concatenate((X[:x], X[x+w_size:]))
		y_train = np.concatenate((y[:x], y[x+w_size:]))
		X_test = X[x:x+w_size]
		y_test = y[x:x+w_size]
		clf.fit(X_train, y_train)
		if(epsilon == None):
			s += clf.score(X_test, y_test)
		else:
			wts = clf.coef_ + np.random.laplace(0, 1.0/epsilon, clf.coef_.shape)
			pred = X_test.dot(wts.T) + clf.intercept_ + np.random.laplace(0, 1.0/epsilon)
			binary = np.where(pred >= 0, 1, 0)
			s += 1 - abs((y_test - binary) * (y_test - binary)).mean()
	return s/windows

Cs = np.arange(0.5, 10, 1)
es = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128])
performance_svm = []
for c in Cs:
	p = np.zeros(len(es))
	clf = svm.SVC(kernel='linear', C=c)
	for x in range(0, 50):
		p += map(lambda x: get_cross_val_score(clf, ir.data, t2, epsilon=x), es)
	performance_svm.append(p/50)
performance_svm = np.array(performance_svm)
performance_logic = []
for c in Cs:
	p = np.zeros(len(es))
	clf = linear_model.LogisticRegression(C=c)
	for x in range(0, 50):
		p += map(lambda x: get_cross_val_score(clf, ir.data, t2, epsilon=x), es)
	performance_logic.append(p/50)
performance_logic = np.array(performance_logic)

#ax = plt.subplot(111)
cx1 = plt.matshow(performance_svm, cmap='Greys')
plt.colorbar(cx1)
cx2 = plt.matshow(performance_logic, cmap='Greys')
plt.colorbar(cx2)

plt.subplot(121)
plt.plot(Cs, performance_svm[:, 3])
plt.plot(Cs, performance_logic[:, 3])
plt.xlabel('C Value')
plt.ylabel('Out-of-Sample Accuracy')
plt.title('Accuracy vs. C value (epsilon=4)')
plt.legend(['SVM', 'Logistic Reg.'], loc='lower right')

ax = plt.subplot(122)
plt.plot(es, performance_svm[-2, :])
plt.plot(es, performance_logic[-2, :])
ax.set_xscale('log')
plt.xlabel('Log Epsilon')
plt.ylabel('Out-of-Sample Accuracy')
plt.title('Accuracy vs. Epsilon (C value=8.5)')
plt.legend(['SVM', 'Logistic Reg.'], loc='lower right')