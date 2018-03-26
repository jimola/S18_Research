import JeevesLib

def gen1(var=1):
	while(True):
		yield np.random.normal(0, var)

def gen2(var=1):
	while(True):
		yield np.random.laplace(0, var)


def mkQueryable(dataset):
	cnt = 0
	data_new = np.zeros(len(dataset))
	while(cnt < len(dataset)):
		r =  np.random.randint(1, 5)
		cnt_next = min(cnt+r, len(dataset))
		avg = np.mean(dataset[cnt:cnt_next])
		for x in xrange(cnt, cnt_next):
			data_new[x] = avg
		cnt = cnt_next
	return data_new

class Query:
	def __init__(self, data):
		self.data = data
	def query(self, a, b):
		return self.data[a:b].sum()

dataset = np.random.random(100)
q_true = Query(dataset)
fuzz1 = Query(mkQueryable(dataset))
fuzz2 = Query(mkQueryable(dataset))

def sample(model, actual):
	l = len(dataset)
	error = 0
	for x in xrange(0, 100):
		r1 = np.random.randint(0, l)
		r2 = np.random.randint(0, l)
		while(r1 != r2):
			r1 = np.random.randint(0, l)
			r2 = np.random.randint(0, l)
		if(r1 > r2):
			temp = r2
			r2 = r1
			r1 = temp
		q_model = model.query(r1, r2)
		q_true = actual.query(r1, r2)
		error += abs(q_model - q_true) / q_true
	return error

acc = JeevesLib.mkLabel()
JeevesLib.restrict(acc, lambda x: sample(x) < 0.5)
