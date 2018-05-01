import pickle
def load(name):
    G = pickle.load(open(name, 'rb'))
    df = pd.DataFrame()
    for g in G:
        df = df.append(g[1].groupby(['eps', 'database', 'lnrow', 'ldomsize', 'nclss', 'unif']).mean())
    df.reset_index(inplace=True)
    return df

def get_best_alg(df):
    best_alg = np.repeat('fs5', len(df))
    best_perf = df.fs5
    mean = (df.fs5 + df.ma5 + df.db1 + df.db3 + df.db7)/5
    for c in ['ma5', 'db1', 'db3', 'db7']:
        perf = df[c]
        best_alg[perf > best_perf] = c
        best_perf[perf > best_perf] = perf[perf > best_perf]

    best_alg_stubbed = best_alg
    best_alg_stubbed[best_perf - mean < 0.03] = 'NA'
    df['best'] = best_perf
    df['best_alg'] = best_alg_stubbed
    return df

t_data = get_best_alg(load('ttt.p'))
b_data = get_best_alg(load('bind.p'))
n_data = get_best_alg(load('nurs.p'))
l_data = get_best_alg(load('loan.p'))

c_dict = {'ma5': 'green', 'db1': 'orange', 'db3': 'yellow', 'db7': 'red', 'fs5': 'blue', 'NA': 'gray'}
def get_scatter(r):
    plt.scatter(r.lnrow, r.eps, c=to_color(r.best_alg, c_dict))
def to_color(r, color_dict):
    for c in color_dict:
        r[r == c] = color_dict[c]
    return r

get_scatter(t_data)
get_scatter(b_data)
get_scatter(n_data)
get_scatter(l_data)
