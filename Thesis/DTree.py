def dtree_helper(DB, depth, budget):
    if(depth = 0 or other things):
        clss_freqs = [noisycount(DB.clss == c)
                for c in DB.clss.attributes]
        return Leaf(pred=argmax(clss_freqs))
    else:
        best_col = choose a column from DB.cols
        L = []
        for attr in DB.best_col.attributes:
            L.append(dtree(DB[best_col==attr], depth-1))
        return Node(children=L)
    return Node(col=best_col, children=C, budget)
def dtree(DB, num_trees, d, u, budget):
    forest = [dtree_helper(DB, d, u, budget/nt) 
                for i in range(1, num_trees)]
    return forest

def do_leaf():
best_class = max over c in clss of
      noisyCount(len(D[clss=c, ]), e/2)
    return Leaf(pred=best_class, size=size)
def do_branch():
    U = [-c_entropy(a, clss, D) for a in attrs]
    best_att = exp_mech(domain=attrs, utilities=U, epsilon=e/2)
    C = [dtree_private(D[best_att=a, ], att-best_att, clss, d-1, e)
            for a in best_att.categories)
    return Node(att=best_att, children=C)

def get_features(db):
    return noisyCount(db), len(db.y.categories), eps

branch = MkChoiceMaker {do_leaf, do_branch} {Public Datasets} {
def dtree_private(D, atts, clss, d, e):
  t = max over a in atts of len(a)
  size = noisyCount(D, e/2)
  branch(D, atts, clss, e)
  #if(d = 0 or len(atts) = 0 or size/(t*len(clss)) < sqrt(2)/e):
  #else:

def dtree_pvt_top(D, atts, clss, d, budget):
  t = dtree_private(D, atts, clss, d, budget/(d+1))
  return C4.5_Prune(t)
