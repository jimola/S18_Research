def dtree(D, atts, clss, d):
  if(d = 0 or len(atts) = 0):
    best_class = max over c in clss of len(D[clss=c, ])
    return Leaf(pred=best_class)
  else:
    best_att = max over A in atts of:
      c_entropy(D, A, clss)
    C = map(best_att, lambda a: dtree(D[best_att=a, ],
      att-best_att, clss, d-1))
    return Node(att=best_att, children=C)

def dtree_private(D, atts, clss, d, e):
  t = max over a in atts of len(a)
  size = noisyCount(D, e/2)
  if(d = 0 or len(atts) = 0 or size/(t*len(clss)) < sqrt(2)/e):
    best_class = max over c in clss of
      noisyCount(len(D[clss=c, ]), e/2)
    return Leaf(pred=best_class, size=size)
  else:
    U = map(attrs, lambda a: -c_entropy(a, clss, D))
    best_att = exp_mech(domain=attrs, utilities=U, epsilon=e/2)
    C = map(best_att, lambda a: dtree_private(D[best_att=a, ],
      att-best_att, clss, d-1, e))
    return Node(att=best_att, children=C)

def dtree_pvt_top(D, atts, clss, d, budget):
  t = dtree_private(D, atts, clss, d, budget/(d+1))
  return C4.5_Prune(t)