NoisyBranch <- function(learner, params){return TRUE} #Stub

dtree_outer <- function(data, attrs, y_name, D_ranges, dep, epsilon){
  dt <- 0
  if(NoisyBranch(L1)){
    dt <- dtree_helper(data, attrs, y_name, D_ranges, dep, epsilon/(2*dep), TRUE)
    prune(dt)
  }else{
    dt <- dtree_helper(data, attrs, y_name, D_ranges, dep, epsilon/(dep), FALSE)
  }
  return(dt)
}

dtree_helper <- function(data, attrs, y_name, D_ranges, dep, epsilon, collect_size){
  dsize <- Inf
  if(collect_size || NoisyBranch(L2)){
    dsize <- noisy_count(data)
  }
  if(dep == 0 || NoisyBranch(L3)){ #May insert NoisyBranch Here
    most_common_class <- ReportNoisyMax(data, epsilon)
    return(Tree$Leaf(size=dsize))
  }
  attr_best <- 0
  if(NoisyBranch(L4)){
    attr_best <- exponential_mechanism(attrs, data, entropy_utility)
  }else{
    attr_best <- random(attrs)
  }
  data_splits <- split(data, attr_best)
  chi <- c()
  for(d in data_splits){
    chi <- c(chi, dtree_helper(d, attrs[-attr_best], y_name, D_ranges, dep-1, epsilon, 
                               collect_size))
  }
  return(Tree$Node(children=chi, size=dsize))
}
