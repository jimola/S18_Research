get_guess <- function(ys, eps){
  m <- data.frame(table(ys))
  counts <- hist_noiser(m$Freq, eps)
  idx <- which.max(counts)
  guess <- m$ys[idx]
  return(guess)
}


dtree_helper_nt <- function(dataset, test, params, alg){
  data <- dataset$data
  x_names <- dataset$x_names
  y_names <- dataset$y_names
  rng <- dataset$rng
  ys <- get(y_names, data)
  params <- alg$update(list(), dataset, params)
  if(params$action == 'stop'){
    g <- get_guess(ys, params$epsilon)
    return(rep(g, nrow(test)))
  }
  atp <- lapply(x_names, function(name) cond_eval(name, data, rng, ys))
  utilities <- sapply(atp, function(x) -x$ent)
  params$attr_best <- exp_mech(atp, utilities, params$node_qb, ent_util$sens)
  params$epsilon <- params$epsilon-params$node_qb
  pars <- split(params$attr_best)
  col <- get(params$attr_best$name, data)
  col2 <- get(params$attr_best$name, test)
  dataset$x_names <- x_names[x_names != params$attr_best$name]
  preds <- rep((get(dataset$y_names, test) %>% levels %>% as.factor)[[1]], nrow(test))
  for(fn in pars){
    subset <- (fn$fn)(col)
    subset2 <- (fn$fn)(col2)
    if(sum(subset2) != 0){
      dataset$data <- data[subset, ]
      preds[subset2] <- dtree_helper_nt(dataset, test[subset2, ], params, alg)
    }
  }
  return(preds)
}

dtree_outer_nt <- function(dataset, dep, epsilon, alg){
  test <- dataset$test$data
  dataset <- dataset$train
  params <- alg$init(dataset, epsilon, dep)
  if(is.null(params$epsilon))
    params$epsilon <- epsilon
  if(is.null(params$dep))
    params$dep <- dep
  if(is.null(params$node_qb))
    params$node_qb <- 0
  if(is.null(params$num_trees))
    params$num_trees <- 1
  L <- sapply(1:params$num_trees,
    function(...){
      return(dtree_helper_nt(dataset, test, params, alg))
    })
  return(L)
}

#Don't think about pruning for now
#Let's test each algorithm and keep track of critical values of the dataset.

#get_pts <- function(dataset, dep, epsilon, algs){
