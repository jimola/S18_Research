library(parallel)
library(class)

predict <- function(t, D, default){
    preds <- rep(default, nrow(D))
    if(nrow(D) == 0)
        return(preds) 
    if(!is.null(t$guess)){
        return(rep(t$guess, nrow(D)))
    }
    for(i in 1:length(t$pars)){
        mask <- t$pars[[i]]$fn(get(t$attr, D))
        preds[mask] <- predict(t$children[[i]], D[mask, ], default)
    }
    return(preds)
}

split_data <- function(D, cutoff=0.8){
    cutoff <- as.integer(nrow(D$data)*cutoff)
    D1 <- D
    D2 <- D
    D1$data <- D$data[1:cutoff, ]
    D2$data <- D$data[(cutoff+1):nrow(D$data), ]
    return(list(train=D1, test=D2))
}
num_matching <- function(D, y_hat){
  ys <- get(D$y_names, D$data)
  return(sum(y_hat == ys) / length(ys))
}
test <- function(D, B, dep, dt_gen){
    dt <- dt_gen(D$train, B, dep)
    ys <- get(D$test$y_names, D$test$data)
    default <- ys[1]
    p <- predict(dt, D$test$data, default)
    return(sum(p == ys) / length(ys))
}
test_tree <- function(t, D){
    ys <- get(D$y_names, D$data)
    default <- ys[1]
    p <- predict(t, D$data, default)
    return(sum(p == ys) / length(ys))
}
predict_maj <- function(ts, D){
  ys <- get(D$y_names, D$data)
  default <- ys[1]
  p <- sapply(ts, function(t) predict(t, D$data, default))
  N <- nrow(D$data)
  pred <- rep(default, N)
  freq <- integer(N)
  for(clss in levels(ys)){
    votes <- rowSums(p == clss)
    pred[votes > freq] <- clss
    freq[votes > freq] <- votes[votes > freq]
  }
  return(pred)
}
get_data <- function(dset, eps_test, predictor, description, d=5, reps=10){
  G <- mcmapply(function(...) sapply(eps_test, function(e){
          t <- dtree_outer(dset$train, d, e, predictor)
          return(test_tree(t, dset$test))
        }), 1:reps)
  return(data.frame(eps=eps_test, performance=rowMeans(G), desc=description))
}
get_plot <- function(df) ggplot(df, aes(x=eps, y=performance, color=desc))+geom_line()

get_leaf_node <- function(ys, node, eps){
    m <- data.frame(table(ys))
    node$counts <- hist_noiser(m$Freq, eps)
    idx <- which.max(node$counts)
    guess <- m$ys[idx]
    node$guess <- guess
    node$name <- paste(node$name, guess, sep=';')
}

split <- function(attr){
    if(attr$type == 'numeric'){
        s <- attr$split
        return(c(list(name=paste('<=', s), fn=function(d){return(d <= s)}),
                 list(name=paste('>', s), fn=function(d){return(d > s)})))
    }else
        return(lapply(attr$split, function(a) list(name=a, fn=equality(a))))
}

get_sigma <- function(data, attrs, y_name){
    sizes <- lapply(data, function(col){
        if(class(col) == 'numeric')
            return(2)
        return(length(levels(col)))
    })
    t <- sizes[attrs] %>% reduce(`max`)
    C <- get(y_name, sizes)
    return(C*t*sqrt(2))
}

generate_data_helper <- function(attr_names, class, attr_sizes, cur_val, nrow, d_continue, p_stop){
    attr_sz <- 0
    attr <- 0
    if(attr_names %>% length > 0){
        attr <- sample(attr_names, 1)
        attr_sz <- get(attr, attr_sizes)
    }
    if(d_continue <= 0 && runif(1) < p_stop || attr_sz > nrow || attr==0){
        cur_val[class] <- runif(1, max=class %>% get(attr_sizes)) %>% as.integer
        df <- lapply(cur_val, function(x) rep(x, nrow)) %>% data.frame
        L <- lapply(attr_names, function(x) runif(nrow, max=x %>% get(attr_sizes)) %>% as.integer)
        names(L) <- attr_names
        df2 <- data.frame(L)
        if(length(df2) > 0)
            df <- cbind(df, df2)
        return(df)
    }
    D <- data.frame()
    attr_names <- attr_names[attr_names != attr]
    r <- 1+rmultinom(1, nrow-attr_sz, integer(attr_sz)+1)
    for(i in 1:attr_sz){
        c <- cur_val
        c[attr] = i-1
        D <- rbind(D, generate_data_helper(attr_names, class, attr_sizes, c, r[i], d_continue-1, p_stop))
    }
    return(D)
}
generate_data <- function(attr_names, class, attr_sizes, nrow, d_continue, p_stop=0.3){
    D <- generate_data_helper(attr_names, class, attr_sizes, list(), nrow, d_continue, p_stop)
    D <- D[sample(1:nrow(D), nrow(D)), ] %>% lapply(function(x) as.factor(x)) %>% data.frame
    return(list(data=D, x_names=attr_names, y_names=class, rng=NA))
}      

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)

  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)

  numPlots = length(plots)

  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                    ncol = cols, nrow = ceiling(numPlots/cols))
  }

 if (numPlots==1) {
    print(plots[[1]])

  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))

    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))

      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
#params needs to have epsilon, dep, node_qb
dtree_helper <- function(t, dataset, params, alg){
    data <- dataset$data
    x_names <- dataset$x_names
    y_names <- dataset$y_names
    rng <- dataset$rng
    ys <- get(y_names, data)
    params <- alg$update(t, dataset, params)
    if(params$action == 'stop'){
        get_leaf_node(ys, t, params$epsilon)
        return()
    }
    atp <- lapply(x_names, function(name) cond_eval(name, data, rng, ys))
    utilities <- sapply(atp, function(x) -x$ent)
    params$attr_best <- exp_mech(atp, utilities, params$node_qb, ent_util$sens)
    params$epsilon <- params$epsilon-params$node_qb
    t$name <- paste(t$name, params$attr_best$name, sep=';')
    t$pars <- split(params$attr_best)
    t$attr <- params$attr_best$name
    col <- get(params$attr_best$name, data)
    dataset$x_names <- x_names[x_names != params$attr_best$name]
    for(fn in t$pars){
        subset <- (fn$fn)(col)
        dataset$data <- data[subset, ]
        dtree_helper(t$AddChild(fn$name), dataset, params, alg)
    }
}

dtree_outer <- function(dataset, dep, epsilon, alg){
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
        dt <- Node$new('')
        dtree_helper(dt, dataset,  params, alg)
        if(params$collect_size){
            prune_tree(dt, dataset$y_name %>% get(dataset$data) %>% levels)
        }
        return(dt)
      })
    return(L)
}
#Friedman and Schuster
alg_1 <- list(
  init=function(dataset, epsilon, dep){
    B <- epsilon/(2*(dep+1))
    sigma <- get_sigma(dataset$data, dataset$x_names, dataset$y_names) / B
    return(list(collect_size=TRUE, node_qb=B, 
                dep=dep, num_trees=1, sigma=sigma))
  },
  update=function(t, dataset, params){
    t$nrows <- hist_noiser(nrow(dataset$data), params$node_qb)
    params$action <- 'stop'
    if(t$nrows >= params$sigma && params$dep > 0)
      params$action <- 'exp'
    params$dep <- params$dep-1
    return(params)
  }
)

#Mohammed et al.
alg_2 <- list(
  init=function(dataset, epsilon, dep){
    B <- epsilon/(dep+1)
    return(list(collect_size=FALSE, node_qb=B, num_trees=1, dep=dep))
  },
  update=function(t, dataset, params){
    params$action <- 'stop'
    if(params$dep > 0)
      params$action <- 'exp'
    params$dep <- params$dep-1
    return(params)
  }
)

#Jagannathan et al.
alg_3 <- list(
  init=function(dataset, epsilon, dep){
    b <- sapply(dataset$data[dataset$x_names], function(r) levels(r) %>% length) %>% mean
    dep_branch <- log(dataset$data %>% length) / log(b) - 1
    dep <- min(length(dataset$x_names)/2, dep_branch) %>% as.integer
    return(list(collect_size=FALSE, dep=dep, num_trees=10, epsilon=epsilon/10))
  },
  update=function(t, dataset, params){
    params$action <- 'stop'
    if(params$dep > 0)
      params$action <- 'exp'
    params$dep <- params$dep-1
    return(params)
  }
)

#Patil & Singh
alg_4 <- list(
  init=function(dataset, epsilon, dep){
    params <- alg_1$init(dataset, epsilon, dep)
    params$num_trees <- 4
    return(params)
  },
  update=function(t, dataset, params){
    return(alg_1$update(t, dataset, params))
  }
)

#Fletcher & Islam
alg_5 <- list(
  init=function(dataset, epsilon, dep){
    C <- get(dataset$y_names, dataset$data) %>% levels %>% length
    return(list(collect_size=FALSE, nrow=length(dataset$data), node_qb=0, epsilon=epsilon,
                stop_pt=sqrt(2)*C / epsilon, num_trees=5))
  },
  update=function(t, dataset, params){
    if(!is.null(params$attr_best)){
      S <- get(params$attr_best$name, dataset$data) %>% levels %>% length
      params$nrow <- params$nrow / S
    }
    if(params$nrow < params$stop_pt)
      params$action <- 'stop'
    else
      params$action <- 'exp'
    return(params)
  }
)
