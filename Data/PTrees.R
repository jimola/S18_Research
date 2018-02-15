library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(DT)
library(tidyr)
library(corrplot)
library(leaflet)
library(lubridate)
library(data.tree)
library(purrr)

inf <- 99999
ent_util <- list(func=function(data){
    return(
        function(col){
            -get_best_val(get(col, data$xs), data$ys, col)$ent
        }
    )
}, sens=(log(1000)+1)/(1000*log(2)))

#Two functions for computing exponential mech.
exp_mech2 <- function(data, domain, util, eps){
    n <- length(domain)
    wghts <- exp(eps*sapply(domain, util$func(data)) / (2*util$sens))
    prob <- wghts / sum(wghts)
    u <- runif(1)
    i <- min(n, findInterval(u, cumsum(prob))+1)
    return(domain[[i]])
}
exp_mech <- function(domain, util, eps, sens=1){
    if(eps == 0){
        return(domain[util==max(util)][[1]])
    }
    util <- util-max(util)
    n <- length(domain)
    wghts <- exp(eps*util / (2*sens))
    if(sum(wghts) < 0.001){
        return(exp_mech(domain, util, 0, sens))
    }
    prob <- wghts / sum(wghts)
    u <- runif(1)
    i <- min(n, findInterval(u, cumsum(prob))+1)
    return(domain[[i]])
}

#Entropy function
get_ent <- function(r){
    if(!is.matrix(r)){
        r <- matrix(r, nrow=1)
    }
    r <- r / rowSums(r)
    ent <- -r*log(r) / log(2)
    ent[is.nan(ent)] = 0
    return(rowSums(ent))
}
#Computes conditional utility of a real attribute as well as the value to split on
cond_eval_num <- function(xs, ys, uti=get_ent, eps=0, min=0, max=100){
    if(length(ys) == 0){
        return(list(ent=inf, split=runif(1, min, max), type='numeric'))
    }
    xs_ord <- order(xs)
    ys_ord <- ys[xs_ord]
    ind <- matrix(sapply(levels(ys), 
                         function(x){cumsum(as.integer(ys_ord == x))}), ncol=length(levels(ys)))
    ind2 <- -sweep(ind, 2, ind[nrow(ind), ])
    changed <- xs[xs_ord] != shift(xs[xs_ord], type='lead')
    changed[is.na(changed)] = TRUE
    ind <- subset(ind, changed)
    ind2 <- subset(ind2, changed)
    d <- nrow(ind)
    ents <- c(uti(ind[d, ]),  (uti(ind) * 1:d + uti(ind2) * (d-1):0)/d)
    if(eps == 0){
        split_ind <- which(ents == min(ents))[[1]]
        return(list(ent=ents[split_ind], split=xs[xs_ord[changed][split_ind]], type='numeric'))
    }
    split_ind <- exp_mech(1:length(ents), -ents, eps, ent_util$sens)
    if(split_ind != 1)
        min = xs[xs_ord[changed][split_ind-1]]
    if(split_ind != d+1)
        max = xs[xs_ord[changed][split_ind]]
    s <- runif(1, min, max)
    return(list(ent=ents[split_ind], split=s, type='numeric'))
}
#Computes conditional utility of a factor variable
cond_eval_fac <- function(xs, ys, uti=get_ent){
    t <- table(xs, ys)
    ents <- uti(table(xs, ys)) * rowSums(t)
    ent <- sum(ents) / sum(t)
    if(is.nan(ent))
        ent <- inf
    return(list(ent=ent, split=levels(xs), type='factor'))
}

#An equality functional
equality <- function(a){
    function(d){
        d == a
    }
}
#Returns a non-private decision tree (helper)
decision_tree <- function(data, pred, attrs, node, d){
    p <- get(pred, data)
    if(length(attrs) == 0 || d <= 0 || get_ent(table(p)) == 0){
        m <- data.frame(table(p))
        idx <- which.max(m$Freq)
        guess <- m$p[idx]
        node$guess <- guess
        node$name <- paste(node$name, guess, sep=';')
        return()
    }
    best_attr <- list(ent=inf)
    for(c in attrs){
        xs <- get(c, data)
        e <- 0
        if(class(xs) == 'factor')
            e <- cond_eval_fac(xs, p)
        else if(class(xs) == 'numeric')
            e <- cond_eval_num(xs, p)
        e$name <- c
        if(e$ent < best_attr$ent){
            best_attr <- e
        }
    }
    node$name <- paste(node$name, best_attr$name, sep=';')
    ba <- best_attr$name
    nl <- attrs[attrs!= ba]
    if(best_attr$type == 'numeric'){
        s1 <- get(ba, data) < best_attr$split
        s <- best_attr$split
        decision_tree(data[s1,], pred, nl,  node$AddChild(paste("<=", s, sep='')), d-1)
        decision_tree(data[!s1,], pred, nl, node$AddChild(paste(">", s, sep='')), d-1)
        node$pars <- c(function(d){return(d <= s)}, function(d){return(d > s)})
    }else if(best_attr$type == 'factor'){
        p <- get(ba, data)
        dec_f <- lapply(best_attr$split, equality)
        for(a in best_attr$split){
            decision_tree(data[p == a,], pred, nl, node$AddChild(a), d-1)
        }
        node$pars <- dec_f
    }
    node$attr <- ba
}
#Returns a decision tree (call this one)
dtree <- function(db, B=0, d=3){
    dt <- Node$new('')
    decision_tree(db$data, db$y_names, db$x_names, dt, d)
    return(dt)
}

#Returns Laplacian Noise
laplacian <- function(epsilon, len=1, sensitivity=1){
    lam <- epsilon/sensitivity
    sign <- 1-2*as.integer(runif(len) < 0.5)
    return(rexp(len, lam)*sign)
}
claplace <- function(x, lambda){
    if(x < 0)
        return(0.5*exp(x*lambda))
    else
        return(1-0.5*exp(-x*lambda))
        
}
#Noises a vector of histogram values (sensitivity 1, final value cannot be negative)
hist_noiser <- function(vals, epsilon=0){
    if(epsilon==0)
        return(vals)
    fuzz <- vals + laplacian(epsilon, length(vals))
    count <- sum(fuzz < 0)
    while(count > 0){
        fuzz[fuzz < 0] <- laplacian(epsilon, count)
        count <- sum(fuzz < 0)
    }
    return(fuzz)
}

#Picks which conditional utility function to use, factor or numeric
#name, ent, split, type
cond_eval <- function(name, data, range_bounds, ys){
    xs <- get(name, data)
    r <- 0
    if(class(xs) == 'numeric'){
        b <- get(name, range_bounds)
        r <- cond_eval_num(xs, ys, min=b$min, max=b$max)
    }else if(class(xs) == 'factor')
        r <- cond_eval_fac(xs, ys)
    r$name <- name
    return(r)
}
#Produces a private decision tree (Helper function)
decision_tree_private <- function(data, range_bounds, pred, attrs, node, eps, d){
    p <- get(pred, data)
    sizes <- lapply(data, function(col){
        if(class(col) == 'numeric')
            return(2)
        return(length(levels(col)))
    })
    t <- sizes[attrs] %>% reduce(`max`)
    C <- get(pred, sizes)
    nrows <- hist_noiser(nrow(data), eps)
    node$nrows <- nrows
    if(length(attrs) == 0 || d <= 0 || nrows / (C*t) < sqrt(2) / eps && eps != 0){
        m <- data.frame(table(p))
        node$counts <-  hist_noiser(m$Freq, eps) #m$Freq+laplacian(eps, nrow(m))
        idx <- which.max(node$counts)
        guess <- m$p[idx]
        node$guess <- guess
        node$name <- paste(node$name, guess, sep=';')
        return()
    }
    atp <- lapply(attrs, function(name) cond_eval(name, data, range_bounds, p))
    best_attr <- exp_mech(atp, sapply(atp, function(x) -x$ent), eps, ent_util$sens)
    node$name <- paste(node$name, best_attr$name, sep=';')
    ba <- best_attr$name
    nl <- attrs[attrs!= ba]
    if(best_attr$type == 'numeric'){
        s1 <- get(ba, data) < best_attr$split
        s <- best_attr$split
        decision_tree_private(data[s1,], range_bounds, pred, nl, node$AddChild(paste("<=", s, sep='')), eps, d-1)
        decision_tree_private(data[!s1,], range_bounds, pred, nl, node$AddChild(paste(">", s, sep='')), eps, d-1)
        node$pars <- c(function(d){return(d <= s)}, function(d){return(d > s)})
    }else if(best_attr$type == 'factor'){
        p <- get(ba, data)
        dec_f <- lapply(best_attr$split, equality)
        for(a in best_attr$split){
            decision_tree_private(data[p == a,], range_bounds, pred, nl, node$AddChild(a), eps, d-1)
        }
        node$pars <- dec_f
    }
    node$attr <- ba
}
#Returns a private decision tree (call this one)
dt_private <- function(db, B, d=3){
    dt_p <- Node$new('')
    eps <- B/(2*(d+1))
    decision_tree_private(db$data, db$rng, db$y_names, db$x_names, dt_p, eps, d)
    return(dt_p)
}

#Predicts the output for a dataset given the 
predict <- function(t, D, default){
    preds <- rep(default, nrow(D))
    if(nrow(D) == 0)
        return(preds)
    if(!is.null(t$guess)){
        return(rep(t$guess, nrow(D)))
    }
    for(i in 1:length(t$pars)){
        mask <- t$pars[[i]](get(t$attr, D))
        preds[mask] <- predict(t$children[[i]], D[mask, ], default)
    }
    return(preds)
}

#Pruning helper method
top_down <- function(t, ratio){
    t$nrows <- ratio*t$nrows
    if(is.null(t$guess)){
        tot_child <- 0
        for(c in t$children){
            tot_child <- tot_child + c$nrows
        }
        for(c in t$children){
            top_down(c, t$nrows / tot_child)
        }
    }
}
#Pruning helper method
bottom_up <- function(t){
    if(!is.null(t$guess)){
        t$counts <- t$counts * t$nrows / sum(t$counts)
    }else{
        cnts <- 0
        for(c in t$children){
            cnts <- cnts + bottom_up(c)
        }
        t$counts <- cnts * t$nrows / sum(cnts)
    }
    err_rate <- 1-max(t$counts) / t$nrows
    t$pess_err_rate <- err_rate + 0.67 * sqrt(err_rate * (1-err_rate) / t$nrows)
    return(t$counts)
}
#Decides if a node should be pruned using the method of C4.5
prune <- function(t, lvls){
    cond_err <- 0
    if(is.null(t$guess)){
        for(c in t$children){
            cond_err <- cond_err + c$pess_err_rate * c$nrows / t$nrows
        }
    }
    if(cond_err > t$pess_err_rate){
        t$guess <- lvls[t$counts == max(t$counts)][[1]]
        t$children <- NULL
    }else{
        for(c in t$children)
            prune(c, lvls)
    }
}
#Does a full prune (call this one)
prune_tree <- function(t, lvls){
    lvls <- as.factor(lvls)
    top_down(t, 1)
    bottom_up(t)
    prune(t, lvls)
}

#Tests the private decision tree
test_private <- function(dtp, data, y_name, cutoff=0.8){
    ys <- get(y_name, data)
    pres <- predict(dtp, data, as.factor(levels(ys)))
    return(sum(pres == ys) / nrow(data))
}
