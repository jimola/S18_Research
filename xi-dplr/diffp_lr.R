library(compiler);
library(foreign);
library(pROC);
library(stringr);

vkoattr <- c("vkorc1=CT", "vkorc1=TT");
cypattr <- c("cyp2c9=13", "cyp2c9=12", "cyp2c9=23", "cyp2c9=33", "cyp2c9=22");

gen.chunks <- function(data, ntrain=24, ntest=1, trainsize=105, testsize=124) {
    
    train.chunks <- list();
    data <- data[sample(nrow(data)),];

    test.chunk <- data[1:testsize,];
    rownames(test.chunk) <- 1:nrow(test.chunk)    
    offset <- testsize + 1;
    
    for(i in 1:ntrain) {
        
        train.chunks[[i]] <- data[offset:(offset+trainsize-1),];
        rownames(train.chunks[[i]]) <- 1:nrow(train.chunks[[i]]);
        offset <- offset + trainsize;        
    }
    
    return(list(trains=train.chunks, test=test.chunk));
}

normalize <- function(data, class) {
    
    data <- as.matrix(data);
    
    mins <- apply(as.array(dimnames(data)[[2]]), 1, function(x) min(data[,x]));
    maxs <- apply(as.array(dimnames(data)[[2]]), 1, function(x) max(data[,x]));
    diff <- maxs - mins;
    diff[diff == 0] <- 1;
    
    data <- sweep(data, 2, mins);
    data <- sweep(data, 2, diff, '/');
    data <- 2*(data - 0.5);
    
    maxnorm <- max(apply(data, 1, function(x) sqrt(sum(x[1:(length(x)-1)]^2))));
    normvect <- append(numeric(length(dimnames(data)[[2]])-1)+maxnorm, 1);    
    data <- sweep(data, 2, normvect, '/');
    
    classidx <- match(class, dimnames(data)[[2]]);
    
    return(list(data = data, untransform = function(x) zapsmall(((x*normvect)/2+0.5)*diff+mins, digits=10), dtransform = function(x) (2*((x-mins)/diff-0.5))/normvect, runtransform = function(x) zapsmall(((x*normvect[classidx])/2+0.5)*diff[classidx]+mins[classidx]), digits=10));  
}

# samples a Laplace-distributed variate
laprnd <- function(n, mu, b) {
    
    # inverse transform
    u <- runif(n, 0, 1) - 0.5;
    y <- mu - b * sign(u) * log(1 - 2*abs(u));
    
    return(y);
}

# samples a noise vector for Xi's algorithm
noiseVect.sample <- function(lambda, n, epsilon, R, d) {

    # uniformly sample a point on the unit ball
    variates.norm <- rnorm(d);
    variates.norm <- variates.norm/sqrt(sum(variates.norm^2));    

    # sample the l2-norm of the noise vector
    vector.l2norm <- rexp(1, rate=(lambda*n*epsilon)/(12*R+8));    

    # scale the ball point to the sampled l2-norm
    sample <- vector.l2norm*variates.norm;    

    return(sample);
}

diffpLR.reverse.xi <- function(data, class, guess, epsilon, R, lambda) {

    odata <- data;
    data <- as.matrix(data);
    dnames <- dimnames(data)[[2]];
    
    norm <- normalize(data, class);
    data <- norm$data;
    n <- nrow(data);
    
    trainx <- data[,!(dimnames(data)[[2]] %in% class)];
    trainy <- as.matrix(data[,class]);

    d <- length(dimnames(data)[[2]])-1;    
    
    noiseobj <- function(w) {

        normw <- sqrt(sum(w^2));
        if(normw > R) {            
            w <- w/(R*normw);
        }        

        w <- as.matrix(w);

        return((1/n)*sum(((trainx %*% w) - trainy)^2) + (lambda/2)*sum(w^2));
    }    
    
    w0 <- as.matrix(rep(0, d));    
        
    optrsol <- optim(w0, noiseobj, method = "BFGS")$par;
    
    normw <- sqrt(sum(optrsol^2));
    if(normw > R) {
        optrsol <- optrsol/(R*normw);
    }

    if(epsilon > 0) {
        noise <- noiseVect.sample(lambda, n, epsilon, R, d);        
        optrsol <- optrsol + noise;
    }

    w <- optrsol[1:d];

    trueval <- norm$dtransform(numeric(length(data[1,]))+1)[1];    

    nonclassnames <- dnames[!(dnames %in% class)];
    nonguessnames <- dnames[!(dnames %in% guess) & !(dnames %in% class)];
    knownnames <- nonclassnames[!(nonclassnames %in% guess)];

    fmls <- apply(as.array(guess), 1, function(name) sprintf("xm[,%d] <= 0", match(name,nonclassnames)));
    fmla <- paste(fmls, collapse=" & ");

    resids <- data[,class] - (data[,nonclassnames] %*% w);
    errmean <- as.numeric(mean(resids));
    errvar <- as.numeric(var(resids));
    errsd <- as.numeric(sd(resids));
    errscale <- sqrt(errvar/2);

    clfun <- cmpfun(function(x) {
        
        x <- norm$dtransform(as.numeric(x));
        
        if(length(x) == d+1) {
            x <- x[!(dimnames(data)[[2]] %in% class)];
        }
        
        return(norm$runtransform(x %*% w));
    });
        
    if(length(guess) > 0) {
        clfun.inverse <- cmpfun(function(x) {
                    
            xorig <- x;             
            x[guess] <- 0;      
                                    
            xorig.transf <- norm$dtransform(as.numeric(xorig));            
            x <- norm$dtransform(as.numeric(x));
            if(length(x) == d+1) {
                dose <- x[match(class, dnames)];
                x <- x[match(nonclassnames, dnames)];
            }
            
            known <- x[match(knownnames, nonclassnames)];
            names(known) <- knownnames;
        
            xm <- matrix(data=x, nrow=length(guess)+1, ncol=length(x), byrow=TRUE);
            for(name in guess) xm[match(name, guess)+1, match(name, dnames)] <- trueval;    
                    
            xmnonmiss <- NA;
            if(length(guess) <= 1) {
                xmnonmiss <- data.frame(xm[,match(guess, nonclassnames)]);
                names(xmnonmiss) <- guess;      
                xmnonmiss <- as.matrix(xmnonmiss);
            } else {
                xmnonmiss <- xm[,match(guess, nonclassnames)];
                colnames(xmnonmiss) <- guess;                   
            }
            
            recon <- merge(t(as.matrix(known)), xmnonmiss);
            xm <- as.matrix(recon[nonclassnames]);

            probs <- apply(as.array(1:nrow(xm)), 1, function(j) getSampleProbability(xm[j,]));
            finalprobs <- dnorm(dose - (xm %*% w), 0, errscale, log=TRUE);            

            whichwild <- which(eval(parse(text=fmla)));
            totmass <- sum(finalprobs[whichwild]);
            totmass <- append(totmass, apply(as.array(guess), 1,
                function(name) sum(finalprobs[which(xm[,match(name, dnames)] > 0)])
            ));
            winner <- which.max(totmass)-1;

            return(winner);
        });
    } else {
        clfun.inverse <- function(x) { return(NA); };
    }
    
    return(list(forward=clfun, inversion=clfun.inverse));
}

vkorc1_nom <- function(s) { 
    
    attrs <- c("vkorc1=CT", "vkorc1=TT");
    
    if(sum(s[attrs]) <= 0) { 
        return("vkorc1=CC");
    } else { 
        return(attrs[which.max(s[attrs])]); 
    }
}

getSampleProbability <- function(sample) {

    if(vkorc1_nom(sample) == "vkorc1=CT") {
        return(0.3633384);
    } else if(vkorc1_nom(sample) == "vkorc1=TT") {
        return(0.3323248);
    } else {
        return(0.3043369);
    }    
}

regstats <- function(clfun, data, target) {
    
    sqerrs <- numeric(0);
    abserrs <- numeric(0);
    relerrs <- numeric(0);
    
    data <- as.matrix(data);
    
    for(i in 1:nrow(data)) {
        
        diff <- data[i,target] - clfun(data[i,]);
        
        sqerrs[i] <- diff^2;
        abserrs[i] <- abs(diff);
        relerrs[i] <- abs(diff / data[i,target]);       
    }
    
    return(list(square = mean(sqerrs), absolute = mean(abserrs), relative = mean(relerrs), cirelative = qt(0.975, df=length(relerrs)-1)*sd(relerrs)/sqrt(length(relerrs))));
}

regstats.predict <- function(clfun, data, target) {
    
    sqerrs <- numeric(0);
    abserrs <- numeric(0);
    relerrs <- numeric(0);    
    
    for(i in 1:nrow(data)) {
        
        diff <- data[i,target] - predict(clfun, data[i,]);
        
        sqerrs[i] <- diff^2;
        abserrs[i] <- abs(diff);
        relerrs[i] <- abs(diff / data[i,target]);       
    }
    
    return(list(square = mean(sqerrs), absolute = mean(abserrs), relative = mean(relerrs), cirelative = qt(0.975, df=length(relerrs)-1)*sd(relerrs)/sqrt(length(relerrs))));
}

getVectorAttrVal <- function(sample, guess) {
    whichguess <- which(sample[guess] > 0);
    if(length(whichguess) == 0) {
        return(0);
    } else {
        return(whichguess)
    }    
}

gm_mean <- function(x, na.rm=TRUE){
  exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
}

rstatsrev <- function(clfun, data, guess) {
    
    abserrs <- numeric(0);
    nguesstypes <- length(guess);
    errortypes <- matrix(0, nrow = nguesstypes+1, ncol = nguesstypes+1);
    
    data <- as.matrix(data);

    truth <- numeric(0);
    predicted <- numeric(0);
    
    for(i in 1:nrow(data)) {
        
        rguess <- clfun(data[i,]);
        real <- getVectorAttrVal(data[i,], guess);
        abserrs[i] <- ifelse(real == rguess, 1, 0);
        errortypes[real+1, rguess+1] <- errortypes[real+1, rguess+1] + 1;
        truth <- append(truth, real);
        predicted <- append(predicted, rguess);
        
    }
    
    return(list(acc=mean(abserrs), auc=as.numeric(multiclass.roc(response=truth, predictor=predicted)$auc), confusion=errortypes));
}

diffpLR.xi.tuned <- function(train, class, guess=vkoattr, epsilon=1.0, trainsize=105, testsize=124, params.R=c(0.25,0.5,1), params.lambda=c(0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256)) {

    chunks <- gen.chunks(train, trainsize=trainsize, testsize=testsize);
    errs <- numeric(0);
    utility <- numeric(0);
    utility.params <- list();
    i <- 0;

    epsilon.expmech <- epsilon/(2*max(params.R)^2);

    for(param.R in params.R) {
        for(param.lambda in params.lambda) {
            i <- i + 1;            
            clfun.xi <- diffpLR.reverse.xi(chunks$trains[[i]], class, guess, epsilon, R=param.R, lambda=param.lambda);
            errs[i] <- regstats(clfun.xi$forward, chunks$test, class)$square;
            utility[i] <- exp(-1*epsilon.expmech*errs[i]);
            utility.params[[i]] <- list(clfun=clfun.xi, R=param.R, lambda=param.lambda, err=errs[i]);
        }
    }

    utility.sorted <- sort(utility/sum(utility), index.return=TRUE);
    utility <- utility.sorted$x;
    ix <- utility.sorted$ix;

    sample.space <- numeric(0);
    for(i in 1:length(utility)) {
        sample.space[i] <- sum(utility[1:i]);
    }    

    variate.unif <- runif(1);
    variate.mechexp <- min((1:length(sample.space))[sample.space > variate.unif]);    

    return(utility.params[[ix[variate.mechexp]]]);
}

run.theoretical <- function(train, test, class, guess, display=TRUE, trials=30, seed=1, epsilons=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,100), param.R=1) {

    set.seed(seed);

    errs <- matrix(NA, nrow=length(epsilons), ncol=2);
    rownames(errs) <- epsilons;
    colnames(errs) <- c("squared", "relative");

    inversion <- matrix(NA, nrow=length(epsilons), ncol=2);
    rownames(inversion) <- epsilons;
    colnames(inversion) <- c("acc", "auc");

    d <- ncol(train)-1;
    n <- nrow(train);

    for(epsilon in epsilons) {    
        squared <- numeric(0);
        relative <- numeric(0);
        acc <- numeric(0);
        auc <- numeric(0);
        if(display) {
            print(sprintf("epsilon=%f ------------------------", epsilon));
        }
        for(i in 1:trials) {

            param.lambda = sqrt(d/(n*epsilon));

            model <- diffpLR.reverse.xi(train, class, guess, epsilon, R=param.R, lambda=param.lambda);
            forward <- model$forward;
            inverse <- model$inversion;

            fstats <- regstats(forward, test, class);
            istats <- rstatsrev(inverse, train, guess);

            squared[i] <- fstats$square;
            relative[i] <- fstats$relative;
            acc[i] <- istats$acc;
            auc[i] <- istats$auc;

            if(display) {
                print(sprintf("squared=%f, relative=%f, acc=%f, auc=%f", median(squared), mean(relative), mean(acc), mean(auc)));
            }
        }

        errs[as.character(epsilon), "squared"] <- median(squared);
        errs[as.character(epsilon), "relative"] <- mean(relative);
        inversion[as.character(epsilon), "acc"] <- mean(acc);
        inversion[as.character(epsilon), "auc"] <- mean(auc);
        
        if(display) {
            print(errs);
            print(inversion);
            print("------------------------------------");
        }
    }

    return(list(errs=errs, inversion=inversion));
}

run.oracle <- function(train, test, class, guess, trials=30, seed=1, epsilons=c(0.1,1,5,10,100), params.R=c(0.25,0.5,1,2), params.lambda=c(0.001,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5)) {

    set.seed(seed);

    err.results <- list();

    inversion.results <- matrix(NA, nrow=length(epsilons), ncol=2);
    rownames(inversion.results) <- epsilons;
    colnames(inversion.results) <- c("acc", "auc");

    best.results <- matrix(NA, nrow=length(epsilons), ncol=6);
    rownames(best.results) <- epsilons;
    colnames(best.results) <- c("R", "lambda", "squared", "relative", "acc", "auc");

    for(epsilon in epsilons) {    

        squared <- matrix(NA, nrow=length(params.R), ncol=length(params.lambda));
        rownames(squared) <- params.R;
        colnames(squared) <- params.lambda;

        relative <- matrix(NA, nrow=length(params.R), ncol=length(params.lambda));
        rownames(relative) <- params.R;
        colnames(relative) <- params.lambda;
        
        for(param.R in params.R) {

            for(param.lambda in params.lambda) {

                squared.cur <- numeric(0);
                relative.cur <- numeric(0);
                
                for(i in 1:trials) {
                    clfun.xi <- diffpLR.reverse.xi(train, class, guess, epsilon, R=param.R, lambda=param.lambda);                    
                    forward <- clfun.xi$forward;                    

                    fstats <- regstats(forward, test, class);                    

                    squared.cur[i] <- fstats$square;
                    relative.cur[i] <- fstats$relative;                    
                }

                print(sprintf("epsilon=%f, R=%f, lambda=%f squared=%f, relative=%f", epsilon, param.R, param.lambda, median(squared.cur), mean(relative.cur)));                

                squared[as.character(param.R), as.character(param.lambda)] <- median(squared.cur);
                relative[as.character(param.R), as.character(param.lambda)] <- mean(relative.cur);
            }
        }
        
        best.params.idx <- which(squared == min(squared), arr.ind=TRUE);
        best.err <- squared[best.params.idx];
        best.rel <- relative[best.params.idx];
        best.R <- params.R[best.params.idx[1]];
        best.lambda <- params.lambda[best.params.idx[2]];

        err.results[[sprintf("eps=%f", epsilon)]] <- list(squared=squared, relative=relative, best=list(err=best.err, R=best.R, lambda=best.lambda));

        print(sprintf("best params @ eps=%f: R=%f, lambda=%f, squared=%f", epsilon, best.R, best.lambda, best.err));

        acc <- numeric(0);
        auc <- numeric(0);        
        for(i in 1:trials) {
            clfun.xi <- diffpLR.reverse.xi(train, class, guess, epsilon, R=best.R, lambda=best.lambda);                    
            inverse <- clfun.xi$inversion;      
            
            istats <- rstatsrev(inverse, train, guess);

            acc[i] <- istats$acc;
            auc[i] <- istats$auc;    

            print(sprintf("    acc=%f, auc=%f", mean(acc), mean(auc)));        
        }

        inversion.results[as.character(epsilon), "acc"] <- mean(acc);
        inversion.results[as.character(epsilon), "auc"] <- mean(auc);

        best.results[as.character(epsilon), "R"] <- best.R;
        best.results[as.character(epsilon), "lambda"] <- best.lambda;
        best.results[as.character(epsilon), "squared"] <- best.err;
        best.results[as.character(epsilon), "relative"] <- best.rel;
        best.results[as.character(epsilon), "acc"] <- mean(acc);
        best.results[as.character(epsilon), "auc"] <- mean(auc);

        print("------------------------------------");
        print(best.results);
        print("------------------------------------");
    }

    return(list(err.results=err.results, inversion.results=inversion.results, best.results=best.results));

}

run.tuned.sizes <- function(train, test, class, guess, sizes=c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9), trials=100, seed=1, epsilons=c(0.5,1), params.R=c(0.25,0.5,1), params.lambda=c(0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256)) {

    errs <- list();
    inversion <- list();

    for(size in sizes) {

        errs.cur <- matrix(0, nrow=length(epsilons), ncol=2);
        rownames(errs.cur) <- epsilons;
        colnames(errs.cur) <- c("squared", "relative");

        inversion.cur <- matrix(0, nrow=length(epsilons), ncol=2);
        rownames(inversion.cur) <- epsilons;
        colnames(inversion.cur) <- c("acc", "auc");

        print(sprintf("training size=%f ----------------------------------------", size));

        for(i in 1:trials) {
            
            results.cur <- run.tuned(train, test, class, guess, display=FALSE, trainsize=round(105*size), testsize=round(124*size), trials=1, seed=i, epsilons=epsilons, params.R=params.R, params.lambda=params.lambda);            

            errs.cur <- errs.cur + results.cur$errs;
            inversion.cur <- inversion.cur + results.cur$inversion;
            
            print(sprintf("iter=%d", i));
            print(errs.cur/i);
            print(inversion.cur/i);
            cat("\n");
        }

        errs.cur <- errs.cur/trials;
        inversion.cur <- inversion.cur/trials;

        errs[[as.character(size)]] <- errs.cur;
        inversion[[as.character(size)]] <- inversion.cur;
    }

    return(list(errs=errs, inversion=inversion));
}    

results.to.latex <- function(results) {

    x.coords <- as.numeric(colnames(results));
    y.coords <- as.numeric(rownames(results));

    i <- 0;     

    for(x in x.coords) {
        j <- 0;
        for(y in y.coords) {
            cat(noquote(sprintf("%f %f %f\n", i, j, results[as.character(y),as.character(x)])));
            j <- j + 1;
        }
        cat(noquote(sprintf("%f %f %f\n", i, j, 0.0)));
        i <- i + 1;        
    }

    j <- 0;
    for(y in y.coords) {
        cat(noquote(sprintf("%f %f %f\n", i, j, 0.0)));
        j <- j + 1;
    }
    cat(noquote(sprintf("%f %f %f\n", i, j, 0.0)));

}

display.tuned <- function(tuned.1, tuned.2, fun, nrow=5, ncol=7) {

    res <- matrix(numeric(nrow*ncol), nrow=nrow, ncol=ncol);
    colnames(res) <- c("epsilon", "r1squared", "r1acc", "r2squared", "r2acc", "funsquared", "funacc");

    res[,"epsilon"] <- as.numeric(rownames(tuned.1$errs));
    res[,"r1squared"] <- tuned.1$errs[,"squared"];
    res[,"r1acc"] <- tuned.1$inversion[,"acc"];
    res[,"r2squared"] <- tuned.2$errs[,"squared"];
    res[,"r2acc"] <- tuned.2$inversion[,"acc"];    
    res[,"funsquared"] <- fun$errs[,"squared"];
    res[,"funacc"] <- fun$inversion[,"acc"];

    return(res);
}
