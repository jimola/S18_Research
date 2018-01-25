diffpLR.reverse.functional <- function(data, class, epsilon, guess) {
    
    odata <- data;
    data <- as.matrix(data);
    dnames <- dimnames(data)[[2]];
        
    norm <- normalize.functional(data, class);
    data <- norm$data;
    
    trainx <- data[,!(dimnames(data)[[2]] %in% class)];
    trainx <- cbind(as.matrix(trainx), numeric(nrow(trainx))+1);
    
    trainy <- data[,class];
    
    d <- length(dimnames(data)[[2]]);
    
    r0 <- t(trainx) %*% trainx;
    r1 <- -2 * (t(trainx) %*% trainy);
    
    sensitivity <- 2*d*d + 4*d;
    
    noisematrix1 <- NA;
    noisematrix2 <- NA;
    if(epsilon <= 0) {
        noisematrix1 <- matrix(0, nrow = d, ncol = d);
        noisematrix2 <- numeric(d);
    } else {
        noisematrix1 <- matrix(laprnd(d*d, 0, sensitivity * 1/epsilon), nrow = d, ncol = d);
        noisematrix2 <- laprnd(d, 0, sensitivity * 1/epsilon);
    }
    
    coe2 <- r0 + noisematrix1;
    coe2 <- 0.5*(t(coe2) + coe2);

    coe1 <- r1 + noisematrix2;

    if(epsilon > 0) {
        coe2 <- coe2 + 4 * sqrt(2) * sensitivity * (1/epsilon) * diag(d);                           
    }
    
    eigs <- eigen(coe2);
    vec <- t(apply(eigs$vectors, 1, rev));
    val <- rev(eigs$values);    
        
    del <- val >= 1e-8;    
    val <- diag(val);
    val <- val[del,];
    val <- val[,del];
        
    vec <- vec[,del];       
    
    coe2 <- val;
    coe1 <- t(vec) %*% coe1;
    
    noiseobj <- function(w) {
        return(t(w) %*% coe2 %*% w + t(coe1) %*% w);
    }
    
    g0 <- runif(d-sum(!del), 0, 1);    
        
    optrsol <- optim(g0, noiseobj, method = "BFGS", control = list(maxit=100000));
    
    g <- optrsol$par;
    
    bestw <- vec %*% g;
    
    w <- bestw[1:length(bestw)-1];
    b <- bestw[length(bestw)];
    
    trueval <- norm$dtransform(numeric(length(data[1,]))+1)[1];
    
    clfun <- cmpfun(function(x) {
        
        x <- norm$dtransform(as.numeric(x));
        
        if(length(x) == d) {
            x <- x[!(dimnames(data)[[2]] %in% class)];
        }
        
        return(norm$runtransform(b + x %*% w));
    });
    return(list(forward=clfun));
}